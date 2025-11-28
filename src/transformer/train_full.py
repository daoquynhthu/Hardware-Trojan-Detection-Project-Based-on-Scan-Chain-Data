import os
import sys
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, precision_score, recall_score, f1_score

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import DATA_DIR, TRANSFORMER_MODEL_DIR
from src.transformer.model import Transformer, ModelArgs
# Import from train.py to reuse Dataset and Utilities
from src.transformer.train import SequenceDataset, FocalLoss, GradScaler, autocast, get_sampler

def train_full():
    # Create output directory
    os.makedirs(TRANSFORMER_MODEL_DIR, exist_ok=True)
    
    dataset_path = os.path.join(DATA_DIR, "dataset.pkl")
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Attempting to build...")
        try:
            from src.dataset_builder import build_dataset
            build_dataset()
        except ImportError:
            print("Error: Could not import build_dataset. Please run dataset_builder.py manually.")
            return

    print("Loading full dataset...")
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
        
    X = data["X"]
    y = data["y"]
    design_ids = data["design_ids"]
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), 'train_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded config from {config_path}")
    else:
        # Fallback
        config = {
            "training": {
                "batch_size": 16, "epochs": 50, "lr": 1e-4, "weight_decay": 0.01,
                "max_len": 512, "pos_weight": [1.0, 5.0]
            },
            "model": {
                "dim": 256, "n_layers": 4, "n_heads": 8, "dropout": 0.2,
                "use_moe": True
            },
            "augmentation": {}
        }

    # Setup Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create Full Dataset
    # Augment=True for Training to maximize generalization
    full_dataset = SequenceDataset(X, y, design_ids, config=config, augment=True)
    
    # Sampler
    sampler = get_sampler(full_dataset)
    
    batch_size = config['training']['batch_size']
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=sampler, pin_memory=pin_memory)
    
    # Setup Model
    model_cfg = config['model']
    model_args = ModelArgs(
        dim=model_cfg['dim'],
        n_layers=model_cfg['n_layers'], 
        n_heads=model_cfg['n_heads'],
        max_seq_len=config['training']['max_len'],
        input_dim=X.shape[1],
        num_classes=2,
        dropout=model_cfg['dropout'],
        use_moe=model_cfg.get('use_moe', True),
        n_routed_experts=model_cfg.get('n_routed_experts', 8),
        n_activated_experts=model_cfg.get('n_activated_experts', 2),
        n_shared_experts=model_cfg.get('n_shared_experts', 1),
        moe_inter_dim=model_cfg.get('moe_inter_dim', 256),
        moe_balance_coef=model_cfg.get('moe_balance_coef', 0.01),
        gate_temperature=model_cfg.get('gate_temperature', 0.5)
    )
    model = Transformer(model_args).to(device)
    
    # Loss
    pos_weight_val = config['training'].get('pos_weight', [1.0, 5.0])
    pos_weight = torch.tensor(pos_weight_val, device=device)
    label_smoothing = config['training'].get('label_smoothing', 0.0)
    
    if config['training'].get('use_focal_loss', False):
        gamma = config['training'].get('focal_gamma', 2.0)
        print(f"Using Focal Loss (gamma={gamma}, weights={pos_weight})")
        criterion = FocalLoss(gamma=gamma, weight=pos_weight).to(device)
    else:
        print(f"Using CrossEntropyLoss weights: {pos_weight}, Label Smoothing: {label_smoothing}")
        criterion = nn.CrossEntropyLoss(weight=pos_weight, label_smoothing=label_smoothing).to(device)
        
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    
    scaler = GradScaler(enabled=(device == 'cuda'))
    
    epochs = config['training']['epochs']
    # We can use CosineAnnealing since we don't have validation set for ReduceLROnPlateau
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"\nStarting Full Volume Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            with autocast(enabled=(device == 'cuda')):
                logits = model(data)
                ce_loss = criterion(logits, target)
                aux = model.get_aux_loss() * model_args.moe_balance_coef if model_args.use_moe else 0.0
                loss = ce_loss + aux
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            aux_val = aux.item() if isinstance(aux, torch.Tensor) else aux
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'aux': f"{aux_val:.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.6f}"})
            
        scheduler.step()
        
    # Save Final Model
    final_model_path = os.path.join(TRANSFORMER_MODEL_DIR, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal Model saved to {final_model_path}")
    
    # --- TESTING ---
    print("\n=== Performing Self-Test on Full Dataset ===")
    test_dataset = SequenceDataset(X, y, design_ids, config=config, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Find optimal threshold based on F1 (since we are evaluating on training data, this is just a check)
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
    f1_scores = 2 * recalls * precisions / (recalls + precisions + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    print(f"Best Threshold found: {best_threshold:.4f}")
    
    preds = (all_probs >= best_threshold).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(all_labels, preds, digits=4))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, preds))

if __name__ == "__main__":
    train_full()
