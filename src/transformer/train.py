import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score, precision_recall_curve, classification_report, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import KFold
from collections import Counter

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import DATA_DIR, TRANSFORMER_MODEL_DIR
from src.transformer.model import Transformer, ModelArgs
from src.transformer.augmentation import DataAugmenter

class SequenceDataset(Dataset):
    def __init__(self, X, y, design_ids, max_len=1024, augment=False):
        self.samples = []
        self.labels = []
        self.max_len = max_len
        self.augment = augment
        self.augmenter = DataAugmenter() if augment else None
        
        unique_designs = np.unique(design_ids)
        
        for design_id in unique_designs:
            mask = design_ids == design_id
            X_design = X[mask]
            y_design = y[mask]
            
            N = X_design.shape[0]
            for start in range(0, N, max_len):
                end = min(start + max_len, N)
                chunk_x = X_design[start:end]
                chunk_y = y_design[start:end]
                
                # Pad if necessary
                if chunk_x.shape[0] < max_len:
                    pad_len = max_len - chunk_x.shape[0]
                    feat_dim = chunk_x.shape[1]
                    chunk_x = np.vstack([chunk_x, np.zeros((pad_len, feat_dim))])
                    chunk_y = np.concatenate([chunk_y, -100 * np.ones(pad_len)])
                
                self.samples.append(chunk_x)
                self.labels.append(chunk_y)
                
        self.samples = np.array(self.samples, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.longlong)
        
        # Identify samples with at least one Trojan label for weighted sampling
        self.has_trojan = np.any(self.labels == 1, axis=1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        y_seq = self.labels[idx]
        
        # Sequence-level label: 1 if any token is 1, else 0
        # Ignore -100 padding
        valid_mask = y_seq != -100
        if np.any(valid_mask):
             y = 1 if np.any(y_seq[valid_mask] == 1) else 0
        else:
             y = 0
        
        if self.augment and self.augmenter:
            x = self.augmenter.augment(x)
            
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def train_fold(fold_idx, train_designs, test_designs, X, y, design_ids):
    print(f"\n=== Starting Fold {fold_idx+1} ===")
    print(f"Train Designs: {train_designs}")
    print(f"Test Designs: {test_designs}")
    
    mask_test = np.isin(design_ids, test_designs)
    mask_train = ~mask_test
    
    X_train, y_train, id_train = X[mask_train], y[mask_train], design_ids[mask_train]
    X_test, y_test, id_test = X[mask_test], y[mask_test], design_ids[mask_test]
    
    # Create Datasets
    max_len = 512
    batch_size = 16
    
    # 启用数据增强
    train_dataset = SequenceDataset(X_train, y_train, id_train, max_len=max_len, augment=True)
    test_dataset = SequenceDataset(X_test, y_test, id_test, max_len=max_len, augment=False)
    
    # 使用 WeightedRandomSampler
    trojan_indices = np.where(train_dataset.has_trojan)[0]
    normal_indices = np.where(~train_dataset.has_trojan)[0]
    
    print(f"Fold {fold_idx+1} Stats - Trojan Sequences: {len(trojan_indices)}, Normal Sequences: {len(normal_indices)}")
    
    weights = np.zeros(len(train_dataset))
    if len(trojan_indices) > 0:
        weights[trojan_indices] = 0.5 / len(trojan_indices)
        weights[normal_indices] = 0.5 / len(normal_indices)
    else:
        weights[:] = 1.0 / len(train_dataset)
        
    sampler = WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=len(weights),
        replacement=True
    )
    
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    
    # Setup Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_args = ModelArgs(
        dim=128,
        n_layers=2, 
        n_heads=4,
        max_seq_len=max_len,
        input_dim=X.shape[1],
        num_classes=2,
        dropout=0.1
    )
    model = Transformer(model_args).to(device)
    
    # Calculate Sequence-level class weights for Loss
    n_pos = len(trojan_indices)
    n_neg = len(normal_indices)
    
    if n_pos > 0:
        pos_weight = n_neg / n_pos
        # Clamp weight to prevent instability
        pos_weight = min(pos_weight, 5.0) 
    else:
        pos_weight = 1.0
        
    seq_class_weights = torch.tensor([1.0, float(pos_weight)], device=device)
    print(f"Using CrossEntropyLoss weights: {seq_class_weights}")
    
    criterion = nn.CrossEntropyLoss(weight=seq_class_weights).to(device)

    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    use_cuda = torch.cuda.is_available()
    scaler = GradScaler(enabled=use_cuda) 
    
    epochs = 50 # Reduced epochs per fold for speed
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best_f1 = 0.0
    best_metrics = {}
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            if use_cuda:
                with autocast():
                    output = model(data) # (bsz, 2)
                    loss = criterion(output, target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()

                probs = torch.softmax(output, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        avg_val_loss = val_loss / len(test_loader) if len(test_loader) > 0 else 0
        
        # Calculate metrics
        if len(all_labels) > 0:
            f1 = f1_score(all_labels, all_preds, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                patience_counter = 0
                
                precision = precision_score(all_labels, all_preds, zero_division=0)
                recall = recall_score(all_labels, all_preds, zero_division=0)
                try:
                    auroc = roc_auc_score(all_labels, all_probs)
                except:
                    auroc = 0.0
                cm = confusion_matrix(all_labels, all_preds)
                
                best_metrics = {
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'auroc': auroc,
                    'cm': cm,
                    'epoch': epoch
                }
                
                # Only save model for the first fold to have a "best" artifact on disk
                if fold_idx == 0:
                    save_path = os.path.join(TRANSFORMER_MODEL_DIR, "best_model.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'f1': f1,
                        'model_args': model_args
                    }, save_path)
            else:
                patience_counter += 1
                
            scheduler.step()
            
            if patience_counter >= patience:
                print(f"  Fold {fold_idx+1}: Early stopping at epoch {epoch+1}")
                break
        
    print(f"Fold {fold_idx+1} Completed. Best F1: {best_f1:.4f}")
    if best_metrics:
        print(f"Best Metrics: F1={best_metrics['f1']:.4f}, Prec={best_metrics['precision']:.4f}, Rec={best_metrics['recall']:.4f}")
        print(f"Confusion Matrix:\n{best_metrics['cm']}")
    
    return best_metrics

def train_model():
    # Create output directory
    os.makedirs(TRANSFORMER_MODEL_DIR, exist_ok=True)
    
    dataset_path = os.path.join(DATA_DIR, "dataset.pkl")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    print("Loading dataset...")
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
        
    X = data["X"]
    y = data["y"]
    design_ids = data["design_ids"]
    
    unique_designs = np.unique(design_ids)
    print(f"Total Designs: {len(unique_designs)}")
    
    # 3-Fold Cross Validation
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(unique_designs)):
        train_designs = unique_designs[train_idx]
        test_designs = unique_designs[test_idx]
        
        metrics = train_fold(fold_idx, train_designs, test_designs, X, y, design_ids)
        if metrics:
            fold_results.append(metrics)
            
    # Aggregate Results
    print("\n" + "="*50)
    print("Cross-Validation Completed!")
    
    f1_scores = [r['f1'] for r in fold_results]
    print(f"F1 Scores per Fold: {[f'{s:.4f}' for s in f1_scores]}")
    print(f"Average F1 Score: {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")
    
    # Sum Confusion Matrices
    total_cm = np.sum([r['cm'] for r in fold_results], axis=0)
    print("Aggregated Confusion Matrix:")
    print(total_cm)

if __name__ == "__main__":
    train_model()
