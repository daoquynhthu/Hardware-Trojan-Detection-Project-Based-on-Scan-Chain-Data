import os
import sys
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
# Handle PyTorch AMP compatibility
# Define dummy classes first with unique names to avoid linter confusion
class _CpuGradScaler:
    def __init__(self, enabled=False): pass
    def scale(self, loss): return loss
    def step(self, optimizer): optimizer.step()
    def update(self): pass
    def unscale_(self, optimizer): pass

class _CpuAutocast:
    def __init__(self, enabled=True, dtype=None, cache_enabled=True): pass
    def __enter__(self): pass
    def __exit__(self, exc_type, exc_val, exc_tb): pass

# Initialize with CPU fallbacks
GradScaler = _CpuGradScaler
autocast = _CpuAutocast

try:
    # Try new API first (PyTorch 2.4+)
    # Linter suggests importing from submodules directly
    from torch.amp.grad_scaler import GradScaler as _AmpGradScaler
    from torch.amp.autocast_mode import autocast as _AmpAutocast
    
    class _WrapperGradScaler(_AmpGradScaler):
        def __init__(self, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True):
            # Default to 'cuda' device for compatibility
            super().__init__('cuda', init_scale, growth_factor, backoff_factor, growth_interval, enabled)
            
    class _WrapperAutocast(_AmpAutocast):
        def __init__(self, enabled=True, dtype=None, cache_enabled=True):
            super().__init__(device_type='cuda', dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)

    GradScaler = _WrapperGradScaler
    autocast = _WrapperAutocast
    
except ImportError:
    try:
        # Fallback to legacy CUDA AMP
        from torch.cuda.amp import GradScaler as _CudaGradScaler, autocast as _CudaAutocast
        GradScaler = _CudaGradScaler
        autocast = _CudaAutocast
    except ImportError:
        # Keep CPU fallbacks
        pass

# Ensure we use the dummies if CUDA is not available at runtime, 
# although torch.cuda.amp might exist but fail or be slow.
# Actually, PyTorch's GradScaler(enabled=False) is fine on CPU.
# So we only need this fallback if the import FAILS (old PyTorch).
# If the import succeeds, we use the real classes.


from sklearn.metrics import f1_score, precision_recall_curve, classification_report, precision_score, recall_score, roc_auc_score, confusion_matrix, average_precision_score
from sklearn.model_selection import KFold
from collections import Counter
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import DATA_DIR, TRANSFORMER_MODEL_DIR
from src.transformer.model import Transformer, ModelArgs
from src.transformer.augmentation import DataAugmenter

class SequenceDataset(Dataset):
    def __init__(self, X, y, design_ids, config, augment=False):
        self.samples = []
        self.labels = []
        self.max_len = config['training']['max_len']
        self.augment = augment
        # Use updated DataAugmenter params if possible, or defaults
        aug_config = config.get('augmentation', {})
        self.augmenter = DataAugmenter(
            noise_std=aug_config.get('noise_std', 0.01), 
            scale_range=(aug_config.get('scale_range_min', 0.95), aug_config.get('scale_range_max', 1.05))
        ) if augment else None
        
        unique_designs = np.unique(design_ids)
        
        for design_id in unique_designs:
            mask = design_ids == design_id
            X_design = X[mask]
            y_design = y[mask]
            
            N = X_design.shape[0]
            # Sliding window with 50% overlap
            stride = self.max_len // 2
            
            for start in range(0, N, stride):
                end = min(start + self.max_len, N)
                chunk_x = X_design[start:end]
                chunk_y = y_design[start:end]
                
                # Sequence-level label: 1 if any token is 1, else 0
                # Ignore -100 padding (handled in __getitem__ but label is set here for oversampling check)
                # Actually, y_design here is raw tokens. -100 is not yet applied (padding logic below).
                # But dataset.pkl y is likely raw 0/1.
                seq_label = 1 if np.any(chunk_y == 1) else 0

                # Pad if necessary
                if chunk_x.shape[0] < self.max_len:
                    pad_len = self.max_len - chunk_x.shape[0]
                    feat_dim = chunk_x.shape[1]
                    chunk_x = np.vstack([chunk_x, np.zeros((pad_len, feat_dim))])
                    chunk_y = np.concatenate([chunk_y, -100 * np.ones(pad_len)])
                
                self.samples.append(chunk_x)
                self.labels.append(chunk_y)
                
                # Oversampling for Trojan samples (Augmentation at dataset creation time)
                # Note: This is different from __getitem__ augmentation.
                # Here we add *more samples* to the list.
                if seq_label == 1 and augment and self.augmenter:
                    # Add 4 augmented copies -> Total 5 samples (1 original + 4 augmented)
                    # Increased from 2 to 4 to handle data imbalance and volume
                    for _ in range(4):
                        augmented_x = self.augmenter.augment(chunk_x.copy())
                        self.samples.append(augmented_x)
                        self.labels.append(chunk_y) # Label remains same
                        
        self.samples = np.array(self.samples, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.longlong)
        
        # Identify samples with at least one Trojan label for weighted sampling
        # Note: Since we already oversampled, WeightedRandomSampler might be less critical or needs adjustment?
        # Standard practice: Use WeightedRandomSampler even with oversampling if balance is still off.
        # But here we hardcoded oversampling. Let's keep WeightedRandomSampler as safety net.
        # But we need to correctly identify trojan samples (including augmented ones).
        # self.labels contains token labels.
        
        # We need a quick way to know if a sample is trojan.
        # Since we added augmented samples with same labels, this check still works.
        # -100 check: 
        self.has_trojan = []
        for i in range(len(self.labels)):
            y_seq = self.labels[i]
            valid_mask = y_seq != -100
            is_trojan = False
            if np.any(valid_mask):
                if np.any(y_seq[valid_mask] == 1):
                    is_trojan = True
            self.has_trojan.append(is_trojan)
        self.has_trojan = np.array(self.has_trojan)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        y_seq = self.labels[idx]
        
        # Sequence-level label: 1 if any token is 1, else 0
        valid_mask = y_seq != -100
        if np.any(valid_mask):
             y = 1 if np.any(y_seq[valid_mask] == 1) else 0
        else:
             y = 0
        
        # Runtime augmentation (optional, but we already did static augmentation for Trojans)
        # If we augment here again, we double augment.
        # The user's code in Enhance.py does:
        # if seq_label == 1 and augment: ... append(augmented)
        # AND in __getitem__: if self.augment ... x = self.augmenter.augment(x)
        # This means augmented samples get augmented AGAIN in __getitem__?
        # Or maybe normal samples get augmented in __getitem__?
        # Let's follow the standard practice: 
        # If we did static augmentation, we might not want random runtime augmentation on top unless intended.
        # But Enhance.py has:
        # if self.augment and self.augmenter and np.random.rand() > 0.5: x = self.augmenter.augment(x)
        # I will enable runtime augmentation with 50% prob as per Enhance.py
        
        if self.augment and self.augmenter and np.random.rand() > 0.5:
             x = self.augmenter.augment(x)
            
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def get_sampler(dataset):
    class_counts = Counter(dataset.has_trojan)
    if 1 not in class_counts:  # 新增
        class_counts[1] = 1  # dummy
    weights = [1.0 / class_counts[label] for label in dataset.has_trojan]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return sampler

def train_fold(fold_idx, train_designs, test_designs, X, y, design_ids, config):
    print(f"\n=== Starting Fold {fold_idx+1} ===")
    print(f"Train Designs: {train_designs}")
    print(f"Test Designs: {test_designs}")
    
    mask_test = np.isin(design_ids, test_designs)
    mask_train = ~mask_test
    
    X_train, y_train, id_train = X[mask_train], y[mask_train], design_ids[mask_train]
    X_test, y_test, id_test = X[mask_test], y[mask_test], design_ids[mask_test]
    
    # Create Datasets
    batch_size = config['training']['batch_size']
    
    # 启用数据增强
    train_dataset = SequenceDataset(X_train, y_train, id_train, config=config, augment=True)
    test_dataset = SequenceDataset(X_test, y_test, id_test, config=config, augment=False)
    
    # 使用 WeightedRandomSampler
    trojan_indices = np.where(train_dataset.has_trojan)[0]
    normal_indices = np.where(np.logical_not(train_dataset.has_trojan))[0]
    
    print(f"Fold {fold_idx+1} Stats - Trojan Sequences: {len(trojan_indices)}, Normal Sequences: {len(normal_indices)}")
    
    sampler = get_sampler(train_dataset)
    
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    
    # Setup Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
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
    
    # Calculate Sequence-level class weights for Loss
    pos_weight_val = config['training'].get('pos_weight', [1.0, 5.0])
    pos_weight = torch.tensor(pos_weight_val, device=device)
    print(f"Using CrossEntropyLoss weights: {pos_weight}")
    
    criterion = nn.CrossEntropyLoss(weight=pos_weight).to(device)

    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    
    use_cuda = torch.cuda.is_available()

    scaler = GradScaler(enabled=use_cuda) 
    
    epochs = config['training']['epochs']
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=config['training'].get('scheduler_factor', 0.5), 
        patience=config['training'].get('scheduler_patience', 3)
    )
    best_f1 = 0.0
    best_metrics = {}
    patience = config['training']['patience']
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Fold {fold_idx+1} Epoch {epoch+1}/{epochs}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            if use_cuda:
                with autocast():
                    logits = model(data) # (bsz, 2)
                    ce_loss = criterion(logits, target)
                    aux = model.get_aux_loss() * model_args.moe_balance_coef if model_args.use_moe else 0.0
                    loss = ce_loss + aux
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(data)
                ce_loss = criterion(logits, target)
                aux = model.get_aux_loss() * model_args.moe_balance_coef if model_args.use_moe else 0.0
                loss = ce_loss + aux
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            aux_val = aux.item() if isinstance(aux, torch.Tensor) else aux
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'aux': f"{aux_val:.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.6f}"})

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
                
                # Store probabilities for class 1 for threshold tuning
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(target.cpu().numpy())

        avg_val_loss = val_loss / len(test_loader) if len(test_loader) > 0 else 0
        
        # Calculate metrics with dynamic thresholding
        if len(all_labels) > 0:
            all_labels = np.array(all_labels)
            all_probs = np.array(all_probs)
            
            # Find optimal threshold for F1
            precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
            f1_scores = 2 * recalls * precisions / (recalls + precisions + 1e-10)
            
            # Handle edge case where thresholds might be empty or all nan
            if len(f1_scores) > 0:
                best_idx = np.argmax(f1_scores)
                best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
                best_epoch_f1 = f1_scores[best_idx]
            else:
                best_threshold = 0.5
                best_epoch_f1 = 0.0

            # Apply best threshold
            preds = (all_probs >= best_threshold).astype(int)
            
            # Standard metrics
            precision = precision_score(all_labels, preds, zero_division=0)
            recall = recall_score(all_labels, preds, zero_division=0)
            try:
                auroc = roc_auc_score(all_labels, all_probs)
                auprc = average_precision_score(all_labels, all_probs)
            except:
                auroc = 0.0
                auprc = 0.0
            cm = confusion_matrix(all_labels, preds)
            
            # Log validation results
            print(f"  Val Loss: {avg_val_loss:.4f} | F1: {best_epoch_f1:.4f} (Thresh: {best_threshold:.3f}) | AUC: {auroc:.4f} | AUPRC: {auprc:.4f}")
            
            if best_epoch_f1 > best_f1:
                best_f1 = best_epoch_f1
                patience_counter = 0
                
                best_metrics = {
                    'f1': best_f1,
                    'precision': precision,
                    'recall': recall,
                    'auroc': auroc,
                    'auprc': auprc,
                    'cm': cm,
                    'epoch': epoch,
                    'threshold': best_threshold
                }
                
                # Only save model for the first fold to have a "best" artifact on disk
                if fold_idx == 0:
                    save_path = os.path.join(TRANSFORMER_MODEL_DIR, "best_model.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'f1': best_f1,
                        'model_args': model_args,
                        'threshold': best_threshold # Save threshold for inference
                    }, save_path)
            else:
                patience_counter += 1
                
            scheduler.step(best_epoch_f1)
            
            if patience_counter >= patience:
                print(f"  Fold {fold_idx+1}: Early stopping at epoch {epoch+1}")
                break
        
    print(f"Fold {fold_idx+1} Completed. Best F1: {best_f1:.4f}")
    if best_metrics:
        print(f"Best Metrics: F1={best_metrics['f1']:.4f}, Prec={best_metrics['precision']:.4f}, Rec={best_metrics['recall']:.4f}, AUPRC={best_metrics['auprc']:.4f}")
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
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), 'train_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded config from {config_path}")
    else:
        print(f"Config not found at {config_path}, using default hardcoded values (fallback)")
        # Fallback config if file missing (safety)
        config = {
            "training": {
                "batch_size": 16, "epochs": 50, "lr": 1e-4, "weight_decay": 0.01,
                "patience": 15, "max_len": 512, "pos_weight": [1.0, 5.0]
            },
            "model": {
                "dim": 256, "n_layers": 4, "n_heads": 8, "dropout": 0.2,
                "use_moe": True
            },
            "augmentation": {}
        }

    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(unique_designs)):
        train_designs = unique_designs[train_idx]
        test_designs = unique_designs[test_idx]
        
        metrics = train_fold(fold_idx, train_designs, test_designs, X, y, design_ids, config)
        if metrics:
            fold_results.append(metrics)
            
    # Aggregate Results
    print("\n" + "="*50)
    print("Cross-Validation Completed!")
    
    f1_scores = [r['f1'] for r in fold_results]
    print(f"F1 Scores per Fold: {[f'{s:.4f}' for s in f1_scores]}")
    print(f"Average F1 Score: {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")
    
    # Sum Confusion Matrices
    total_cm = np.sum(np.array([r['cm'] for r in fold_results]), axis=0)
    print("Aggregated Confusion Matrix:")
    print(total_cm)

if __name__ == "__main__":
    train_model()
