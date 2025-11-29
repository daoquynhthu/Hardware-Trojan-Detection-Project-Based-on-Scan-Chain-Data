import os
import sys
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold
from collections import Counter
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import DATA_DIR
# Note: We import model and augmentation from CURRENT directory (token_level)
from src.token_level_transformer.model import Transformer, ModelArgs
from src.token_level_transformer.augmentation import DataAugmenter

# Handle PyTorch AMP compatibility
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs: (B, C, L)
        # targets: (B, L)
        
        # Flatten
        # (B, C, L) -> (B, L, C) -> (N, C)
        inputs = inputs.transpose(1, 2).reshape(-1, inputs.shape[1])
        targets = targets.reshape(-1)
        
        # Mask ignored index
        mask = targets != self.ignore_index
        inputs = inputs[mask]
        targets = targets[mask]
        
        if len(targets) == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()

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

GradScaler = _CpuGradScaler
autocast = _CpuAutocast

try:
    from torch.amp.grad_scaler import GradScaler as _AmpGradScaler
    from torch.amp.autocast_mode import autocast as _AmpAutocast
    
    class _WrapperGradScaler(_AmpGradScaler):
        def __init__(self, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True):
            super().__init__('cuda', init_scale, growth_factor, backoff_factor, growth_interval, enabled)
            
    class _WrapperAutocast(_AmpAutocast):
        def __init__(self, enabled=True, dtype=None, cache_enabled=True):
            super().__init__(device_type='cuda', dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)

    GradScaler = _WrapperGradScaler
    autocast = _WrapperAutocast
    
except ImportError:
    try:
        from torch.cuda.amp import GradScaler as _CudaGradScaler, autocast as _CudaAutocast
        GradScaler = _CudaGradScaler
        autocast = _CudaAutocast
    except ImportError:
        pass

class TokenLevelDataset(Dataset):
    def __init__(self, X, y, design_ids, config, augment=False):
        self.samples = []
        self.labels = [] # Now stores FULL sequences
        self.max_len = config['training']['max_len']
        self.augment = augment
        aug_config = config.get('augmentation', {})
        
        self.augmenter = DataAugmenter(
            noise_std=aug_config.get('noise_std', 0.01), 
            scale_range=(aug_config.get('scale_range_min', 0.95), aug_config.get('scale_range_max', 1.05)),
            mask_prob=aug_config.get('mask_prob', 0.05)
        ) if augment else None
        
        unique_designs = np.unique(design_ids)
        
        for design_id in unique_designs:
            mask = design_ids == design_id
            X_design = X[mask]
            y_design = y[mask]
            
            N = X_design.shape[0]
            
            # Stride logic
            cursor = 0
            while cursor < N:
                end = min(cursor + self.max_len, N)
                chunk_y = y_design[cursor:end]
                
                has_trojan = np.any(chunk_y == 1)
                
                if has_trojan and self.augment: 
                    step = 32 # Dense sampling for training with Trojans
                elif has_trojan and not self.augment:
                    step = self.max_len // 2 
                else:
                    step = self.max_len // 2
                
                chunk_x = X_design[cursor:end]
                
                # Pad if necessary
                if chunk_x.shape[0] < self.max_len:
                    pad_len = self.max_len - chunk_x.shape[0]
                    feat_dim = chunk_x.shape[1]
                    chunk_x = np.vstack([chunk_x, np.zeros((pad_len, feat_dim))])
                    # Pad labels with -100 (Ignore Index)
                    chunk_y = np.concatenate([chunk_y, -100 * np.ones(pad_len)])
                
                self.samples.append(chunk_x)
                self.labels.append(chunk_y)
                
                # In-memory Augmentation for Trojan sequences
                if has_trojan and augment and self.augmenter:
                    for _ in range(2):
                        augmented_x = self.augmenter.augment(chunk_x.copy())
                        self.samples.append(augmented_x)
                        self.labels.append(chunk_y)
                
                cursor += step
                if cursor >= N: break
                        
        self.samples = np.array(self.samples, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.longlong)
        
        # Identify samples containing trojans for sampler
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
        y = self.labels[idx] # Returns (Seq_Len,)
        
        if self.augment and self.augmenter and np.random.rand() > 0.5:
             x = self.augmenter.augment(x)
            
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def get_sampler(dataset):
    class_counts = Counter(dataset.has_trojan)
    if 1 not in class_counts:
        class_counts[1] = 1
    weights = [1.0 / class_counts[label] for label in dataset.has_trojan]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return sampler

def calculate_metrics(preds, targets, ignore_index=-100):
    """
    Calculate token-level metrics, ignoring padding.
    preds: (N,)
    targets: (N,)
    """
    mask = targets != ignore_index
    preds_filtered = preds[mask]
    targets_filtered = targets[mask]
    
    if len(targets_filtered) == 0:
        return 0, 0, 0, np.zeros((2,2))

    p = precision_score(targets_filtered, preds_filtered, zero_division=0)
    r = recall_score(targets_filtered, preds_filtered, zero_division=0)
    f1 = f1_score(targets_filtered, preds_filtered, zero_division=0)
    cm = confusion_matrix(targets_filtered, preds_filtered, labels=[0, 1])
    
    return p, r, f1, cm

def train_fold(fold_idx, train_designs, test_designs, X, y, design_ids, config):
    print(f"\n=== Starting Fold {fold_idx+1} ===")
    
    mask_test = np.isin(design_ids, test_designs)
    mask_train = ~mask_test
    
    X_train, y_train, id_train = X[mask_train], y[mask_train], design_ids[mask_train]
    X_test, y_test, id_test = X[mask_test], y[mask_test], design_ids[mask_test]
    
    batch_size = config['training']['batch_size']
    
    train_dataset = TokenLevelDataset(X_train, y_train, id_train, config=config, augment=True)
    test_dataset = TokenLevelDataset(X_test, y_test, id_test, config=config, augment=False)
    
    sampler = get_sampler(train_dataset)
    
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    
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
    
    pos_weight_val = config['training'].get('pos_weight', [1.0, 5.0])
    pos_weight = torch.tensor(pos_weight_val, device=device)
    
    if config['training'].get('use_focal_loss', False):
        gamma = config['training'].get('focal_gamma', 2.0)
        print(f"Using Focal Loss (gamma={gamma})")
        criterion = FocalLoss(gamma=gamma, weight=pos_weight, ignore_index=-100).to(device)
    else:
        print(f"Using CrossEntropyLoss")
        criterion = nn.CrossEntropyLoss(weight=pos_weight, ignore_index=-100).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    scaler = GradScaler(enabled=(device=='cuda'))
    
    epochs = config['training']['epochs']
    best_f1 = 0.0
    # Initialize best_metrics with zeros to avoid KeyError if no update happens
    best_metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'cm': np.zeros((2,2))}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Fold {fold_idx+1} Epoch {epoch+1}/{epochs}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            with autocast(enabled=(device=='cuda')):
                logits = model(data) # (B, L, C)
                
                # Loss calculation: Permute logits to (B, C, L)
                ce_loss = criterion(logits.transpose(1, 2), target)
                aux = model.get_aux_loss() * model_args.moe_balance_coef if model_args.use_moe else 0.0
                loss = ce_loss + aux
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Validation
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                logits = model(data) # (B, L, C)
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1) # (B, L)
                
                all_preds.extend(preds.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        p, r, f1, cm = calculate_metrics(all_preds, all_targets)
        
        print(f"Val - P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        
        if f1 >= best_f1:
            best_f1 = f1
            best_metrics = {'precision': p, 'recall': r, 'f1': f1, 'cm': cm}
            # Save best model for this fold
            model_path = os.path.join(os.path.dirname(__file__), f"token_model_fold_{fold_idx+1}.pth")
            torch.save(model.state_dict(), model_path)
            
    return best_metrics

def train_model():
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
    
    # Balanced Split Logic (Simplified from original train.py)
    design_counts = []
    for d in unique_designs:
        mask = design_ids == d
        count = np.sum(y[mask] == 1)
        design_counts.append((d, count))
    design_counts.sort(key=lambda x: x[1], reverse=True)
    
    n_folds = 3
    folds = [[] for _ in range(n_folds)]
    fold_sums = [0] * n_folds
    for d, c in design_counts:
        min_idx = np.argmin(fold_sums)
        folds[min_idx].append(d)
        fold_sums[min_idx] += c
        
    config_path = os.path.join(os.path.dirname(__file__), 'train_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    fold_results = []
    
    for fold_idx in range(n_folds):
        test_designs = np.array(folds[fold_idx])
        train_designs = []
        for i in range(n_folds):
            if i != fold_idx:
                train_designs.extend(folds[i])
        train_designs = np.array(train_designs)
        
        metrics = train_fold(fold_idx, train_designs, test_designs, X, y, design_ids, config)
        fold_results.append(metrics)
        
    # Summary
    f1_scores = [r['f1'] for r in fold_results]
    print(f"\nAverage F1: {np.mean(f1_scores):.4f}")

if __name__ == "__main__":
    train_model()
