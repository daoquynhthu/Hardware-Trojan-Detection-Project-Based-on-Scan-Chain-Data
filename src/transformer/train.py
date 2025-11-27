import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score, precision_recall_curve, classification_report, precision_score, recall_score, roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from imblearn.over_sampling import SMOTE

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
                    # Pad with zeros
                    chunk_x = np.vstack([chunk_x, np.zeros((pad_len, feat_dim))])
                    # Pad labels with -100 (ignored index)
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
        y = self.labels[idx]
        
        if self.augment and self.augmenter:
            x = self.augmenter.augment(x)
            
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = torch.ones_like(targets, dtype=torch.float32) * (1 - self.alpha)
            alpha_t[targets == 1] = self.alpha
            loss = alpha_t * focal_term * ce_loss
        else:
            loss = focal_term * ce_loss
            
        return loss.mean()

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
    
    # Split designs
    unique_designs = np.unique(design_ids)
    np.random.seed(42)
    np.random.shuffle(unique_designs)
    
    test_size = max(1, int(len(unique_designs) * 0.2))
    test_designs = unique_designs[:test_size]
    train_designs = unique_designs[test_size:]
    
    print(f"Train designs: {train_designs}")
    print(f"Test designs: {test_designs}")
    
    mask_test = np.isin(design_ids, test_designs)
    mask_train = ~mask_test
    
    X_train, y_train, id_train = X[mask_train], y[mask_train], design_ids[mask_train]
    X_test, y_test, id_test = X[mask_test], y[mask_test], design_ids[mask_test]
    
    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    # After resampling, we need to regenerate design_ids for the new samples.
    # A simple approach is to assign the original design_id to the resampled data.
    # This is not perfect, but it's a common practice.
    id_train_res, _ = smote.fit_resample(id_train.reshape(-1, 1), y_train)
    id_train_res = id_train_res.flatten()

    # Create Datasets
    max_len = 1024
    batch_size = 8 # From original code
    
    # Disable augmentation for debugging
    train_dataset = SequenceDataset(X_train_res, y_train_res, id_train_res, max_len=max_len, augment=False)
    test_dataset = SequenceDataset(X_test, y_test, id_test, max_len=max_len, augment=False)
    
    # DataLoader arguments for GPU optimization
    # Note: On Windows, num_workers > 0 can be problematic, so we default to 0.
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    
    # Setup Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        # Enable cudnn benchmark for optimized kernel selection
        torch.backends.cudnn.benchmark = True
    
    model_args = ModelArgs(
        dim=128,
        n_layers=4, # Increased layers
        n_heads=4,
        max_seq_len=max_len,
        input_dim=X.shape[1], # Dynamic input dim
        num_classes=2,
        dropout=0.2
    )
    model = Transformer(model_args).to(device)
    
    # loss function - standard CE loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.01, ignore_index=-100).to(device)

    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Enable scaler only if CUDA is available
    use_cuda = torch.cuda.is_available()
    scaler = GradScaler(enabled=use_cuda)
    
    epochs = 100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best_f1 = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # Use mixed precision training
            if use_cuda:
                with autocast():
                    output = model(data)
                    loss = criterion(output.view(-1, 2), target.view(-1))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = criterion(output.view(-1, 2), target.view(-1))
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

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output.view(-1, 2), target.view(-1))
                val_loss += loss.item()

                # Mask out padding
                mask = target.view(-1) != -100
                preds = torch.argmax(output.view(-1, 2), dim=1)[mask]
                labels = target.view(-1)[mask]
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(test_loader)
        
        # Calculate metrics
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        auroc = roc_auc_score(all_labels, all_preds)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUROC: {auroc:.4f}")

        # Update learning rate
        scheduler.step()

                
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            save_path = os.path.join(TRANSFORMER_MODEL_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model to {save_path} with F1: {f1:.4f}")

if __name__ == "__main__":
    train_model()
