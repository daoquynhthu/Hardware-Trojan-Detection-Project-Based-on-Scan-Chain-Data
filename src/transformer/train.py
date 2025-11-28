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
        y = self.labels[idx]
        
        if self.augment and self.augmenter:
            x = self.augmenter.augment(x)
            
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

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
    
    # 分析类别分布
    unique, counts = np.unique(y, return_counts=True)
    print(f"Label distribution: {dict(zip(unique, counts))}")
    trojan_ratio = counts[1] / counts[0] if len(counts) > 1 else 0
    print(f"Trojan/Normal ratio: {trojan_ratio:.4f}")
    
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
    
    # Create Datasets
    max_len = 512
    batch_size = 16
    
    # 启用数据增强，但不要太激进
    train_dataset = SequenceDataset(X_train, y_train, id_train, max_len=max_len, augment=True)
    test_dataset = SequenceDataset(X_test, y_test, id_test, max_len=max_len, augment=False)
    
    # 使用 WeightedRandomSampler 来处理类别不平衡
    # 确保每个 Batch 中包含 Trojan 的样本和不包含的样本比例均衡
    trojan_indices = np.where(train_dataset.has_trojan)[0]
    normal_indices = np.where(~train_dataset.has_trojan)[0]
    
    print(f"Trojan sequences: {len(trojan_indices)}, Normal sequences: {len(normal_indices)}")
    
    weights = np.zeros(len(train_dataset))
    # 给予 Trojan 样本更高的权重，使得它们被采样的概率与 Normal 样本相当
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
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    
    model_args = ModelArgs(
        dim=128,
        n_layers=2, # 回到 2 层以加快收敛和避免过拟合
        n_heads=4,
        max_seq_len=max_len,
        input_dim=X.shape[1],
        num_classes=2,
        dropout=0.1
    )
    model = Transformer(model_args).to(device)
    
    # 计算 Token 级别的类别权重
    # 虽然 Sampler 平衡了序列，但序列内部大部分是 Normal Token
    print("Calculating token-level class weights...")
    all_train_labels = np.concatenate([y for _, y in train_dataset])
    valid_mask = all_train_labels != -100
    valid_labels = all_train_labels[valid_mask]
    n_neg = (valid_labels == 0).sum()
    n_pos = (valid_labels == 1).sum()
    
    if n_pos > 0:
        pos_weight = n_neg / n_pos
        print(f"Token stats: Neg={n_neg}, Pos={n_pos}, Calculated Pos Weight={pos_weight:.2f}")
        # 为了稳定性，可以限制最大权重，例如 100
        pos_weight = min(pos_weight, 100.0) 
    else:
        print("Warning: No positive tokens in training set!")
        pos_weight = 1.0
        
    token_class_weights = torch.tensor([1.0, float(pos_weight)], device=device)
    print(f"Using CrossEntropyLoss weights: {token_class_weights}")
    
    criterion = nn.CrossEntropyLoss(weight=token_class_weights, ignore_index=-100).to(device)

    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    use_cuda = torch.cuda.is_available()
    scaler = GradScaler(enabled=use_cuda) # 使用默认参数
    
    epochs = 100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best_f1 = 0.0
    patience = 20
    patience_counter = 0
    
    # Initialize metric containers for final evaluation scope
    all_preds = []
    all_labels = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
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
        all_probs = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output.view(-1, 2), target.view(-1))
                val_loss += loss.item()

                # Mask out padding
                mask = target.view(-1) != -100
                if mask.sum() > 0:
                    probs = torch.softmax(output.view(-1, 2), dim=1)[mask]
                    preds = torch.argmax(probs, dim=1)
                    labels = target.view(-1)[mask]
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs[:, 1].cpu().numpy())

        avg_val_loss = val_loss / len(test_loader) if len(test_loader) > 0 else 0
        
        # Calculate metrics
        if len(all_labels) > 0:
            f1 = f1_score(all_labels, all_preds, zero_division=0)
            precision = precision_score(all_labels, all_preds, zero_division=0)
            recall = recall_score(all_labels, all_preds, zero_division=0)
            
            if len(np.unique(all_labels)) > 1:
                auroc = roc_auc_score(all_labels, all_probs)
            else:
                auroc = 0.0
                
            pred_dist = Counter(all_preds)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            print(f"  F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUROC: {auroc:.4f}")
            print(f"  Predictions - Normal: {pred_dist.get(0, 0)}, Trojan: {pred_dist.get(1, 0)}")

            scheduler.step()
                    
            # Save best model
            if f1 > best_f1:
                best_f1 = f1
                patience_counter = 0
                save_path = os.path.join(TRANSFORMER_MODEL_DIR, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'f1': f1,
                    'model_args': model_args
                }, save_path)
                print(f"  ✓ Saved new best model with F1: {f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {patience} epochs without improvement")
                    break
        else:
            print(f"Epoch {epoch+1}/{epochs}: No validation data found.")

    # Final evaluation
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Best F1 Score: {best_f1:.4f}")
    if len(all_labels) > 0:
        print(classification_report(all_labels, all_preds, target_names=['Normal', 'Trojan'], zero_division=0))

if __name__ == "__main__":
    train_model()
