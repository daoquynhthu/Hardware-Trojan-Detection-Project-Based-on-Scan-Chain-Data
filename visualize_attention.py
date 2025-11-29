import os
import sys
import pickle
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR, TRANSFORMER_MODEL_DIR
from src.transformer.model import Transformer, ModelArgs
from src.transformer.train import SequenceDataset

def visualize_attention():
    # 1. Load Config and Model
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'transformer', 'train_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Recreate Model Args
    model_cfg = config['model']
    # We need input_dim from dataset, let's load dataset first
    
    dataset_path = os.path.join(DATA_DIR, "dataset.pkl")
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
    
    X = data["X"]
    y = data["y"] # Full label sequence
    design_ids = data["design_ids"]
    
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
    
    # Load Weights
    model_path = os.path.join(TRANSFORMER_MODEL_DIR, "final_model.pth")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
        
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully.")

    # 2. Prepare Dataset (No Augmentation)
    dataset = SequenceDataset(X, y, design_ids, config=config, augment=False)
    
    # 3. Find Trojan Samples
    print("Searching for Trojan samples...")
    trojan_indices = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label == 1:
            trojan_indices.append(i)
            if len(trojan_indices) >= 5: # Just take 5 examples
                break
    
    print(f"Found {len(trojan_indices)} Trojan samples for visualization.")
    
    # 4. Visualize
    output_dir = "attention_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    for idx in trojan_indices:
        # Get data
        x_tensor, label = dataset[idx]
        # Add batch dim
        x_batch = x_tensor.unsqueeze(0).to(device)
        
        # Run Inference
        with torch.no_grad():
            logits, attentions = model(x_batch, return_attentions=True)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            
        # Extract Attentions
        # pooling_weights: (1, seqlen, 1) -> (seqlen,)
        pooling_weights = attentions['pooling_weights'].squeeze().cpu().numpy()
        
        # Get Ground Truth Trojan Locations
        # We need to reconstruct the exact window from dataset
        # SequenceDataset logic is complex (strides), but since we have the index, 
        # we can rely on the fact that dataset.samples[idx] is X window.
        # BUT dataset.labels[idx] is just 0/1. We don't have the per-token labels stored in Dataset.
        # We need to recover the original slice.
        # Fortunately, SequenceDataset stores (start_idx, end_idx) or copies data?
        # Let's look at SequenceDataset implementation in train.py
        # It stores `self.samples` which are the X windows.
        # It DOES NOT store the y windows, only the scalar label.
        # This makes overlaying ground truth hard unless we hack SequenceDataset.
        
        # HACK: Let's modify this script to manually slice X and y to get the ground truth window.
        # We will iterate X, y similar to Dataset but keep track of y_window.
        pass

    # ---------------------------------------------------------
    # Custom Loop to get Y-Window
    # ---------------------------------------------------------
    # We will replicate the dataset building loop but just for the found indices?
    # No, indices are opaque.
    # Let's just search the dataset again with our own loop to find samples and keep y.
    
    max_len = config['training']['max_len']
    unique_designs = np.unique(design_ids)
    
    found_count = 0
    
    for design_id in unique_designs:
        mask = design_ids == design_id
        X_design = X[mask]
        y_design = y[mask] # (N,)
        
        N = X_design.shape[0]
        cursor = 0
        while cursor < N:
            end = min(cursor + max_len, N)
            chunk_y = y_design[cursor:end]
            
            if np.any(chunk_y == 1): # Trojan Sample
                # Pad if necessary
                if len(chunk_y) < max_len:
                    pad_len = max_len - len(chunk_y)
                    chunk_y = np.concatenate([chunk_y, np.zeros(pad_len)])
                    
                chunk_x = X_design[cursor:end]
                if chunk_x.shape[0] < max_len:
                    pad_len = max_len - chunk_x.shape[0]
                    chunk_x = np.vstack([chunk_x, np.zeros((pad_len, chunk_x.shape[1]))])
                
                # Run Model
                x_tensor = torch.tensor(chunk_x, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits, attentions = model(x_tensor, return_attentions=True)
                    probs = torch.softmax(logits, dim=1)
                    pred = torch.argmax(probs, dim=1).item()
                
                pooling_weights = attentions['pooling_weights'].squeeze().cpu().numpy()
                
                # Plot
                plt.figure(figsize=(12, 6))
                
                # Plot 1: Ground Truth
                plt.subplot(2, 1, 1)
                plt.plot(chunk_y, label='Ground Truth Trojan', color='red', linewidth=2)
                plt.title(f'Design {design_id} - Window {cursor}-{end} (Pred: {pred}, Prob: {probs[0,1]:.4f})')
                plt.ylabel('Trojan Label')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # Plot 2: Attention Weights
                plt.subplot(2, 1, 2)
                plt.plot(pooling_weights, label='Attention Pooling Weights', color='blue', linewidth=1)
                plt.fill_between(range(len(pooling_weights)), pooling_weights, color='blue', alpha=0.3)
                plt.ylabel('Attention Score')
                plt.xlabel('Time Step')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                save_path = os.path.join(output_dir, f'vis_design{design_id}_step{cursor}.png')
                plt.tight_layout()
                plt.savefig(save_path)
                plt.close()
                print(f"Saved plot to {save_path}")
                
                found_count += 1
                if found_count >= 10:
                    return

            cursor += max_len // 2 # Stride
            
if __name__ == "__main__":
    visualize_attention()
