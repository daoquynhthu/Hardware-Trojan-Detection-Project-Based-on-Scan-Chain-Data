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

def analyze_features():
    # 1. Load Config and Model
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'transformer', 'train_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    dataset_path = os.path.join(DATA_DIR, "dataset.pkl")
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
    
    X = data["X"]
    y = data["y"]
    design_ids = data["design_ids"]
    
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
    
    model_path = os.path.join(TRANSFORMER_MODEL_DIR, "final_model.pth")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
        
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 2. Find a suitable Trojan Sample (Design 3 is interesting based on previous check)
    # We want a window that CONTAINS the trigger point.
    # Previous check showed trigger around 6600 for Design 3?
    # Let's search for a window with 0->1 transition in labels.
    
    print("Searching for a sample with Trojan trigger transition...")
    
    target_design = 3
    max_len = config['training']['max_len']
    
    mask = design_ids == target_design
    X_design = X[mask]
    y_design = y[mask]
    
    found_sample = None
    found_y = None
    trigger_offset = 0
    
    # Scan with stride
    cursor = 0
    while cursor < len(y_design):
        end = min(cursor + max_len, len(y_design))
        chunk_y = y_design[cursor:end]
        
        # Look for 0 -> 1 transition
        # This means we capture the moment it turns ON
        if np.any(chunk_y == 0) and np.any(chunk_y == 1):
            # Find first 1
            first_one = np.argmax(chunk_y == 1)
            if first_one > 50 and first_one < max_len - 50: # Ensure some context
                # Found it!
                chunk_x = X_design[cursor:end]
                
                # Pad if needed
                if chunk_x.shape[0] < max_len:
                    pad_len = max_len - chunk_x.shape[0]
                    chunk_x = np.vstack([chunk_x, np.zeros((pad_len, chunk_x.shape[1]))])
                    chunk_y = np.concatenate([chunk_y, np.zeros(pad_len)])
                
                found_sample = chunk_x
                found_y = chunk_y
                trigger_offset = first_one
                print(f"Found trigger at relative step {trigger_offset} in window starting at {cursor}")
                break
        
        cursor += max_len // 2

    if found_sample is None:
        print("Could not find a suitable trigger transition sample.")
        return

    # 3. Calculate Saliency (Input Gradients)
    # Make input require grad
    x_tensor = torch.tensor(found_sample, dtype=torch.float32).unsqueeze(0).to(device)
    x_tensor.requires_grad = True
    
    # Forward
    model.zero_grad()
    logits = model(x_tensor)
    
    # We want to explain why it predicted Class 1 (Trojan)
    # Score for class 1
    score = logits[0, 1]
    
    # Backward
    score.backward()
    
    # Get gradients
    # shape: (1, seq_len, n_features)
    if x_tensor.grad is None:
        print("Error: Gradients are None. Backward pass failed.")
        return

    gradients = x_tensor.grad.data.cpu().numpy()[0] # (seq_len, n_features)
    
    # Take absolute value (magnitude of influence)
    saliency = np.abs(gradients)
    
    # Normalize for visualization
    # Ensure we don't divide by zero and handle types correctly
    s_min = float(saliency.min())
    s_max = float(saliency.max())
    saliency = (saliency - s_min) / (s_max - s_min + 1e-9)

    # 4. Visualization
    output_dir = "fine_grained_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    if found_y is None:
        print("Error: found_y is None.")
        return

    # Plot 1: Ground Truth Label
    plt.subplot(3, 1, 1)
    plt.plot(found_y, color='red', linewidth=2, label='Ground Truth')
    plt.axvline(x=float(trigger_offset), color='black', linestyle='--', alpha=0.5, label='Trigger Start')
    plt.title(f'Trojan Activation (Design {target_design})')
    plt.ylabel('Label')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Feature Saliency Heatmap
    plt.subplot(3, 1, 2)
    # Transpose to have Features on Y axis, Time on X axis
    # saliency shape (Time, Feat) -> (Feat, Time)
    sns_plot = plt.imshow(saliency.T, aspect='auto', cmap='inferno', origin='lower', interpolation='nearest')
    plt.colorbar(sns_plot, label='Gradient Magnitude')
    plt.title('Feature-Level Saliency Map (Which Register Triggered It?)')
    plt.ylabel('Feature Index (0-34)')
    plt.xlabel('Time Step')
    # Mark the trigger line
    plt.axvline(x=float(trigger_offset), color='cyan', linestyle='--', linewidth=1)
    
    # Plot 3: Top 3 Suspicious Features
    plt.subplot(3, 1, 3)
    # Sum saliency over time window around trigger (e.g., +/- 20 steps)
    start_f = max(0, trigger_offset - 50)
    end_f = min(max_len, trigger_offset + 10)
    local_importance = np.sum(saliency[start_f:end_f, :], axis=0)
    
    # Get top 5 features
    top_indices = np.argsort(local_importance)[-5:][::-1]
    
    plt.bar(range(len(top_indices)), local_importance[top_indices], color='orange')
    plt.xticks(range(len(top_indices)), [f"Feat {i}" for i in top_indices])
    plt.title(f'Most Suspicious Features around Trigger (Step {start_f}-{end_f})')
    plt.ylabel('Cumulative Saliency')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'feature_saliency_map.png')
    plt.savefig(save_path)
    print(f"Analysis saved to {save_path}")
    
    # Text Report
    print("\n=== Fine-Grained Analysis Report ===")
    print(f"Trigger detected at relative time step: {trigger_offset}")
    print("Top 5 Most Suspicious Features (Potential Trojan Triggers):")
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. Feature {idx} (Score: {local_importance[idx]:.4f})")
    print("=====================================")

if __name__ == "__main__":
    analyze_features()
