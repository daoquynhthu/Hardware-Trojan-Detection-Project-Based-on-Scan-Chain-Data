import os
import sys
import pickle
import numpy as np
import json
from collections import Counter
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from config import DATA_DIR

def analyze_dataset():
    dataset_path = os.path.join(DATA_DIR, "dataset.pkl")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
        
    X = data["X"]
    y = data["y"]
    design_ids = data["design_ids"]
    
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    # 1. Global Stats
    total_samples = len(y)
    trojan_samples = np.sum(y == 1)
    normal_samples = total_samples - trojan_samples
    trojan_ratio = trojan_samples / total_samples
    
    print(f"\n[Global Class Distribution]")
    print(f"Total Samples (Tokens/Cycles): {total_samples}")
    print(f"Normal Samples (0): {normal_samples} ({normal_samples/total_samples:.2%})")
    print(f"Trojan Samples (1): {trojan_samples} ({trojan_ratio:.2%})")
    print(f"Imbalance Ratio (Normal:Trojan): {normal_samples/trojan_samples:.1f}:1")
    
    # 2. Design Stats
    unique_designs = np.unique(design_ids)
    print(f"\n[Design Analysis]")
    print(f"Total Designs: {len(unique_designs)}")
    
    design_stats = []
    for d_id in unique_designs:
        mask = design_ids == d_id
        y_d = y[mask]
        t_count = np.sum(y_d == 1)
        n_count = len(y_d)
        ratio = t_count / n_count
        design_stats.append({
            "id": d_id,
            "total": n_count,
            "trojan": t_count,
            "ratio": ratio
        })
        
    # Sort by trojan ratio
    design_stats.sort(key=lambda x: x["ratio"], reverse=True)
    
    print(f"{'Design ID':<10} | {'Total':<10} | {'Trojan':<10} | {'Ratio':<10}")
    print("-" * 50)
    for s in design_stats:
        print(f"{s['id']:<10} | {s['total']:<10} | {s['trojan']:<10} | {s['ratio']:.4f}")
        
    # 3. Feature Analysis
    print(f"\n[Feature Analysis]")
    print(f"Feature Matrix Shape: {X.shape}")
    
    # Check for NaN/Inf
    if np.isnan(X).any():
        print("WARNING: Dataset contains NaNs!")
    if np.isinf(X).any():
        print("WARNING: Dataset contains Infs!")
        
    # Stats per feature
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    
    print(f"{'Feat#':<6} | {'Mean':<8} | {'Std':<8} | {'Min':<8} | {'Max':<8}")
    print("-" * 50)
    for i in range(min(X.shape[1], 10)): # Show first 10 features
        print(f"{i:<6} | {means[i]:.4f}   | {stds[i]:.4f}   | {mins[i]:.4f}   | {maxs[i]:.4f}")
    if X.shape[1] > 10:
        print(f"... and {X.shape[1]-10} more features.")
        
    # 4. Sequence Length Analysis (Simulation)
    # Based on config.max_len, how many "positive" sequences vs "negative" sequences?
    print(f"\n[Sequence Simulation (max_len=512, stride=256)]")
    # Default params from train.py logic
    max_len = 512
    stride = 256
    
    seq_trojan = 0
    seq_normal = 0
    
    for d_id in unique_designs:
        mask = design_ids == d_id
        y_d = y[mask]
        N = len(y_d)
        
        for start in range(0, N, stride):
            end = min(start + max_len, N)
            chunk_y = y_d[start:end]
            
            # Pad if needed (just for counting logic)
            if len(chunk_y) < max_len:
                # usually padded, but here we just check if ANY 1 exists
                pass
                
            if np.any(chunk_y == 1):
                seq_trojan += 1
            else:
                seq_normal += 1
                
    total_seqs = seq_trojan + seq_normal
    print(f"Total Sequences: {total_seqs}")
    print(f"Normal Sequences: {seq_normal} ({seq_normal/total_seqs:.2%})")
    print(f"Trojan Sequences: {seq_trojan} ({seq_trojan/total_seqs:.2%})")
    if seq_trojan > 0:
        print(f"Sequence Imbalance: {seq_normal/seq_trojan:.1f}:1")
    else:
        print("Sequence Imbalance: Infinite (No Trojan sequences!)")

if __name__ == "__main__":
    analyze_dataset()
