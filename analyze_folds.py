import numpy as np
from sklearn.model_selection import KFold
import pickle
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from config import DATA_DIR

def analyze_folds():
    dataset_path = os.path.join(DATA_DIR, "dataset.pkl")
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
    
    design_ids = data["design_ids"]
    y = data["y"]
    unique_designs = np.unique(design_ids)
    
    # Replicate the split logic from train.py
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    print(f"Total Designs: {len(unique_designs)}")
    print(f"Design List: {unique_designs}")
    
    # Calculate Trojan stats per design first
    design_stats = {}
    for d in unique_designs:
        mask = design_ids == d
        y_d = y[mask]
        trojan_tokens = np.sum(y_d == 1)
        design_stats[d] = trojan_tokens
        
    print("\nDesign Trojan Counts:")
    # Sort by count desc
    sorted_stats = sorted(design_stats.items(), key=lambda x: x[1], reverse=True)
    for d, count in sorted_stats:
        print(f"  Design {d}: {count} tokens")

    print("\n" + "="*60)
    print("FOLD ANALYSIS (Simulated)")
    print("="*60)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(unique_designs)):
        train_designs = unique_designs[train_idx]
        test_designs = unique_designs[test_idx]
        
        # Train Stats
        train_trojans = sum(design_stats[d] for d in train_designs)
        train_designs_with_trojan = [d for d in train_designs if design_stats[d] > 0]
        
        # Test Stats
        test_trojans = sum(design_stats[d] for d in test_designs)
        test_designs_with_trojan = [d for d in test_designs if design_stats[d] > 0]
        
        print(f"\n[FOLD {fold_idx+1}]")
        print(f"  TRAIN Designs: {train_designs}")
        print(f"  TRAIN Trojan Tokens: {train_trojans}")
        print(f"  TRAIN Designs w/ Trojan: {train_designs_with_trojan}")
        
        print(f"  TEST  Designs: {test_designs}")
        print(f"  TEST  Trojan Tokens: {test_trojans}")
        print(f"  TEST  Designs w/ Trojan: {test_designs_with_trojan}")
        
        if train_trojans == 0:
            print("  WARNING: NO TROJANS IN TRAINING SET!")
        elif test_trojans == 0:
            print("  WARNING: NO TROJANS IN TEST SET! (Metrics will be undefined/perfect)")
        else:
            print(f"  Train/Test Trojan Ratio: {train_trojans}/{test_trojans}")

if __name__ == "__main__":
    analyze_folds()
