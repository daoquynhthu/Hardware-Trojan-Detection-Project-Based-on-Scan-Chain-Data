import os
import sys
import pickle
import numpy as np
from collections import Counter

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR

def diagnose():
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
    
    print(f"\nTotal Samples (Tokens): {len(y)}")
    print(f"Total Designs: {len(np.unique(design_ids))}")
    
    unique_designs = np.unique(design_ids)
    
    print("\n--- Design Analysis ---")
    trojan_designs = []
    normal_designs = []
    
    for design_id in unique_designs:
        mask = design_ids == design_id
        y_design = y[mask]
        
        has_trojan = np.any(y_design == 1)
        total_tokens = len(y_design)
        trojan_tokens = np.sum(y_design == 1)
        
        status = "TROJAN" if has_trojan else "Normal"
        if has_trojan:
            trojan_designs.append(design_id)
        else:
            normal_designs.append(design_id)
            
        print(f"Design {design_id}: {status} | Total Tokens: {total_tokens} | Trojan Tokens: {trojan_tokens}")

    print("\n--- Summary ---")
    print(f"Total Designs: {len(unique_designs)}")
    print(f"Trojan Designs ({len(trojan_designs)}): {trojan_designs}")
    print(f"Normal Designs ({len(normal_designs)}): {normal_designs}")
    
    if len(trojan_designs) == 0:
        print("\nCRITICAL: No Trojan designs found in the dataset!")
    elif len(trojan_designs) < 2:
        print("\nWARNING: Only 1 Trojan design found. Cannot split into Train/Test effectively without leakage or missing class.")

    # Simulate current random split
    print("\n--- Current Random Split Simulation (20% Test) ---")
    np.random.seed(42)
    shuffled_designs = np.array(unique_designs)
    np.random.shuffle(shuffled_designs)
    
    test_size = max(1, int(len(unique_designs) * 0.2))
    test_designs = shuffled_designs[:test_size]
    train_designs = shuffled_designs[test_size:]
    
    train_has_trojan = any(d in trojan_designs for d in train_designs)
    test_has_trojan = any(d in trojan_designs for d in test_designs)
    
    print(f"Test Designs: {test_designs}")
    print(f"Train Designs: {train_designs}")
    print(f"Trojan in Train: {train_has_trojan}")
    print(f"Trojan in Test: {test_has_trojan}")

    if not train_has_trojan:
        print("🚨 CRITICAL: Training set has NO Trojan designs!")
    if not test_has_trojan:
        print("🚨 CRITICAL: Test set has NO Trojan designs!")

if __name__ == "__main__":
    diagnose()
