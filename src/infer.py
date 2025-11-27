import argparse
import numpy as np
import lightgbm as lgb
import os
import sys
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODEL_PATH
from src.utils import load_npz
from src.feature_extractor import extract_features_from_design

def infer(npz_path, top_k=20, model_path=MODEL_PATH):
    if not os.path.exists(npz_path):
        print(f"Error: File {npz_path} not found.")
        return

    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found. Train model first.")
        return

    # Load Model
    try:
        bst = lgb.Booster(model_file=model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Processing {npz_path}...")
    try:
        data = load_npz(npz_path)
        seqs = data["seqs"]
        # Try to get reg2row if it exists
        if "reg2row" in data:
            reg2row = data["reg2row"]
        else:
            reg2row = None
    except Exception as e:
        print(f"Error reading npz: {e}")
        return

    # Extract
    print("Extracting features...")
    feats = extract_features_from_design(seqs)
    
    # Normalize (Fit on self - Instance Normalization)
    print("Normalizing features...")
    scaler = StandardScaler()
    feats_norm = scaler.fit_transform(feats)
    
    # Predict
    print("Predicting...")
    # For binary classification, predict returns probability of class 1
    scores = bst.predict(feats_norm)
    # Fix: Ensure scores is numpy array
    scores = np.array(scores)
    
    # Top K
    idx_sorted = np.argsort(-scores)

    top_indices = idx_sorted[:top_k]
    top_scores = scores[top_indices]
    
    print(f"\nTop-{top_k} Suspicious Registers:")
    print("-" * 50)
    print(f"{'Rank':<5} | {'Index':<8} | {'Prob':<8} | {'Name'}")
    print("-" * 50)
    
    # Inverse map reg2row if possible

    row2name = {}
    if reg2row is not None:
        # Handle if reg2row is numpy 0-d array containing dict
        if isinstance(reg2row, np.ndarray) and reg2row.shape == ():
            reg2row = reg2row.item()
        
        if isinstance(reg2row, dict):
            row2name = {v: k for k, v in reg2row.items()}
            
    for rank, (idx, score) in enumerate(zip(top_indices, top_scores)):
        name = row2name.get(idx, "N/A")
        print(f"{rank+1:<5} | {idx:<8} | {score:.4f}   | {name}")
        
    return top_indices, top_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to .npz file")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top suspicious registers")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to model file")
    
    args = parser.parse_args()
    infer(args.file, args.top_k, args.model)
