import os
import numpy as np
import pickle
import sys
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Add src to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SEQS_DATA_DIR, MODELS_DIR, DATA_DIR
from src.utils import load_labels, load_npz
from src.feature_extractor import extract_features_from_design

def build_dataset():
    # List files
    if not os.path.exists(SEQS_DATA_DIR):
        print(f"Error: {SEQS_DATA_DIR} does not exist.")
        return

    files = sorted([f for f in os.listdir(SEQS_DATA_DIR) if f.endswith('.npz')])
    labels_map = load_labels()
    
    X_all = []
    y_all = []
    design_ids = []
    
    # Directory to save individual scalers
    scalers_dir = os.path.join(MODELS_DIR, "scalers")
    os.makedirs(scalers_dir, exist_ok=True)
    
    # Checkpoint Directory
    interim_dir = os.path.join(DATA_DIR, "interim")
    os.makedirs(interim_dir, exist_ok=True)
    
    print(f"Found {len(files)} files in {SEQS_DATA_DIR}")
    
    for idx, filename in enumerate(tqdm(files)):
        interim_path = os.path.join(interim_dir, f"{filename}.pkl")
        
        # Checkpoint: Load if exists
        if os.path.exists(interim_path):
            try:
                with open(interim_path, "rb") as f:
                    data_interim = pickle.load(f)
                    X_all.append(data_interim["X"])
                    y_all.append(data_interim["y"])
                    design_ids.append(data_interim["design_ids"])
                    # Scaler is already saved separately, but that's fine
                    continue
            except Exception as e:
                print(f"Error loading checkpoint for {filename}, reprocessing. Error: {e}")
        
        file_path = os.path.join(SEQS_DATA_DIR, filename)
        design_id = str(idx) # "0", "1", ...
        
        # Load Data
        try:
            data = load_npz(file_path)
            seqs = data["seqs"]
            
            # --- Transpose Logic ---
            # Check if seqs is (N_regs, N_cycles) instead of (N_cycles, N_regs)
            # Use reg2row to verify
            if "reg2row" in data:
                reg2row = data["reg2row"]
                if isinstance(reg2row, np.ndarray) and reg2row.shape == ():
                    reg2row = reg2row.item()
                
                if isinstance(reg2row, dict) and reg2row:
                    max_row_idx = max(reg2row.values())
                    
                    # If max index is closer to dim 0 than dim 1, and dim 0 > dim 1
                    # Or simply if max_row_idx >= seqs.shape[1] (implies indices are out of bounds for dim 1)
                    if max_row_idx >= seqs.shape[1]:
                        # Must be (N_regs, N_cycles)
                        # print(f"Transposing {filename}: {seqs.shape} -> ({seqs.shape[1]}, {seqs.shape[0]})")
                        seqs = seqs.T
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
            
        # Extract Features
        # print(f"Extracting features for {filename}...")
        features = extract_features_from_design(seqs) # [N_regs, N_feats]
        
        # Create Labels
        n_regs = features.shape[0]
        y = np.zeros(n_regs)
        
        if design_id in labels_map:
            l_start, l_end = labels_map[design_id]
            # Updated to use Python slicing (exclusive end) as requested
            # User provided code: label[start:end] = 1
            if l_end <= n_regs:
                y[l_start : l_end] = 1
            else:
                # Handle edge case where label index might exceed reg count
                actual_end = min(l_end, n_regs)
                y[l_start : actual_end] = 1
                if l_start < n_regs:
                    print(f"Warning: Label end {l_end} > n_regs {n_regs} for {filename}, clipped.")
                else:
                    print(f"Warning: Label start {l_start} >= n_regs {n_regs} for {filename}, no labels set.")

        else:
            print(f"Warning: No labels found for design index {design_id} ({filename})")
            
        # Normalize per design
        scaler = StandardScaler()
        features_norm = scaler.fit_transform(features)
        
        # Save Scaler
        scaler_path = os.path.join(scalers_dir, f"{filename}_scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
            
        # Checkpoint: Save interim result
        try:
            with open(interim_path, "wb") as f:
                pickle.dump({
                    "X": features_norm,
                    "y": y,
                    "design_ids": np.full(n_regs, idx)
                }, f)
        except Exception as e:
            print(f"Warning: Failed to save checkpoint for {filename}: {e}")
            
        X_all.append(features_norm)
        y_all.append(y)
        design_ids.append(np.full(n_regs, idx))
        
    # Concatenate all data
    if X_all:
        X_concat = np.vstack(X_all)
        y_concat = np.concatenate(y_all)
        ids_concat = np.concatenate(design_ids)
        
        # Save Dataset
        dataset_path = os.path.join(DATA_DIR, "dataset.pkl")
        print(f"Saving dataset to {dataset_path}...")
        print(f"Total shape: {X_concat.shape}")
        
        with open(dataset_path, "wb") as f:
            pickle.dump({
                "X": X_concat, 
                "y": y_concat, 
                "design_ids": ids_concat, 
                "filenames": files
            }, f)
            
        print("Dataset built successfully.")
    else:
        print("No data processed.")

if __name__ == "__main__":
    build_dataset()
