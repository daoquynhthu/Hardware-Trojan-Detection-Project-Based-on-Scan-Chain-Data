import os
import numpy as np
import pickle
import sys
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import logging

# Add src to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SEQS_DATA_DIR, MODELS_DIR, DATA_DIR
from src.utils import load_labels, load_npz
from src.feature_extractor import extract_features_from_design

# Configure logging
logging.basicConfig(
    filename='dataset_builder.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
            
            # --- Transpose Logic & Shape Verification ---
            # Expected shape: (N_cycles, N_regs)
            # We use reg2row (map of reg_name -> index) to determine N_regs
            
            n_regs_metadata = None
            
            if "reg2row" in data:
                reg2row = data["reg2row"]
                if isinstance(reg2row, np.ndarray) and reg2row.shape == ():
                    reg2row = reg2row.item()
                
                if isinstance(reg2row, dict) and reg2row:
                    # Handle nested structure where 'reg' key holds the actual map
                    if "reg" in reg2row and isinstance(reg2row["reg"], dict):
                        reg_map = reg2row["reg"]
                    else:
                        reg_map = reg2row
                    
                    if reg_map:
                        max_val = 0
                        for val in reg_map.values():
                            if isinstance(val, (list, tuple, np.ndarray)):
                                # If list [start, end], end is the upper bound
                                curr = max(val) if len(val) > 0 else 0
                                if curr > max_val: max_val = curr
                            elif isinstance(val, (int, float, np.integer)):
                                # If int index, n_regs = index + 1
                                if val + 1 > max_val: max_val = val + 1
                        
                        n_regs_metadata = max_val
            
            # Heuristic: If we have metadata, use it to validate dimensions
            if n_regs_metadata is not None:
                dim0, dim1 = seqs.shape
                
                # Check if dim1 matches n_regs (Standard case)
                if dim1 == n_regs_metadata:
                    pass # Correct shape (T, N)
                # Check if dim0 matches n_regs (Transposed case)
                elif dim0 == n_regs_metadata:
                    # print(f"Transposing {filename} based on metadata: {seqs.shape} -> ({dim1}, {dim0})")
                    seqs = seqs.T
                else:
                    print(f"Warning: Dimensions {seqs.shape} do not match metadata N_regs={n_regs_metadata} for {filename}. Using heuristic.")
                    # Fallback Heuristic: Time usually > Regs, but not always.
                    # If dim0 < dim1, it might be (N, T) if N is small and T is large? No.
                    # Usually T >> N or T ~ N. 
                    # If metadata fails, we rely on the previous logic or assume (T, N)
            
            # Fallback if no metadata or mismatch: Check for obvious transpose (N_regs > N_cycles is rare but possible)
            # The previous logic: if max_row_idx >= seqs.shape[1], then seqs.shape[1] is too small to be N_regs.
            elif n_regs_metadata is not None and n_regs_metadata >= seqs.shape[1]:
                 seqs = seqs.T
                 
        except Exception as e:
            error_msg = f"Error loading {filename}: {e}"
            print(error_msg)
            logging.error(error_msg)
            continue
            
        # Extract Features
        # print(f"Extracting features for {filename}...")
        try:
            features = extract_features_from_design(seqs) # [N_regs, N_feats]
        except Exception as e:
             error_msg = f"Error extracting features for {filename}: {e}"
             print(error_msg)
             logging.error(error_msg)
             continue
        
        # Create Labels
        n_regs = features.shape[0]
        y = np.zeros(n_regs, dtype=int)
        
        has_labels = False
        if design_id in labels_map:
            l_start, l_end = labels_map[design_id]
            # Updated to use Python slicing.
            # Assuming labels.json uses INCLUSIVE indexing [start, end] (Hardware convention)
            # Python slicing is [start, end), so we need end + 1.
            
            l_end_slice = l_end + 1
            
            if l_end_slice <= n_regs:
                y[l_start : l_end_slice] = 1
                has_labels = True
            else:
                # Critical: If labels are out of bounds, we might be using the wrong file or labels.
                # Do NOT clip and pretend it's fine. Skip this design to avoid poisoning the dataset.
                error_msg = f"Skipping {filename}: Label indices [{l_start}, {l_end}] out of bounds for N_regs={n_regs}."
                print(f"Warning: {error_msg}")
                logging.warning(error_msg)
                continue

        else:
            print(f"Warning: No labels found for design index {design_id} ({filename})")
            # If no labels exist in map, we assume clean? Or skip?
            # Assuming clean (y=0) if not in map is risky if map is incomplete.
            # But here we are iterating over known files.
            pass
            
        # Normalize per design
        # We normalize per design (Instance Normalization) to handle process variation between chips/simulations.
        # This means the model learns relative patterns, not absolute values.
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
