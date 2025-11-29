import os
import pickle
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import DATA_DIR

def check_design_3_len():
    dataset_path = os.path.join(DATA_DIR, "dataset.pkl")
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
    
    design_ids = data["design_ids"]
    mask = design_ids == 3
    print(f"Design 3 Total Length: {np.sum(mask)}")
    
    # Also check Design 0, 1, 2
    for i in range(5):
        mask = design_ids == i
        print(f"Design {i} Total Length: {np.sum(mask)}")

if __name__ == "__main__":
    check_design_3_len()
