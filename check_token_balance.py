import os
import sys
import pickle
import numpy as np
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import DATA_DIR

def analyze_token_balance():
    dataset_path = os.path.join(DATA_DIR, "dataset.pkl")
    if not os.path.exists(dataset_path):
        print("Dataset not found.")
        return

    print("Loading dataset...")
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
        
    y = data["y"]
    
    total_tokens = len(y)
    positive_tokens = np.sum(y == 1)
    negative_tokens = np.sum(y == 0)
    
    print(f"Total Tokens: {total_tokens}")
    print(f"Positive (Trojan): {positive_tokens}")
    print(f"Negative (Normal): {negative_tokens}")
    
    if positive_tokens > 0:
        ratio = negative_tokens / positive_tokens
        print(f"Ratio Normal:Trojan = {ratio:.2f}:1")
        print(f"Trojan Percentage: {positive_tokens/total_tokens*100:.4f}%")
    else:
        print("No positive tokens found!")

if __name__ == "__main__":
    analyze_token_balance()
