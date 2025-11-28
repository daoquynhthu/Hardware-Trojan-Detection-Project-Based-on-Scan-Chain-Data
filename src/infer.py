import argparse
import os
import sys
import pickle
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import LGBM_MODEL_DIR, TRANSFORMER_MODEL_DIR

def load_lgbm_model():
    import lightgbm as lgb
    model_path = os.path.join(LGBM_MODEL_DIR, "lgbm_model.txt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LGBM model not found at {model_path}")
    return lgb.Booster(model_file=model_path)

def load_transformer_model(device):
    import torch
    from src.transformer.model import Transformer, ModelArgs
    
    model_path = os.path.join(TRANSFORMER_MODEL_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Transformer model not found at {model_path}")
        
    checkpoint = torch.load(model_path, map_location=device)
    
    # Try to load args from checkpoint
    if isinstance(checkpoint, dict) and 'model_args' in checkpoint:
        print("Loading ModelArgs from checkpoint...")
        model_args = checkpoint['model_args']
        # Ensure dropout is 0 for inference
        if hasattr(model_args, 'dropout'):
             model_args.dropout = 0.0
    else:
        print("ModelArgs not found in checkpoint. Using defaults.")
        # Assuming default args for now.
        model_args = ModelArgs(
            dim=128,
            n_layers=2,
            n_heads=4,
            max_seq_len=1024,
            input_dim=35, # Default based on current dataset
            num_classes=2,
            dropout=0.0 # No dropout for inference
        )
    
    model = Transformer(model_args).to(device)
    
    # Load state dict
    state_dict = checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        print("Attempting strict=False...")
        model.load_state_dict(state_dict, strict=False)
        
    model.eval()
    return model

def infer_lgbm(model, X):
    # X shape: (N, features)
    y_pred = model.predict(X)
    return y_pred

def infer_transformer(model, X, device):
    import torch
    
    max_len = 1024
    N = X.shape[0]
    preds = []
    
    with torch.no_grad():
        for start in range(0, N, max_len):
            end = min(start + max_len, N)
            chunk = X[start:end]
            
            # Pad if necessary
            original_len = chunk.shape[0]
            if original_len < max_len:
                pad_len = max_len - original_len
                chunk = np.vstack([chunk, np.zeros((pad_len, chunk.shape[1]))])
            
            chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device) # Batch size 1
            output = model(chunk_tensor) # (1, 1024, 2)
            probs = torch.softmax(output, dim=-1)[:, :, 1] # (1, 1024)
            
            # Extract valid predictions
            chunk_probs = probs[0, :original_len].cpu().numpy()
            preds.extend(chunk_probs)
            
    return np.array(preds)

def main():
    parser = argparse.ArgumentParser(description="Inference for HT Detector")
    parser.add_argument("--model", type=str, required=True, choices=["lgbm", "transformer"], help="Model type to use")
    parser.add_argument("--input", type=str, required=True, help="Path to input data (.pkl or .npy)")
    parser.add_argument("--output", type=str, required=True, help="Path to save predictions (.npy)")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    if args.input.endswith(".pkl"):
        with open(args.input, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, dict) and "X" in data:
                X = data["X"]
            else:
                X = data
    elif args.input.endswith(".npy"):
        X = np.load(args.input)
    else:
        print("Error: Unsupported file format. Use .pkl or .npy")
        return

    # Explicitly cast to numpy array to resolve linter errors and ensure correct type
    X = np.array(X)

    # Ensure X is 2D
    if len(X.shape) != 2:
        print(f"Error: Expected 2D input (N, features), got {X.shape}")
        return
        
    print(f"Data shape: {X.shape}")
    
    # Initialize preds to avoid unbound variable error
    preds = None
    
    if args.model == "lgbm":
        print("Loading LightGBM model...")
        try:
            model = load_lgbm_model()
            print("Running inference...")
            preds = infer_lgbm(model, X)
        except Exception as e:
            print(f"Error running LightGBM inference: {e}")
            return
        
    elif args.model == "transformer":
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Transformer model on {device}...")
        try:
            model = load_transformer_model(device)
            print("Running inference...")
            preds = infer_transformer(model, X, device)
        except Exception as e:
            print(f"Error running Transformer inference: {e}")
            return
        
    # Save results
    if preds is not None:
        print(f"Saving results to {args.output}...")
        # Explicitly cast to numpy array to satisfy linter and ensure compatibility
        preds_array = np.array(preds)
        np.save(args.output, preds_array)
        print("Done.")
    else:
        print("Error: Inference failed or model type not recognized.")

if __name__ == "__main__":
    main()
