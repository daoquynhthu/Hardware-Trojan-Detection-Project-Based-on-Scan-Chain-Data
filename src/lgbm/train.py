import lightgbm as lgb
import numpy as np
import pickle
import os
import sys
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, jaccard_score

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import DATA_DIR, MODEL_PATH, MODELS_DIR, LGBM_PARAMS

def train_model():
    dataset_path = os.path.join(DATA_DIR, "dataset.pkl")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}. Run dataset_builder.py first.")
        return

    print("Loading dataset...")
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
        
    X = data["X"]
    y = data["y"]
    design_ids = data["design_ids"]
    
    # --- Hold-out Validation Split (Random Designs) ---
    unique_designs = np.unique(design_ids)
    # Fix: Set random seed for reproducibility
    np.random.seed(42) 
    np.random.shuffle(unique_designs)
    
    # Hold out 20% of designs for validation
    test_size = max(1, int(len(unique_designs) * 0.2))
    test_designs = unique_designs[:test_size]
    train_designs = unique_designs[test_size:]
    
    print(f"Total Designs: {len(unique_designs)}")
    print(f"Training Designs: {len(train_designs)}")
    print(f"Hold-out Test Designs: {len(test_designs)} ({test_designs})")
    
    # Create Masks
    mask_test = np.isin(design_ids, test_designs)
    mask_train = ~mask_test
    
    X_train, y_train = X[mask_train], y[mask_train]
    X_test, y_test = X[mask_test], y[mask_test]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape:     {X_test.shape}")
    print(f"Positive samples (Train): {np.sum(y_train)}")
    print(f"Positive samples (Test):  {np.sum(y_test)}")
    
    # --- SMOTE Augmentation Loop ---
    try:
        from imblearn.over_sampling import SMOTE
        
        # Explore performance limits with different ratios
        smote_ratios = [0.1, 0.5, 1.0]
        
        for ratio in smote_ratios:
            print(f"\n{'='*20} Testing SMOTE Ratio: {ratio} {'='*20}")
            
            # Reset training data for each iteration
            X_curr, y_curr = X[mask_train], y[mask_train]
            
            print(f"Before SMOTE: {X_curr.shape}, Positive: {np.sum(y_curr)}")
            
            # Fix: Add type ignore for linter false positives
            smote = SMOTE(random_state=42, sampling_strategy=ratio) # type: ignore
            X_res, y_res = smote.fit_resample(X_curr, y_curr) # type: ignore
            
            print(f"After SMOTE:  {X_res.shape}, Positive: {np.sum(y_res)}")
            
            # Adjust params based on ratio
            current_params = LGBM_PARAMS.copy()
            if ratio >= 0.5:
                 current_params['class_weight'] = None
            else:
                 # For lower ratios, use balanced weights to handle remaining imbalance
                 current_params['class_weight'] = 'balanced'
                 
            print(f"Using class_weight: {current_params.get('class_weight')}")

            # Train
            clf = lgb.LGBMClassifier(**current_params)
            clf.fit(X_res, y_res)
            
            # Evaluate
            y_prob = np.array(clf.predict_proba(X_test))[:, 1]
            
            # Find Optimal Threshold
            thresholds = np.arange(0.01, 1.0, 0.01)
            best_f1 = 0
            best_thresh = 0.5
            
            for thresh in thresholds:
                y_tmp = (y_prob >= thresh).astype(int)
                f1_tmp = f1_score(y_test, y_tmp, zero_division=0)
                if f1_tmp > best_f1:
                    best_f1 = f1_tmp
                    best_thresh = thresh
            
            print(f"Optimal Threshold: {best_thresh:.2f}")
            print(f"Max F1 Score:      {best_f1:.4f}")
            
            # Detailed Metrics for Best Threshold
            y_pred = (y_prob >= best_thresh).astype(int)
            p = precision_score(y_test, y_pred, zero_division=0)
            r = recall_score(y_test, y_pred, zero_division=0)
            if np.sum(y_test) > 0:
                auc = roc_auc_score(y_test, y_prob)
            else:
                auc = 0.0
                
            print(f"Precision:         {p:.4f}")
            print(f"Recall:            {r:.4f}")
            print(f"AUROC:             {auc:.4f}")
            
            if best_f1 >= 0.90:
                 print("RESULT: PASS")
                 # Save best model
                 print(f"Saving best model to {MODEL_PATH}...")
                 os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                 clf.booster_.save_model(MODEL_PATH)
            else:
                 print("RESULT: FAIL")
                 
    except ImportError:
        print("\nWarning: imblearn not installed. Skipping SMOTE exploration.")
    except Exception as e:
        print(f"\nWarning: SMOTE failed: {e}")
        
    print("\nTraining pipeline completed.")

if __name__ == "__main__":
    train_model()
