import lightgbm as lgb
import numpy as np
import pickle
import os
import sys
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, jaccard_score

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR, MODEL_PATH, MODELS_DIR

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
    
    # Parameters from instructions
    clf = lgb.LGBMClassifier(
        num_leaves=48,
        learning_rate=0.03,
        n_estimators=2000,
        min_data_in_leaf=20,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        class_weight={0:1, 1:50},
        objective="binary",
        verbose=-1
    )
    
    print("Training LightGBM model on Training Set...")
    clf.fit(X_train, y_train)
    
    # --- Evaluation on Hold-out Set ---
    print("\nEvaluating on Hold-out Test Set...")
    y_pred = clf.predict(X_test)
    y_prob = np.array(clf.predict_proba(X_test))[:, 1]
    
    y_pred = np.array(y_pred)
    
    p = precision_score(y_test, y_pred, zero_division=0)
    r = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    if np.sum(y_test) > 0:
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = 0.0
        
    iou = jaccard_score(y_test, y_pred, zero_division=0)
    
    print("-" * 50)
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUROC:     {auc:.4f}")
    print(f"IoU:       {iou:.4f}")
    print("-" * 50)
    
    # Check if passed
    if f1 >= 0.90:
        print("RESULT: PASS (F1 >= 0.90)")
    else:
        print("RESULT: FAIL (F1 < 0.90)")

    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print(f"Saving model to {MODEL_PATH}...")
    # Save as text format as requested
    clf.booster_.save_model(MODEL_PATH)
    
    print("Training pipeline completed.")

if __name__ == "__main__":
    train_model()
