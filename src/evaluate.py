import numpy as np
import pickle
import os
import sys
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, jaccard_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR

def evaluate_lodo():
    dataset_path = os.path.join(DATA_DIR, "dataset.pkl")
    if not os.path.exists(dataset_path):
        print("Dataset not found. Run dataset_builder.py first.")
        return

    print("Loading dataset...")
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
        
    X = data["X"]
    y = data["y"]
    design_ids = data["design_ids"]
    filenames = data.get("filenames", [])
    
    unique_designs = np.unique(design_ids)
    results = []
    
    print(f"Starting Leave-One-Design-Out (LODO) evaluation on {len(unique_designs)} designs...")
    print("-" * 110)
    print(f"{'Design':<25} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'AUC':<6} | {'IoU':<6} | {'P@20':<6} | {'R@20':<6}")
    print("-" * 110)
    
    params = {
        'num_leaves': 48,
        'learning_rate': 0.03,
        'n_estimators': 2000,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'class_weight': {0:1, 1:50},
        'objective': 'binary',
        'verbose': -1,
        'n_jobs': 1 # Avoid overloading if running many loops, or use -1
    }
    
    for d_id in unique_designs:
        d_id = int(d_id)
        # Split
        mask_test = (design_ids == d_id)
        mask_train = ~mask_test
        
        X_train, y_train = X[mask_train], y[mask_train]
        X_test, y_test = X[mask_test], y[mask_test]
        
        if len(y_test) == 0:
            continue
            
        # Train
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_train, y_train)
        
        # Predict
        y_pred = clf.predict(X_test)
        y_prob = np.array(clf.predict_proba(X_test))[:, 1]
        
        # Explicitly cast to numpy array to satisfy linter
        y_pred = np.array(y_pred)

        # Metrics
        p = precision_score(y_test, y_pred, zero_division=0)
        r = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # AUC (Check if there are positive samples in test)
        if np.sum(y_test) > 0 and np.sum(y_test) < len(y_test):
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = 0.0 # Undefined
            
        # IoU (Jaccard)
        iou = jaccard_score(y_test, y_pred, zero_division=0)

        
        # Top-K Metrics (K=20)
        k = 20
        # Handle case where n_regs < k
        k_actual = min(k, len(y_prob))
        top_k_indices = np.argsort(y_prob)[-k_actual:]
        
        tp_top_k = np.sum(y_test[top_k_indices])
        total_pos = np.sum(y_test)
        
        precision_top20 = tp_top_k / k_actual if k_actual > 0 else 0.0
        recall_top20 = tp_top_k / total_pos if total_pos > 0 else 0.0
        
        fname = filenames[d_id] if d_id < len(filenames) else str(d_id)
        # Truncate fname for display
        fname_disp = (fname[:22] + '..') if len(fname) > 22 else fname
        
        print(f"{fname_disp:<25} | {p:.3f}  | {r:.3f}  | {f1:.3f}  | {auc:.3f}  | {iou:.3f}  | {precision_top20:.3f}  | {recall_top20:.3f}")
        
        results.append({
            "design": d_id,
            "precision": p,
            "recall": r,
            "f1": f1,
            "auc": auc,
            "iou": iou,
            "precision_top20": precision_top20,
            "recall_top20": recall_top20
        })
        
    print("-" * 110)
    # Average
    avg_f1 = np.mean([r["f1"] for r in results])
    avg_r = np.mean([r["recall"] for r in results])
    avg_p = np.mean([r["precision"] for r in results])
    avg_auc = np.mean([r["auc"] for r in results])
    avg_iou = np.mean([r["iou"] for r in results])
    
    print(f"Average Precision: {avg_p:.3f}")
    print(f"Average Recall:    {avg_r:.3f}")
    print(f"Average F1:        {avg_f1:.3f}")
    print(f"Average AUC:       {avg_auc:.3f}")
    print(f"Average IoU:       {avg_iou:.3f}")
    
    if avg_f1 >= 0.90:
        print("RESULT: PASS (F1 >= 0.90)")
    else:
        print("RESULT: FAIL (F1 < 0.90) - Consider tuning parameters or adding features.")

if __name__ == "__main__":
    evaluate_lodo()
