import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SEQS_DATA_DIR = os.path.join(DATA_DIR, "seqs_data")
LABELS_PATH = os.path.join(DATA_DIR, "labels.json")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LGBM_MODEL_DIR = os.path.join(MODELS_DIR, "lgbm")
TRANSFORMER_MODEL_DIR = os.path.join(MODELS_DIR, "transformer")
MODEL_PATH = os.path.join(LGBM_MODEL_DIR, "lgbm_model.txt")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# Hyperparameters
LGBM_PARAMS = {
    'num_leaves': 48,
    'learning_rate': 0.03,
    'n_estimators': 2000,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'class_weight': {0: 1, 1: 100},
    'objective': 'binary',
    'verbose': -1,
    'n_jobs': 1,
    'random_state': 42  # Fixed seed for reproducibility
}
