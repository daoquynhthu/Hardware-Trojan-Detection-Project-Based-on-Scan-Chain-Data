import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SEQS_DATA_DIR = os.path.join(DATA_DIR, "seqs_data")
LABELS_PATH = os.path.join(DATA_DIR, "labels.json")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "lgbm_model.txt")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
