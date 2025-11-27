import numpy as np
import json
import os
import sys

# Ensure we can import config if running from this directory or parent
try:
    from config import LABELS_PATH
except ImportError:
    from src.config import LABELS_PATH

def load_labels():
    with open(LABELS_PATH, "r") as f:
        return json.load(f)

def load_npz(path):
    return np.load(path)
