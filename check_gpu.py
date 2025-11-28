import torch
import sys
import os

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is NOT available.")
    # Check if it's a CPU-only build
    if "+cpu" in torch.__version__:
        print("Reason: You have a CPU-only version of PyTorch installed.")
    else:
        print("Reason: PyTorch could not find a GPU. Possible driver issue or CUDA mismatch.")
