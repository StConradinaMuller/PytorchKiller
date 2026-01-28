"""
CI smoke test: install requirements and run a minimal import + tiny forward pass on CPU.
This script is intended to run quickly on GitHub Actions runners.
"""
import torch
import torchvision
import numpy as np

def smoke():
    print("torch version:", torch.__version__)
    # tiny model forward pass
    from models.simple_cnn import SimpleCNN
    model = SimpleCNN(in_channels=1, num_classes=10)
    model.eval()
    x = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        out = model(x)
    print("forward ok, output shape:", out.shape)
    # basic numpy check
    a = np.array([1,2,3])
    print("numpy ok, sum:", a.sum())

if __name__ == "__main__":
    smoke()
