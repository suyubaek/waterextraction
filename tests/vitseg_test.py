import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from models.vitseg import ViT_UNet


if __name__ == "__main__":
    model = ViT_UNet()
    inp = torch.randn(2, 3, 512, 512)
    out = model(inp)
    print(out.shape)  # (2, 1, 512, 512)