import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from models.vitseg import ViTseg
from datasets.waterseg_dataset import get_loader


if __name__ == "__main__":
    # 替换为你的数据集文件夹路径
    folder = "/Users/songyufei1/Documents/look_code_syf/water-extract-dl/data/water_seg"
    loader = get_loader(folder, batch_size=2, img_size=512, shuffle=False, num_workers=0)

    for i, (imgs, masks) in enumerate(loader):
        print(f"Batch {i}:")
        print(f"  imgs.shape: {imgs.shape}")   # (B, 3, 512, 512)
        print(f"  masks.shape: {masks.shape}") # (B, 1, 512, 512)
        print(f"  imgs dtype: {imgs.dtype}, masks dtype: {masks.dtype}")
        print(f"  imgs min/max: {imgs.min().item():.3f}/{imgs.max().item():.3f}")
        print(f"  masks unique: {masks.unique()}")
        if i >= 1:  # 只测试前2个batch
            break