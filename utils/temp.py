import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.waterseg_dataset import get_loader

if __name__ == "__main__":
    root_dir = "/d/chenhongxian/traingf2"  # 替换为你的数据集根目录
    loader = get_loader(root_dir, batch_size=2, img_size=256, shuffle=True, num_workers=0)
    for i, (imgs, masks) in enumerate(loader):
        print(f"Batch {i}:")
        print("  imgs.shape:", imgs.shape)
        print("  masks.shape:", masks.shape)
        if i == 2:  # 只查看前3个batch
            break