import os
from torch.utils.data import Dataset, DataLoader
import torch
import cv2
import numpy as np

class WaterSegDataset(Dataset):
    def __init__(self, samples, img_size=256, transform=None):
        self.samples = samples
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        # 读取原图（RGB三通道）
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"影像读取失败: {img_path}")
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0  # 归一化
        img = torch.from_numpy(img).permute(2, 0, 1)  # (C, H, W)

        # 读取mask（RGB三通道，实际为二值图）
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"掩码读取失败: {mask_path}")
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        # 转为单通道二值图
        if len(mask.shape) == 3:
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)

        if self.transform:
            img, mask = self.transform(img, mask)
        return img, mask

def make_dataset(root_dir, split_ratio=0.8, shuffle=True):
    img_dir = os.path.join(root_dir, 'images')
    mask_dir = os.path.join(root_dir, 'masks')
    samples = []
    for fname in os.listdir(img_dir):
        if fname.lower().endswith('.tif'):
            img_path = os.path.join(img_dir, fname)
            mask_name = fname.replace('_image_', '_mask_')
            mask_path = os.path.join(mask_dir, mask_name)
            if os.path.exists(mask_path):
                samples.append((img_path, mask_path))
    if shuffle:
        np.random.shuffle(samples)
    split = int(len(samples) * split_ratio)
    train_samples = samples[:split]
    test_samples = samples[split:]
    return train_samples, test_samples

def get_loaders(root_dir, batch_size=4, img_size=256, split_ratio=0.8, shuffle=True, num_workers=2, transform=None):
    train_samples, test_samples = make_dataset(root_dir, split_ratio, shuffle)
    train_set = WaterSegDataset(train_samples, img_size=img_size, transform=transform)
    test_set = WaterSegDataset(test_samples, img_size=img_size, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader