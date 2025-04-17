import os
from torch.utils.data import Dataset, DataLoader
import torch
import cv2
import numpy as np

class WaterSegDataset(Dataset):
    def __init__(self, folder, img_size=512, transform=None):
        self.folder = folder
        self.img_size = img_size
        self.transform = transform
        self.samples = []
        for fname in os.listdir(folder):
            if fname.lower().endswith('.png'):
                mask_path = os.path.join(folder, fname)
                img_name = os.path.splitext(fname)[0] + '.tif'
                img_path = os.path.join(folder, img_name)
                if os.path.exists(img_path):
                    self.samples.append((img_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        # 读取影像（tif），BGR转RGB
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"影像读取失败: {img_path}")
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # (C, H, W)

        # 读取mask
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"掩码读取失败: {mask_path}")
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        if len(mask.shape) == 3:
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)

        if self.transform:
            img, mask = self.transform(img, mask)
        return img, mask

def get_loader(folder, batch_size=4, img_size=512, shuffle=True, num_workers=2, transform=None):
    dataset = WaterSegDataset(folder, img_size=img_size, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)