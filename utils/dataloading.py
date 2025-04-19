from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
import torch
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.datapretreat import normalize_like_original
#这个主要包含数据的加载

#这个类针对图像型的数据输入
class SegmentationDataset(Dataset):
    """
    Dataset class for loading image and mask pairs for segmentation tasks.
    """
    def __init__(self, image_dir, mask_dir, image_names, transform=None, mask_transform=None):
        """
        Initialize the dataset.

        Args:
            image_dir (str): Directory containing images.
            mask_dir (str): Directory containing masks.
            image_names (list): List of image filenames.
            transform (callable, optional): Transform to be applied to images.
            mask_transform (callable, optional): Transform to be applied to masks.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = image_names
        self.transform = transform
        self.mask_transform = mask_transform if mask_transform else transforms.ToTensor()

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.image_names)

    def __getitem__(self, idx):
        """
        Get an image and its corresponding mask.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (image, mask) pair.
        """
        img_name = self.image_names[idx]
        mask_name = img_name.replace('image', 'mask')
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        #这个地方应该没问题了，这个地方用的PIL库，默认就是RGB格式
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.transform:
            image = self.transform(image)
        mask = self.mask_transform(mask)
        return image, mask
#这里也需要小心，可能会有问题
#这个类主要针对npy格式的数据输入
class NpySegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_names, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = image_names
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        mask_name = img_name.replace('image', 'mask')
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Load numpy arrays
        image = np.load(img_path, allow_pickle=True)  # Assume shape is (H, W, C)
        mask = np.load(mask_path, allow_pickle=True)  # Assume shape is (H, W)
        #print(image.shape, mask.shape)  # Debugging line to check shapes

        # Select specific bands (e.g., bands 4, 3, 2)
        #重大问题调整
        image = image[:, :, [3, 2, 1]]  # 调整为通道顺序 3, 2, 1  RGB
        
        #转为标准化的tensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()         
        image=normalize_like_original(image)  # Normalize to ImageNet standards  
        mask = mask.astype(np.float32)  # 确保 NumPy 数组为浮点类型
        mask = torch.from_numpy(mask)  # 转换为 PyTorch 张量
        mask = (mask > 0).float()  # 转换为二值图像并确保为浮点类型
        #mask = torch.from_numpy(mask).float().unsqueeze(0)  # Add channel dimension
        # 修复：确保 mask 的形状为 [batch_size, channels, height, width]       
        return image, mask