#这个文件主要包含图像和数据预处理的函数
import torch
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
#处理npy格式数据，这里还存疑，后面需要调整
def normalize_like_original(image):
    """
    Normalize the input image to ImageNet standards.
    If the input range is 0-2000+, first normalize it to 0-1.
    
    Args:
        image (torch.Tensor): Input image tensor with shape (C, H, W).
        
    Returns:
        torch.Tensor: Normalized image tensor.
    """
    # Step 1: Normalize to [0, 1] range
    if image.max() > 0:  # Avoid division by zero
        image = image / image.max()
    
    # Step 2: Apply ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = (image - mean) / std
    
    return image
def analyze_npy_files(folder_path):
        """
        Analyzes all .npy files in a folder and reports their value ranges.
        
        Args:
            folder_path (str): Path to the folder containing .npy files
        """
        npy_files = glob(os.path.join(folder_path, '*.npy'))
        
        if not npy_files:
            print(f"No .npy files found in {folder_path}")
            return
        
        print(f"Found {len(npy_files)} .npy files in {folder_path}")
        
        min_values = []
        max_values = []
        mean_values = []
        std_values = []
        
        for file_path in npy_files:
            data = np.load(file_path)
            data=data[:, :, [3, 2, 1]] # Adjust to channel order 3, 2, 1 RGB
            file_name = os.path.basename(file_path)
            
            min_val = data.min()
            max_val = data.max()
            mean_val = data.mean()
            std_val = data.std()
            
            min_values.append(min_val)
            max_values.append(max_val)
            mean_values.append(mean_val)
            std_values.append(std_val)
            
            # print(f"File: {file_name}")
            # print(f"  Shape: {data.shape}")
            # print(f"  Data type: {data.dtype}")
            # print(f"  Min: {min_val}")
            # print(f"  Max: {max_val}")
            # print(f"  Mean: {mean_val}")
            # print(f"  Std: {std_val}\n")
        
        # Overall statistics
        print("Overall Statistics:")
        print(f"  Min value across all files: {min(min_values)}")
        print(f"  Max value across all files: {max(max_values)}")
        print(f"  Average mean value: {sum(mean_values)/len(mean_values)}")
        print(f"  Average standard deviation: {sum(std_values)/len(std_values)}")
        
        # Optional: visualize distribution of a sample file
        if len(npy_files) > 0:
            sample_data = np.load(npy_files[0]).flatten()
            plt.figure(figsize=(10, 6))
            plt.hist(sample_data, bins=50)
            plt.title(f"Value distribution for {os.path.basename(npy_files[0])}")
            plt.xlabel("Pixel values")
            plt.ylabel("Frequency")
            plt.show()
if __name__ == "__main__":
    # 测试函数
    
    # Test the function with a sample folder path
    folder_path = "E:\Thesis2025\SWED\SWED\\train\\filter_images"
    analyze_npy_files(folder_path)