#这个文件主要包含图像和数据预处理的函数
import torch

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

