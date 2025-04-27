import numpy as np
import torch
from PIL import Image
import os
import matplotlib.pyplot as plt

def create_padding_mask(image, threshold=1):
    """
    创建填充掩码，标记图像中的黑色填充区域。
    
    Args:
        image: PIL Image或numpy数组
        threshold: 像素值小于此阈值被视为填充区域
    
    Returns:
        掩码数组，1表示填充区域，0表示有效区域
    """
    # 确保图像是numpy数组
    if isinstance(image, Image.Image):
        image_array = np.array(image)
    else:
        image_array = image
    
    # 如果是三通道图像，求和判断是否全为0
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        padding_mask = (image_array.sum(axis=2) < threshold).astype(np.float32)
    else:
        padding_mask = (image_array < threshold).astype(np.float32)
    
    return padding_mask

def create_weighted_loss_mask(image, mask=None):
    """
    创建用于损失函数的权重掩码，降低填充区域的权重。
    
    Args:
        image: 输入图像(PIL Image或numpy数组)
        mask: 可选的已有掩码
    
    Returns:
        权重掩码，填充区域权重为0，有效区域为1
    """
    if mask is None:
        padding_mask = create_padding_mask(image)
    else:
        padding_mask = mask
    
    # 反转掩码，使填充区域权重为0，有效区域为1
    weight_mask = 1.0 - padding_mask
    
    return weight_mask

class WeightedBCEWithLogitsLoss(torch.nn.Module):
    """
    加权BCE损失，可以根据权重掩码减少填充区域的影响
    """
    def __init__(self, reduction='mean'):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, input, target, weight=None):
        """
        计算加权BCE损失
        
        Args:
            input: 模型输出的预测值
            target: 真实标签
            weight: 权重掩码，形状与input相同，填充区域为0，有效区域为1
        """
        # 标准BCE损失
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            input, target, reduction='none'
        )
        
        # 如果提供了权重掩码，则应用权重
        if weight is not None:
            loss = loss * weight
            
        # 应用reduction
        if self.reduction == 'mean':
            # 计算有效区域(权重>0)的平均损失
            if weight is not None:
                # 避免除以0
                valid_pixels = torch.sum(weight > 0)
                if valid_pixels > 0:
                    return torch.sum(loss) / valid_pixels
                else:
                    return torch.sum(loss) * 0.0  # 返回0损失
            else:
                return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:  # 'none'
            return loss

def visualize_padding_mask(image_path, save_path=None):
    """
    可视化图像和其填充掩码，用于调试
    
    Args:
        image_path: 图像路径
        save_path: 可选的保存路径
    """
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)
    
    # 创建填充掩码
    padding_mask = create_padding_mask(image_array)
    
    # 可视化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image_array)
    plt.title('原始图像')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(padding_mask, cmap='gray')
    plt.title('填充掩码 (白色=填充区域)')
    plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"可视化结果已保存至 {save_path}")
    plt.show()
