import torch

def binary_iou(pred, target, threshold=0.5, eps=1e-6):
    """
    计算二分类分割的IoU（Intersection over Union）
    pred: (N, H, W) 或 (N, 1, H, W) 预测概率或二值
    target: (N, H, W) 或 (N, 1, H, W) 真实标签（0/1）
    threshold: 概率阈值
    """
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)
    pred_bin = (pred > threshold).float()
    target_bin = (target > 0.5).float()
    intersection = (pred_bin * target_bin).sum(dim=(1,2))
    union = (pred_bin + target_bin - pred_bin * target_bin).sum(dim=(1,2))
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()

# ...existing code...

# if __name__ == "__main__":
#     # 构造一个简单的预测和标签
#     pred = torch.tensor([
#         [[0.8, 0.2],
#          [0.4, 0.9]]
#     ])  # shape: (1, 2, 2)
#     target = torch.tensor([
#         [[1, 0],
#          [1, 1]]
#     ])  # shape: (1, 2, 2)

#     iou = binary_iou(pred, target, threshold=0.5)
#     print(f"Demo IoU: {iou:.4f}")