from tqdm import tqdm
import torch
from utils.model_eval import calculate_accuracy, calculate_iou

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, total_epochs, stage):
    """训练一个epoch并返回平均损失和指标"""
    model.train()
    train_loss = 0.0
    train_accuracy = 0.0
    train_iou = 0.0
    
    train_pbar = tqdm(dataloader, desc=f"Stage {stage}, Epoch {epoch+1}/{total_epochs} [Train]")
    for images, masks in train_pbar:
        images, masks = images.to(device), masks.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 计算指标
        train_loss += loss.item()
        probs = torch.sigmoid(outputs)
        batch_accuracy = calculate_accuracy(probs, masks)
        batch_iou = calculate_iou(probs, masks)
        
        train_accuracy += batch_accuracy
        train_iou += batch_iou
        
        # 更新进度条
        train_pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'accuracy': f"{batch_accuracy:.4f}",
            'IoU': f"{batch_iou:.4f}"
        })
    
    # 计算平均指标
    avg_train_loss = train_loss / len(dataloader)
    avg_train_accuracy = train_accuracy / len(dataloader)
    avg_train_iou = train_iou / len(dataloader)
    
    return avg_train_loss, avg_train_accuracy, avg_train_iou