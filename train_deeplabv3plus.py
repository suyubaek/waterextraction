import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from models.DeepLabV3plus import DeepLabV3Plus
from utils.dataloading import SegmentationDataset, NpySegmentationDataset
from utils.model_eval import calculate_accuracy, calculate_iou, validate_model

def train_deeplabv3_plus(
    data_path: str,
    model: nn.Module,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    val_percent: float = 0.2,
    start_epoch: int = 0,
    history_save_path: str = "",
    is_npy_dataset: bool = False,
):
    """
    训练DeepLabV3+模型进行图像分割。
    
    Args:
        data_path (str): 数据集路径
        model (nn.Module): DeepLabV3+模型实例
        device (torch.device): 训练设备 (CPU/GPU)
        epochs (int): 训练轮数
        batch_size (int): 批次大小
        learning_rate (float): 学习率
        val_percent (float): 验证集所占比例
        start_epoch (int): 起始轮数（用于恢复训练）
        history_save_path (str): 模型和历史记录保存路径
        is_npy_dataset (bool): 是否为 .npy 格式的数据集
    
    Returns:
        tuple: 训练后的模型和训练历史记录
    """
    # 创建输出目录
    os.makedirs(os.path.dirname(history_save_path), exist_ok=True)
    
    # 设置数据路径
    image_dir = os.path.join(data_path, "images")
    mask_dir = os.path.join(data_path, "masks")
    
    # 图像变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 准备数据集
    image_names = [f for f in os.listdir(image_dir) if f.endswith(('.tif', '.jpg', '.png', '.npy'))]
    train_names, val_names = train_test_split(image_names, test_size=val_percent, random_state=42)
    
    # 创建数据集和加载器
    if is_npy_dataset:
        train_dataset = NpySegmentationDataset(image_dir, mask_dir, train_names, transform=transform)
        val_dataset = NpySegmentationDataset(image_dir, mask_dir, val_names, transform=transform)
    else:
        train_dataset = SegmentationDataset(image_dir, mask_dir, train_names, transform=transform)
        val_dataset = SegmentationDataset(image_dir, mask_dir, val_names, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    
    # 将模型移到设备
    model.to(device)
    
    # 初始化训练历史记录
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'train_iou': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_iou': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'learning_rates': []
    }
    
    # 记录最佳验证损失，用于保存最佳模型
    best_val_loss = float('inf')
    
    # 训练循环
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        train_iou = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch + epochs} [Train]")
        for images, masks in train_pbar:
            images, masks = images.to(device), masks.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss = criterion(outputs, masks)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 计算指标
            train_loss += loss.item()
            
            # 转换输出为概率
            probs = torch.sigmoid(outputs)
            
            # 计算准确率和IoU
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
        
        # 计算训练集上的平均指标
        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)
        train_iou /= len(train_loader)
        
        # 验证阶段
        val_loss, val_accuracy, val_iou, val_precision, val_recall, val_f1 = validate_model(
            model, val_loader, criterion, device
        )
        
        # 更新学习率
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 保存指标到历史记录
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_iou'].append(val_iou)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        history['learning_rates'].append(current_lr)
        
        # 输出本轮结果
        print(f"Epoch {epoch+1}/{start_epoch + epochs} - "
              f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}, "
              f"LR: {current_lr:.8f}")
        
        # # 保存当前轮次模型
        # epoch_save_path = os.path.join(history_save_path, f"checkpoint_epoch_{epoch+1}.pth")
        # torch.save({
        #     'epoch': epoch + 1,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'scheduler_state_dict': scheduler.state_dict(),
        #     'train_loss': train_loss,
        #     'val_loss': val_loss,
        #     'val_iou': val_iou,
        # }, epoch_save_path)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(history_save_path, "best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_iou': val_iou,
                'val_loss': val_loss,
            }, best_model_path)
            print(f"保存最佳模型 (Val Loss: {val_loss:.4f})")
    
    # 保存完整训练历史
    history_file = os.path.join(history_save_path, "training_history.pth")
    torch.save(history, history_file)
    
    # 绘制训练历史曲线
    plot_training_history(history, os.path.dirname(history_save_path))
    
    return model, history

def plot_training_history(history, output_dir):
    """
    绘制训练历史曲线
    
    Args:
        history (dict): 包含训练历史数据的字典
        output_dir (str): 输出目录
    """
    plt.figure(figsize=(18, 10))
    
    # 绘制损失曲线
    plt.subplot(2, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制IoU曲线
    plt.subplot(2, 3, 2)
    plt.plot(history['train_iou'], label='Train IoU')
    plt.plot(history['val_iou'], label='Val IoU')
    plt.title('IoU Curves')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率曲线
    plt.subplot(2, 3, 3)
    plt.plot(history['train_accuracy'], label='Train Acc')
    plt.plot(history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 绘制精确率、召回率和F1值曲线
    plt.subplot(2, 3, 4)
    plt.plot(history['val_precision'], label='Precision')
    plt.plot(history['val_recall'], label='Recall')
    plt.plot(history['val_f1'], label='F1 Score')
    plt.title('Precision/Recall/F1 Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    # 绘制学习率曲线
    plt.subplot(2, 3, 5)
    plt.semilogy(history['learning_rates'], label='Learning Rate')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate (log scale)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

def load_model_and_resume_training(
    data_path, 
    checkpoint_path, 
    device, 
    epochs=30, 
    batch_size=8,
    learning_rate=1e-4,
    history_save_path="",
    is_npy_dataset=False
):
    """
    加载现有模型并继续训练
    
    Args:
        data_path (str): 数据集路径
        checkpoint_path (str): 检查点路径
        device (torch.device): 训练设备
        epochs (int): 继续训练的轮数
        batch_size (int): 批次大小
        learning_rate (float): 学习率
        history_save_path (str): 历史记录保存路径
        is_npy_dataset (bool): 是否为.npy格式数据集
        
    Returns:
        tuple: 训练后的模型和训练历史记录
    """
    # 初始化模型
    model = DeepLabV3Plus(in_channels=3, num_classes=1)
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint.get('epoch', 0)
    
    print(f"加载模型成功，从第 {start_epoch} 轮继续训练")
    
    # 继续训练
    return train_deeplabv3_plus(
        data_path=data_path,
        model=model,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        start_epoch=start_epoch,
        history_save_path=history_save_path,
        is_npy_dataset=is_npy_dataset
    )

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检查GPU可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 数据集路径
    data_path = "E:\\Thesis2025\\all_clip_256\\filter0585"
    
    # 输出目录
    output_dir = "E:\\Thesis2025\\deeplabv3plus_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化模型
    model = DeepLabV3Plus(in_channels=3, num_classes=1)
    
    # 训练参数
    epochs = 3
    batch_size = 8
    learning_rate = 1e-4
    
    # 训练模型
    print("开始训练DeepLabV3+模型...")
    model, history = train_deeplabv3_plus(
        data_path=data_path,
        model=model,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        history_save_path=os.path.join(output_dir),
        is_npy_dataset=False  # 根据实际数据集格式修改
    )
    
    print("训练完成！")
