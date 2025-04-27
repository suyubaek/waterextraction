import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from models.CBAM_U_net import UNet_CBAM
from utils.dataloading import SegmentationDataset, NpySegmentationDataset
from utils.model_eval import calculate_accuracy, calculate_iou, validate_model, load_model_if_exists
from configs.paths import PRETRAINED_MODEL_DIR
from configs.trainconfig import TrainConfig

# 直接在这里定义DiceLoss类
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
            
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)  # 应用sigmoid激活
            
        # 展平预测和真实值
        inputs = inputs.view(-1)
        targets = targets.view(-1)
            
        # 计算交集
        intersection = (inputs * targets).sum()
            
        # 计算Dice系数
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
            
        # 返回Dice损失
        return 1 - dice

def train_model_with_gradual_unfreezing(
    data_path: str,
    model: nn.Module,
    device: torch.device,
    epochs_per_stage: list,
    batch_size: int = 8,
    learning_rates: list = [1e-4, 5e-5, 1e-5],
    val_percent: float = 0.2,
    history_save_path: str = "",
    is_npy_dataset: bool = False,
):
    """
    训练模型，采用逐层解冻的方式进行迁移学习
    
    Args:
        data_path: 数据集路径
        model: 预训练模型
        device: 计算设备
        epochs_per_stage: 每个解冻阶段的训练轮数列表
        batch_size: 批次大小
        learning_rates: 每个阶段的学习率列表
        val_percent: 验证集比例
        history_save_path: 模型保存路径
        is_npy_dataset: 是否为npy格式数据集
    
    Returns:
        tuple: (训练后的模型, 训练历史记录)
    """
    # 创建输出目录
    os.makedirs(os.path.dirname(history_save_path), exist_ok=True)
    
    # 设置数据路径
    if is_npy_dataset:
        image_dir = os.path.join(data_path, "filter_images")
        mask_dir = os.path.join(data_path, "filter_masks")
    else:
        image_dir = os.path.join(data_path, "images")
        mask_dir = os.path.join(data_path, "masks")
    
    # 定义图像变换
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
    
    
    
    criterion = DiceLoss()
    
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
    
    # 设置最佳验证损失的初始值
    best_val_loss = float('inf')
    start_epoch = 0
    
    # 定义解冻阶段，根据CBAM_U_NET.py中的实际模块名称调整
    unfreezing_stages = [
        {'name': 'Only train classifier', 'params_to_unfreeze': []},
        {'name': 'Unfreeze up4', 'params_to_unfreeze': ['up4']},  # 最后一个解码器层
        {'name': 'Unfreeze up3', 'params_to_unfreeze': ['up3']},
        {'name': 'Unfreeze up2', 'params_to_unfreeze': ['up2']},
        {'name': 'Unfreeze up1', 'params_to_unfreeze': ['up1']},  # 第一个解码器层
        {'name': 'Unfreeze all', 'params_to_unfreeze': ['down4', 'down3', 'down2', 'down1', 'inc']},  # 所有编码器层
    ]
    
    # 确保学习率列表长度与解冻阶段数量匹配
    if len(learning_rates) < len(unfreezing_stages):
        learning_rates.extend([learning_rates[-1]] * (len(unfreezing_stages) - len(learning_rates)))
        
    # 确保epochs_per_stage列表长度与解冻阶段数量匹配
    if len(epochs_per_stage) < len(unfreezing_stages):
        epochs_per_stage.extend([epochs_per_stage[-1]] * (len(unfreezing_stages) - len(epochs_per_stage)))
    
    # 逐阶段解冻并训练
    for stage_idx, stage in enumerate(unfreezing_stages):
        print(f"\n{'='*20} 阶段 {stage_idx + 1}: {stage['name']} {'='*20}")
        
        # 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False
            
        # 解冻分类器（最后一层）
        for param in model.outc.parameters():
            param.requires_grad = True
            
        # 解冻当前阶段和之前阶段的参数
        for i in range(stage_idx + 1):
            for param_name in unfreezing_stages[i]['params_to_unfreeze']:
                if hasattr(model, param_name):
                    for param in getattr(model, param_name).parameters():
                        param.requires_grad = True
                    print(f"已解冻: {param_name}")
                else:
                    print(f"警告：模型中找不到模块 {param_name}，无法解冻！")
        # 检查哪些层已解冻
        trainable_params = 0
        total_params = 0
        for name, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"可训练参数: {trainable_params:,} / {total_params:,} = {100 * trainable_params / total_params:.2f}%")
        
        # 设置优化器和学习率调度器
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                              lr=learning_rates[stage_idx])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )
        
        # 确定当前阶段的训练轮数
        current_epochs = epochs_per_stage[stage_idx]
        
        # 训练当前阶段
        for epoch in range(current_epochs):
            model.train()
            train_loss = 0.0
            train_accuracy = 0.0
            train_iou = 0.0
            
            train_pbar = tqdm(train_loader, desc=f"阶段 {stage_idx+1}/{len(unfreezing_stages)}, 轮次 {epoch+1}/{current_epochs} [训练]")
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
            print(f"阶段 {stage_idx+1}/{len(unfreezing_stages)}, 轮次 {epoch+1}/{current_epochs} - "
                  f"训练损失: {train_loss:.4f}, 训练IoU: {train_iou:.4f}, "
                  f"验证损失: {val_loss:.4f}, 验证IoU: {val_iou:.4f}, "
                  f"学习率: {current_lr:.8f}")
            
            # 保存当前模型
            global_epoch = start_epoch + sum(epochs_per_stage[:stage_idx]) + epoch
            # epoch_save_path = os.path.join(history_save_path, f"checkpoint_epoch_{global_epoch+1}.pth")
            
            # torch.save({
            #     'epoch': global_epoch + 1,
            #     'stage': stage_idx,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'scheduler_state_dict': scheduler.state_dict(),
            #     'train_loss': train_loss,
            #     'val_loss': val_loss,
            #     'val_iou': val_iou,
            #     'history': history
            # }, epoch_save_path)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(os.path.dirname(history_save_path), "best_model.pth")
                torch.save({
                    'epoch': global_epoch + 1,
                    'stage': stage_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_iou': val_iou,
                    'val_loss': val_loss,
                }, best_model_path)
                print(f"已保存新的最佳模型 (验证损失: {val_loss:.4f})")
        
        # 更新开始轮次
        start_epoch += current_epochs
    
    # 保存完整训练历史
    history_file = os.path.join(os.path.dirname(history_save_path), "training_history.pth")
    torch.save(history, history_file)
    
    # 绘制训练历史曲线
    plot_training_history(history, os.path.dirname(history_save_path))
    
    return model, history

def plot_training_history(history, output_dir):
    """
    绘制训练历史曲线
    
    Args:
        history (dict): 训练历史记录
        output_dir (str): 输出目录
    """
    plt.figure(figsize=(18, 10))
    
    # 绘制损失曲线
    plt.subplot(2, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制IoU曲线
    plt.subplot(2, 3, 2)
    plt.plot(history['train_iou'], label='Train IoU')
    plt.plot(history['val_iou'], label='Validation IoU')
    plt.title('IoU Curves')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率曲线
    plt.subplot(2, 3, 3)
    plt.plot(history['train_accuracy'], label='Train Acc')
    plt.plot(history['val_accuracy'], label='Validation Acc')
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
    epochs_per_stage=[10, 10, 5, 5, 5, 3],
    batch_size=8,
    learning_rates=[1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6],
    history_save_path="",
    is_npy_dataset=False
):
    """
    加载现有模型并继续训练
    
    Args:
        data_path: 数据集路径
        checkpoint_path: 检查点路径
        device: 训练设备
        epochs_per_stage: 每个解冻阶段的训练轮数
        batch_size: 批次大小
        learning_rates: 每个阶段的学习率
        history_save_path: 模型保存路径
        is_npy_dataset: 是否为npy格式数据集
        
    Returns:
        tuple: (训练后的模型, 训练历史记录)
    """
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 初始化模型
    model = UNet_CBAM(in_channels=3, out_channels=1)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # 获取当前阶段和轮次
    current_stage = checkpoint.get('stage', 0)
    current_epoch = checkpoint.get('epoch', 0)
    print(f"从阶段 {current_stage+1}, 轮次 {current_epoch} 继续训练")
    
    # 修改epochs_per_stage，跳过已完成的阶段
    epochs_per_stage = epochs_per_stage[current_stage:]
    learning_rates = learning_rates[current_stage:]
    
    # 继续训练
    return train_model_with_gradual_unfreezing(
        data_path=data_path,
        model=model,
        device=device,
        epochs_per_stage=epochs_per_stage,
        batch_size=batch_size,
        learning_rates=learning_rates,
        history_save_path=history_save_path,
        is_npy_dataset=is_npy_dataset
    )

if __name__ == "__main__":
    # 设置随机种子，确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 定义数据路径
    #source_data_path = "E:\Thesis2025\\all_clip_256\\filter0585"
    target_data_path = "E:\\Thesis2025\\SWED\\SWED\\train"
    output_dir = "E:\Thesis2025\\autodle"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化 CBAM-UNet 模型
    model = UNet_CBAM(in_channels=3, out_channels=1)
    
    # 尝试加载预训练模型
    pretrained_model_path = "E:\\Thesis2025\\autodl\\best_model_gf_v3.pth"
    model, loaded, _ = load_model_if_exists(model, pretrained_model_path, device)
    
    if loaded:
        print("成功加载预训练模型")
    else:
        print("未找到预训练模型，将使用随机初始化权重")
    
    # 设置训练参数[10, 8, 6, 6, 4, 4] 
    epochs_per_stage = [1, 1, 1, 1, 1, 1]  # 每个阶段训练的轮数
    learning_rates = [1e-5, 5e-6, 2e-6, 1e-6, 5e-7, 2e-7]  # 每个阶段的学习率
    batch_size = 8
    
    # 执行逐层解冻训练
    print("开始 CBAM-UNet 模型的逐层解冻迁移学习训练")
    model, history = train_model_with_gradual_unfreezing(
        data_path=target_data_path,
        model=model,
        device=device,
        epochs_per_stage=epochs_per_stage,
        batch_size=batch_size,
        learning_rates=learning_rates,
        history_save_path=os.path.join(output_dir, "training_history_cbam_v2.pth"),
        is_npy_dataset=True  # 设置为True如果目标数据集是npy格式
    )
    
    print("训练完成！最终模型已保存。")
