import torch
import torch.nn as nn
import os
from sklearn.model_selection import train_test_split
from utils.dataloading import NpySegmentationDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from train import transform
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.model_eval import  validate_model
from utils.train_one_epoch import train_one_epoch
from utils.history import plot_training_history
from models.U_net import UNet
from configs.paths import PRETRAINED_MODEL_DIR, DATASET_DIR
from utils.model_eval import load_model_if_exists

data_path=DATASET_DIR
def train_model_with_progressive_unfreezing(
    data_path: str,
    model: nn.Module,
    device: torch.device,
    epochs_per_stage: int = 5,  # 每个解冻阶段的训练轮数
    batch_size: int = 8,
    base_lr: float = 5e-6,
    val_percent: float = 0.2,
    history_save_path: str = "",
    in_channels: int = 3,
    out_channels: int = 1,
):
    # 数据加载部分保持不变
    image_dir = os.path.join(data_path, "images")
    mask_dir = os.path.join(data_path, "masks")
    output_dir = os.path.join(data_path, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    image_names = [f for f in os.listdir(image_dir) if f.endswith('.tif') or f.endswith('.npy') or f.endswith('.png')]
    train_names, val_names = train_test_split(image_names, test_size=val_percent, random_state=42)
    train_dataset = NpySegmentationDataset(image_dir, mask_dir, train_names, transform=transform)
    val_dataset = NpySegmentationDataset(image_dir, mask_dir, val_names, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 初始化模型（如果未提供）
    if model is None:
        model = UNet(in_channels=in_channels, out_channels=out_channels)
    model.to(device)
    
    # 损失函数
    criterion = nn.BCEWithLogitsLoss()
    
    # 历史记录
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
        'stage': []  # 记录当前是哪个解冻阶段
    }
    
    best_val_loss = float('inf')
    start_epoch = 0
    current_stage = 0
    total_epochs = 0
    
    # ===== 阶段1：仅训练编码器，冻结所有解码器 =====
    print("Stage 0: Training encoder only, all decoders frozen")
    current_stage = 0
    
    # 冻结所有解码器层
    for param in model.decoder1.parameters():
        param.requires_grad = False
    for param in model.decoder2.parameters():
        param.requires_grad = False
    for param in model.decoder3.parameters():
        param.requires_grad = False
    for param in model.decoder4.parameters():
        param.requires_grad = False
    
    # 为编码器部分设置较高的学习率
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                               lr=base_lr, weight_decay=1e-5)
    
    # 学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # 训练编码器部分
    for epoch in range(epochs_per_stage):
        total_epochs += 1
        train_loss, train_accuracy, train_iou = train_one_epoch(
            model, train_loader, optimizer, criterion, device, 
            epoch, epochs_per_stage, current_stage
        )
        
        val_loss, val_accuracy, val_iou, val_precision, val_recall, val_f1 = validate_model(
            model, val_loader, criterion, device
        )
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存历史记录
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_iou'].append(val_iou)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        history['stage'].append(current_stage)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(history_save_path, "best_model.pth")
            torch.save({
                'epoch': total_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_iou': val_iou,
                'val_loss': val_loss,
                'stage': current_stage
            }, best_model_path)
        
        # 保存每个epoch的检查点
        checkpoint_path = os.path.join(history_save_path, f"checkpoint_epoch_{total_epochs}.pth")
        torch.save({
            'epoch': total_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_iou': val_iou,
            'stage': current_stage
        }, checkpoint_path)
    
    # ===== 阶段2：解冻decoder1（最浅层解码器）=====
    print("Stage 1: Unfreezing decoder1 (shallowest decoder layer)")
    current_stage = 1
    
    # 解冻decoder1，其他decoder保持冻结
    for param in model.decoder1.parameters():
        param.requires_grad = True
    
    # 降低学习率，因为我们现在训练更少的参数
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr * 0.5
    
    # 重新初始化学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # 训练阶段2：编码器+decoder1
    for epoch in range(epochs_per_stage):
        total_epochs += 1
        train_loss, train_accuracy, train_iou = train_one_epoch(
            model, train_loader, optimizer, criterion, device, 
            epoch, epochs_per_stage, current_stage
        )
        
        val_loss, val_accuracy, val_iou, val_precision, val_recall, val_f1 = validate_model(
            model, val_loader, criterion, device
        )
        
        scheduler.step(val_loss)
        
        # 保存历史记录和模型（与阶段1相同）
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_iou'].append(val_iou)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        history['stage'].append(current_stage)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(history_save_path, "best_model.pth")
            torch.save({
                'epoch': total_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_iou': val_iou,
                'val_loss': val_loss,
                'stage': current_stage
            }, best_model_path)
        
        checkpoint_path = os.path.join(history_save_path, f"checkpoint_epoch_{total_epochs}.pth")
        torch.save({
            'epoch': total_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_iou': val_iou,
            'stage': current_stage
        }, checkpoint_path)
    
    # ===== 阶段3：解冻decoder2 =====
    print("Stage 2: Unfreezing decoder2")
    current_stage = 2
    
    # 解冻decoder2，其他保持状态
    for param in model.decoder2.parameters():
        param.requires_grad = True
    
    # 再次降低学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr * 0.25
    
    # 重新初始化学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # 训练阶段3
    for epoch in range(epochs_per_stage):
        total_epochs += 1
        train_loss, train_accuracy, train_iou = train_one_epoch(
            model, train_loader, optimizer, criterion, device, 
            epoch, epochs_per_stage, current_stage
        )
        
        val_loss, val_accuracy, val_iou, val_precision, val_recall, val_f1 = validate_model(
            model, val_loader, criterion, device
        )
        
        scheduler.step(val_loss)
        
        # 保存历史记录和模型
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_iou'].append(val_iou)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        history['stage'].append(current_stage)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(history_save_path, "best_model.pth")
            torch.save({
                'epoch': total_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_iou': val_iou,
                'val_loss': val_loss,
                'stage': current_stage
            }, best_model_path)
        
        checkpoint_path = os.path.join(history_save_path, f"checkpoint_epoch_{total_epochs}.pth")
        torch.save({
            'epoch': total_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_iou': val_iou,
            'stage': current_stage
        }, checkpoint_path)
    
    # ===== 阶段4：解冻decoder3 =====
    print("Stage 3: Unfreezing decoder3")
    current_stage = 3
    
    for param in model.decoder3.parameters():
        param.requires_grad = True
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr * 0.1
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # 训练阶段4
    for epoch in range(epochs_per_stage):
        total_epochs += 1
        train_loss, train_accuracy, train_iou = train_one_epoch(
            model, train_loader, optimizer, criterion, device, 
            epoch, epochs_per_stage, current_stage
        )
        
        val_loss, val_accuracy, val_iou, val_precision, val_recall, val_f1 = validate_model(
            model, val_loader, criterion, device
        )
        
        scheduler.step(val_loss)
        
        # 保存历史记录和模型
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_iou'].append(val_iou)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        history['stage'].append(current_stage)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(history_save_path, "best_model.pth")
            torch.save({
                'epoch': total_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_iou': val_iou,
                'val_loss': val_loss,
                'stage': current_stage
            }, best_model_path)
        
        checkpoint_path = os.path.join(history_save_path, f"checkpoint_epoch_{total_epochs}.pth")
        torch.save({
            'epoch': total_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_iou': val_iou,
            'stage': current_stage
        }, checkpoint_path)
    
    # ===== 阶段5：解冻decoder4（最后一层） =====
    print("Stage 4: Unfreezing decoder4 (deepest decoder layer)")
    current_stage = 4
    
    for param in model.decoder4.parameters():
        param.requires_grad = True
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr * 0.05
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # 训练阶段5：全部解冻
    for epoch in range(epochs_per_stage):
        total_epochs += 1
        train_loss, train_accuracy, train_iou = train_one_epoch(
            model, train_loader, optimizer, criterion, device, 
            epoch, epochs_per_stage, current_stage
        )
        
        val_loss, val_accuracy, val_iou, val_precision, val_recall, val_f1 = validate_model(
            model, val_loader, criterion, device
        )
        
        scheduler.step(val_loss)
        
        # 保存历史记录和模型
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_iou'].append(val_iou)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        history['stage'].append(current_stage)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(history_save_path, "best_model.pth")
            torch.save({
                'epoch': total_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_iou': val_iou,
                'val_loss': val_loss,
                'stage': current_stage
            }, best_model_path)
        
        checkpoint_path = os.path.join(history_save_path, f"checkpoint_epoch_{total_epochs}.pth")
        torch.save({
            'epoch': total_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_iou': val_iou,
            'stage': current_stage
        }, checkpoint_path)
    
    # ===== 阶段6：全模型低学习率微调 =====
    print("Stage 5: Final fine-tuning with low learning rate")
    current_stage = 5
    
    # 所有层都可训练，使用非常低的学习率进行最终微调
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr * 0.01
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True)
    
    # 最终微调
    for epoch in range(epochs_per_stage):
        total_epochs += 1
        train_loss, train_accuracy, train_iou = train_one_epoch(
            model, train_loader, optimizer, criterion, device, 
            epoch, epochs_per_stage, current_stage
        )
        
        val_loss, val_accuracy, val_iou, val_precision, val_recall, val_f1 = validate_model(
            model, val_loader, criterion, device
        )
        
        scheduler.step(val_loss)
        
        # 保存历史记录和模型
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_iou'].append(val_iou)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        history['stage'].append(current_stage)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(history_save_path, "best_model.pth")
            torch.save({
                'epoch': total_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                train_loss: train_loss,
                'val_iou': val_iou,
                'val_loss': val_loss,
                'stage': current_stage
            }, best_model_path)
        
        checkpoint_path = os.path.join(history_save_path, f"checkpoint_epoch_{total_epochs}.pth")
        torch.save({
            'epoch': total_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_iou': val_iou,
            'stage': current_stage
        }, checkpoint_path)
    
    print(f"Training completed! Total epochs: {total_epochs}")
    
    # 保存完整训练历史
    history_file = os.path.join(history_save_path, "training_history.pth")
    torch.save(history, history_file)
    
    return model, history

if __name__ == "__main__":
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
   
    # 初始化模型
    model = UNet(in_channels=3, out_channels=1)
    
    # 加载预训练模型权重
    pretrained_model_path = PRETRAINED_MODEL_DIR
    model, loaded, checkpoint = load_model_if_exists(model, pretrained_model_path, device)
    
    if loaded:
        print("成功加载预训练模型权重！")
    else:
        print("未能加载预训练模型，使用随机初始化的模型。")
    
    # 设置保存路径
    history_save_path_s2= os.path.join(data_path, "S2output")
    
    # 运行渐进式解冻训练
    model, history = train_model_with_progressive_unfreezing(
        data_path=data_path,
        model=model,
        device=device,
        epochs_per_stage=1,  # 每个阶段训练5个epoch
        batch_size=8,
        base_lr=5e-6,
        val_percent=0.2,
        history_save_path=history_save_path_s2,
        in_channels=3,
        out_channels=1
    )
    output_dir="datasets\\output"