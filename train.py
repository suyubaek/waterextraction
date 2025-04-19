import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision import transforms
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils.dataloading import SegmentationDataset, NpySegmentationDataset
from utils.model_eval import calculate_accuracy, calculate_iou, validate_model
from models.U_Net import UNet
import numpy as np
# Define paths
data_path = "E:\Thesis2025\SWED\SWED\\train"

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),  # 添加类型转换
    #transforms.Lambda(standardize_tensor), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
def train_model(
    data_path: str,
    model: nn.Module,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 1,
    learning_rate: float = 1e-6,
    val_percent: float = 0.2,
    start_epoch:int=0,
    history_save_path:str="",
    in_channels:int=3,
    out_channels:int=1,
):
    image_dir = os.path.join(data_path, "images")
    mask_dir = os.path.join(data_path, "masks")
    output_dir = os.path.join(data_path, "output")
    os.makedirs(output_dir, exist_ok=True) 
    # Load dataset

    image_names = [f for f in os.listdir(image_dir) if f.endswith('.tif') or f.endswith('.jpg') or f.endswith('.png')]
    train_names, val_names = train_test_split(image_names, test_size=val_percent, random_state=42)
    train_dataset = SegmentationDataset(image_dir, mask_dir, train_names, transform=transform)
    val_dataset = SegmentationDataset(image_dir, mask_dir, val_names, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Define optimizer and loss function
    model = UNet(in_channels=in_channels, out_channels=out_channels)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    

    model.to(device)
    
    
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'train_iou': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_iou': []
    }
    #用loss存储最好的模型
    best_val_loss = float('inf')  # Set to a very high value initially

    # Define learning rate scheduler  2个epoch后学习率衰减为原来的十分之一
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    
    # Ensure save path directory exists
    os.makedirs(os.path.dirname(history_save_path), exist_ok=True)
    
    for epoch in range(start_epoch, start_epoch + epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        train_iou = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch + epochs} [Train]")
        for images, masks in train_pbar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            train_loss += loss.item()
            
            # Convert outputs to probabilities
            probs = torch.sigmoid(outputs)
            
            # Calculate accuracy and IoU for the batch
            batch_accuracy = calculate_accuracy(probs, masks)
            batch_iou = calculate_iou(probs, masks)
            
            train_accuracy += batch_accuracy
            train_iou += batch_iou
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'accuracy': f"{batch_accuracy:.4f}",
                'IoU': f"{batch_iou:.4f}"
            })
        
        # Calculate average metrics for the epoch
        train_loss = train_loss / len(train_loader)
        train_accuracy = train_accuracy / len(train_loader)
        train_iou = train_iou / len(train_loader)
        
        # Validation phase
        val_loss, val_accuracy, val_iou,val_precision, val_recall, val_f1 = validate_model(model, val_loader, criterion, device)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Save metrics to history
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_iou'].append(val_iou)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        
        # Save the model for the current epoch
        epoch_save_path = os.path.join(history_save_path, f"checkpoint_epoch_{epoch+1}.pth")
        os.makedirs(os.path.dirname(epoch_save_path), exist_ok=True)  # Ensure directory exists
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_iou': val_iou,
        }, epoch_save_path)
        
        # Save the best model based on validation IoU
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(history_save_path, "best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_loss': val_loss,
            }, best_model_path)
def train_model_npy(
    data_path: str,
    model: nn.Module,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 1,
    learning_rate: float = 1e-6,
    val_percent: float = 0.2,
    start_epoch:int=0,
    history_save_path:str="",
    in_channels:int=3,
    out_channels:int=1,
):
    image_dir = os.path.join(data_path, "filter_images")
    mask_dir = os.path.join(data_path, "binary_filter_masks")
    output_dir = os.path.join(data_path, "output")
    os.makedirs(output_dir, exist_ok=True) 
    # Load dataset

    image_names = [f for f in os.listdir(image_dir) if f.endswith('.tif') or f.endswith('.npy') or f.endswith('.png')]
    train_names, val_names = train_test_split(image_names, test_size=val_percent, random_state=42)
    train_dataset = NpySegmentationDataset(image_dir, mask_dir, train_names, transform=transform)
    val_dataset = NpySegmentationDataset(image_dir, mask_dir, val_names, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Define optimizer and loss function
    model = UNet(in_channels=in_channels, out_channels=out_channels)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    

    model.to(device)
    
    
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
    }
    #用loss存储最好的模型
    best_val_loss = float('inf')  # Set to a very high value initially

    # Define learning rate scheduler  3个epoch后学习率衰减为原来的十分之一
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    
    # Ensure save path directory exists
    os.makedirs(os.path.dirname(history_save_path), exist_ok=True)
    
    for epoch in range(start_epoch, start_epoch + epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        train_iou = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch + epochs} [Train]")
        for images, masks in train_pbar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            train_loss += loss.item()
            
            # Convert outputs to probabilities
            probs = torch.sigmoid(outputs)
            
            # Calculate accuracy and IoU for the batch
            batch_accuracy = calculate_accuracy(probs, masks)
            batch_iou = calculate_iou(probs, masks)
            
            train_accuracy += batch_accuracy
            train_iou += batch_iou
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'accuracy': f"{batch_accuracy:.4f}",
                'IoU': f"{batch_iou:.4f}"
            })
        
        # Calculate average metrics for the epoch
        train_loss = train_loss / len(train_loader)
        train_accuracy = train_accuracy / len(train_loader)
        train_iou = train_iou / len(train_loader)
        
        # Validation phase
        val_loss, val_accuracy, val_iou,val_precision, val_recall, val_f1 = validate_model(model, val_loader, criterion, device)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Save metrics to history
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_iou'].append(val_iou)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        
        # Save the model for the current epoch
        epoch_save_path = os.path.join(history_save_path, f"checkpoint_epoch_{epoch+1}.pth")
        os.makedirs(os.path.dirname(epoch_save_path), exist_ok=True)  # Ensure directory exists
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_iou': val_iou,
        }, epoch_save_path)
        
        # Save the best model based on validation LOSS
        #怀疑会过拟合，因此选用loss的最小作为最好的模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(history_save_path, "best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_loss': val_loss,
            }, best_model_path)


def visualize_predictions(model, dataset, device, num_samples=3, save_path=None):
    """
    Visualize model predictions on a few samples.
    
    Args:
        model (nn.Module): Trained model
        dataset (Dataset): Dataset to sample from
        device (torch.device): Device to run the model on
        num_samples (int): Number of samples to visualize
        save_path (str, optional): Path to save the visualization
    """
    # Get a few random samples
    indices = [674, 520, 611]
    plt.figure(figsize=(15, 5 * num_samples))
    
    for i, idx in enumerate(indices):
        image, mask = dataset[idx]
        
        # Make prediction
        pred = torch.sigmoid(model(image.unsqueeze(0).to(device))).squeeze().detach().cpu().numpy()
        
        # Denormalize image if needed
        if isinstance(image, torch.Tensor):
            # Assuming normalization with ImageNet mean and std
            image = image.cpu().numpy().transpose(1, 2, 0)
            image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            image = np.clip(image, 0, 1)
        
        # Convert mask to numpy array if needed
        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze().cpu().numpy()
        
        # Calculate IoU
        iou = calculate_iou(pred, mask)
        
        # Plot image, ground truth, and prediction
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(image)
        plt.title(f"Sample {idx}")
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(mask, cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(pred, cmap='gray')
        plt.title(f"Prediction (IoU: {iou:.4f})")
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

# -------------------------------
# Model Checkpoint Functions
# -------------------------------
def load_checkpoint(filepath, map_location):
    """
    Load a checkpoint without file locking.
    
    Args:
        filepath (str): Path to the checkpoint.
        map_location (torch.device): Device to map the checkpoint.
        
    Returns:
        dict: Loaded checkpoint state.
    """
    try:
        checkpoint = torch.load(filepath, map_location=map_location)
        print(f"Checkpoint loaded from {filepath}")
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint from {filepath}: {str(e)}")
        return None

def load_model_if_exists(model, model_path, device):
    try:
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            checkpoint = load_checkpoint(model_path, map_location=device)
            if checkpoint:
                model = model.to(device)  # Ensure model is on device before loading weights
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")
                return model, True, checkpoint  # Return checkpoint as well
            else:
                print(f"Failed to load model from {model_path}")
                return model, False, None
        else:
            print(f"No model found at {model_path}")
            return model, False, None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return model, False, None

def train_model_if_disrupt(
    data_path: str,
    model: nn.Module,
    device: torch.device,
    checkpoint_path: str,
    epochs: int = 20,
    batch_size: int = 1,
    learning_rate: float = 1e-6,
    val_percent: float = 0.2,
    history_save_path: str = "",
    in_channels: int = 3,
    out_channels: int = 1,
):
    """
    Resume training from a checkpoint if it exists, otherwise start training from scratch.
    """
    # Initialize model and optimizer
    model = UNet(in_channels=in_channels, out_channels=out_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start_epoch = 0

    # Check if checkpoint exists and load it
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, map_location=device)
        if checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Move optimizer state to the correct device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
                        start_epoch = checkpoint['epoch']
                        print(f"Resumed training from epoch {start_epoch}")
                    else:
                        print("Failed to load checkpoint. Starting training from scratch.")
        else:
            print("No checkpoint found. Starting training from scratch.")

        # Train the model
        train_model(
            data_path=data_path,
            model=model,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            val_percent=val_percent,
            start_epoch=start_epoch,
            history_save_path=history_save_path,
            in_channels=in_channels,
            out_channels=out_channels,
        )

def predict(image, model, device):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        prediction = torch.sigmoid(output).squeeze().cpu().numpy()
    return prediction   
if __name__ == "__main__":
    # Check if GPU is available and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define model parameters
    in_channels = 3  # Number of input channels (e.g., RGB image)
    out_channels = 1  # Number of output channels (e.g., binary segmentation)

    # Define paths for saving history
    history_save_path = os.path.join(data_path, "output", "unet_model_finetuned.pth")
    
    # Train the model
    # 加载预训练模型
    #pretrained_model_path = "E:\Thesis2025\\all_clip_256\\filter2560\output\\unet_model_best.pth\\best_model.pth"#"D:\学术相关\毕设参考\Pytorch-UNet-master\my-unet\\best_model_pixel2560.pth"  # 替换为你的预训练模型路径
   #checkpointpath="E:\Thesis2025\\all_clip_256\\filter2560\\output\\unet_model_finetuned.pth\\checkpoint_epoch_50.pth"
    #checkpointpath="E:\Thesis2025\\all_clip_256\\filter2560\\output\\unet_model_finetuned.pth\\best_model.pth"
    pretrained_model_path = "E:\\Thesis2025\\SWED\\SWED\\train\\output\\unet_model_finetuned.pth\\checkpoint_epoch_23.pth"
    model = UNet(in_channels=3, out_channels=1)  # 假设输入为3通道，输出为1通道（如二分类任务）
    model, loaded,checkpoint = load_model_if_exists(model,  pretrained_model_path , device)

    if loaded:
        print("预训练模型加载成功！")
    else:
        print("未能加载预训练模型，使用随机初始化的模型。")
    #冻结吧，基本的水体特征应该没啥问题

    # 冻结编码器部分（可选）
   

    # 定义新的优化器和损失函数
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-06)

   
    
   
  
    # 微调模型
    
    
    print("冻结解码器训练开始")
    for param in model.decoder1.parameters():
        param.requires_grad = False
    for param in model.decoder2.parameters():
        param.requires_grad = False
    for param in model.decoder3.parameters():
        param.requires_grad = False
    for param in model.decoder4.parameters():
        param.requires_grad = False

    # Train for the first few epochs with frozen decoder
    # data_path: str,
    # model: nn.Module,
    # device: torch.device,
    # epochs: int = 20,
    # batch_size: int = 1,
    # learning_rate: float = 1e-6,
    # val_percent: float = 0.2,
    # start_epoch:int=0,
    # history_save_path:str="",
    # in_channels:int=3,
    # out_channels:int=1,
    train_model_npy(data_path,model, device, 12, 8, 5e-6, 0.2, 23, history_save_path, in_channels, out_channels)

    # Unfreeze the decoder layers for the remaining epochs
    print("Unfreezing decoder layers for further training.")
    for param in model.decoder1.parameters():
        param.requires_grad = True
    for param in model.decoder2.parameters():
        param.requires_grad = True
    for param in model.decoder3.parameters():
        param.requires_grad = True
    for param in model.decoder4.parameters():
        param.requires_grad = True

    # Train for the remaining epochs with unfrozen decoder
    train_model_npy(data_path,model, device, 30, 8, 1e-6, 0.2, 35, history_save_path, in_channels, out_channels)

    # Combine the training history
    print("Training completed. Combining training history.")

