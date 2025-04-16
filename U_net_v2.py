import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

"""
U-Net implementation in PyTorch for image segmentation tasks.

Classes:
    UNet: Defines the U-Net architecture.
    SegmentationDataset: Dataset class for loading image and mask pairs.

Functions:
    calculate_iou: Calculates Intersection over Union metric.
    calculate_accuracy: Calculates pixel-wise accuracy.
    calculate_metrics_from_confusion_matrix: Calculates precision, recall, and F1 score.
    train_model: Trains the U-Net model.
    validate_model: Validates the model on the validation set.
    plot_training_history: Plots training and validation metrics.
    visualize_predictions: Visualizes predictions on a few samples.
    load_checkpoint: Loads a model checkpoint.
    load_model_if_exists: Loads a model if a checkpoint exists.
"""

# -------------------------------
# U-Net Model Definition
# -------------------------------
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        """
        Initialize the U-Net model.

        Args:
            in_channels (int): Number of input channels (default: 3 for RGB images).
            out_channels (int): Number of output channels (default: 1 for binary segmentation).
        """
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)
        self.decoder4 = self.conv_block(1024 + 512, 512)
        self.decoder3 = self.conv_block(512 + 256, 256)
        self.decoder2 = self.conv_block(256 + 128, 128)
        self.decoder1 = self.conv_block(128 + 64, 64)
        self.final_layer = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """
        Create a convolutional block with two convolutional layers and batch normalization.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            nn.Sequential: A block of convolutional layers with batch normalization.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # 添加批量归一化
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # 添加批量归一化
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass of the U-Net model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        dec4 = self.decoder4(torch.cat((F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=True), enc4), dim=1))
        dec3 = self.decoder3(torch.cat((F.interpolate(dec4, scale_factor=2, mode='bilinear', align_corners=True), enc3), dim=1))
        dec2 = self.decoder2(torch.cat((F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=True), enc2), dim=1))
        dec1 = self.decoder1(torch.cat((F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True), enc1), dim=1))
        return self.final_layer(dec1)

# -------------------------------
# Dataset Class
# -------------------------------
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
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.transform:
            image = self.transform(image)
        mask = self.mask_transform(mask)
        return image, mask
    
# 直接从原始值缩放到ImageNet标准化范围
#这是为了进行标准化处理
def normalize_like_original(image):
    # 首先归一化到[0,1]范围
    # if image.max() > 0:  # 避免除以零
    #     #那应该是每一张图像都要进行归一化处理
    #     image = image / image.max()  # 使用最大可能值
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # 最后应用ImageNet标准化,使分布接近于ImageNet数据集
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = (image - mean) / std
    
    return image
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

        # Select specific bands (e.g., bands 2, 3, 4)
        image = image[:, :, [1, 2, 3]]  # Select RGB bands
        
        #转为标准化的tensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        # print(f"归一化前")
        # print(f"Image shape: {image.shape}")
        # print(f"Image min/max: {image.min()}/{image.max()}")
        # print(f"Image mean/std: {image.mean()}/{image.std()}")
        # if self.transform:
        #     # Normalize to [0, 1]
        #     image = image / 255.0
        #     # Apply additional normalization
        #     mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)  # Shape: (3, 1, 1)
        #     std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)    # Shape: (3, 1, 1)
        #     image = (image - mean) / std
        # 确保首先将值范围归一化到[0,1]
       
        
        image=normalize_like_original(image)  # Normalize to ImageNet standards  
        # print(f"归一化后")
        # print(f"Image shape: {image.shape}")
        # print(f"Image min/max: {image.min()}/{image.max()}")
        # print(f"Image mean/std: {image.mean()}/{image.std()}")
        # Process mask
        mask = mask.astype(np.float32)  # 确保 NumPy 数组为浮点类型
        mask = torch.from_numpy(mask)  # 转换为 PyTorch 张量
        mask = (mask > 0).float()  # 转换为二值图像并确保为浮点类型
        #mask = torch.from_numpy(mask).float().unsqueeze(0)  # Add channel dimension
        # 修复：确保 mask 的形状为 [batch_size, channels, height, width]
        
        return image, mask  # 移除多余的维度

# -------------------------------
# Metric Calculation Functions
# -------------------------------
def calculate_iou(pred_mask, true_mask, threshold=0.5):
    """
    Calculate Intersection over Union (IoU) metric.
    
    Args:
        pred_mask (numpy.ndarray): Predicted mask
        true_mask (numpy.ndarray): Ground truth mask
        threshold (float): Threshold for binarizing the predicted mask
        
    Returns:
        float: IoU score
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.detach().cpu().numpy()
    
    # Binarize the masks
    pred_mask = (pred_mask > threshold).astype(np.float32)
    true_mask = (true_mask > threshold).astype(np.float32)
    
    # Calculate intersection and union
    intersection = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum() - intersection
    
    # Return IoU
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def predict(image, model, device):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        prediction = torch.sigmoid(output).squeeze().cpu().numpy()
    return prediction
def calculate_accuracy(pred_mask, true_mask, threshold=0.5):
    """
    Calculate pixel-wise accuracy.
    
    Args:
        pred_mask (numpy.ndarray): Predicted mask
        true_mask (numpy.ndarray): Ground truth mask
        threshold (float): Threshold for binarizing the predicted mask
        
    Returns:
        float: Accuracy score
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.detach().cpu().numpy()
    
    # Binarize the predicted mask
    pred_mask = (pred_mask > threshold).astype(np.float32)
    true_mask = (true_mask > threshold).astype(np.float32)
    
    # Calculate accuracy
    correct = (pred_mask == true_mask).sum()
    total = pred_mask.size
    
    return correct / total


def calculate_metrics_from_confusion_matrix(pred_mask, true_mask, threshold=0.5):
    """
    Calculate recall, precision, and F1 score using confusion matrix.
    
    Args:
        pred_mask (numpy.ndarray): Predicted mask
        true_mask (numpy.ndarray): Ground truth mask
        threshold (float): Threshold for binarizing the predicted mask
        
    Returns:
        tuple: (recall, precision, F1 score)
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.detach().cpu().numpy()
    
    # Binarize the masks
    pred_mask = (pred_mask > threshold).astype(np.int32).flatten()
    true_mask = (true_mask > threshold).astype(np.int32).flatten()
    
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(true_mask, pred_mask, labels=[0, 1]).ravel()
    
    # Calculate metrics
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return recall, precision, f1_score

# -------------------------------
# Training and Validation Functions
# -------------------------------
#这里的train_model_with_arrays函数是为了直接使用numpy数组进行训练

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, save_path, start_epoch=0):
    """
    Train the U-Net model.
    
    Args:
        model (nn.Module): U-Net model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to run the model on
        num_epochs (int): Number of training epochs
        save_path (str): Path to save the model
        start_epoch (int): Starting epoch number (default: 0)
        
    Returns:
        dict: Training history
    """
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

    # Define learning rate scheduler  五个epoch后学习率衰减为原来的十分之一
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    
    # Ensure save path directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        train_iou = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch + num_epochs} [Train]")
        for images, masks in train_pbar:
            #print(images.shape)  # 检查输入图像的尺寸
            #break
            images, masks = images.to(device), masks.to(device)
            #images=images.permute(0, 2, 3, 1)  # 调整维度顺序为 (batch_size, height, width, channels)
            #print(images.shape) 
            # Forward pass
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
        
        # Save the model for the current epoch
        epoch_save_path = os.path.join(save_path, f"checkpoint_epoch_{epoch+1}.pth")
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
            best_model_path = os.path.join(save_path, "best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_loss': val_loss,
            }, best_model_path)
    
    return history

def validate_model(model, val_loader, criterion, device):
    """
    Validate the model on the validation set.
    
    Args:
        model (nn.Module): U-Net model
        val_loader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        device (torch.device): Device to run the model on
        
    Returns:
        tuple: (val_loss, val_accuracy, val_iou, val_precision, val_recall, val_f1)
    """
    
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    val_iou = 0.0
    val_precision = 0.0
    val_recall = 0.0
    val_f1 = 0.0
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc="Validation")
        for images, masks in val_pbar:
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            val_loss += loss.item()
            
            # Convert outputs to probabilities
            probs = torch.sigmoid(outputs)
            
            # Calculate accuracy and IoU for the batch
            batch_accuracy = calculate_accuracy(probs, masks)
            batch_iou = calculate_iou(probs, masks)
            batch_recall, batch_precision, batch_f1 = calculate_metrics_from_confusion_matrix(probs, masks)
            val_accuracy += batch_accuracy
            val_iou += batch_iou
            val_precision += batch_precision
            val_recall += batch_recall
            val_f1 += batch_f1
            # Update progress bar
            val_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'accuracy': f"{batch_accuracy:.4f}",
                'IoU': f"{batch_iou:.4f}",
                'precision': f"{batch_precision:.4f}",
                'recall': f"{batch_recall:.4f}",
                'f1': f"{batch_f1:.4f}"
            })
       
    
    # Calculate average metrics
    val_loss = val_loss / len(val_loader)
    val_accuracy = val_accuracy / len(val_loader)
    val_iou = val_iou / len(val_loader)
    val_precision = val_precision / len(val_loader)
    val_recall = val_recall / len(val_loader)
    val_f1 = val_f1 / len(val_loader)

    
    return val_loss, val_accuracy, val_iou, val_precision, val_recall, val_f1

# -------------------------------
# Visualization Functions
# -------------------------------
def plot_training_history(history, save_path=None):
    """
    Plot training and validation metrics.
    
    Args:
        history (dict): Training history
        save_path (str, optional): Path to save the plot
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_accuracy'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot IoU
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['train_iou'], 'b-', label='Training IoU')
    plt.plot(epochs, history['val_iou'], 'r-', label='Validation IoU')
    plt.title('IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


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

# 将标准化函数移到全局作用域
def standardize_tensor(x):
    """Standardize a tensor by subtracting the mean and dividing by the standard deviation."""
    return (x - x.min()) / (x.max() - x.min())
    #return (x - x.mean()) / x.std()  

# Main Execution
# -------------------------------
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define paths
    data_path = "E:\\Thesis2025\\all_clip_256\\filter2560"
    image_dir = os.path.join(data_path, "images")
    mask_dir = os.path.join(data_path, "masks")
    output_dir = os.path.join(data_path, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Define model parameters
    in_channels = 3
    out_channels = 1

    # Define training parameters
    batch_size = 8
    num_epochs =100
    learning_rate = 0.00002  #e-5的学习率

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),  # 添加类型转换
        #transforms.Lambda(standardize_tensor),  # 使用全局定义的函数
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets and dataloaders
    image_names = [f for f in os.listdir(image_dir) if f.endswith('.tif') or f.endswith('.jpg') or f.endswith('.png')]
    train_names, val_names = train_test_split(image_names, test_size=0.2, random_state=42)
    train_dataset = SegmentationDataset(image_dir, mask_dir, train_names, transform=transform)
    val_dataset = SegmentationDataset(image_dir, mask_dir, val_names, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # print(f"Train dataset size: {len(train_dataset)}")
    # print(f"Validation dataset size: {len(val_dataset)}")

    # Create model, loss function, and optimizer
    model = UNet(in_channels=in_channels, out_channels=out_channels)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # # Train the model
    # model_save_path = os.path.join(output_dir, "unet_model_best.pth")

    #history = train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, model_save_path)
    # Resume training from checkpoint if it exists
    # checkpoint_path = "E:\Thesis2025\\all_clip_256\\filter2560\output\\unet_model_best.pth\\checkpoint_epoch_29.pth" 
    # #Replace with your checkpoint file
    # if os.path.exists(checkpoint_path):
    #     print(f"Resuming training from checkpoint: {checkpoint_path}")
    #     checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    #     if checkpoint:
    #         model.load_state_dict(checkpoint['model_state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #         # Now manually move optimizer state to the correct device
    #         for state in optimizer.state.values():
    #             for k, v in state.items():
    #                 if isinstance(v, torch.Tensor):
    #                     state[k] = v.to(device)
    #         start_epoch = checkpoint['epoch']
    #         print(f"Resumed training from epoch {start_epoch}")
    #     else:
    #         print("Failed to load checkpoint. Starting training from scratch.")
    #         start_epoch = 0
    # else:
    #     print("No checkpoint found. Starting training from scratch.")
    #     start_epoch = 0

    # # Continue training from the loaded checkpoint or from scratch

    # history = train_model(model, train_loader, val_loader, criterion, optimizer, device, 71 - start_epoch, model_save_path)
    # Load the best model
    #modelfilepath="E:\Thesis2025\\all_clip_256\\filter2560\output\\unet_model_best.pth\\best_model.pth"
    #modelhistory1path="E:\Thesis2025\\all_clip_256\\filter2560\output\\unet_model_finetuned.pth\\checkpoint_epoch_82.pth"
    #model, loaded, checkpoint = load_model_if_exists(model, modelhistory1path, device)
    # model.eval()

    # # Visualize predictions
    # predictions_plot_path = os.path.join(output_dir, "predictions.png")
    # visualize_predictions(model, val_dataset, device, num_samples=3, save_path=predictions_plot_path)

    # # Evaluate the model
   
    # 加载预训练模型
    pretrained_model_path = "E:\Thesis2025\\all_clip_256\\filter2560\output\\unet_model_best.pth\\best_model.pth"#"D:\学术相关\毕设参考\Pytorch-UNet-master\my-unet\\best_model_pixel2560.pth"  # 替换为你的预训练模型路径
    #checkpointpath="E:\Thesis2025\\all_clip_256\\filter2560\\output\\unet_model_finetuned.pth\\checkpoint_epoch_50.pth"
   #checkpointpath="E:\Thesis2025\\all_clip_256\\filter2560\\output\\unet_model_finetuned.pth\\best_model.pth"
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
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

   
    new_data_path ="E:\Thesis2025\SWED\SWED\\train"
    new_image_dir = os.path.join(new_data_path, "filterrgb")
    new_mask_dir = os.path.join(new_data_path, "filtermaskimages")
    new_image_names = [f for f in os.listdir(new_image_dir) if f.endswith('.tif')]
    #训练集：验证集=8:2
    new_train_names, new_val_names = train_test_split(new_image_names, test_size=0.2, random_state=42)
    new_train_dataset = SegmentationDataset(new_image_dir, new_mask_dir, new_train_names, transform=transform)
    new_val_dataset = SegmentationDataset(new_image_dir, new_mask_dir, new_val_names, transform=transform)

    new_train_loader = DataLoader(new_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    new_val_loader = DataLoader(new_val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
  
    # 微调模型
    print(f"S2Train dataset size: {len(new_train_dataset)}")
    print(f"S2Validation dataset size: {len(new_val_dataset)}")
    new_model_save_path = os.path.join(output_dir, "unet_model_finetuned.pth") 
                                      #model, train_images, train_loader, val_loader, criterion, optimizer, device, num_epochs, save_path
    # Freeze the encoder layers for the first 50 epochs
    # for param in model.encoder1.parameters():
    #     param.requires_grad = False
    # for param in model.encoder2.parameters():
    #     param.requires_grad = False
    # for param in model.encoder3.parameters():
    #     param.requires_grad = False
    # for param in model.encoder4.parameters():
    #     param.requires_grad = False
    # Freeze the decoder layers for the first few epochs
    print("冻结解码器训练")
    for param in model.decoder1.parameters():
        param.requires_grad = False
    for param in model.decoder2.parameters():
        param.requires_grad = False
    for param in model.decoder3.parameters():
        param.requires_grad = False
    for param in model.decoder4.parameters():
        param.requires_grad = False

    # Train for the first few epochs with frozen decoder
    history_part1 = train_model(model, new_train_loader, new_val_loader, criterion, optimizer, device, 50, new_model_save_path, start_epoch=0)

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
    history_part2 = train_model(
        model,
        new_train_loader,
        new_val_loader,
        criterion,
        optimizer,
        device,
        50,
        new_model_save_path,
        start_epoch=50
    )

    # Combine the training history
    history = {
        key: history_part1[key] + history_part2[key]
        for key in history_part1
    }
 

    #所有的savepath全是文件夹
    new_model="E:\Thesis2025\\all_clip_256\\filter2560\output\\unet_model_finetuned.pth\\best_model.pth"
    model, loaded,checkpoint = load_model_if_exists(model, new_model, device)
    model.eval()
    torch.set_grad_enabled(False)
    val_loss, val_accuracy, val_iou, val_precision, val_recall, val_f1 = validate_model(model, new_val_loader, criterion, device)
    print(f"Validation Results - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, IoU: {val_iou:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # input_folder = "E:\imagedata\\new\s2test\\testrgb"
    # output_folder = "E:\imagedata\\new\s2test\\prediction255"
    # Visualize predictions
    predictions_plot_path = os.path.join(output_dir, "predictions_transfer.png")
    visualize_predictions(model, val_dataset, device, num_samples=3, save_path=predictions_plot_path)
    print("Prediction completed. Results saved to output folder.")

