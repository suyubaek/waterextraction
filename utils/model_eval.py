#这个文件主要包含对于模型的评价指标
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
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
#模型训练后的验证函数
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