#这个文件主要包含对于模型的评价指标
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from PIL import Image
#模型的预测函数
def predict(image, model, device):
    """
    Perform prediction using the given model and input image.

    Args:
        image (torch.Tensor): Input image tensor with shape (1, C, H, W).
        model (nn.Module): Trained model.
        device (torch.device): Device to run the model on.

    Returns:
        np.ndarray: Predicted binary mask as a NumPy array.
    """
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        prediction = torch.sigmoid(output).squeeze().cpu().numpy()
    return prediction


def predict_folder(input_folder, model, device, transform, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_list = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
    for filename in tqdm(file_list, desc="Processing images"):
        file_path = os.path.join(input_folder, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            image = Image.open(file_path).convert('RGB')
            input_image = transform(image).unsqueeze(0).to(device)
            prediction = predict(input_image, model, device)
            # Save the prediction array as a .npy file
            prediction_array_path = os.path.join(output_folder, f"array_{os.path.splitext(filename)[0]}.npy")
            np.save(prediction_array_path, prediction)
            #生成的预测图得调整一下
            threshold = 0.722  # Set the threshold,后面可以优化阈值
            prediction = (prediction > threshold).astype(np.uint8)  # Convert to binary image
            prediction_image = Image.fromarray((prediction * 255).astype('uint8'))
            output_path = os.path.join(output_folder, f"trans_{filename}")
            prediction_image.save(output_path)

    print("Prediction completed. Results saved to output folder.")

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


#从验证集上挑选出一些样本进行可视化
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