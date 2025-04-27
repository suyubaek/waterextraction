import os
import torch
def read_history(history_file, epoch=None):
    """
    Read training history from a .pth file and print specified epoch metrics
    
    Args:
        history_file (str): Path to the history .pth file
        epoch (int, optional): Specific epoch to display. If None, print the last one.
    """
    # Load history data
    if not os.path.exists(history_file):
        print(f"Error: File {history_file} does not exist.")
        return
    
    try:
        history = torch.load(history_file)
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Check if history has expected format with the correct keys
    if not isinstance(history, dict) or 'train_loss' not in history or 'train_iou' not in history:
        print("Invalid history file format. Expected dict with 'train_loss' and 'train_iou' keys.")
        return
    
    loss_history = history['train_loss']
    iou_history = history['train_iou']
    
    epochs = len(loss_history)
    
    if epochs == 0:
        print("History file contains no data.")
        return
    
    # If epoch is not specified, use the last one
    if epoch is None:
        epoch = epochs - 1
    elif epoch < 0 or epoch >= epochs:
        print(f"Error: Epoch {epoch} out of range (0-{epochs-1})")
        return
    
    # Print requested epoch metrics
    print(f"Epoch {epoch} metrics:")
    print(f"train Loss: {loss_history[epoch]:.6f}")
    print(f"train IoU:  {iou_history[epoch]:.6f}")
    
    return history
if __name__ == "__main__":
        # Example usage
        history_file_path = "E:\Thesis2025\\autodl\\training_history_cbam_trans_v2.pth"

        
        # Read and display a specific epoch (e.g., epoch 5)
        read_history(history_file_path, epoch=35)
        # Store the returned history dictionary if needed
        history_data = read_history(history_file_path)
        if history_data:
            print(f"Total epochs in history: {len(history_data['train_loss'])}")
