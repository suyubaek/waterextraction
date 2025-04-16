import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

from U_net_v2 import UNet, calculate_iou
from Unet import predict

def predict_folder(folder_path, model, device, transform, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    iou_list = []  # 移动到循环外

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            image = Image.open(file_path).convert('RGB')
            input_image = transform(image).unsqueeze(0).to(device)
            prediction = predict(input_image, model, device)

            prediction_image = Image.fromarray((prediction * 255).astype('uint8'))

            # IoU 计算
            true_mask_path = os.path.join(folder_path.replace("images", "masks"), filename.replace("image","mask"))
           

            if os.path.exists(true_mask_path):
                #print(f"Found true mask: {true_mask_path}")
                true_mask = Image.open(true_mask_path).convert("L")

                prediction_array = np.array(prediction_image)
                true_mask_array = np.array(true_mask)

               # print(f"Prediction shape: {prediction_array.shape}, True mask shape: {true_mask_array.shape}")
                iou = calculate_iou(prediction_array, true_mask_array)
                print(f"{filename} IoU: {iou:.4f}")
                iou_list.append(iou)

            output_path = os.path.join(output_folder, f"pred_{filename}")
            prediction_image.save(output_path)

    if iou_list:
        print(f"Average IoU: {np.mean(iou_list):.4f}")
    else:
        print("No IoU values calculated.")

def predict_binary(image, model, device, threshold=0.8):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        prediction = (torch.sigmoid(output) > threshold).float().squeeze().cpu().numpy()
    return prediction
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
    """
    Load a model from a checkpoint if it exists.
    
    Args:
        model (nn.Module): Model to load weights into
        model_path (str): Path to the checkpoint
        device (torch.device): Device to load the model onto
        
    Returns:
        tuple: (model, bool) where bool indicates whether the model was loaded
    """
    try:
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            checkpoint = load_checkpoint(model_path, map_location=device)
            if checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(device)
                print(f"Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")
                return model, True
            else:
                print(f"Failed to load model from {model_path}")
                return model, False
        else:
            print(f"No model found at {model_path}")
            return model, False
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return model, False
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    model = UNet(in_channels=3, out_channels=1)
    user_home = os.path.expanduser("~")  # Get user's home directory
    #path=os.path.join(user_home, "unet_output","best_model_pixel2560.pth")
    path="E:\Thesis2025\\all_clip_256\\filter2560\\output\\unet_model_finetuned.pth\\best_model.pth"
    #path="E:\Thesis2025\\all_clip_256\\filter2560\\output\\unet_model_finetuned.pth\\best_model_frozen.pth"
    #path="E:\\Thesis2025\\all_clip_256\\best_model_pixel2560.pth"
    #path="E:\Thesis2025\\all_clip_256\\filter2560\\output\\unet_model_best.pth\\best_model.pth"
    #path="E:\\Thesis2025\\all_clip_256\\filter2560\\output\\unet_model_best.pth\\best_model.pth"
    model,loaded=load_model_if_exists(model, path, device)
    model.to(device)
    model.eval()
    
    torch.set_grad_enabled(False)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    input_folder = "E:\imagedata\\new\s2test\\testrgb"
    #input_folder="E:\\Thesis2025\\SWED_sample\\test\\images"
    output_folder = "E:\imagedata\\new\s2test\\prediction255"
    #output_folder="E:\\Thesis2025\\SWED_sample\\test\labels"

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            image = Image.open(file_path).convert('RGB')
            input_image = transform(image).unsqueeze(0).to(device)
            prediction = predict(input_image, model, device)
            #生成的预测图得调整一下
            threshold = 0.5  # Set the threshold
            prediction = (prediction > threshold).astype(np.uint8)  # Convert to binary image
            prediction_image = Image.fromarray((prediction * 255).astype('uint8'))

            output_path = os.path.join(output_folder, f"pred_trans_{filename}")
            prediction_image.save(output_path)

    print("Prediction completed. Results saved to output folder.")
    # Visualize and save original and predicted images side by side
    # for filename in os.listdir(input_folder):
    #     file_path = os.path.join(input_folder, filename)
    #     if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
    #         image = Image.open(file_path).convert('RGB')
    #         input_image = transform(image).unsqueeze(0).to(device)
    #         prediction = predict(input_image, model, device)

    #         prediction_image = Image.fromarray((prediction * 255).astype('uint8'))

    #         # Plot original and predicted images
    #         fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    #         axes[0].imshow(image)
    #         axes[0].set_title("Original Image")
    #         axes[0].axis("off")

    #         axes[1].imshow(prediction_image, cmap="gray")
    #         axes[1].set_title("Predicted Image")
    #         axes[1].axis("off")

    #         # Save the subplot
    #         subplot_output_path = os.path.join(output_folder, f"subplot_{filename}.png")
    #         plt.savefig(subplot_output_path)
    #         plt.close(fig)

    # print("Subplots saved to output folder.")
