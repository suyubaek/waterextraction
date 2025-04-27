import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from models.U_net import UNet
from utils.model_eval import predict, load_model_if_exists
from datetime import datetime
from tqdm import tqdm
from osgeo import gdal
from models.CBAM_U_net import UNet_CBAM
from models.DeepLabV3plus import DeepLabV3Plus
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

def predict_with_nodata_mask(input_folder,model, device, transform, output_folder):
    """
    使用模型预测图像，并考虑无数据掩码。
    无数据区域直接标记为非水体（0）。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取所有图像文件  无图像掩码和图像放置在一个文件夹
    file_list = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f)) 
                and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')) and "_image_" in f]
    
    for filename in tqdm(file_list, desc="Processing images"):
        file_path = os.path.join(input_folder, filename)
        # 尝试查找对应的无数据掩码
        nodata_mask_path = file_path.replace("_image_","_nodata_mask_")
        
        # 加载图像
        image = Image.open(file_path).convert('RGB')
        input_image = transform(image).unsqueeze(0).to(device)
        
        # 模型预测
        prediction = predict(input_image, model, device)
        
        # 检查是否有无数据掩码
        if os.path.exists(nodata_mask_path):
            # 加载无数据掩码
            try:
                nodata_ds = gdal.Open(nodata_mask_path, gdal.GA_ReadOnly)
                if nodata_ds:
                    nodata_mask = nodata_ds.GetRasterBand(1).ReadAsArray()
                    nodata_ds = None
                    
                    # 在无数据区域（掩码值为1）将预测结果设置为0（非水体）
                    prediction[nodata_mask == 1] = 0
                    #print(f"已应用无数据掩码: {nodata_mask_path}")
            except Exception as e:
                print(f"加载无数据掩码出错 {nodata_mask_path}: {str(e)}")
        else:
            print(f"未找到无数据掩码: {nodata_mask_path}")
        
        # 保存预测数组
        prediction_array_path = os.path.join(output_folder, f"array_{os.path.splitext(filename)[0]}.npy")
        np.save(prediction_array_path, prediction)
        
        # 应用阈值并保存预测图像,阈值建议调整，还是建议学习
        threshold = 0.722  # 阈值
        binary_prediction = (prediction > threshold).astype(np.uint8)
        prediction_image = Image.fromarray((binary_prediction * 255).astype('uint8'))
        output_path = os.path.join(output_folder, f"trans_{filename}")
        prediction_image.save(output_path)
    
    print("预测完成。结果已保存到输出文件夹。")

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    model = DeepLabV3Plus(in_channels=3, num_classes=1)
    user_home = os.path.expanduser("~")  # Get user's home directory
    path="E:\\Thesis2025\\autodl\\best_model_dlv3+.pth"  #从autodl上训练的模型 逐层解冻的模型
    #path_gf="datasets\\GFoutput\\best_model.pth"
    model,loaded,checkpoint=load_model_if_exists(model, path, device)
    model.to(device)
    model.eval()
    
    torch.set_grad_enabled(False)

    # Define transformations
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),  # 添加类型转换
        #transforms.Lambda(standardize_tensor), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    # file_path="E:\Thesis2025\encoder_heatmaps\GF2_PMS1__L1A0000564539-MSS1_image_2560_3072.tif"
    # output_folder="E:\Thesis2025\encoder_heatmaps"
    # image = Image.open(file_path).convert('RGB')
    # input_image = transform(image).unsqueeze(0).to(device)
    # prediction=predict(input_image,model,device)
    # threshold = 0.722  # 阈值
    # binary_prediction = (prediction > threshold).astype(np.uint8)
    # prediction_image = Image.fromarray((binary_prediction * 255).astype('uint8'))
    # output_path = os.path.join(output_folder, f"gf_predict.tif")
    # prediction_image.save(output_path)
    input_folder="E:\imagedata\\2305-06\clipped_images\\images"
    output_folder="E:\imagedata\\2305-06\\clipped_images\\prediction_dlv3+"

    # 使用处理无数据区域的预测函数
    predict_with_nodata_mask(input_folder, model, device, transform, output_folder)
    

