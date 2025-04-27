# config/paths.py
import torch
import torchvision.transforms as transforms
DATASET_DIR = "datasets/"  # 默认路径，用户可自定义
MODEL_SAVE_DIR = "saved_models/"
PREDICTION_DIR = "predictions/"
OUTPUT_DIR = "output/"
PRETRAINED_MODEL_DIR = "datasets\\GFoutput\\best_model.pth"


transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),  # 添加类型转换
        #transforms.Lambda(standardize_tensor), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
