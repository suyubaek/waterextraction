#这个文件主要包含可视化的相关函数
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
#绘制训练损失曲线,Iou曲线
def plot_loss_curve(folder_path, output_dir,curve_name):
    """
    Plot the training and validation loss curves from checkpoint files in a given folder.
    
    Args:
        folder_path (str): Path to the folder containing checkpoint files.
        output_dir (str): Directory where the plot image will be saved.
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    # 设置文件夹路径
    #user_home = os.path.expanduser("~")  # Get user's home directory   
    #output_dir = os.path.join(user_home, "unet_output")
    # output_dir="E:\\Thesis2025\\transferlearning"
    # folder_path = "E:\\Thesis2025\\all_clip_256\\filter2560\\output\\unet_model_finetuned.pth"
 
    # 初始化存储指标的列表
    train_losses = []
    val_losses = []
    val_ious = []

    # 遍历文件夹中的所有文件
    # 使用tqdm可视化进度
    for root, _, files in tqdm(os.walk(folder_path), desc="Processing folders"):
        # 过滤文件名，确保最后一部分是整数
        valid_files = [f for f in files if f.startswith('checkpoint_epoch_') and f.endswith('.pth') and f.split('_')[-1].split('.')[0].isdigit()]
        files = sorted(valid_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        # 使用tqdm可视化文件处理进度
        files = tqdm(files, desc=f"Processing files in {root}")
        for file in files:
            epoch_num = int(file.split('_')[-1].split('.')[0])  # 提取epoch编号
            checkpoint_path = os.path.join(root, file)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            # 假设checkpoint中包含'train_loss', 'val_loss', 'val_IoU'
            train_losses.append(checkpoint['train_loss'])
            val_losses.append(checkpoint['val_loss'])
            val_ious.append(checkpoint['val_iou'])

    # 绘制变化曲线
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))

    # 绘制训练损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss')
    plt.legend()

    # 绘制验证损失曲线
    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()

    # 绘制验证IoU曲线
    plt.subplot(1, 3, 3)
    plt.plot(epochs, val_ious, label='Validation IoU', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('Validation IoU')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 保存图片到指定路径
    output_image_path = os.path.join(output_dir, curve_name)
    plt.savefig(output_image_path)
    print(f"Training history plot saved to {output_image_path}")

    