import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.vitseg import ViTseg
from utils.metrics import binary_iou
from datasets.waterseg_dataset import get_loader

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_iou = 0
    for imgs, masks in tqdm(loader):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_iou += binary_iou(outputs.detach().cpu(), masks.detach().cpu())
    return total_loss / len(loader), total_iou / len(loader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ViTseg().to(device)
    # 替换为你的数据集文件夹路径
    folder = "/Users/songyufei1/Documents/look_code_syf/water-extract-dl/data/water_seg"
    loader = get_loader(folder, batch_size=2, img_size=512, shuffle=True, num_workers=2)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        loss, iou = train_one_epoch(model, loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, IoU={iou:.4f}")

if __name__ == "__main__":
    main()