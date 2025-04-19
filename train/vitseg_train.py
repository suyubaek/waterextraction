import sys
import os
import random
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.vitseg import ViTseg
from utils.metrics import binary_iou
from datasets.waterseg_dataset import get_loaders
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    with torch.no_grad():
        for imgs, masks in tqdm(loader):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            total_iou += binary_iou(outputs.cpu(), masks.cpu())
    return total_loss / len(loader), total_iou / len(loader)

def plot_and_save(train_losses, test_losses, train_ious, test_ious, save_path):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.subplot(1, 2, 2)
    plt.plot(train_ious, label='Train IoU')
    plt.plot(test_ious, label='Test IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.title('IoU Curve')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    folder = "/d/chenhongxian/traingf2"
    train_loader, test_loader = get_loaders(folder, batch_size=8, img_size=256, shuffle=True, num_workers=2)
    model = ViTseg().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_losses, test_losses, train_ious, test_ious = [], [], [], []
    for epoch in range(1, 101):
        train_loss, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_iou = eval_one_epoch(model, test_loader, criterion, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_ious.append(train_iou)
        test_ious.append(test_iou)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train IoU={train_iou:.4f} | Test Loss={test_loss:.4f}, Test IoU={test_iou:.4f}")

    # 保存曲线
    os.makedirs("../data", exist_ok=True)
    plot_and_save(train_losses, test_losses, train_ious, test_ious, "../data/loss_iou_curve.png")

if __name__ == "__main__":
    main()