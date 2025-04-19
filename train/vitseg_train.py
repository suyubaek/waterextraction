import sys
import os
import random
import numpy as np
import logging
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

def setup_logger(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler()
        ]
    )

def run_training(seed, run_idx):
    set_seed(seed)
    os.makedirs("data", exist_ok=True)
    log_file = f"data/train_seed_{seed}_run_{run_idx}.log"
    curve_file = f"data/loss_iou_curve_seed_{seed}_run_{run_idx}.png"
    # 重新设置logger，避免多次调用basicConfig无效
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    setup_logger(log_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    folder = "/d/chenhongxian/traingf2"
    train_loader, test_loader = get_loaders(folder, batch_size=8, img_size=256, shuffle=True, num_workers=2)
    model = ViTseg().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    train_losses, test_losses, train_ious, test_ious = [], [], [], []
    for epoch in range(1, 101):
        train_loss, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_iou = eval_one_epoch(model, test_loader, criterion, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_ious.append(train_iou)
        test_ious.append(test_iou)
        scheduler.step(test_loss)
        logging.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train IoU={train_iou:.4f} | Test Loss={test_loss:.4f}, Test IoU={test_iou:.4f} | LR={optimizer.param_groups[0]['lr']:.6f}")

    plot_and_save(train_losses, test_losses, train_ious, test_ious, curve_file)

if __name__ == "__main__":
    seeds = random.sample(range(1, 101), 30)
    for idx, seed in enumerate(seeds):
        print(f"Run {idx+1}/30, Seed={seed}")
        run_training(seed, idx+1)