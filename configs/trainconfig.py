# config/train_config.py
import torch
from torchvision import transforms
class TrainConfig:
    # 训练参数
    EPOCHS = 60
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-5
    START_EPOCH = 0  # 从第0个epoch开始训练
    VAL_PERCENT = 0.9  # 方差百分比
    # 其他参数（可选）
    OPTIMIZER = "Adam"
    
