# config/train_config.py
import torch
class TrainConfig:
    # 训练参数
    EPOCHS = 12
    BATCH_SIZE = 8
    LEARNING_RATE = 5e-6
    START_EPOCH = 0  # 从第0个epoch开始训练
    VAL_PERCENT = 0.2  # 方差百分比
    # 其他参数（可选）
    OPTIMIZER = "Adam"
