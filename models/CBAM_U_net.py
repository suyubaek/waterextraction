import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module focuses on 'what' is meaningful in the input feature map.
    It emphasizes important feature channels while suppressing less useful ones.
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        # Global average pooling and max pooling to gather channel-wise statistics
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Average pooling across spatial dimensions
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Max pooling across spatial dimensions
        
        # Shared multi-layer perceptron (MLP) to reduce parameters
        self.shared_mlp = nn.Sequential(
            # Reduce channel dimension by reduction ratio
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            # Restore original channel dimension
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()  # Normalize attention values between 0-1
        
    def forward(self, x):
        # Generate channel attention through two paths (avg and max)
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        # Combine both features
        out = avg_out + max_out
        # Return channel attention map
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module focuses on 'where' is meaningful in the input feature map.
    It emphasizes important spatial locations in the feature map.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Conv layer to generate spatial attention map from pooled features
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()  # Normalize attention values between 0-1
        
    def forward(self, x):
        # Generate spatial descriptors using average and max pooling across channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Average pooling across channels
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Max pooling across channels
        # Concatenate both features
        out = torch.cat([avg_out, max_out], dim=1)
        # Generate spatial attention map
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM) combines channel and spatial attention.
    It sequentially applies channel and spatial attention to refine feature maps.
    """
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        # Apply channel attention first
        x = x * self.channel_attention(x)  # Element-wise multiplication with channel attention
        # Then apply spatial attention
        x = x * self.spatial_attention(x)  # Element-wise multiplication with spatial attention
        return x


class DoubleConv(nn.Module):
    """
    Double Convolution block with optional CBAM attention.
    Consists of two 3x3 convolutions each followed by BatchNorm and ReLU.
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, use_attention=True):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        # Standard double convolution block
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Optional attention mechanism
        self.use_attention = use_attention
        if use_attention:
            self.cbam = CBAM(out_channels)
        
    def forward(self, x):
        x = self.double_conv(x)
        if self.use_attention:
            x = self.cbam(x)  # Apply attention if enabled
        return x


class Down(nn.Module):
    """
    Downsampling block that combines maxpooling with double convolution.
    Used in the encoder part of U-Net.
    """
    def __init__(self, in_channels, out_channels, use_attention=True):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # Downsampling by factor of 2
            DoubleConv(in_channels, out_channels, use_attention=use_attention)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upsampling block that combines upsampling with double convolution.
    Used in the decoder part of U-Net.
    """
    def __init__(self, in_channels, out_channels, bilinear=True, use_attention=True):
        super(Up, self).__init__()
        
        # Two options for upsampling
        if bilinear:
            # Bilinear interpolation (smoother, but may lose detail)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, use_attention=use_attention)
        else:
            # Transposed convolution (learnable upsampling)
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, use_attention=use_attention)

    def forward(self, x1, x2):
        x1 = self.up(x1)  # Upsample the feature map
        
        # Handle different spatial dimensions with padding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection with upsampled features
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final convolution layer to map features to the desired number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet_CBAM(nn.Module):
    """
    U-Net architecture enhanced with CBAM attention mechanism.
    Consists of an encoder path (downsampling) and a decoder path (upsampling)
    with skip connections between corresponding encoder and decoder blocks.
    """
    def __init__(self, in_channels=3, out_channels=1, bilinear=True):
        super(UNet_CBAM, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = bilinear

        # Encoder path
        self.inc = DoubleConv(in_channels, 64, use_attention=False)  # Initial block without attention
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1  # Adjusts channels based on upsampling method
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder path with skip connections，跳跃连接
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Final output convolution
        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        # Encoder path and store features for skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path with skip connections from encoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Final output projection
        logits = self.outc(x)
        return logits
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
                
# 定义损失函数
    # 定义Dice损失函数
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
            
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)  # 应用sigmoid激活
            
        # 展平预测和真实值
        inputs = inputs.view(-1)
        targets = targets.view(-1)
            
        # 计算交集
        intersection = (inputs * targets).sum()
            
        # 计算Dice系数
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
            
        # 返回Dice损失
        return 1 - dice