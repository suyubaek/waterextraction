import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    标准卷积块：Conv2d + BatchNorm + ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                             stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ASPPConv(nn.Sequential):
    """
    ASPP (Atrous Spatial Pyramid Pooling) 卷积模块
    使用不同的扩张率进行卷积，捕捉多尺度上下文信息
    """
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    """
    ASPP 池化模块
    使用全局平均池化捕获全局上下文
    """
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        # 上采样回原始特征图尺寸
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    """
    ASPP (Atrous Spatial Pyramid Pooling) 模块
    包含多个扩张卷积和一个池化分支，用于多尺度特征提取
    """
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        # 1x1 卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # 多个扩张卷积
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
            
        # 全局池化分支
        modules.append(ASPPPooling(in_channels, out_channels))
        
        self.convs = nn.ModuleList(modules)
        
        # 合并所有分支的输出
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class ResNetBlock(nn.Module):
    """
    ResNet 基本块
    包含两个卷积层和一个残差连接
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResNetBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

class ResNetEncoder(nn.Module):
    """
    简化版 ResNet 作为 DeepLabV3+ 的编码器
    """
    def __init__(self, in_channels=3):
        super(ResNetEncoder, self).__init__()
        
        # 初始卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet 层
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride, downsample))
        
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        low_level_feat = x  # 保存低层特征用于解码器
        
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x, low_level_feat

class Decoder(nn.Module):
    """
    DeepLabV3+ 解码器
    将编码器的高层特征与低层特征融合
    """
    def __init__(self, low_level_channels, encoder_channels):
        super(Decoder, self).__init__()
        
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        self.output_conv = nn.Sequential(
            nn.Conv2d(encoder_channels + 48, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
    
    def forward(self, x, low_level_feat):
        # 处理低层特征
        low_level_feat = self.low_level_conv(low_level_feat)
        
        # 上采样高层特征
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=False)
        
        # 拼接特征
        x = torch.cat([x, low_level_feat], dim=1)
        
        # 输出卷积
        x = self.output_conv(x)
        return x

class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ 网络
    用于语义分割任务
    """
    def __init__(self, in_channels=3, num_classes=1, output_stride=16):
        super(DeepLabV3Plus, self).__init__()
        
        # 根据output_stride确定扩张率
        if output_stride == 16:
            atrous_rates = [6, 12, 18]
        elif output_stride == 8:
            atrous_rates = [12, 24, 36]
        else:
            raise ValueError("output_stride 必须为 8 或 16!")
        
        # 编码器
        self.encoder = ResNetEncoder(in_channels)
        
        # ASPP 模块
        self.aspp = ASPP(512, atrous_rates)
        
        # 解码器
        self.decoder = Decoder(64, 256)
        
        # 分类头
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 输入大小
        input_size = x.size()[2:]
        
        # 编码器前向传播
        enc_feat, low_level_feat = self.encoder(x)
        
        # ASPP 模块
        aspp_feat = self.aspp(enc_feat)
        
        # 解码器前向传播
        dec_feat = self.decoder(aspp_feat, low_level_feat)
        
        # 分类头
        output = self.classifier(dec_feat)
        
        # 上采样到原始图像大小
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)
        
        return output

# 测试代码
if __name__ == "__main__":
    # 创建一个示例输入
    x = torch.randn(2, 3, 256, 256)
    
    # 初始化模型
    model = DeepLabV3Plus(in_channels=3, num_classes=1)
    
    # 前向传播
    output = model(x)
    
    # 打印输出形状
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
