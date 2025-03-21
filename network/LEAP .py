import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=kernel_size // 2, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class LEAP(nn.Module):
    def __init__(self, num_keypoints=17):
        super().__init__()
        # MobileNetV2 轻量化主干
        self.backbone = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True).features

        # 自定义头部（热力图回归）
        self.head = nn.Sequential(
            DepthwiseSeparableConv(1280, 256),  # MobileNet最后一层输出通道1280
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_keypoints, 1)  # 输出热力图
        )

    def forward(self, x):
        x = self.backbone(x)  # (B,1280,7,7) 输入256x256时
        x = F.interpolate(x, scale_factor=4, mode='bilinear')  # 上采样到28x28
        return self.head(x)  # (B, num_keypoints, 28, 28)
