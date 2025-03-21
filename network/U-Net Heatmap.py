import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, num_keypoints, in_channels=3):
        super().__init__()
        # 编码器
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256)

        # 解码器
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)  # 256=128(up)+128(skip)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)  # 128=64(up)+64(skip)

        # 输出层
        self.final_conv = nn.Conv2d(64, num_keypoints, 1)

    def forward(self, x):
        # 编码
        e1 = self.enc1(x)  # (B,64,H,W)
        e2 = self.enc2(self.pool1(e1))  # (B,128,H/2,W/2)
        e3 = self.enc3(self.pool2(e2))  # (B,256,H/4,W/4)

        # 解码
        d3 = self.up3(e3)  # (B,128,H/2,W/2)
        d3 = torch.cat([d3, e2], dim=1)  # 跳跃连接
        d3 = self.dec3(d3)

        d2 = self.up2(d3)  # (B,64,H,W)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        return self.final_conv(d2)  # (B, num_keypoints, H, W)