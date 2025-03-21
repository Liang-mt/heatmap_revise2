import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    """标准残差块（支持通道数变化）"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 跳跃连接适配维度
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class HRNet(nn.Module):
    def __init__(self, num_keypoints, base_channel=32):
        super().__init__()
        # Stage 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Stage 2
        self.stage2 = self._make_stage(64, base_channel, num_blocks=4)  # 输出: (B,32,64,64)
        # Stage 3
        self.stage3 = self._make_stage(base_channel, base_channel * 2, num_blocks=4)  # 输出: (B,64,32,32)
        # Stage 4
        self.stage4 = self._make_stage(base_channel * 2, base_channel * 4, num_blocks=4)  # 输出: (B,128,16,16)

        # 多分辨率特征融合
        self.fuse_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(base_channel, base_channel * 4, kernel_size=1),  # 32 -> 128
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            ),
            nn.Sequential(
                nn.Conv2d(base_channel * 2, base_channel * 4, kernel_size=1),  # 64 -> 128
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            ),
            nn.Sequential(
                nn.Identity(),
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            )
        ])

        # 输出头
        self.final_conv = nn.Conv2d(base_channel * 4 * 3, num_keypoints, kernel_size=1)

    def _make_stage(self, in_channels, out_channels, num_blocks):
        layers = [Residual(in_channels, out_channels, stride=1)]
        for _ in range(1, num_blocks):
            layers.append(Residual(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Stage 1
        x = F.relu(self.bn1(self.conv1(x)))  # (B,64,128,128)
        x = F.relu(self.bn2(self.conv2(x)))  # (B,64,64,64)

        # Stage 2
        x2 = self.stage2(x)  # (B,32,64,64)

        # Stage 3
        x3 = F.max_pool2d(x2, kernel_size=2, stride=2)  # (B,32,32,32)
        x3 = self.stage3(x3)  # (B,64,32,32)

        # Stage 4
        x4 = F.max_pool2d(x3, kernel_size=2, stride=2)  # (B,64,16,16)
        x4 = self.stage4(x4)  # (B,128,16,16)

        # 特征融合
        x2_fuse = self.fuse_layers[0](x2)  # (B,128,64,64)
        x3_fuse = self.fuse_layers[1](x3)  # (B,128,64,64)
        x4_fuse = self.fuse_layers[2](x4)  # (B,128,64,64)

        fused = torch.cat([x2_fuse, x3_fuse, x4_fuse], dim=1)  # (B, 384, 64,64)
        return self.final_conv(fused)  # (B, num_keypoints, 64,64)


# 测试代码
if __name__ == "__main__":
    # 初始化模型
    model = HRNet(num_keypoints=1)  # 17个关键点
    input_tensor = torch.randn(1, 3, 128, 128)  # 输入尺寸 (B, C, H, W)

    # 前向传播
    output = model(input_tensor)
    print("输出热力图尺寸:", output.shape)  # 预期输出: (2, 17, 64, 64)