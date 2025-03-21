import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 1)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, 1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.conv3(out)
        return F.relu(out + identity, inplace=True)


class Hourglass(nn.Module):
    def __init__(self, depth, features, num_keypoints):
        super().__init__()
        self.depth = depth
        self.up1 = Residual(features, features)
        self.pool = nn.MaxPool2d(2, 2)
        self.low1 = Residual(features, features)
        if depth > 1:
            self.low2 = Hourglass(depth - 1, features, num_keypoints)
        else:
            self.low2 = Residual(features, features)
        self.low3 = Residual(features, features)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1 = self.up1(x)
        pool = self.pool(x)
        low1 = self.low1(pool)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2


class StackedHourglass(nn.Module):
    def __init__(self, nstack=4, num_keypoints=17, in_channels=3, feat_channels=256):
        super().__init__()
        self.nstack = nstack
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.res1 = Residual(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.res2 = Residual(128, 128)
        self.res3 = Residual(128, feat_channels)

        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass(depth=4, features=feat_channels, num_keypoints=num_keypoints),
                Residual(feat_channels, feat_channels),
                nn.Conv2d(feat_channels, num_keypoints, 1)
            ) for _ in range(nstack)
        ])

    def forward(self, x):
        x = self.conv1(x)  # (B,64,H/2,W/2)
        x = self.res1(x)  # (B,128,H/2,W/2)
        x = self.pool(x)  # (B,128,H/4,W/4)
        x = self.res2(x)  # (B,128,H/4,W/4)
        x = self.res3(x)  # (B,256,H/4,W/4)

        outputs = []
        for i in range(self.nstack):
            y = self.hgs[i](x)
            outputs.append(y)
            if i < self.nstack - 1:
                x = x + self.hgs[i][-2](y)  # 特征融合
        return torch.stack(outputs, dim=1)  # (B, nstack, C, H, W)


# 使用示例
model = StackedHourglass(nstack=4, num_keypoints=17)
input_tensor = torch.randn(2, 3, 256, 256)
output = model(input_tensor)  # (2,4,17,64,64)