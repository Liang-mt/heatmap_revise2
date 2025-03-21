import torch
import torch.nn as nn
import torch.nn.functional as F

class OpenPose(nn.Module):
    def __init__(self, num_keypoints=18, num_paf=38, stages=6):
        super().__init__()
        # VGG19 前10层作为特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128x128
            # ... 完整VGG19前10层结构
        )

        # 多阶段预测网络
        self.stages = stages
        self.stage_blocks = nn.ModuleList()
        in_channels = 256  # VGG最终通道数
        for _ in range(stages):
            # 每个阶段预测热力图和PAF
            self.stage_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, num_keypoints + num_paf, 1)  # 联合输出
            ))
            in_channels = 256 + num_keypoints + num_paf  # 特征拼接

    def forward(self, x):
        features = self.feature_extractor(x)  # (B,256,32,32)
        pafs = []
        heatmaps = []

        for i in range(self.stages):
            pred = self.stage_blocks[i](features)  # (B, C, 32, 32)
            paf = pred[:, :38, :, :]  # PAF通道数38
            heatmap = pred[:, 38:, :, :]  # 关键点热力图通道数18

            # 特征拼接用于下一阶段
            if i < self.stages - 1:
                features = torch.cat([features, paf, heatmap], dim=1)

            pafs.append(paf)
            heatmaps.append(heatmap)

        return torch.stack(pafs, dim=1), torch.stack(heatmaps, dim=1)  # (B, stages, 38, 32,32), (B, stages, 18,32,32)