import torch
import torch.nn as nn
import torch.nn.functional as F
from HRNet import HRNet

class DARKpose(nn.Module):
    def __init__(self, num_keypoints=17, backbone='hrnet'):
        super().__init__()
        # 使用HRNet作为基础网络
        self.backbone = HRNet(num_keypoints)  # 前述HRNet代码

    def forward(self, x):
        heatmaps = self.backbone(x)
        return heatmaps

    # DARKpose的核心改进在解码阶段，需自定义解码函数
    @staticmethod
    def decode_heatmap(heatmaps, kernel_size=5):
        """
        输入: heatmaps - (B, C, H, W) 网络输出的原始热力图
        输出: coords - (B, C, 2) 精确坐标 (x,y)
        """
        B, C, H, W = heatmaps.shape
        # 1. 生成修正偏移量
        offset_x = (F.avg_pool2d(heatmaps, kernel_size, 1, padding=kernel_size // 2)
                    - F.avg_pool2d(heatmaps, kernel_size, 1, padding=kernel_size // 2).detach())
        offset_y = (F.avg_pool2d(heatmaps.transpose(2, 3), kernel_size, 1, padding=kernel_size // 2)
                    - F.avg_pool2d(heatmaps.transpose(2, 3), kernel_size, 1, padding=kernel_size // 2).detach())

        # 2. 原始坐标预测
        max_vals, indices = torch.max(heatmaps.view(B, C, -1), dim=2)
        y = indices // W
        x = indices % W

        # 3. 应用偏移修正
        x = x.float() + offset_x.view(B, C, -1).gather(2, indices.unsqueeze(2)).squeeze(2)
        y = y.float() + offset_y.view(B, C, -1).gather(2, indices.unsqueeze(2)).squeeze(2)

        return torch.stack([x, y], dim=2)  # (B, C, 2)