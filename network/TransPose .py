import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, nhead=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (L, B, C)
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)


class TransPose(nn.Module):
    def __init__(self, num_keypoints):
        super().__init__()
        # Backbone (ResNet-50)
        self.backbone = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # 移除最后两层

        # Transformer
        self.transformer = nn.Sequential(
            TransformerBlock(d_model=2048),  # ResNet-50最终特征维度
            TransformerBlock(d_model=2048)
        )

        # 预测头
        self.head = nn.Sequential(
            nn.Conv2d(2048, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, num_keypoints, 1)
        )

    def forward(self, x):
        # CNN特征提取
        features = self.backbone(x)  # (B,2048,H/32,W/32)
        B, C, H, W = features.shape

        # 转换为Transformer输入格式 (L, B, C)
        features = features.view(B, C, -1).permute(2, 0, 1)  # (HW, B, C)
        features = self.transformer(features)

        # 恢复空间维度
        features = features.permute(1, 2, 0).view(B, C, H, W)
        return self.head(features)  # (B, num_keypoints, H/32, W/32)