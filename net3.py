from torch import nn


# 需要修正的网络结构（示例）
class KeypointDetector_v2(nn.Module):
    def __init__(self, num_keypoints):
        super(KeypointDetector_v2, self).__init__()

        # 编码器（下采样）
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # 输入通道3，输出32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x64

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 8x8
        )

        # 解码器（上采样）
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),  # 16x16
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 2, stride=2),  # 32x32
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 2, stride=2),  # 64x64
            nn.ReLU(),

            nn.ConvTranspose2d(32, num_keypoints, 2, stride=2),  # 128x128
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)  # 输出形状应为 [B, C, 128, 128]
        return x


class KeypointDetector_v2_1(nn.Module):
    def __init__(self, num_keypoints):
        super(KeypointDetector_v2_1, self).__init__()
        self.num_keypoints = num_keypoints

        # 编码器（下采样）
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 8x8
        )

        # 解码器（上采样）
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 16x16
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 32x32
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 64x64
            nn.ReLU(),

            nn.ConvTranspose2d(32, num_keypoints, kernel_size=2, stride=2),  # 128x128
            nn.Sigmoid()  # 输出概率图
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)  # 输出形状 [B, C, 128, 128]


class KeypointDetector_v2_heatmap(nn.Module):
    def __init__(self, num_keypoints):
        super(KeypointDetector_v2_heatmap, self).__init__()
        self.num_keypoints = num_keypoints

        # 特征提取主干网络
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出尺寸: (32, 64, 64)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出尺寸: (64, 32, 32)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出尺寸: (128, 16, 16)

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 输出尺寸: (256, 8, 8)
        )

        # 上采样部分
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 输出尺寸: (128, 16, 16)
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 输出尺寸: (64, 32, 32)
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 输出尺寸: (32, 64, 64)
            nn.ReLU(),

            nn.ConvTranspose2d(32, num_keypoints, kernel_size=2, stride=2),  # 输出尺寸: (num_keypoints, 128, 128)
            nn.Sigmoid()  # 输出值在 [0, 1] 范围内
        )

    def forward(self, x):
        features = self.backbone(x)  # 提取特征
        heatmaps = self.upsample(features)  # 上采样生成热力图
        return heatmaps

