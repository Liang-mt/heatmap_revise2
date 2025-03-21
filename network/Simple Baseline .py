import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleBaseline(nn.Module):
    def __init__(self, num_keypoints, backbone='resnet50'):
        super().__init__()
        orig_resnet = torch.hub.load('pytorch/vision', backbone, pretrained=True)
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu = orig_resnet.relu
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1  # 输出通道256
        self.layer2 = orig_resnet.layer2  # 输出通道512
        self.layer3 = orig_resnet.layer3  # 输出通道1024
        self.layer4 = orig_resnet.layer4  # 输出通道2048

        # 反卷积头
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(2048, 256, 4, 2, 1),  # 输出尺寸加倍
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),  # 再次加倍
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),  # 最终输出尺寸为输入的1/4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_keypoints, 1)  # 关键点热力图
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # (B,2048,H/32,W/32)
        x = self.deconv_layers(x)  # (B, num_keypoints, H/4, W/4)
        return x