import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class SpatialSoftmax(nn.Module):
    def __init__(self, height, width, channel):
        super().__init__()
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1., 1., width),
            torch.linspace(-1., 1., height),
            indexing='xy'
        )
        self.register_buffer('pos_x', pos_x.reshape(1, 1, height * width))
        self.register_buffer('pos_y', pos_y.reshape(1, 1, height * width))
        self.height = height
        self.width = width
        self.channel = channel

    def forward(self, feature):
        B, C, H, W = feature.shape
        feature = feature.view(B, C, H * W)
        softmax = F.softmax(feature, dim=-1)
        x = (softmax * self.pos_x).sum(dim=2)
        y = (softmax * self.pos_y).sum(dim=2)
        return torch.cat([x, y], dim=1)  # (B, 2*C)

class TimMResNet18Encoder(nn.Module):
    def __init__(self, pretrained=True, latent_dim=128):
        super().__init__()
        self.backbone = timm.create_model('resnet18', pretrained=pretrained, features_only=True)
        self.spatial_softmax = SpatialSoftmax(7, 7, 512)  # output 1024 dims
        self.projector = nn.Linear(1024, latent_dim)  # project down to 128

    def forward(self, x):
        features = self.backbone(x)[-1]  # (B, 512, 7, 7)
        ss = self.spatial_softmax(features)  # (B, 1024)
        latent = self.projector(ss)  # (B, 128)
        return latent


