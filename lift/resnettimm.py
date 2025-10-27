import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class SpatialSoftmax(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1., 1., width),
            torch.linspace(-1., 1., height),
            indexing='xy'
        )
        self.register_buffer('pos_x', pos_x.reshape(1, 1, height * width))  # shape (1,1,H*W)
        self.register_buffer('pos_y', pos_y.reshape(1, 1, height * width))  # shape (1,1,H*W)

    def forward(self, feature):
        B, C, H, W = feature.shape
        feature = feature.view(B, C, H * W)
        softmax = F.softmax(feature, dim=-1)
        x = (softmax * self.pos_x).sum(dim=2)
        y = (softmax * self.pos_y).sum(dim=2)
        return torch.cat([x, y], dim=1)  # shape: (B, 2*C)

class TimMResNet18Encoder(nn.Module):
    def __init__(self, pretrained=True, latent_dim=128):
        super().__init__()
        self.backbone = timm.create_model('resnet18', pretrained=pretrained, features_only=True)

        # Dummy forward to get feature shape
        dummy_input = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            dummy_feat = self.backbone(dummy_input)[-1]
            _, C, H, W = dummy_feat.shape

        self.spatial_softmax = SpatialSoftmax(H, W)  # dynamically use spatial shape
        self.projector = nn.Linear(2 * C, latent_dim)  # 2*C from spatial softmax

    def forward(self, x):
        features = self.backbone(x)[-1]  # get final ResNet feature map
        softmax_features = self.spatial_softmax(features)  # → (B, 2*C)
        latent = self.projector(softmax_features)  # → (B, latent_dim)
        return latent
