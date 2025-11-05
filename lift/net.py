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
        self.register_buffer('pos_x', pos_x.reshape(1, 1, height * width))
        self.register_buffer('pos_y', pos_y.reshape(1, 1, height * width))  

    def forward(self, feature):
        B, C, H, W = feature.shape
        feature = feature.view(B, C, H * W)
        softmax = F.softmax(feature, dim=-1)
        x = (softmax * self.pos_x).sum(dim=2)
        y = (softmax * self.pos_y).sum(dim=2)
        return torch.cat([x, y], dim=1)

class TimMResNet18Encoder(nn.Module):
    def __init__(self, pretrained=True, latent_dim=128):
        super().__init__()
        self.backbone = timm.create_model('resnet18', pretrained=pretrained, features_only=True)

        dummy_input = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            dummy_feat = self.backbone(dummy_input)[-1]
            _, C, H, W = dummy_feat.shape

        self.spatial_softmax = SpatialSoftmax(H, W)  
        self.projector = nn.Linear(2 * C, latent_dim)

    def forward(self, x):
        features = self.backbone(x)[-1] 
        softmax_features = self.spatial_softmax(features) 
        latent = self.projector(softmax_features) 
        return latent

class StateEncoder(nn.Module):
    def __init__(self, input_dim=14, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        return self.net(x)
