import torch
import torch.nn as nn
from model import SqueezeNet, SENet


class BaselineClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.squeezenet_views = nn.ModuleList([SqueezeNet() for _ in range(4)])
        self.senet_views = nn.ModuleList([SENet(c=512) for _ in range(4)])
        self.conv10_views = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1),
                nn.ReLU()
            ) for _ in range(4)
        ])
        self.gap_views = nn.ModuleList([nn.AdaptiveAvgPool2d((1, 1)) for _ in range(4)])

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels=num_classes * 4, out_channels=num_classes, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fused_senet = SENet(c=num_classes)
        self.fused_gap = nn.AdaptiveAvgPool2d((1, 1))

    def process_view(self, view_images, v):
        features = self.squeezenet_views[v](view_images)  # [B, C, H', W']
        features = self.senet_views[v](features)         # [B, 512, H', W']
        features = self.conv10_views[v](features)        # [B, 5, H', W']
        logits = self.gap_views[v](features).view(features.size(0), -1)  # [B, 5]
        return features, logits

    def forward(self, images):
        """
        images: [B, 4, 3, 200, 200] (Batch, Views, Channels, Height, Width)
        """
        B, V, C, H, W = images.size()  # B: Batch, V: Views
        assert V == len(self.squeezenet_views), "This model is designed for 4 views only."

        features_list, outputs_list = [], []
        for v in range(V):
            view_images = images[:, v, :, :, :]  # [B, 3, H, W]
            features, logits = self.process_view(view_images, v)
            features_list.append(features)
            outputs_list.append(logits)

        fused_features = torch.cat(features_list, dim=1)  # [B, 5*4, H', W']
        fused_features = self.fusion_conv(fused_features)  # [B, 5, H', W']
        fused_features = self.fused_senet(fused_features)  # [B, 5, H', W']
        fused_outputs = self.fused_gap(fused_features).view(B, -1)  # [B, 5]

        return outputs_list, fused_outputs
