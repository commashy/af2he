"""Whole-slide thumbnail encoder.

ImageNet ResNet-18 reads the 1024 thumbnail and keeps its 32x32 map.
Attention pooling makes one slide vector. One head is benign versus
malignant. The other head is the slide subtype.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class ThumbEncoder(nn.Module):
    def __init__(self, n_subtypes: int, size: int = 1024):
        super().__init__()
        self.size = int(size)
        self.backbone = timm.create_model(
            "resnet18", pretrained=True, num_classes=0, global_pool="",
        )
        channels = 512
        self.attn = nn.Conv2d(channels, 1, kernel_size=1)
        self.drop = nn.Dropout(0.3)
        self.bin_head = nn.Linear(channels, 2)
        self.sub_head = nn.Linear(channels, int(n_subtypes))
        nn.init.zeros_(self.attn.weight)
        nn.init.zeros_(self.attn.bias)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if image.ndim != 4 or image.shape[1:] != (3, self.size, self.size):
            raise ValueError(f"thumbnail must be (N, 3, {self.size}, {self.size}), got {tuple(image.shape)}")
        features = self.backbone(image)
        side = self.size // 32
        if features.shape[-2:] != (side, side):
            raise RuntimeError(f"thumbnail map must be {side}x{side}, got {tuple(features.shape)}")
        weights = torch.softmax(self.attn(features).flatten(1), dim=1)
        weights = weights.view(features.shape[0], 1, side, side)
        pooled = (features * weights).sum(dim=(2, 3))
        hidden = self.drop(pooled)
        return self.bin_head(hidden), self.sub_head(hidden), pooled
