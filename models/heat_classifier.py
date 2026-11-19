"""Real-H&E malignant fraction.

ImageNet ResNet-18 with one logit, the Camelyon patch classifier in the
MONAI tumor-detection tutorial (``resnet18``, ``num_classes=1``). The input
is a 1024 H&E field in [-1, 1]. The target is the soft fraction from
``data.heat_label``.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class HeatClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = timm.create_model("resnet18", pretrained=True, num_classes=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image).flatten(0)
