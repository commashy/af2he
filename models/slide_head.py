"""Benign/malignant head on the collage class token."""

from __future__ import annotations

import torch.nn as nn


class SlideDiagnosisHead(nn.Module):
    def __init__(self, dim: int, n_classes: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(int(dim))
        self.linear = nn.Linear(int(dim), int(n_classes))

    def forward(self, token):
        return self.linear(self.norm(token))
