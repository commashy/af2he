"""CUT patch projector. One MLP per encoder map, then L2 normalization."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchProjector(nn.Module):
    def __init__(self, channels: list[int], width: int = 256):
        super().__init__()
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(int(channel), int(width)),
                    nn.ReLU(),
                    nn.Linear(int(width), int(width)),
                )
                for channel in channels
            ]
        )
        for mlp in self.mlps:
            for module in mlp.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, 0.0, 0.02)
                    nn.init.zeros_(module.bias)

    def _project(
        self,
        feats: list[torch.Tensor],
        num_patches: int,
        patch_ids: list[torch.Tensor] | None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if len(feats) != len(self.mlps):
            raise ValueError(f"{len(feats)} feature maps for {len(self.mlps)} projectors")
        sampled: list[torch.Tensor] = []
        ids: list[torch.Tensor] = []
        for index, feat in enumerate(feats):
            batch, channels, height, width = feat.shape
            flat = feat.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
            if patch_ids is None:
                count = min(int(num_patches), flat.shape[1])
                choice = torch.randperm(flat.shape[1], device=feat.device)[:count]
            else:
                choice = patch_ids[index]
            picked = flat[:, choice, :].reshape(-1, channels).float()
            projected = F.normalize(self.mlps[index](picked), dim=1, eps=1e-7)
            sampled.append(projected)
            ids.append(choice)
        return sampled, ids

    def forward(self, groups: list[list[torch.Tensor]], num_patches: int) -> list[list[torch.Tensor]]:
        """Project several feature groups with one shared set of patch locations."""
        ids: list[torch.Tensor] | None = None
        outputs = []
        for feats in groups:
            projected, ids = self._project(feats, num_patches, ids)
            outputs.append(projected)
        return outputs
