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


class DomainAdapter(nn.Module):
    """Per-patch LayerNorm, then a domain scale and shift. Two vectors per domain."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(int(in_dim), elementwise_affine=False)
        self.gamma_af = nn.Parameter(torch.ones(1, 1, int(in_dim)))
        self.beta_af = nn.Parameter(torch.zeros(1, 1, int(in_dim)))
        self.gamma_he = nn.Parameter(torch.ones(1, 1, int(in_dim)))
        self.beta_he = nn.Parameter(torch.zeros(1, 1, int(in_dim)))

    def forward(self, patches: torch.Tensor, domain: str) -> torch.Tensor:
        aligned = self.norm(patches.float())
        if domain == "af":
            return aligned * self.gamma_af + self.beta_af
        if domain == "he":
            return aligned * self.gamma_he + self.beta_he
        raise ValueError(f"domain must be af or he, got {domain}")


class DomainAdaptedPatchProjector(nn.Module):
    """One domain adapter and one shared MLP for each encoder map."""

    def __init__(self, channels: list[int], width: int = 256):
        super().__init__()
        self.adapters = nn.ModuleList([DomainAdapter(int(channel)) for channel in channels])
        self.shared_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(int(channel), int(width)),
                    nn.ReLU(),
                    nn.Linear(int(width), int(width)),
                )
                for channel in channels
            ]
        )
        for mlp in self.shared_mlps:
            for module in mlp.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, 0.0, 0.02)
                    nn.init.zeros_(module.bias)

    def forward(self, sampled: list[torch.Tensor], domain: str) -> list[torch.Tensor]:
        if len(sampled) != len(self.adapters):
            raise ValueError(f"{len(sampled)} patch groups for {len(self.adapters)} adapters")
        outputs = []
        for patches, adapter, mlp in zip(sampled, self.adapters, self.shared_mlps):
            hidden = mlp(adapter(patches, domain))
            outputs.append(F.normalize(hidden, p=2, dim=-1))
        return outputs

    def param_groups(self, lr: float, adapter_wd: float = 1e-4) -> list[dict]:
        return [
            {"params": list(self.adapters.parameters()), "lr": lr, "weight_decay": adapter_wd},
            {"params": list(self.shared_mlps.parameters()), "lr": lr, "weight_decay": 0.0},
        ]
