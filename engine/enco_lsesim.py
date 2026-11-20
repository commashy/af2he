"""Local self-similarity between the AF encoder and the pre-skip decoder.

The descriptor at a site is its cosine with the surrounding neighborhood, not
the feature vector. A dim nucleus in bright tissue keeps the same pattern
after the stain colors change. The encoder side is detached. The decoder side
is taken before the skip join, so the loss cannot be satisfied by copying the
encoder skip.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnCoLSeSimLoss(nn.Module):
    def __init__(
        self,
        num_layers: int,
        kernel_size: int = 7,
        dilation: int = 1,
        hidden_dim: int = 128,
        tau: float = 0.07,
        fg_ratio: float = 0.85,
    ):
        super().__init__()
        if int(kernel_size) % 2 != 1:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        if float(tau) <= 0:
            raise ValueError(f"temperature must be positive, got {tau}")
        self.k = int(kernel_size)
        self.dilation = int(dilation)
        self.pad = (self.k // 2) * self.dilation
        self.tau = float(tau)
        self.fg_ratio = float(fg_ratio)
        radius = self.k // 2
        offsets = torch.arange(-radius, radius + 1) * self.dilation
        dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
        dy = dy.reshape(-1)
        dx = dx.reshape(-1)
        keep = (dy != 0) | (dx != 0)
        self.register_buffer("dy", dy[keep].view(1, 1, -1))
        self.register_buffer("dx", dx[keep].view(1, 1, -1))
        width = int(keep.sum().item())
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(width, int(hidden_dim)),
                    nn.ReLU(),
                    nn.Linear(int(hidden_dim), int(hidden_dim)),
                )
                for _ in range(int(num_layers))
            ]
        )
        for mlp in self.mlps:
            for module in mlp.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, 0.0, 0.02)
                    nn.init.zeros_(module.bias)

    def _sample_coords(
        self,
        batch: int,
        height: int,
        width: int,
        num_patches: int,
        device: torch.device,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = min(int(num_patches), height * width)
        tissue = None
        if mask is not None:
            tissue = F.interpolate(mask.float(), size=(height, width), mode="nearest")
            tissue = tissue.view(batch, height * width) > 0.5
        chosen = []
        for index in range(batch):
            if tissue is None:
                sites = torch.randperm(height * width, device=device)[:count]
            else:
                foreground = torch.nonzero(tissue[index], as_tuple=False).flatten()
                background = torch.nonzero(~tissue[index], as_tuple=False).flatten()
                n_fg = min(int(count * self.fg_ratio), int(foreground.numel()))
                n_bg = min(count - n_fg, int(background.numel()))
                remain = count - (n_fg + n_bg)
                if remain > 0 and int(foreground.numel()) > n_fg:
                    n_fg = min(int(foreground.numel()), n_fg + remain)
                elif remain > 0 and int(background.numel()) > n_bg:
                    n_bg = min(int(background.numel()), n_bg + remain)
                parts = []
                if n_fg > 0:
                    order = torch.randperm(int(foreground.numel()), device=device)[:n_fg]
                    parts.append(foreground[order])
                if n_bg > 0:
                    order = torch.randperm(int(background.numel()), device=device)[:n_bg]
                    parts.append(background[order])
                sites = torch.cat(parts, dim=0) if parts else torch.randperm(height * width, device=device)[:count]
            if int(sites.numel()) < count:
                extra = sites[torch.randint(0, int(sites.numel()), (count - int(sites.numel()),), device=device)]
                sites = torch.cat((sites, extra), dim=0)
            chosen.append(sites)
        flat = torch.stack(chosen, dim=0)
        return torch.div(flat, width, rounding_mode="floor"), flat % width

    def _local_self_sim(self, feat: torch.Tensor, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        batch, _channels, _height, _width = feat.shape
        normalized = F.normalize(feat.float(), p=2, dim=1, eps=1e-6)
        padded = F.pad(normalized, (self.pad, self.pad, self.pad, self.pad), mode="reflect")
        _batch, _channels, _padded_h, padded_w = padded.shape
        flat = padded.permute(0, 2, 3, 1).reshape(batch, -1, padded.shape[1])
        rows = torch.arange(batch, device=feat.device)[:, None]
        center_index = (y + self.pad) * padded_w + (x + self.pad)
        center = flat[rows, center_index]
        neighbor_y = y.unsqueeze(-1) + self.pad + self.dy
        neighbor_x = x.unsqueeze(-1) + self.pad + self.dx
        neighbor_index = neighbor_y * padded_w + neighbor_x
        batch_index = rows.unsqueeze(-1).expand_as(neighbor_index)
        neighbors = flat[batch_index, neighbor_index]
        return (center.unsqueeze(2) * neighbors).sum(dim=-1)

    def forward(
        self,
        enc_feats: list[torch.Tensor],
        dec_feats: list[torch.Tensor],
        num_patches: int = 256,
        mask_fg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if len(enc_feats) != len(dec_feats) or len(enc_feats) != len(self.mlps):
            raise ValueError(
                f"layer counts enc={len(enc_feats)} dec={len(dec_feats)} mlp={len(self.mlps)}"
            )
        total = enc_feats[0].new_zeros((), dtype=torch.float32)
        for enc, dec, mlp in zip(enc_feats, dec_feats, self.mlps):
            if enc.shape[-2:] != dec.shape[-2:]:
                raise ValueError(f"spatial mismatch {tuple(enc.shape[-2:])} vs {tuple(dec.shape[-2:])}")
            batch, _channels, height, width = enc.shape
            y, x = self._sample_coords(
                batch, height, width, int(num_patches), enc.device, mask_fg,
            )
            structure_af = self._local_self_sim(enc.detach(), y, x)
            structure_he = self._local_self_sim(dec, y, x)
            key = F.normalize(mlp(structure_af), p=2, dim=-1, eps=1e-6)
            query = F.normalize(mlp(structure_he), p=2, dim=-1, eps=1e-6)
            count = query.shape[1]
            logits = torch.bmm(query, key.transpose(1, 2)) / self.tau
            labels = torch.arange(count, device=query.device).unsqueeze(0).expand(batch, count)
            total = total + F.cross_entropy(logits.reshape(batch * count, count), labels.reshape(batch * count))
        return total / len(self.mlps)
