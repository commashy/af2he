"""Cosine distance between local self-similarity maps.

Each site is the cosine of its feature with the other cells of a 7×7
neighborhood. The encoder map is detached. The decoder map is the pre-skip
tensor, so copying the skip cannot satisfy the loss. The generator is trained
with the distance 1 − cos between those two neighborhood vectors, which is
the FSeSim generator term. There is no MLP and no InfoNCE.

Features follow PatchSim: subtract the spatial mean, then divide each
location by its length. The neighborhood vector is then centered, so a faint
pattern and a strong pattern of the same shape match. A neighborhood whose
centered length is below ``flat_norm`` is left out, and the length used in
the denominator never falls below that floor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnCoFSeSimLoss(nn.Module):
    def __init__(
        self,
        dilations: tuple[int, ...] | list[int],
        kernel_size: int = 7,
        fg_ratio: float = 0.85,
        flat_norm: float = 0.05,
    ):
        super().__init__()
        if int(kernel_size) % 2 != 1:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        chosen = tuple(int(value) for value in dilations)
        if not chosen or any(value < 1 for value in chosen):
            raise ValueError(f"dilations must be positive, got {dilations}")
        if float(flat_norm) <= 0:
            raise ValueError(f"flat_norm must be positive, got {flat_norm}")
        self.k = int(kernel_size)
        self.dilations = chosen
        self.fg_ratio = float(fg_ratio)
        self.flat_norm = float(flat_norm)
        radius = self.k // 2
        offsets = torch.arange(-radius, radius + 1)
        dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
        dy = dy.reshape(-1)
        dx = dx.reshape(-1)
        keep = (dy != 0) | (dx != 0)
        self.register_buffer("base_dy", dy[keep].view(1, 1, -1))
        self.register_buffer("base_dx", dx[keep].view(1, 1, -1))

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

    def _local_self_sim(
        self,
        feat: torch.Tensor,
        y: torch.Tensor,
        x: torch.Tensor,
        dilation: int,
    ) -> torch.Tensor:
        batch, _channels, _height, _width = feat.shape
        centered = feat.float() - feat.float().mean(dim=(-2, -1), keepdim=True)
        normalized = F.normalize(centered, p=2, dim=1, eps=1e-6)
        pad = (self.k // 2) * int(dilation)
        padded = F.pad(normalized, (pad, pad, pad, pad), mode="reflect")
        _batch, _channels, _padded_h, padded_w = padded.shape
        flat = padded.permute(0, 2, 3, 1).reshape(batch, -1, padded.shape[1])
        rows = torch.arange(batch, device=feat.device)[:, None]
        center = flat[rows, (y + pad) * padded_w + (x + pad)]
        neighbor_y = y.unsqueeze(-1) + pad + self.base_dy * int(dilation)
        neighbor_x = x.unsqueeze(-1) + pad + self.base_dx * int(dilation)
        neighbor_index = neighbor_y * padded_w + neighbor_x
        batch_index = rows.unsqueeze(-1).expand_as(neighbor_index)
        neighbors = flat[batch_index, neighbor_index]
        return (center.unsqueeze(2) * neighbors).sum(dim=-1)

    def _distance(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
        source = source - source.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)
        source_norm = source.norm(dim=-1)
        target_norm = target.norm(dim=-1)
        floor = source.new_tensor(self.flat_norm)
        valid = source_norm > floor
        if int(valid.sum()) == 0:
            return None
        source_hat = source / source_norm.clamp(min=floor).unsqueeze(-1)
        target_hat = target / target_norm.clamp(min=floor).unsqueeze(-1)
        cosine = (source_hat * target_hat).sum(dim=-1).clamp(-1.0, 1.0)
        distance = (1.0 - cosine) * valid.detach().to(cosine.dtype)
        return distance.sum() / valid.detach().sum().clamp(min=1)

    def forward(
        self,
        enc_feats: list[torch.Tensor],
        dec_feats: list[torch.Tensor],
        num_patches: int = 256,
        mask_fg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if len(enc_feats) != len(dec_feats) or len(enc_feats) != len(self.dilations):
            raise ValueError(
                f"layer counts enc={len(enc_feats)} dec={len(dec_feats)} dilations={len(self.dilations)}"
            )
        terms = []
        for enc, dec, dilation in zip(enc_feats, dec_feats, self.dilations):
            if enc.shape[-2:] != dec.shape[-2:]:
                raise ValueError(f"spatial mismatch {tuple(enc.shape[-2:])} vs {tuple(dec.shape[-2:])}")
            batch, _channels, height, width = enc.shape
            y, x = self._sample_coords(
                batch, height, width, int(num_patches), enc.device, mask_fg,
            )
            source = self._local_self_sim(enc.detach(), y, x, dilation)
            target = self._local_self_sim(dec, y, x, dilation)
            distance = self._distance(source, target)
            if distance is not None:
                terms.append(distance)
        if not terms:
            return enc_feats[0].new_zeros((), dtype=torch.float32)
        return torch.stack(terms).mean()
