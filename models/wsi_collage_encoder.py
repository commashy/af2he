"""WSI collage encoder.

ResNet-18 reads the 2048 collage. Each 256 cell becomes one token after the
network has seen the inside of that cell. Gated attention, as in CLAM,
pools the 64 tokens into the slide vector.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class GatedAttention(nn.Module):
    """CLAM ``Attn_Net_Gated``, over the cell tokens of one slide."""

    def __init__(self, channels: int = 512, hidden: int = 256, dropout: float = 0.25):
        super().__init__()
        self.attention_a = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.attention_b = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.attention_c = nn.Linear(hidden, 1)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention_c(self.attention_a(tokens) * self.attention_b(tokens))
        weights = torch.softmax(scores.squeeze(-1), dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), tokens).squeeze(1)
        return pooled, weights


class WSICollageEncoder(nn.Module):
    def __init__(self, n_subtypes: int, size: int = 2048, grid: int = 8):
        super().__init__()
        self.size = int(size)
        self.grid = int(grid)
        self.backbone = timm.create_model(
            "resnet18", pretrained=True, num_classes=0, global_pool="",
        )
        channels = 512
        self.cell_score = nn.Conv2d(channels, 1, kernel_size=1)
        self.attention = GatedAttention(channels)
        self.drop = nn.Dropout(0.3)
        self.bin_head = nn.Linear(channels, 2)
        self.sub_head = nn.Linear(channels, int(n_subtypes))
        nn.init.zeros_(self.cell_score.weight)
        nn.init.zeros_(self.cell_score.bias)

    def cell_tokens(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1:] != (3, self.size, self.size):
            raise ValueError(f"collage must be (N, 3, {self.size}, {self.size}), got {tuple(image.shape)}")
        features = self.backbone(image)
        batch, channels, height, width = features.shape
        side = self.size // 32
        cell = side // self.grid
        if (height, width) != (side, side) or cell < 1:
            raise RuntimeError(f"feature map {height}x{width} does not match a {self.grid} grid")
        grouped = features.view(batch, channels, self.grid, cell, self.grid, cell)
        grouped = grouped.permute(0, 2, 4, 1, 3, 5).contiguous()
        grouped = grouped.view(batch * self.grid * self.grid, channels, cell, cell)
        scores = self.cell_score(grouped).flatten(1)
        weights = torch.softmax(scores, dim=1).view(-1, 1, cell, cell)
        pooled = (grouped * weights).sum(dim=(2, 3))
        return pooled.view(batch, self.grid * self.grid, channels)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.cell_tokens(image)
        pooled, _weights = self.attention(tokens)
        hidden = self.drop(pooled)
        return self.bin_head(hidden), self.sub_head(hidden), tokens


class WSICollageAdapter(nn.Module):
    """New collage encoder as the generator's slide conditioning.

    The slide vector is the FiLM condition. The bottleneck memory is that
    vector plus the 3x3 cells around the tile. A neighbor past the grid is
    the encoder's token for a black cell.
    """

    def __init__(self, n_subtypes: int, bottleneck_dim: int = 384):
        super().__init__()
        self.encoder = WSICollageEncoder(n_subtypes)
        self.token_proj = nn.Linear(512, bottleneck_dim)
        self.slide_proj = nn.Linear(512, bottleneck_dim)

    def load_pretrained(self, path: str) -> int:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.encoder.load_state_dict(payload["model"])
        return int(payload["epoch"])

    def forward(
        self,
        collage: torch.Tensor,
        cells: torch.Tensor,
        black_index: torch.Tensor,
        group: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Training batches repeat one collage across four quadrants. A seam
        # window is a single tile, so it passes group=1.
        if group < 1 or collage.shape[0] % group != 0:
            raise ValueError(f"collage batch {collage.shape[0]} is not divisible by group {group}")
        if cells.shape[0] != collage.shape[0]:
            raise ValueError(f"{cells.shape[0]} cells for {collage.shape[0]} collages")
        unique = collage[::group]
        unique_cells = cells[::group]
        del black_index
        frozen = all(not parameter.requires_grad for parameter in self.encoder.parameters())
        if frozen:
            with torch.no_grad():
                tokens = self.encoder.cell_tokens(unique)
                pooled, _weights = self.encoder.attention(tokens)
                hidden = self.encoder.drop(pooled)
                binary = self.encoder.bin_head(hidden)
                subtype = self.encoder.sub_head(hidden)
            tokens = tokens.detach()
            pooled = pooled.detach()
            binary = binary.detach()
            subtype = subtype.detach()
        else:
            tokens = self.encoder.cell_tokens(unique)
            pooled, _weights = self.encoder.attention(tokens)
            hidden = self.encoder.drop(pooled)
            binary = self.encoder.bin_head(hidden)
            subtype = self.encoder.sub_head(hidden)
        self.last_pooled = pooled.detach()
        projected = self.token_proj(tokens)
        slide = self.slide_proj(pooled)
        memory, keep = self._neighborhood(projected, unique_cells, slide)
        return (
            memory.repeat_interleave(group, dim=0),
            slide.repeat_interleave(group, dim=0),
            binary,
            subtype,
            keep.repeat_interleave(group, dim=0),
        )

    def _neighborhood(
        self,
        tokens: torch.Tensor,
        cells: torch.Tensor,
        slide: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local = []
        keep_local = []
        batch = torch.arange(tokens.shape[0], device=tokens.device)
        outside = cells[:, 0] < 0
        for offset_row in (-1, 0, 1):
            for offset_col in (-1, 0, 1):
                row = cells[:, 0] + offset_row
                col = cells[:, 1] + offset_col
                inside = (~outside) & (row >= 0) & (row < 8) & (col >= 0) & (col < 8)
                index = row.clamp(0, 7) * 8 + col.clamp(0, 7)
                picked = tokens[batch, index]
                local.append(torch.where(inside[:, None], picked, torch.zeros_like(picked)))
                keep_local.append(inside)
        memory = torch.cat((slide.unsqueeze(1), torch.stack(local, dim=1)), dim=1)
        keep = torch.cat(
            (
                torch.ones(tokens.shape[0], 1, dtype=torch.bool, device=tokens.device),
                torch.stack(keep_local, dim=1),
            ),
            dim=1,
        )
        return memory, keep
