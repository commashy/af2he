"""CONCH field critic following Diffusion-FFPE's vision-aided head.

The backbone is frozen CONCH ViT-B/16. The released weights use 448 input,
so the 1024 field is resized to 448. The trainable head is their
``MultiLevelDViT`` for ``cv_type='conch'``: two convolutional levels and one
linear level. Their class embedding is unused (``num_classes=0``). The slide
vector is added on that last level in the same dot-product place.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

CHECKPOINT = Path(
    "/home/taek/.cache/huggingface/hub/models--MahmoodLab--conch/"
    "snapshots/f9ca9f877171a28ade80228fb195ac5d79003357/pytorch_model.bin"
)


class _BlurPool(nn.Module):
    """Binomial blur used by their head. ``pad_off`` matches their calls."""

    def __init__(self, channels: int, stride: int = 1, pad_off: int = 0):
        super().__init__()
        kernel = torch.tensor([1.0, 3.0, 3.0, 1.0])
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel / kernel.sum()
        self.register_buffer("filt", kernel[None, None].repeat(int(channels), 1, 1, 1))
        pad = 1 + int(pad_off)
        self.pad = nn.ZeroPad2d(pad)
        self.stride = int(stride)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.conv2d(self.pad(image), self.filt, stride=self.stride, groups=image.shape[1])


class _SpatialLevel(nn.Module):
    """One convolutional level of ``MultiLevelDViT`` with ``down=2``."""

    def __init__(self, in_channels: int = 768, width: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            _BlurPool(in_channels, stride=1, pad_off=1),
            spectral_norm(nn.Conv2d(in_channels, width, kernel_size=3, stride=2, padding=0)),
            nn.LeakyReLU(0.2, inplace=True),
            _BlurPool(width, stride=1, pad_off=0),
            spectral_norm(nn.Conv2d(width, 1, kernel_size=1, stride=2)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(1)


class ConchFieldDiscriminator(nn.Module):
    def __init__(self, slide_dim: int | None = 512, image_size: int = 448, checkpoint: str | Path = CHECKPOINT):
        super().__init__()
        from conch.open_clip_custom import create_model_from_pretrained

        if not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
        self.image_size = int(image_size)
        model = create_model_from_pretrained(
            "conch_ViT-B-16",
            str(checkpoint),
            device="cpu",
            force_image_size=self.image_size,
            return_transform=False,
        )
        self.visual = model.visual
        del model
        self.visual.eval()
        for parameter in self.visual.parameters():
            parameter.requires_grad_(False)
        mean = torch.tensor(self.visual.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(self.visual.image_std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)
        self.spatial = nn.ModuleList([_SpatialLevel() for _ in range(2)])
        self.global_hidden = nn.Sequential(
            spectral_norm(nn.Linear(768, 128)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.global_score = spectral_norm(nn.Linear(128, 1))
        self.slide = None if slide_dim is None else nn.Linear(int(slide_dim), 128, bias=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.visual.eval()
        return self

    def state_dict(self, *args, **kwargs):
        full = super().state_dict(*args, **kwargs)
        return {key: value for key, value in full.items() if not key.startswith("visual.")}

    def load_state_dict(self, state_dict, strict: bool = True):
        return super().load_state_dict(state_dict, strict=False)

    def _levels(self, image: torch.Tensor):
        resized = F.interpolate(
            image.float().mul(0.5).add(0.5), size=(self.image_size, self.image_size), mode="area",
        )
        normalized = (resized - self.mean) / self.std
        with torch.no_grad():
            layers = self.visual.trunk.get_intermediate_layers(
                normalized, n=8, return_prefix_tokens=True,
            )
        picked = [layers[index] for index in (0, 4, -1)]
        maps = []
        for tokens, _prefix in picked[:2]:
            batch, count, channels = tokens.shape
            side = int(count ** 0.5)
            if side * side != count or channels != 768:
                raise RuntimeError(f"CONCH map is {tuple(tokens.shape)}, expected 196 x 768")
            maps.append(tokens.permute(0, 2, 1).reshape(batch, channels, side, side).float())
        cls = picked[-1][1][:, 0, :].float()
        if cls.shape[-1] != 768:
            raise RuntimeError(f"CONCH class token is {tuple(cls.shape)}")
        return maps[0], maps[1], cls

    def forward(self, image: torch.Tensor, slide: torch.Tensor | None = None) -> list[torch.Tensor]:
        if image.ndim != 4 or image.shape[-2:] != (1024, 1024):
            raise ValueError(f"field must be (N, 3, 1024, 1024), got {tuple(image.shape)}")
        first, second, cls = self._levels(image)
        predictions = [level(features) for level, features in zip(self.spatial, (first, second))]
        hidden = self.global_hidden(cls)
        score = self.global_score(hidden)
        if self.slide is not None:
            if slide is None or slide.ndim != 2 or slide.shape != (image.shape[0], self.slide.in_features):
                width = None if self.slide is None else self.slide.in_features
                raise ValueError(
                    f"slide vector must be ({image.shape[0]}, {width}), got {None if slide is None else tuple(slide.shape)}"
                )
            score = score + (self.slide(slide.to(dtype=hidden.dtype)) * hidden).sum(dim=1, keepdim=True)
        predictions.append(score)
        return predictions


def multilevel_bce(predictions: list[torch.Tensor], real: bool, weight: torch.Tensor) -> torch.Tensor:
    """Their ``multilevel_sigmoid``: BCE-with-logits summed across levels."""
    target_value = 1.0 if real else 0.0
    total = None
    for prediction in predictions:
        target = torch.full_like(prediction.float(), target_value)
        loss = F.binary_cross_entropy_with_logits(prediction.float(), target, reduction="none")
        if loss.ndim > 2:
            loss = loss.mean(dim=tuple(range(1, loss.ndim)))
        elif loss.ndim == 2:
            loss = loss.mean(dim=1)
        total = loss if total is None else total + loss
    return (total * weight).sum() / weight.sum().clamp_min(1.0)
