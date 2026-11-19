"""UVCGAN v2 patch mask. Kept patches stay native. Dropped patches are -1."""

from __future__ import annotations

import torch


def mask_patches(
    image: torch.Tensor,
    patch: int = 64,
    fraction: float = 0.4,
    fill: float = -1.0,
) -> torch.Tensor:
    """Mask about ``fraction`` of the non-overlapping patches.

    ``fraction`` 0.4 keeps a patch when a uniform random draw is above 0.4,
    which is the UVCGAN v2 rule. The fill is -1, the bottom of the tanh range.
    """
    if image.ndim != 4:
        raise ValueError(f"image must be NCHW, got {tuple(image.shape)}")
    height, width = image.shape[-2:]
    if patch < 1 or height % patch != 0 or width % patch != 0:
        raise ValueError(f"patch {patch} does not divide {(height, width)}")
    if not 0.0 <= float(fraction) <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    grid_h, grid_w = height // patch, width // patch
    keep = (
        torch.rand((image.shape[0], 1, grid_h, grid_w), device=image.device) > float(fraction)
    ).to(dtype=image.dtype)
    keep = keep.repeat_interleave(patch, dim=2).repeat_interleave(patch, dim=3)
    filled = torch.full_like(image, float(fill))
    return image * keep + filled * (1.0 - keep)
