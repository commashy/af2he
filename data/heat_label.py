"""Soft malignant target from a jet heatmap.

The heatmaps use the jet path. The ends measured in the files are dark blue
``(0, 0, 128)`` and dark red ``(159, 0, 0)``. Those ends are 0 and 1. Cyan,
green, and yellow stay on the path between them. The tile target is the mean.
A benign slide with no map is 0.
"""

from __future__ import annotations

import numpy as np

# Jet position of the dark-red end (159, 0, 0) in these heatmap files.
RED_END = 0.9716


def jet_score_cube() -> np.ndarray:
    """32x32x32 cube. Index with ``rgb // 8``. Values are in [0, 1]."""
    import matplotlib

    jet = matplotlib.colormaps["jet"]
    level = np.linspace(0.0, 0.999, 256)
    rgb = np.array([jet(float(v))[:3] for v in level], dtype=np.float32) * 255.0
    score = np.clip(level / RED_END, 0.0, 1.0).astype(np.float32)
    step = 256.0 / 32.0
    center = (np.arange(32, dtype=np.float32) + 0.5) * step
    points = np.stack(np.meshgrid(center, center, center, indexing="ij"), axis=-1).reshape(-1, 3)
    nearest = ((points[:, None, :] - rgb[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    return score[nearest].reshape(32, 32, 32).astype(np.float32)


# Yellow through red. Blue, cyan, and green are the other class.
MALIGNANT_MIN = 0.60


def malignant_pixels(score: np.ndarray) -> np.ndarray:
    """1 on yellow through red, 0 on every bluer color."""
    return (score >= MALIGNANT_MIN).astype(np.float32)


def tile_is_malignant(score: np.ndarray) -> float:
    """1 when at least half the tile is yellow through red, otherwise 0."""
    return float(malignant_pixels(score).mean() >= 0.5)


def score_image(rgb: np.ndarray, cube: np.ndarray) -> np.ndarray:
    """RGB uint8 image -> float32 jet level in [0, 1]."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB, got {rgb.dtype} {rgb.shape}")
    quant = rgb >> 3
    return cube[quant[:, :, 0], quant[:, :, 1], quant[:, :, 2]]


def paired_weight(amount: float, place: float, target: float) -> float:
    """Down-weight a paired tile when the sections disagree or the map is mixed.

    ``amount`` is 1 when the epoch-10 image and the real H&E have the same
    tissue amount. ``place`` is the overlap of those two tissue masks.
    ``|2t-1|`` is 0 for a half-and-half tile and 1 when the tile is all low
    or all high.
    """
    return float(amount) * float(place) * abs(2.0 * float(target) - 1.0)
