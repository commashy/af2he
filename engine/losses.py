"""UVCGAN v2 reconstruction: L1 between the full output and the full unmasked image."""

from __future__ import annotations

import torch


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} != target {tuple(target.shape)}")
    loss = (pred - target).abs().mean()
    value = float(loss.detach())
    return loss, {"loss": value, "l1": value}
