"""Checkpoint save and load. Weights, optimizer, epoch, step. No GradScaler."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def unwrap(model: nn.Module) -> nn.Module:
    while True:
        if hasattr(model, "module"):
            model = model.module
            continue
        orig = getattr(model, "_orig_mod", None)
        if orig is not None:
            model = orig
            continue
        return model


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    global_step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    pretrain_discriminator: nn.Module | None = None,
    pretrain_disc_optimizer: torch.optim.Optimizer | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    module = unwrap(model)
    payload = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model": module.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    # This critic belongs to reconstruction pretraining only. Stage-2 staining
    # builds a new discriminator and must not load these keys.
    if pretrain_discriminator is not None and pretrain_disc_optimizer is not None:
        payload["pretrain_discriminator"] = unwrap(pretrain_discriminator).state_dict()
        payload["pretrain_disc_optimizer"] = pretrain_disc_optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[int, int]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    missing = [key for key in ("epoch", "global_step", "model") if key not in payload]
    if missing:
        raise KeyError(f"{path} is missing {missing}")
    module = unwrap(model)
    incompatible = module.load_state_dict(payload["model"])
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"{path} does not match the generator: "
            f"missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
        )
    if optimizer is not None:
        if "optimizer" not in payload:
            raise KeyError(f"{path} has no optimizer state")
        optimizer.load_state_dict(payload["optimizer"])
    return int(payload["epoch"]), int(payload["global_step"])
