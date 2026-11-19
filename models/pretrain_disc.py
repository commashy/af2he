"""Pretraining PatchGAN, separate from the virtual-staining critic.

The loss schedule follows RAE and UAE: hinge discriminator, non-saturating
generator loss, and the adaptive weight from Esser et al. 2021. The network
itself is the pix2pix PatchGAN and sees the 512 tile at full resolution.
Stage-2 staining builds a new discriminator and must not load this one.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def hinge_discriminator_loss(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """VQGAN hinge loss, the RAE discriminator objective."""
    return 0.5 * (F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean())


def vanilla_generator_loss(fake: torch.Tensor) -> torch.Tensor:
    """Non-saturating generator loss used by RAE: negative mean logit."""
    return -fake.mean()


def adaptive_weight(
    reconstruction: torch.Tensor,
    adversarial: torch.Tensor,
    last_layer: nn.Parameter,
    maximum: float = 1e4,
) -> torch.Tensor:
    """Esser et al. 2021 balance, as used by RAE. Detached scalar."""
    recon_grad = torch.autograd.grad(reconstruction, last_layer, retain_graph=True)[0]
    gan_grad = torch.autograd.grad(adversarial, last_layer, retain_graph=True)[0]
    scale = recon_grad.detach().float().norm() / (gan_grad.detach().float().norm() + 1e-6)
    return scale.clamp(0.0, maximum).detach()


def diff_augment(images: torch.Tensor) -> torch.Tensor:
    """Zhao et al. 2020 color and translation augment. Cutout stays off, as in RAE."""
    images = _color(images)
    return _translation(images)


def _color(images: torch.Tensor) -> torch.Tensor:
    batch = images.shape[0]
    brightness = torch.empty(batch, 1, 1, 1, device=images.device).uniform_(0.6, 1.4)
    images = images * brightness
    mean = images.mean(dim=1, keepdim=True)
    saturation = torch.empty(batch, 1, 1, 1, device=images.device).uniform_(0.6, 1.4)
    images = (images - mean) * saturation + mean
    gray = images.mean(dim=(1, 2, 3), keepdim=True)
    contrast = torch.empty(batch, 1, 1, 1, device=images.device).uniform_(0.6, 1.4)
    return (images - gray) * contrast + gray


def _translation(images: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    batch, _channels, height, width = images.shape
    shift_y = int(height * ratio + 0.5)
    shift_x = int(width * ratio + 0.5)
    delta_y = torch.randint(-shift_y, shift_y + 1, (batch,), device=images.device)
    delta_x = torch.randint(-shift_x, shift_x + 1, (batch,), device=images.device)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, height, device=images.device),
        torch.linspace(-1, 1, width, device=images.device),
        indexing="ij",
    )
    grid_y = grid_y.unsqueeze(0) - delta_y.view(batch, 1, 1) * (2.0 / max(height - 1, 1))
    grid_x = grid_x.unsqueeze(0) - delta_x.view(batch, 1, 1) * (2.0 / max(width - 1, 1))
    grid = torch.stack((grid_x, grid_y), dim=-1)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="reflection", align_corners=True)


class PretrainDiscriminator(nn.Module):
    """pix2pix PatchGAN. The receptive field is 70 pixels on the 512 tile."""

    def __init__(self, channels: int = 64, layers: int = 3):
        super().__init__()
        blocks: list[nn.Module] = [
            nn.Conv2d(3, channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        width = channels
        for index in range(1, layers):
            nxt = min(width * 2, 512)
            blocks += [
                nn.Conv2d(width, nxt, 4, stride=2, padding=1),
                nn.InstanceNorm2d(nxt, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            width = nxt
        blocks += [
            nn.Conv2d(width, width, 4, stride=1, padding=1),
            nn.InstanceNorm2d(width, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(width, 1, 4, stride=1, padding=1),
        ]
        self.net = nn.Sequential(*blocks)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)
