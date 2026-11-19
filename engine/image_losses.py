"""Gaussian pyramid, multiscale LSGAN, and amplitude-only frequency distribution loss."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianPyramidLoss(nn.Module):
    """L1 on Gaussian pyramid levels. Index 0 is the finest blurred image."""

    def __init__(self, num_high: int = 5, weights: list[float] | None = None):
        super().__init__()
        self.num_high = int(num_high)
        if self.num_high < 1:
            raise ValueError(f"num_high must be >= 1, got {num_high}")
        expected = self.num_high + 1
        if weights is None:
            weights = [1.0] * expected
        if len(weights) != expected:
            raise ValueError(f"pyramid weights length {len(weights)} != {expected}")
        self.weights = [float(value) for value in weights]
        kernel = torch.tensor(
            [
                [1.0, 4.0, 6.0, 4.0, 1.0],
                [4.0, 16.0, 24.0, 16.0, 4.0],
                [6.0, 24.0, 36.0, 24.0, 6.0],
                [4.0, 16.0, 24.0, 16.0, 4.0],
                [1.0, 4.0, 6.0, 4.0, 1.0],
            ],
            dtype=torch.float32,
        ) / 256.0
        self.register_buffer("kernel", kernel.view(1, 1, 5, 5))

    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.kernel.to(dtype=x.dtype).repeat(x.shape[1], 1, 1, 1)
        return F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), weight, groups=x.shape[1])

    def _pyramid(self, x: torch.Tensor) -> list[torch.Tensor]:
        current = x
        levels = []
        for _ in range(self.num_high):
            filtered = self._blur(current)
            levels.append(filtered)
            current = filtered[:, :, ::2, ::2]
        levels.append(current)
        return levels

    def forward(self, fake: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if fake.shape != target.shape:
            raise ValueError(f"fake {tuple(fake.shape)} != target {tuple(target.shape)}")
        fake_f = fake.float()
        target_f = target.detach().float()
        losses = [
            F.l1_loss(left, right) * weight
            for weight, left, right in zip(self.weights, self._pyramid(fake_f), self._pyramid(target_f))
        ]
        return sum(losses) / len(losses)


def lsgan(prediction: torch.Tensor | list[torch.Tensor], real: bool) -> torch.Tensor:
    """Mean squared error to 1 (real) or 0 (fake). Scales are summed, as in pix2pixHD."""
    target_value = 1.0 if real else 0.0
    maps = prediction if isinstance(prediction, (list, tuple)) else [prediction]
    losses = [F.mse_loss(item.float(), torch.full_like(item.float(), target_value)) for item in maps]
    return sum(losses)


_RGB_FROM_HED = torch.tensor(
    [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]],
    dtype=torch.float32,
)
_HED_FROM_RGB = torch.linalg.inv(_RGB_FROM_HED)


def hed_he(image: torch.Tensor, detach: bool = False) -> torch.Tensor:
    """Hematoxylin and eosin optical density, shape (N, 2, H, W).

    The residual third channel is left out. ``detach`` is for targets. The
    generated image keeps its gradient.
    """
    src = image.detach() if detach else image
    rgb = ((src.float() + 1.0) * 0.5).clamp(1e-6, 1.0)
    optical = -torch.log(rgb)
    hed = torch.matmul(
        optical.permute(0, 2, 3, 1),
        _HED_FROM_RGB.to(device=optical.device, dtype=optical.dtype).T,
    ).permute(0, 3, 1, 2).clamp_min(0.0)
    return hed[:, :2]


def hed_stain(image: torch.Tensor) -> torch.Tensor:
    """Hematoxylin plus eosin optical density, shape (N, H, W)."""
    channels = hed_he(image, detach=True)
    return channels[:, 0] + channels[:, 1]


def hed_tissue_mask(image: torch.Tensor, threshold: float = 0.15) -> torch.Tensor:
    """True where hematoxylin plus eosin optical density exceeds ``threshold``."""
    return (hed_stain(image) > float(threshold)).unsqueeze(1)


def tissue_score(
    image: torch.Tensor,
    threshold: float = 0.15,
    temperature: float = 0.05,
    detach: bool = False,
) -> torch.Tensor:
    """Soft tissue score in (0, 1), shape (N, 1, H, W). Midpoint is ``threshold``.

    Hematoxylin and eosin are added, so the score is where stain is, not which
    stain it is. ``detach`` is for maps built from targets.
    """
    channels = hed_he(image, detach=detach)
    stain = (channels[:, 0] + channels[:, 1]).unsqueeze(1)
    return torch.sigmoid((stain - float(threshold)) / float(temperature))


def hed_tissue_prob(
    image: torch.Tensor,
    threshold: float = 0.15,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Detached tissue score. Targets and masks use this."""
    return tissue_score(image, threshold, temperature, detach=True)


def af_presence(
    image: torch.Tensor,
    abs_lo: float = 0.025,
    abs_hi: float = 0.10,
    blur: int = 9,
) -> torch.Tensor:
    """Absolute AF presence in [0, 1], shape (N, 1, H, W).

    This is the source-domain vote from the clean hallucination loss. Empty
    tiles stay near zero because the thresholds are absolute, not a percentile.
    """
    gray = ((image.detach().float() + 1.0) * 0.5).mean(dim=1, keepdim=True)
    kernel = int(blur) if int(blur) % 2 == 1 else int(blur) + 1
    if kernel > 1:
        gray = F.avg_pool2d(gray, kernel_size=kernel, stride=1, padding=kernel // 2)
    span = max(float(abs_hi) - float(abs_lo), 1e-6)
    return ((gray - float(abs_lo)) / span).clamp(0.0, 1.0)


def tissue_probability(
    af: torch.Tensor,
    pseudo: torch.Tensor,
    teacher: torch.Tensor,
    threshold: float = 0.15,
    temperature: float = 0.05,
    abs_lo: float = 0.025,
    abs_hi: float = 0.10,
    blur: int = 9,
) -> torch.Tensor:
    """AF must agree before either H&E target can mark a pixel as tissue."""
    presence = af_presence(af, abs_lo=abs_lo, abs_hi=abs_hi, blur=blur)
    voted = torch.maximum(
        hed_tissue_prob(pseudo, threshold, temperature),
        hed_tissue_prob(teacher, threshold, temperature),
    )
    return presence * voted


def gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur. ``sigma`` is in pixels."""
    sigma = float(sigma)
    if sigma <= 0:
        return image
    radius = max(1, int(round(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=image.device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel = kernel / kernel.sum()
    channels = image.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    vertical = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    blurred = F.conv2d(F.pad(image, (radius, radius, 0, 0), mode="reflect"), horizontal, groups=channels)
    return F.conv2d(F.pad(blurred, (0, 0, radius, radius), mode="reflect"), vertical, groups=channels)


def hed_recon(
    fake: torch.Tensor,
    target: torch.Tensor,
    sigma: float,
    tissue_only: bool = False,
    eosin_only: bool = False,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Blurred stain of the output against the epoch-10 image.

    ``eosin_only`` compares eosin. Otherwise hematoxylin and eosin are both
    compared. The residual channel is left out. With ``tissue_only``, epoch-10
    glass is removed before the blur, so stain added there is not an error.
    ``weight`` is a per-pixel multiplier, detached, used as a weighted mean.
    """
    def stains(image: torch.Tensor, detach: bool) -> torch.Tensor:
        src = image.detach() if detach else image
        rgb = ((src.float() + 1.0) * 0.5).clamp(1e-6, 1.0)
        optical = -torch.log(rgb)
        hed = torch.matmul(
            optical.permute(0, 2, 3, 1),
            _HED_FROM_RGB.to(device=optical.device, dtype=optical.dtype),
        ).permute(0, 3, 1, 2).clamp_min(0.0)
        if eosin_only:
            return hed[:, 1:2]
        return hed[:, :2]

    left = stains(fake, False)
    right = stains(target, True)
    if tissue_only:
        mass = right if right.shape[1] == 1 else right[:, 0:1] + right[:, 1:2]
        mask = (mass > 0.15).to(dtype=left.dtype)
        left = left * mask
        right = right * mask
    if sigma > 0:
        left = gaussian_blur(left, sigma)
        right = gaussian_blur(right, sigma)
    error = (left - right).abs().mean(dim=1, keepdim=True)
    if weight is None:
        if tissue_only:
            return _gated_mean(error, mask)
        return error.mean()
    scale = weight.detach().float()
    if scale.ndim == 3:
        scale = scale.unsqueeze(1)
    if scale.shape[-2:] != error.shape[-2:]:
        raise ValueError(f"recon weight {tuple(scale.shape)} does not match {tuple(error.shape)}")
    if tissue_only:
        scale = scale * mask
    total = scale.sum()
    if float(total) <= 0.0:
        return error.sum() * 0.0
    return (error * scale).sum() / total


def _hed_mass(image: torch.Tensor, detach: bool) -> torch.Tensor:
    """Hematoxylin plus eosin optical density, shape (N, 1, H, W)."""
    src = image.detach() if detach else image
    rgb = ((src.float() + 1.0) * 0.5).clamp(1e-6, 1.0)
    optical = -torch.log(rgb)
    hed = torch.matmul(
        optical.permute(0, 2, 3, 1),
        _HED_FROM_RGB.to(device=optical.device, dtype=optical.dtype),
    ).permute(0, 3, 1, 2).clamp_min(0.0)
    return hed[:, 0:1] + hed[:, 1:2]


def stain_edge_loss(
    fake: torch.Tensor,
    target: torch.Tensor,
    sigma: float = 4.0,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Sobel edges of blurred hematoxylin-plus-eosin mass.

    Both images are reduced to one stain-mass channel and blurred by the
    same sigma before the gradient. Magnitude is Charbonnier. Orientation is
    ``1 - cos``, so a reversed edge is an error. The target is detached.
    Orientation is averaged on the target's edge strength, so blank pixels
    do not dominate.
    """
    left = _hed_mass(fake, False)
    right = _hed_mass(target, True)
    if sigma > 0:
        left = gaussian_blur(left, sigma)
        right = gaussian_blur(right, sigma)
    smooth = torch.tensor([1.0, 2.0, 1.0], device=left.device, dtype=left.dtype)
    diff = torch.tensor([1.0, 0.0, -1.0], device=left.device, dtype=left.dtype)
    kernel_h = (smooth.view(3, 1) @ diff.view(1, 3)).view(1, 1, 3, 3)
    kernel_v = (diff.view(3, 1) @ smooth.view(1, 3)).view(1, 1, 3, 3)

    def gradients(mass: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        padded = F.pad(mass, (1, 1, 1, 1), mode="reflect")
        horizontal = F.conv2d(padded, kernel_h)
        vertical = F.conv2d(padded, kernel_v)
        magnitude = torch.sqrt(horizontal.square() + vertical.square() + epsilon)
        return horizontal, vertical, magnitude

    src_h, src_v, src_mag = gradients(right)
    fake_h, fake_v, fake_mag = gradients(left)
    magnitude = torch.sqrt((src_mag - fake_mag).square() + epsilon).mean()
    src_norm = torch.sqrt(src_h.square() + src_v.square() + epsilon)
    fake_norm = torch.sqrt(fake_h.square() + fake_v.square() + epsilon)
    cosine = ((src_h * fake_h + src_v * fake_v) / (src_norm * fake_norm)).clamp(-1.0, 1.0)
    weight = src_mag.detach()
    orientation = ((1.0 - cosine) * weight).sum() / weight.sum().clamp_min(epsilon)
    return magnitude + orientation


def tone_loss(
    fake: torch.Tensor,
    teacher: torch.Tensor,
    pseudo: torch.Tensor,
    prob: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Match a wide blur to the teacher on tissue and the pseudo label on glass."""
    blurred_fake = gaussian_blur(fake.float(), sigma)
    blurred_teacher = gaussian_blur(teacher.detach().float(), sigma)
    blurred_pseudo = gaussian_blur(pseudo.detach().float(), sigma)
    target = prob.detach() * blurred_teacher + (1.0 - prob.detach()) * blurred_pseudo
    return (blurred_fake - target).abs().mean()


def af_brightness(
    image: torch.Tensor,
    lo: float = 0.20,
    hi: float = 0.30,
    blur: int = 9,
) -> torch.Tensor:
    """0 below ``lo``, 1 at ``hi``. Presence already saturates at gray 0.10."""
    gray = ((image.detach().float() + 1.0) * 0.5).mean(dim=1, keepdim=True)
    kernel = int(blur) if int(blur) % 2 == 1 else int(blur) + 1
    if kernel > 1:
        gray = F.avg_pool2d(gray, kernel_size=kernel, stride=1, padding=kernel // 2)
    span = max(float(hi) - float(lo), 1e-6)
    return ((gray - float(lo)) / span).clamp(0.0, 1.0)


def rgb_tissue_recon(
    fake: torch.Tensor,
    teacher: torch.Tensor,
    mask: torch.Tensor,
    sigma: float,
    field_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """RGB error on teacher tissue. The mask is fixed before any blur."""
    if sigma > 0:
        left = gaussian_blur(fake.float(), sigma)
        right = gaussian_blur(teacher.detach().float(), sigma)
    else:
        left = fake.float()
        right = teacher.detach().float()
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    mask = mask.detach().float()
    error = (left - right).abs() * mask
    covered = mask.flatten(1).sum(dim=1)
    per_field = error.flatten(1).sum(dim=1) / covered.clamp(min=1.0) / left.shape[1]
    per_field = per_field.masked_fill(covered <= 0, 0.0)
    alive = (covered > 0).float()
    if field_weight is not None:
        alive = alive * field_weight.detach().float().reshape(-1)
    if float(alive.sum()) <= 0.0:
        return left.sum() * 0.0
    return (per_field * alive).sum() / alive.sum()


def hed_ruifrok(image: torch.Tensor) -> torch.Tensor:
    """Hematoxylin and eosin with the Ruifrok matrix, shape (N, 2, H, W)."""
    rgb = ((image.float() + 1.0) * 0.5).clamp(1e-6, 1.0)
    optical = -torch.log(rgb)
    hed = torch.matmul(
        optical.permute(0, 2, 3, 1),
        _HED_FROM_RGB.to(device=optical.device, dtype=optical.dtype),
    ).permute(0, 3, 1, 2).clamp_min(0.0)
    return hed[:, :2]


def _gated_mean(error: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    total = gate.sum()
    if float(total) <= 0.0:
        return error.sum() * 0.0
    return (error * gate).sum() / total


def _floor_mean(error: torch.Tensor, mask: torch.Tensor, floor: float = 0.02) -> torch.Tensor:
    """Mean over the mask, with the denominator held at ``floor`` of the image.

    A pure mask mean gives a handful of pixels a huge step. A pure image mean
    ignores them. The floor keeps the per-pixel step at most ``1 / floor``
    times an image-wide mean, and the logged value shrinks when the mask is
    smaller than that.
    """
    total = mask.sum()
    if float(total) <= 0.0:
        return error.sum() * 0.0
    normalizer = total.clamp_min(float(floor) * mask.numel())
    return (error * mask).sum() / normalizer


def _af_gray(image: torch.Tensor) -> torch.Tensor:
    """Mean channel in [0, 1], detached. Shape (N, 1, H, W)."""
    return ((image.detach().float() + 1.0) * 0.5).mean(dim=1, keepdim=True)


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if int(radius) <= 0:
        return mask
    kernel = int(radius) * 2 + 1
    return F.max_pool2d(mask, kernel, stride=1, padding=int(radius))


def _window_mean(value: torch.Tensor, mask: torch.Tensor, radius: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean of ``value`` over ``mask`` inside a square window. ``den`` is the mask fraction."""
    kernel = int(radius) * 2 + 1
    weight = mask.float()
    numerator = F.avg_pool2d(value * weight, kernel, stride=1, padding=int(radius), count_include_pad=False)
    denominator = F.avg_pool2d(weight, kernel, stride=1, padding=int(radius), count_include_pad=False)
    return numerator / denominator.clamp_min(1e-6), denominator


def recon_bright_weight(
    af: torch.Tensor,
    epoch10: torch.Tensor,
    lo: float = 0.30,
    hi: float = 0.55,
    gain: float = 3.0,
    pale_od: float = 0.30,
) -> torch.Tensor:
    """1 on dim AF, ``gain`` on bright AF whose epoch 10 is still stained.

    The ramp is full at ``hi``. Epoch 10 below ``pale_od`` keeps weight 1,
    so an empty bright pixel is not pulled harder toward white.
    """
    gray = _af_gray(af)
    span = max(float(hi) - float(lo), 1e-6)
    ramp = ((gray - float(lo)) / span).clamp(0.0, 1.0)
    reference = hed_ruifrok(epoch10).detach()
    optical = reference[:, 0:1] + reference[:, 1:2]
    ramp = ramp * (optical >= float(pale_od)).to(dtype=ramp.dtype)
    return 1.0 + (float(gain) - 1.0) * ramp


def bright_neighbor_fill(
    fake: torch.Tensor,
    epoch10: torch.Tensor,
    af: torch.Tensor,
    gray_min: float = 0.60,
    pale_od: float = 0.30,
    radius: int = 32,
    dilate: int = 4,
    support: float = 0.005,
) -> torch.Tensor:
    """Fill AF at or above gray 0.60 where epoch 10 is pale.

    That is where epoch 10 turns white. Brighter pixels below 0.60 stay on
    the reconstruction loss, which already has a pink epoch-10 target there.
    The borrowed color is nearby epoch-10 tissue with eosin at least 0.45,
    eosin at least hematoxylin, and optical density at least 0.55. A thin
    neighborhood uses that color over the field. The hole widens by a few
    pixels and stops at darker AF and at epoch-10 nuclei.
    """
    gray = _af_gray(af)
    reference = hed_ruifrok(epoch10).detach()
    predicted = hed_ruifrok(fake)
    hematoxylin = reference[:, 0:1]
    eosin = reference[:, 1:2]
    optical = hematoxylin + eosin
    source = ((eosin >= 0.45) & (eosin >= hematoxylin) & (optical >= 0.55)).to(dtype=reference.dtype)
    local, density = _window_mean(reference, source, int(radius))
    field_sum = (reference * source).sum(dim=(2, 3))
    field_count = source.sum(dim=(2, 3)).clamp_min(1.0)
    field = (field_sum / field_count).view(reference.shape[0], 2, 1, 1)
    has_source = (source.sum(dim=(2, 3)) >= 1.0).to(dtype=reference.dtype).view(-1, 1, 1, 1)
    target = torch.where(density >= float(support), local, field)
    nucleus = (hematoxylin >= 0.40) & (hematoxylin >= eosin)
    hole = (gray >= float(gray_min)) & (optical < float(pale_od))
    mask = _dilate(hole.to(dtype=reference.dtype), int(dilate))
    shoulder = max(float(gray_min) - 0.15, 0.0)
    mask = mask * (gray >= shoulder).to(dtype=mask.dtype) * (~nucleus).to(dtype=mask.dtype) * has_source
    gap = torch.relu(target - predicted).mean(dim=1, keepdim=True)
    # These pixels are about 0.01% of a field, so the floor is 0.1%.
    return _floor_mean(gap, mask, floor=0.001)


def bright_af_keep(
    fake: torch.Tensor,
    af: torch.Tensor,
    gray_min: float = 0.60,
    floor_od: float = 0.35,
) -> torch.Tensor:
    """Keep some stain on AF at or above gray 0.60.

    The penalty is the missing hematoxylin-plus-eosin up to ``floor_od`` and
    then stops. A brighter pixel is not asked to be darker, and the color is
    not specified. Epoch 10 is often white here, so reconstruction does not
    supply this.
    """
    gray = _af_gray(af)
    optical = hed_ruifrok(fake).sum(dim=1, keepdim=True)
    mask = (gray >= float(gray_min)).to(dtype=optical.dtype)
    gap = torch.relu(float(floor_od) - optical)
    return _floor_mean(gap, mask, floor=0.01)




def recon_epoch_settings(
    epoch_index: int,
    sharp_epochs: int,
    sharp_weight: float,
    sharp_sigma: float,
    late_weight: float,
    late_sigmas: list[float],
) -> tuple[float, float]:
    """Weight and blur for one 0-based epoch.

    Displayed epoch ``epoch_index + 1``. The first ``sharp_epochs`` displayed
    epochs use the sharp weight and sigma. Each later epoch takes the next
    late sigma, then holds the last one.
    """
    if int(epoch_index) < 0:
        raise ValueError(f"epoch_index must be >= 0, got {epoch_index}")
    if int(sharp_epochs) < 1:
        raise ValueError(f"sharp_epochs must be >= 1, got {sharp_epochs}")
    if not late_sigmas:
        raise ValueError("late sigma schedule is empty")
    if epoch_index < int(sharp_epochs):
        return float(sharp_weight), float(sharp_sigma)
    offset = int(epoch_index) - int(sharp_epochs)
    sigma = late_sigmas[min(offset, len(late_sigmas) - 1)]
    return float(late_weight), float(sigma)


def recon_loss(
    fake: torch.Tensor,
    teacher: torch.Tensor,
    pseudo: torch.Tensor,
    prob: torch.Tensor,
    sigma: float,
    bright: torch.Tensor | None = None,
    bright_gain: float = 1.0,
    field_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match blurred hematoxylin and eosin, not RGB pixels.

    Tissue follows the teacher and glass follows the pseudo label, using
    ``prob``. Both sides are blurred, so detail finer than ``sigma`` is not
    a target and is not an error. The two stain channels stay separate, so
    exchanging hematoxylin and eosin is an error. Dim pixels keep weight 1.
    Bright AF pixels are multiplied by ``bright_gain``.
    """
    blurred_fake = gaussian_blur(hed_he(fake, detach=False), sigma)
    blurred_teacher = gaussian_blur(hed_he(teacher, detach=True), sigma)
    blurred_pseudo = gaussian_blur(hed_he(pseudo, detach=True), sigma)
    gate = prob.detach()
    target = gate * blurred_teacher + (1.0 - gate) * blurred_pseudo
    error = (blurred_fake - target).abs()
    gain = float(bright_gain)
    if bright is not None and gain > 1.0:
        error = error * (1.0 + (gain - 1.0) * bright.detach())
    per_field = error.flatten(1).mean(dim=1)
    if field_weight is None:
        return per_field.mean()
    weight = field_weight.detach().float().reshape(-1)
    if weight.shape[0] != per_field.shape[0]:
        raise ValueError(f"field weight {tuple(weight.shape)} for {per_field.shape[0]} fields")
    if float(weight.sum()) <= 0.0:
        return per_field.sum() * 0.0
    return (per_field * weight).sum() / weight.sum()


def stain_polarity(image: torch.Tensor, detach: bool = False, eps: float = 1e-3) -> torch.Tensor:
    """Rough hematoxylin-versus-eosin direction in [-1, 1].

    Positive is hematoxylin, negative is eosin. Dividing by the sum removes
    stain strength, so a pale field and a dark field with the same mix agree.
    """
    channels = hed_he(image, detach=detach)
    total = channels[:, 0:1] + channels[:, 1:2]
    return (channels[:, 0:1] - channels[:, 1:2]) / (total + float(eps))


def structure_loss(
    fake: torch.Tensor,
    reference: torch.Tensor,
    sigma: float,
    polarity_sigma: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tissue occupancy and H/E polarity against an aligned reference.

    The reference is the epoch-10 image. Occupancy is where stain is, not
    which stain or how dark. Polarity is the blurred H-versus-E direction,
    and only reference tissue counts, so glass and stain style do not.
    ``polarity_sigma`` defaults to ``sigma``.
    """
    direction_sigma = float(sigma if polarity_sigma is None else polarity_sigma)
    predicted_tissue = tissue_score(fake, detach=False)
    reference_tissue = tissue_score(reference, detach=True)
    occupancy = (gaussian_blur(predicted_tissue, sigma) - gaussian_blur(reference_tissue, sigma)).abs().mean()
    predicted_polarity = gaussian_blur(stain_polarity(fake, detach=False), direction_sigma)
    reference_polarity = gaussian_blur(stain_polarity(reference, detach=True), direction_sigma)
    # The soft score has a glass tail. Only clearly stained reference pixels vote.
    weight = (gaussian_blur(reference_tissue, direction_sigma) >= 0.5).float()
    difference = (predicted_polarity - reference_polarity).abs()
    denominator = weight.sum()
    if float(denominator) <= 0.0:
        polarity = predicted_polarity.sum() * 0.0
    else:
        polarity = (difference * weight).sum() / denominator
    return occupancy, polarity


def occupancy_loss(
    fake: torch.Tensor,
    prob: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Match blurred tissue probability. Stain color is not a target.

    ``prob`` is the AF presence times the stronger of the pseudo-label and
    teacher tissue scores, on the same field. Both maps are blurred, so the
    loss sees occupancy averaged over about ``sigma`` pixels. A stain wiggle
    inside that radius is not an error. A missing region still counts for
    its area. Once a pixel is clearly tissue or clearly glass the score
    saturates, and a darker stain does not reduce the loss.
    """
    predicted = tissue_score(fake, detach=False)
    error = gaussian_blur(predicted, sigma) - gaussian_blur(prob.detach().float(), sigma)
    return error.abs().mean()


def detail_loss(
    fake: torch.Tensor,
    teacher: torch.Tensor,
    prob: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Match fine residual detail to the teacher, averaged on the tissue map."""
    residual_fake = fake.float() - gaussian_blur(fake.float(), sigma)
    residual_teacher = teacher.detach().float() - gaussian_blur(teacher.detach().float(), sigma)
    diff = (residual_fake - residual_teacher).abs()
    weight = prob.detach()
    numer = (diff * weight).sum()
    denom = weight.sum() * diff.shape[1]
    if float(denom) == 0.0:
        return fake.new_zeros(())
    return numer / denom


def hed_tissue_fraction(image: torch.Tensor, threshold: float = 0.15) -> torch.Tensor:
    """Fraction of pixels with hematoxylin plus eosin above ``threshold``.

    The threshold is absolute hematoxylin plus eosin optical density, so a
    flat gray field scores near zero. A within-image percentile cannot.
    """
    return hed_tissue_mask(image, threshold).flatten(1).float().mean(dim=1)


def masked_l1(fake: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean absolute error on pixels where ``mask`` is true.

    Glass pixels are left out, so an empty mask contributes zero instead of
    training the generator to copy a target there.
    """
    if fake.shape != target.shape:
        raise ValueError(f"fake {tuple(fake.shape)} != target {tuple(target.shape)}")
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[0] != fake.shape[0] or mask.shape[1] != 1 or mask.shape[-2:] != fake.shape[-2:]:
        raise ValueError(f"mask {tuple(mask.shape)} does not match fake {tuple(fake.shape)}")
    diff = (fake.float() - target.detach().float()).abs()
    weight = mask.detach().float()
    numer = (diff * weight).sum()
    denom = weight.sum() * diff.shape[1]
    if float(denom) == 0.0:
        return fake.new_zeros(())
    return numer / denom


def content_gate(pseudo: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """Scalar in [0, 1] per image. 1 when the H&E tissue amounts match.

    This multiplies the whole FDL term. It is not a spatial mask: the real
    H&E crop is not registered to the pseudo label.
    """
    if pseudo.shape != real.shape:
        raise ValueError(f"pseudo {tuple(pseudo.shape)} != real {tuple(real.shape)}")
    gap = (hed_tissue_fraction(pseudo) - hed_tissue_fraction(real)).abs()
    return (1.0 - gap).clamp(0.0, 1.0)


def sliced_wasserstein(left: torch.Tensor, right: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    """1D Wasserstein distance averaged over random convolutional projections."""
    projected_left = torch.sort(F.conv2d(left, projection).flatten(2), dim=-1).values
    projected_right = torch.sort(F.conv2d(right, projection).flatten(2), dim=-1).values
    return (projected_left - projected_right).abs().mean(dim=(1, 2))


class AmplitudeFDL(nn.Module):
    """Sliced Wasserstein on VGG Fourier magnitude and phase.

    Magnitude is always fully weighted. Phase is sensitive to the shift
    between the generated field and real H&E, so ``phase_gate`` scales only
    that weight, per image.
    """

    CHANNELS = (64, 128, 256, 512, 512)

    def __init__(
        self,
        num_proj: int = 256,
        patch_size: int = 5,
        input_size: int = 512,
        phase_weight: float = 0.1,
        seed: int = 0,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.phase_weight = float(phase_weight)
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        for index, channels in enumerate(self.CHANNELS):
            rand = torch.randn(
                int(num_proj), int(channels), int(patch_size), int(patch_size), generator=generator,
            )
            rand = rand / rand.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-8)
            self.register_buffer(f"proj_{index}", rand)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.vgg: nn.ModuleList | None = None

    def _features(self, x01: torch.Tensor) -> list[torch.Tensor]:
        if self.vgg is None:
            from torchvision.models import VGG19_Weights, vgg19
            backbone = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
            cuts = (4, 9, 18, 27, 36)
            stages = []
            start = 0
            for cut in cuts:
                stages.append(nn.Sequential(*[backbone[i] for i in range(start, cut)]))
                start = cut
            self.vgg = nn.ModuleList(stages).to(device=x01.device)
            self.vgg.eval()
            for parameter in self.vgg.parameters():
                parameter.requires_grad = False
        if next(self.vgg.parameters()).device != x01.device:
            self.vgg.to(device=x01.device)
        h = (x01 - self.mean.to(dtype=x01.dtype)) / self.std.to(dtype=x01.dtype)
        outputs = []
        for stage in self.vgg:
            h = stage(h)
            outputs.append(h)
        return outputs

    def forward(
        self,
        fake: torch.Tensor,
        real: torch.Tensor,
        phase_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fake01 = ((fake.float() + 1.0) * 0.5).clamp(0.0, 1.0)
        real01 = ((real.detach().float() + 1.0) * 0.5).clamp(0.0, 1.0)
        if self.input_size > 0:
            fake01 = F.interpolate(fake01, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
            real01 = F.interpolate(real01, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        total = fake01.new_zeros(fake01.shape[0])
        for index, (left, right) in enumerate(zip(self._features(fake01), self._features(real01))):
            freq_left = torch.fft.fftn(left, dim=(-2, -1))
            freq_right = torch.fft.fftn(right, dim=(-2, -1))
            projection = getattr(self, f"proj_{index}").to(dtype=freq_left.real.dtype)
            total = total + sliced_wasserstein(freq_left.abs(), freq_right.abs(), projection)
            if self.phase_weight > 0:
                phase = sliced_wasserstein(freq_left.angle(), freq_right.angle(), projection)
                weight = phase.new_full(phase.shape, self.phase_weight)
                if phase_gate is not None:
                    gate = phase_gate.detach().float().reshape(-1)
                    if gate.shape[0] != phase.shape[0]:
                        raise ValueError(f"phase gate {tuple(gate.shape)} for {phase.shape[0]} images")
                    weight = weight * gate
                total = total + weight * phase
        return total
