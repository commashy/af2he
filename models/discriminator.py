"""Three-scale PatchGAN. LSGAN, spectral norm, antialiased stride-2."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def _blur_kernel() -> torch.Tensor:
    base = torch.tensor([1.0, 2.0, 1.0])
    kernel = base[:, None] * base[None, :]
    return kernel / kernel.sum()


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.pad = nn.ReflectionPad2d(1)
        filt = _blur_kernel()[None, None].repeat(int(channels), 1, 1, 1)
        self.register_buffer("filt", filt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.filt.to(dtype=x.dtype)
        return F.conv2d(self.pad(x), weight, stride=2, groups=x.shape[1])


class SlideFiLM(nn.Module):
    """Channel scale and shift from a slide vector, after instance norm.

    ``alpha`` starts at 0, so the normalized features are unchanged until the
    gate opens. The linear map is small and random; its gradient arrives once
    ``alpha`` moves, which is the same schedule as a zero residual.
    """

    def __init__(self, embed_dim: int, channels: int):
        super().__init__()
        self.to_gamma_beta = nn.Linear(int(embed_dim), 2 * int(channels))
        nn.init.normal_(self.to_gamma_beta.weight, std=0.02)
        nn.init.zeros_(self.to_gamma_beta.bias)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_gamma_beta(embed.float()).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        out = x.float() * (1.0 + self.alpha * gamma) + self.alpha * beta
        return out.to(dtype=x.dtype)


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc: int = 3, ndf: int = 64, n_layers: int = 3, num_classes: int = 0):
        super().__init__()
        layers: list[nn.Module] = [
            spectral_norm(nn.Conv2d(input_nc, ndf, 4, 1, 1)),
            nn.LeakyReLU(0.2, True),
            Downsample(ndf),
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                spectral_norm(nn.Conv2d(ndf * prev, ndf * nf_mult, 4, 1, 1)),
                nn.InstanceNorm2d(ndf * nf_mult, affine=False, track_running_stats=False),
                nn.LeakyReLU(0.2, True),
                Downsample(ndf * nf_mult),
            ]
        prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            spectral_norm(nn.Conv2d(ndf * prev, ndf * nf_mult, 4, 1, 1)),
            nn.InstanceNorm2d(ndf * nf_mult, affine=False, track_running_stats=False),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1)),
        ]
        self.model = nn.Sequential(*layers)
        self.cls = None
        if int(num_classes) > 0:
            self.cls = spectral_norm(nn.Conv2d(self.model[-1].in_channels, int(num_classes), 4, 1, 1))

    def forward(
        self,
        x: torch.Tensor,
        films: nn.ModuleList | None = None,
        embed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if films is None and self.cls is None:
            return self.model(x), None
        if films is not None and embed is None:
            raise RuntimeError("slide FiLM requires an embedding")
        hidden = x
        film_index = 0
        layers = list(self.model)
        for layer in layers[:-1]:
            hidden = layer(hidden)
            if films is not None and isinstance(layer, nn.InstanceNorm2d):
                if film_index >= len(films):
                    raise RuntimeError("more instance norms than slide FiLM layers")
                hidden = films[film_index](hidden, embed)
                film_index += 1
        if films is not None and film_index != len(films):
            raise RuntimeError(f"slide FiLM used {film_index} of {len(films)} layers")
        classes = self.cls(hidden) if self.cls is not None else None
        return layers[-1](hidden), classes


def _is_slide_film_key(name: str) -> bool:
    return name.startswith("slide_embed.") or name.startswith("slide_films.")


def _is_slide_aux_key(name: str) -> bool:
    return _is_slide_film_key(name) or ".cls." in name


class FieldDiscriminator(nn.Module):
    """One score for a 1024 field, projected onto the frozen slide vector.

    No instance normalization, so the field's overall color and cellularity
    remain visible. The slide vector is 512 numbers from the collage encoder.
    """

    def __init__(self, slide_dim: int = 512, ndf: int = 64):
        super().__init__()
        channels = [ndf, ndf * 2, ndf * 4, ndf * 8]
        layers: list[nn.Module] = []
        incoming = 3
        for outgoing in channels:
            layers += [
                spectral_norm(nn.Conv2d(incoming, outgoing, kernel_size=4, stride=2, padding=1)),
                nn.LeakyReLU(0.2, True),
            ]
            incoming = outgoing
        self.features = nn.Sequential(*layers)
        width = channels[-1]
        self.score = spectral_norm(nn.Linear(width, 1))
        self.slide = spectral_norm(nn.Linear(int(slide_dim), width, bias=False))

    def forward(self, image: torch.Tensor, slide: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[-2:] != (1024, 1024):
            raise ValueError(f"field must be (N, 3, 1024, 1024), got {tuple(image.shape)}")
        if slide.ndim != 2 or slide.shape[0] != image.shape[0] or slide.shape[1] != self.slide.in_features:
            raise ValueError(
                f"slide vector must be ({image.shape[0]}, {self.slide.in_features}), got {tuple(slide.shape)}"
            )
        hidden = self.features(image).mean(dim=(2, 3))
        realism = self.score(hidden).squeeze(1)
        conditioned = self.slide(slide.to(dtype=hidden.dtype))
        return realism + (hidden * conditioned).sum(dim=1)


class MultiScaleDiscriminator(nn.Module):
    """One PatchGAN per scale. Later scales see average-pooled images.

    ``slide_count > 0`` adds a learned id to every instance norm on the
    coarsest scale only. Finer scales stay unconditional. The FiLM gate
    starts closed, so a discriminator loaded from an unconditional
    checkpoint produces the same logits.
    """

    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        num_scales: int = 3,
        slide_count: int = 0,
        embed_dim: int = 128,
        slide_classes: int = 0,
        class_scales: str = "all",
        film_scales: list[int] | None = None,
    ):
        super().__init__()
        if int(num_scales) < 1:
            raise ValueError(f"num_scales must be >= 1, got {num_scales}")
        self.slide_classes = int(slide_classes)
        if self.slide_classes < 0:
            raise ValueError(f"slide_classes must be >= 0, got {self.slide_classes}")
        if class_scales not in ("all", "coarsest"):
            raise ValueError(f"class_scales must be all or coarsest, got {class_scales}")
        self.class_scales = class_scales
        last = int(num_scales) - 1
        self.discriminators = nn.ModuleList(
            [
                NLayerDiscriminator(
                    input_nc, ndf, n_layers,
                    num_classes=self.slide_classes if class_scales == "all" or index == last else 0,
                )
                for index in range(int(num_scales))
            ]
        )
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
        self.slide_embed = None
        self.slide_films = nn.ModuleDict()
        self.n_slides = int(slide_count)
        if self.n_slides < 0:
            raise ValueError(f"slide_count must be >= 0, got {self.n_slides}")
        if self.n_slides == 0:
            return
        if self.n_slides == 1 or len(self.discriminators) < 2:
            raise ValueError("slide conditioning needs a background id and at least two scales")
        self.slide_embed = nn.Embedding(self.n_slides, int(embed_dim))
        nn.init.normal_(self.slide_embed.weight, std=0.02)
        if film_scales is None:
            chosen = [len(self.discriminators) - 1]
        else:
            chosen = [int(scale) for scale in film_scales]
            if not chosen:
                raise ValueError("film_scales is empty")
        for scale in chosen:
            if scale < 0 or scale >= len(self.discriminators):
                raise ValueError(f"film scale {scale} is outside 0..{len(self.discriminators) - 1}")
            norms = [layer for layer in self.discriminators[scale].model if isinstance(layer, nn.InstanceNorm2d)]
            if not norms:
                raise RuntimeError(f"scale {scale} has no instance norm to condition")
            self.slide_films[str(scale)] = nn.ModuleList(
                [SlideFiLM(int(embed_dim), int(layer.num_features)) for layer in norms]
            )

    def film_alpha(self) -> float:
        alphas = [float(film.alpha.detach().abs()) for films in self.slide_films.values() for film in films]
        if not alphas:
            return 0.0
        return sum(alphas) / len(alphas)

    def forward(self, x: torch.Tensor, slide_idx: torch.Tensor | None = None) -> list[torch.Tensor]:
        embed = None
        if self.slide_embed is not None:
            if slide_idx is None:
                raise RuntimeError("slide-conditioned discriminator requires slide_idx")
            if slide_idx.shape != (x.shape[0],):
                raise ValueError(f"slide_idx must be ({x.shape[0]},), got {tuple(slide_idx.shape)}")
            if slide_idx.dtype != torch.long:
                slide_idx = slide_idx.long()
            if bool((slide_idx < 0).any() or (slide_idx >= self.n_slides).any()):
                raise ValueError(f"slide_idx out of range [0, {self.n_slides})")
            embed = self.slide_embed(slide_idx)
        elif slide_idx is not None:
            raise RuntimeError("slide_idx passed to an unconditional discriminator")
        outputs = []
        class_maps = []
        current = x
        for index, disc in enumerate(self.discriminators):
            films = self.slide_films[str(index)] if str(index) in self.slide_films else None
            logit, classes = disc(current, films, embed)
            outputs.append(logit)
            if classes is not None:
                class_maps.append(classes)
            if index < len(self.discriminators) - 1:
                current = self.pool(current)
        if class_maps:
            return outputs, class_maps
        return outputs


def backbone_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [param for name, param in module.named_parameters() if not _is_slide_aux_key(name)]


def film_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [param for name, param in module.named_parameters() if _is_slide_film_key(name)]


def class_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [param for name, param in module.named_parameters() if ".cls." in name]


def field_subtype_loss(
    class_maps: list[torch.Tensor],
    target: torch.Tensor,
    include: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """One label per field, from the coarsest class map. Skipped fields stay connected."""
    logits = class_maps[-1].float().mean(dim=(2, 3))
    loss = logits.sum() * 0.0
    if include.dtype != torch.bool:
        include = include.bool()
    if bool(include.any()):
        loss = loss + F.cross_entropy(logits[include], target[include], weight=weight)
    return loss


def slide_classification_loss(
    class_maps: list[torch.Tensor],
    slide_idx: torch.Tensor,
    mask: torch.Tensor,
    cutoff: float = 0.15,
) -> torch.Tensor:
    """Cross-entropy of each scale, only on tissue positions of ``mask``.

    ``mask`` is a field-resolution map in ``[0, 1]``. A spatial cell whose
    pooled tissue is below ``cutoff`` is glass and contributes nothing. An
    empty mask contributes zero and stays connected to the logits.
    """
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    total = class_maps[0].float().sum() * 0.0
    used = 0
    for logits in class_maps:
        pooled = F.adaptive_avg_pool2d(mask.float(), logits.shape[-2:])
        keep = pooled[:, 0] >= float(cutoff)
        connected = logits.float().sum() * 0.0
        if bool(keep.any()):
            target = slide_idx.view(-1, 1, 1).expand_as(keep)
            connected = connected + F.cross_entropy(logits.float().permute(0, 2, 3, 1)[keep], target[keep])
        total = total + connected
        used += 1
    return total / used


def load_discriminator_state(module: MultiScaleDiscriminator, state: dict) -> bool:
    """Load D weights. An unconditional checkpoint leaves the new FiLM at init.

    Returns True when the slide layers were absent from ``state``.
    """
    current = module.state_dict()
    missing = sorted(set(current) - set(state))
    unexpected = sorted(set(state) - set(current))
    if unexpected:
        raise RuntimeError(f"checkpoint discriminator has unexpected keys: {unexpected[:8]}")
    if not missing:
        module.load_state_dict(state)
        return False
    foreign = [key for key in missing if not _is_slide_aux_key(key)]
    if foreign:
        raise RuntimeError(f"checkpoint discriminator is missing backbone keys: {foreign[:8]}")
    module.load_state_dict({key: value for key, value in state.items() if key in current}, strict=False)
    return True
