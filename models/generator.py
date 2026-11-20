"""AF generator.

Native 512 tile in, tanh output in [-1, 1]. No input stretch.
No local field and no text token. The collage CLS applies a zero-init FiLM
after per-pixel channel RMS. The 16 collage tokens are cross-attention memory.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import NUM_REGISTERS, NUM_STYLE_TOKENS, BottleneckBlock

TILE = 512
COLLAGE = 1024
WIDTHS = (96, 192, 384, 576)
# stem, then the four encoder skips, before each downsample.
FEATURE_CHANNELS = (96, 96, 192, 384, 576)
BOTTLENECK_DIM = 384
BOTTLENECK_HEADS = 6
BOTTLENECK_MLP = 1536
BOTTLENECK_BLOCKS = 8
COLLAGE_DIM = 256
COLLAGE_HEADS = 4
COLLAGE_BLOCKS = 4
COLLAGE_MLP = 1024
COLLAGE_TOKENS = 16
COLLAGE_STRIDE = 256
LEAKY = 0.01


class PixelRMS(nn.Module):
    """Per-pixel channel RMS. Does not mix pixels or remove a spatial offset."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, int(channels), 1, 1))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().square().mean(dim=1, keepdim=True) + self.eps)
        return (x.float() * scale).to(dtype=x.dtype) * self.weight.to(dtype=x.dtype)


class FiLM(nn.Module):
    """Global channel affine from the collage CLS. Zero weights stay at identity."""

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.linear = nn.Linear(int(cond_dim), int(channels) * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.linear(cond).chunk(2, dim=1)
        gamma = gamma.to(dtype=x.dtype)[:, :, None, None]
        beta = beta.to(dtype=x.dtype)[:, :, None, None]
        return x * (1.0 + gamma) + beta


class EncLevel(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode="reflect")
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode="reflect")
        self.rms = PixelRMS(out_ch)
        self.down = nn.Conv2d(out_ch, out_ch, kernel_size=2, stride=2)
        self.act = nn.LeakyReLU(LEAKY)
        self.film = FiLM(BOTTLENECK_DIM, out_ch)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h = self.rms(h)
        h = self.film(h, cond)
        return self.down(h), h


class StyleScale(nn.Module):
    def __init__(self, mod_dim: int, channels: int):
        super().__init__()
        self.affine = nn.Linear(mod_dim, channels)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, mod: torch.Tensor) -> torch.Tensor:
        return 1.0 + self.alpha.to(dtype=mod.dtype) * self.affine(mod)


class ModulatedConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, padding: int, padding_mode: str):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel, padding=padding, padding_mode=padding_mode,
        )

    def forward(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        y = self.conv(x * scale[:, :, None, None])
        weight = self.conv.weight.float()
        weight_sq = weight.square().sum(dim=(2, 3))
        sigma_sq = (scale.float().square().unsqueeze(1) * weight_sq.unsqueeze(0)).sum(dim=2)
        demod = torch.rsqrt(sigma_sq + 1e-6).to(dtype=y.dtype)
        return y * demod[:, :, None, None]


class Up(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, padding_mode="reflect")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


class DecBlock(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, mod_dim: int):
        super().__init__()
        self.scale1 = StyleScale(mod_dim, in_ch)
        self.conv1 = ModulatedConv2d(in_ch, mid_ch, 3, padding=1, padding_mode="reflect")
        self.scale2 = StyleScale(mod_dim, mid_ch)
        self.conv2 = ModulatedConv2d(mid_ch, out_ch, 3, padding=1, padding_mode="reflect")
        self.act = nn.LeakyReLU(LEAKY)

    def forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x, self.scale1(mod)))
        x = self.act(self.conv2(x, self.scale2(mod)))
        return x


class CollageBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"collage dim {dim} must be divisible by heads {heads}")
        self.heads = int(heads)
        self.head_dim = dim // heads
        self.norm_sa = _RMS(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.ls_sa = _LayerScale(dim)
        self.norm_ffn = _RMS(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )
        self.ls_ffn = _LayerScale(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm_sa(x)
        batch, length, dim = h.shape
        qkv = self.qkv(h).reshape(batch, length, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], dropout_p=0.0)
        attn = attn.transpose(1, 2).reshape(batch, length, dim)
        x = x + self.ls_sa(self.proj(attn))
        x = x + self.ls_ffn(self.ffn(self.norm_ffn(x)))
        return x


class _RMS(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(dtype=x.dtype) * self.weight.to(dtype=x.dtype)


class _LayerScale(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), 1e-4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma.to(dtype=x.dtype) * x


def collage_grid(size: int) -> int:
    """Cells along one side. The stem stride is 256, so 1024 -> 4 and 2048 -> 8."""
    if int(size) < COLLAGE_STRIDE or int(size) % COLLAGE_STRIDE != 0:
        raise ValueError(f"collage size {size} is not a multiple of {COLLAGE_STRIDE}")
    return int(size) // COLLAGE_STRIDE


def resize_collage_positions(weight: torch.Tensor, tokens: int) -> torch.Tensor:
    """Bilinear resize of a square collage position table. The class token stays."""
    if weight.ndim != 3 or weight.shape[0] != 1:
        raise ValueError(f"collage positions must be (1, tokens, dim), got {tuple(weight.shape)}")
    cls, spatial = weight[:, :1], weight[:, 1:]
    old = int(spatial.shape[1] ** 0.5)
    new = int(int(tokens) ** 0.5)
    if old * old != spatial.shape[1] or new * new != int(tokens):
        raise ValueError(f"cannot resize {spatial.shape[1]} positions to {tokens}")
    if old == new:
        return weight
    grid = spatial.reshape(1, old, old, spatial.shape[-1]).permute(0, 3, 1, 2)
    grid = F.interpolate(grid.float(), size=(new, new), mode="bilinear", align_corners=False)
    spatial = grid.permute(0, 2, 3, 1).reshape(1, int(tokens), spatial.shape[-1]).to(dtype=weight.dtype)
    return torch.cat((cls, spatial), dim=1)


class CollageEncoder(nn.Module):
    """Collage image -> one token per 256-pixel cell. Strided convs, then a small transformer."""

    def __init__(self, size: int = COLLAGE):
        super().__init__()
        self.size = int(size)
        self.grid = collage_grid(self.size)
        self.tokens = self.grid * self.grid
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=8),
            nn.LeakyReLU(LEAKY),
            nn.Conv2d(32, 64, kernel_size=4, stride=4),
            nn.LeakyReLU(LEAKY),
            nn.Conv2d(64, 128, kernel_size=4, stride=4),
            nn.LeakyReLU(LEAKY),
            nn.Conv2d(128, COLLAGE_DIM, kernel_size=2, stride=2),
            nn.LeakyReLU(LEAKY),
        )
        self.cls = nn.Parameter(torch.empty(1, 1, COLLAGE_DIM))
        self.pos = nn.Parameter(torch.empty(1, 1 + self.tokens, COLLAGE_DIM))
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            [CollageBlock(COLLAGE_DIM, COLLAGE_HEADS, COLLAGE_MLP) for _ in range(COLLAGE_BLOCKS)]
        )
        self.proj = nn.Linear(COLLAGE_DIM, BOTTLENECK_DIM)

    def forward(self, collage: torch.Tensor) -> torch.Tensor:
        if collage.shape[-2:] != (self.size, self.size):
            raise ValueError(f"collage must be {self.size}x{self.size}, got {tuple(collage.shape)}")
        feat = self.cnn(collage)
        if feat.shape[-2:] != (self.grid, self.grid):
            raise RuntimeError(f"collage map must be {self.grid}x{self.grid}, got {tuple(feat.shape)}")
        patches = feat.flatten(2).transpose(1, 2)
        # Multiply by 1 so the class-token gradient keeps the parameter's own
        # strides. A plain repeat of one row leaves a view into the token
        # sequence, and DDP then recopies that view into its bucket.
        cls = (self.cls * 1).expand(patches.shape[0], -1, -1)
        tokens = torch.cat((cls, patches), dim=1)
        # repeat, not expand: expand leaves a non-contiguous grad and DDP recopies it.
        tokens = tokens + self.pos.repeat(tokens.shape[0], 1, 1).to(dtype=tokens.dtype)
        for block in self.blocks:
            tokens = block(tokens)
        projected = self.proj(tokens)
        self.last_pooled = projected[:, 0].detach()
        return projected[:, 1:], projected[:, 0]


class LocalEncoder(CollageEncoder):
    """1024 neighborhood, RGB plus a coverage mask, one token per 256 input pixels."""

    def __init__(self):
        super().__init__(COLLAGE)
        self.expand_mask_channel()

    def expand_mask_channel(self) -> None:
        old = self.cnn[0]
        if not isinstance(old, nn.Conv2d) or old.in_channels not in (3, 4):
            raise TypeError(f"local stem is {type(old)} with {getattr(old, 'in_channels', None)} channels")
        if old.in_channels == 4:
            return
        new = nn.Conv2d(4, old.out_channels, kernel_size=old.kernel_size, stride=old.stride)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :3].copy_(old.weight)
            if old.bias is not None and new.bias is not None:
                new.bias.copy_(old.bias)
        self.cnn[0] = new

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 4 or image.shape[-2:] != (self.size, self.size):
            raise ValueError(f"neighborhood must be (N, 4, {self.size}, {self.size}), got {tuple(image.shape)}")
        tokens, _cls = super().forward(image)
        return tokens


class Bottleneck(nn.Module):
    def __init__(self, local: bool = False):
        super().__init__()
        self.in_proj = nn.Linear(WIDTHS[-1], BOTTLENECK_DIM)
        self.out_proj = nn.Linear(BOTTLENECK_DIM, WIDTHS[-1])
        self.style = nn.Parameter(torch.empty(1, NUM_STYLE_TOKENS, BOTTLENECK_DIM))
        self.registers = nn.Parameter(torch.empty(1, NUM_REGISTERS, BOTTLENECK_DIM))
        nn.init.normal_(self.style, std=0.02)
        nn.init.normal_(self.registers, std=0.02)
        self.context_type = nn.Parameter(torch.empty(1, 1, BOTTLENECK_DIM))
        nn.init.normal_(self.context_type, std=0.02)
        self.local_type = None
        if local:
            self.local_type = nn.Parameter(torch.empty(1, 1, BOTTLENECK_DIM))
            nn.init.normal_(self.local_type, std=0.02)
        self.blocks = nn.ModuleList(
            [
                BottleneckBlock(BOTTLENECK_DIM, BOTTLENECK_HEADS, BOTTLENECK_MLP)
                for _ in range(BOTTLENECK_BLOCKS)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        local_memory: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, height, width = x.shape
        if channels != WIDTHS[-1] or height != 32 or width != 32:
            raise ValueError(f"bottleneck map must be {WIDTHS[-1]}x32x32, got {tuple(x.shape)}")
        if memory.ndim != 3 or memory.shape[0] != batch or memory.shape[-1] != BOTTLENECK_DIM:
            raise ValueError(
                f"collage memory must be ({batch}, tokens, {BOTTLENECK_DIM}), got {tuple(memory.shape)}"
            )
        spatial = self.in_proj(x.flatten(2).transpose(1, 2))
        # repeat copies the parameter before it is concatenated. expand().cat()
        # leaves the style grad as a slice of the 5-token block (stride 5*384),
        # and DDP recopies that grad on every backward.
        style = self.style.repeat(batch, 1, 1).to(dtype=spatial.dtype)
        registers = self.registers.repeat(batch, 1, 1).to(dtype=spatial.dtype)
        tokens = torch.cat((spatial, style, registers), dim=1)
        context = self.context_type.repeat(batch, memory.shape[1], 1).to(dtype=memory.dtype)
        memory = memory + context
        if local_memory is not None:
            if self.local_type is None:
                raise RuntimeError("local memory requires Generator(local=True)")
            if local_memory.shape[0] != batch or local_memory.shape[-1] != BOTTLENECK_DIM:
                raise ValueError(
                    f"local memory must be ({batch}, tokens, {BOTTLENECK_DIM}), got {tuple(local_memory.shape)}"
                )
            local_context = self.local_type.repeat(batch, local_memory.shape[1], 1).to(dtype=local_memory.dtype)
            memory = torch.cat((memory, local_memory + local_context), dim=1)
            if memory_mask is not None:
                extra = torch.ones(
                    batch, local_memory.shape[1], dtype=torch.bool, device=memory_mask.device,
                )
                memory_mask = torch.cat((memory_mask, extra), dim=1)
        for block in self.blocks:
            tokens = block(tokens, memory, height, width, memory_mask)
        spatial_len = height * width
        out = self.out_proj(tokens[:, :spatial_len])
        out = out.transpose(1, 2).reshape(batch, WIDTHS[-1], height, width)
        mod = tokens[:, spatial_len]
        return out, mod


class Generator(nn.Module):
    def __init__(self, collage_size: int = COLLAGE, local: bool = False):
        super().__init__()
        self.collage_size = int(collage_size)
        self.stem = nn.Conv2d(3, WIDTHS[0], 3, padding=1, padding_mode="reflect")
        self.stem_act = nn.LeakyReLU(LEAKY)
        self.stem_rms = PixelRMS(WIDTHS[0])
        self.stem_film = FiLM(BOTTLENECK_DIM, WIDTHS[0])
        channels = (WIDTHS[0], *WIDTHS)
        self.levels = nn.ModuleList(
            [EncLevel(channels[i], channels[i + 1]) for i in range(4)]
        )
        self.collage_encoder = CollageEncoder(self.collage_size)
        self.local_encoder = LocalEncoder() if local else None
        self.bottleneck = Bottleneck(local=local)
        self.up3 = Up(576)
        self.dec3 = DecBlock(1152, 576, 384, BOTTLENECK_DIM)
        self.up2 = Up(384)
        self.dec2 = DecBlock(768, 384, 192, BOTTLENECK_DIM)
        self.up1 = Up(192)
        self.dec1 = DecBlock(384, 192, 96, BOTTLENECK_DIM)
        self.up0 = Up(96)
        self.dec0 = DecBlock(192, 96, 96, BOTTLENECK_DIM)
        self.head = nn.Conv2d(96, 3, kernel_size=1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0.0, 0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.modules():
            if isinstance(module, FiLM):
                nn.init.zeros_(module.linear.weight)
                nn.init.zeros_(module.linear.bias)
        if self.local_encoder is not None:
            with torch.no_grad():
                self.local_encoder.cnn[0].weight[:, 3:].zero_()

    def _local_memory(self, local: torch.Tensor, tiles: int) -> torch.Tensor:
        if self.local_encoder is None:
            raise RuntimeError("local neighborhood requires Generator(local=True)")
        if local.ndim != 4 or local.shape[1:] != (4, COLLAGE, COLLAGE):
            raise ValueError(f"neighborhood must be (N, 4, {COLLAGE}, {COLLAGE}), got {tuple(local.shape)}")
        if local.shape[0] == tiles:
            fields = local
            group = 1
        elif tiles % local.shape[0] == 0:
            fields = local
            group = tiles // local.shape[0]
        else:
            raise ValueError(f"{local.shape[0]} neighborhoods for {tiles} tiles")
        tokens = self.local_encoder(fields)
        if group == 1:
            return tokens
        return tokens.repeat_interleave(group, dim=0)

    def _check(
        self,
        x: torch.Tensor,
        collage: torch.Tensor,
        sizes: tuple[int, ...] = (TILE,),
    ) -> None:
        side = x.shape[-1] if x.ndim == 4 else 0
        if x.ndim != 4 or x.shape[1] != 3 or x.shape[-2] != side or side not in sizes:
            allowed = ", ".join(str(size) for size in sizes)
            raise ValueError(f"image must be (N, 3, {allowed}), got {tuple(x.shape)}")
        if collage.shape[0] != x.shape[0] or collage.shape[1:] != (3, self.collage_size, self.collage_size):
            raise ValueError(
                f"collage must be (N, 3, {self.collage_size}, {self.collage_size}), got {tuple(collage.shape)}"
            )

    def encode(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        h = self.stem_film(self.stem_rms(self.stem_act(self.stem(x))), cond)
        feats = [h]
        skips = []
        for level in self.levels:
            h, skip = level(h, cond)
            feats.append(skip)
            skips.append(skip)
        return h, skips, feats

    def encoder_features(
        self,
        x: torch.Tensor,
        collage: torch.Tensor,
        cells: torch.Tensor | None = None,
        black_index: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Encoder maps for contrastive query and key. Generate stays 512.

        A 1024 field is accepted here so PatchNCE can sample across the stitch.
        The key caller detaches AF. This path does not run the bottleneck.
        """
        self._check(x, collage, sizes=(TILE, TILE * 2))
        if cells is None:
            _, cond = self.collage_encoder(collage)
        else:
            _, cond, _binary, _subtype, _keep = self.collage_encoder(collage, cells, black_index)
        _, _, feats = self.encode(x, cond)
        return feats

    def decode(
        self,
        x: torch.Tensor,
        skips: list[torch.Tensor],
        mod: torch.Tensor,
        return_preskip: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Decode. Pre-skip maps are the upsampled tensors before each skip join.

        Keys 4, 3, 2, 1 match encoder feature indices on a 512 tile.
        """
        s0, s1, s2, s3 = skips
        preskip: dict[int, torch.Tensor] = {}
        up = self.up3(x)
        preskip[4] = up
        x = self.dec3(torch.cat((up, s3), dim=1), mod)
        up = self.up2(x)
        preskip[3] = up
        x = self.dec2(torch.cat((up, s2), dim=1), mod)
        up = self.up1(x)
        preskip[2] = up
        x = self.dec1(torch.cat((up, s1), dim=1), mod)
        up = self.up0(x)
        preskip[1] = up
        x = self.dec0(torch.cat((up, s0), dim=1), mod)
        image = torch.tanh(self.head(x))
        if return_preskip:
            return image, preskip
        return image

    def forward(
        self,
        x: torch.Tensor,
        collage: torch.Tensor,
        local: torch.Tensor | None = None,
        return_features: bool = False,
        return_cond: bool = False,
        cells: torch.Tensor | None = None,
        black_index: torch.Tensor | None = None,
        return_aux: bool = False,
        quadrant_group: int = 4,
        return_preskip: bool = False,
    ) -> torch.Tensor | tuple:
        self._check(x, collage)
        aux = None
        memory_mask = None
        if cells is None:
            memory, cond = self.collage_encoder(collage)
        else:
            if black_index is None:
                raise ValueError("the WSI collage encoder needs a black-cell index")
            memory, cond, binary, subtype, memory_mask = self.collage_encoder(
                collage, cells, black_index, group=quadrant_group,
            )
            aux = (binary, subtype)
        hidden, skips, feats = self.encode(x, cond)
        local_memory = None if local is None else self._local_memory(local, x.shape[0])
        hidden, mod = self.bottleneck(hidden, memory, local_memory, memory_mask)
        if return_preskip:
            image, preskip = self.decode(hidden, skips, mod, return_preskip=True)
        else:
            image = self.decode(hidden, skips, mod)
            preskip = None
        if return_aux:
            return image, aux
        if return_preskip and return_cond:
            return image, cond, feats, preskip
        if return_preskip:
            return image, feats, preskip
        if return_features and return_cond:
            return image, feats, cond
        if return_cond:
            return image, cond
        if return_features:
            return image, feats
        return image
