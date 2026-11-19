"""Bottleneck attention: axial 2D RoPE, unrotated style token, register RoPE."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SPATIAL_ROPE_THETA = 10_000.0
REGISTER_ROPE_THETA = 100.0
NUM_STYLE_TOKENS = 1
NUM_REGISTERS = 4
LAYERSCALE_INIT = 1e-4


class TokenRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(dtype=x.dtype) * self.weight.to(dtype=x.dtype)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init: float = LAYERSCALE_INIT):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), float(init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma.to(dtype=x.dtype) * x


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    x1 = pairs[..., 0]
    x2 = pairs[..., 1]
    return torch.stack((-x2, x1), dim=-1).reshape(*x.shape)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


def build_2d_rope(
    head_dim: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    theta: float = SPATIAL_ROPE_THETA,
    pt_seq_len: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 4 != 0:
        raise ValueError(f"head_dim must be divisible by 4, got {head_dim}")
    axis_dim = head_dim // 2
    n_freq = axis_dim // 2
    index = torch.arange(0, axis_dim, 2, device=device, dtype=torch.float32)[:n_freq]
    inv_freq = 1.0 / (theta ** (index / axis_dim))
    t_h = torch.arange(height, device=device, dtype=torch.float32)
    t_h = t_h / max(height, 1) * float(pt_seq_len)
    t_w = torch.arange(width, device=device, dtype=torch.float32)
    t_w = t_w / max(width, 1) * float(pt_seq_len)
    freqs_h = torch.outer(t_h, inv_freq).repeat_interleave(2, dim=-1)
    freqs_w = torch.outer(t_w, inv_freq).repeat_interleave(2, dim=-1)
    freqs = torch.cat(
        (
            freqs_h[:, None, :].expand(height, width, axis_dim),
            freqs_w[None, :, :].expand(height, width, axis_dim),
        ),
        dim=-1,
    )
    flat = freqs.reshape(height * width, head_dim)
    return flat.cos().to(dtype=dtype), flat.sin().to(dtype=dtype)


def split_rope(
    query: torch.Tensor,
    key: torch.Tensor | None,
    *,
    height: int,
    width: int,
    pt_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Spatial RoPE, unrotated style token, 2x2 register RoPE."""
    length = query.shape[-2]
    head_dim = query.shape[-1]
    spatial = height * width
    if length != spatial + NUM_STYLE_TOKENS + NUM_REGISTERS:
        raise ValueError(
            f"token length {length} != {height}*{width}+{NUM_STYLE_TOKENS + NUM_REGISTERS}"
        )
    cos_s, sin_s = build_2d_rope(
        head_dim, height, width,
        device=query.device, dtype=query.dtype, pt_seq_len=pt_seq_len,
    )
    cos_r, sin_r = build_2d_rope(
        head_dim, 2, 2,
        device=query.device, dtype=query.dtype,
        theta=REGISTER_ROPE_THETA, pt_seq_len=2,
    )
    ones = torch.ones(NUM_STYLE_TOKENS, head_dim, device=query.device, dtype=query.dtype)
    zeros = torch.zeros(NUM_STYLE_TOKENS, head_dim, device=query.device, dtype=query.dtype)
    cos = torch.cat((cos_s, ones, cos_r), dim=0).view(1, 1, length, head_dim)
    sin = torch.cat((sin_s, zeros, sin_r), dim=0).view(1, 1, length, head_dim)
    query = apply_rotary(query, cos, sin)
    if key is not None:
        key = apply_rotary(key, cos, sin)
    return query, key


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, pt_seq_len: int = 64):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim {dim} must be divisible by heads {heads}")
        self.heads = int(heads)
        self.head_dim = dim // heads
        self.pt_seq_len = int(pt_seq_len)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = TokenRMSNorm(self.head_dim)
        self.k_norm = TokenRMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, length, dim = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = split_rope(
            query, key, height=height, width=width, pt_seq_len=self.pt_seq_len,
        )
        out = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(batch, length, dim)
        return self.proj(out)


class CrossAttention(nn.Module):
    """Queries use the spatial RoPE. Memory keys stay unrotated."""

    def __init__(self, dim: int, heads: int, pt_seq_len: int = 64):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim {dim} must be divisible by heads {heads}")
        self.heads = int(heads)
        self.head_dim = dim // heads
        self.pt_seq_len = int(pt_seq_len)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=False)
        self.q_norm = TokenRMSNorm(self.head_dim)
        self.k_norm = TokenRMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        memory: torch.Tensor,
        height: int,
        width: int,
        memory_keep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, dim = tokens.shape
        ctx = memory.shape[1]
        query = self.q_proj(tokens).reshape(batch, length, self.heads, self.head_dim)
        query = query.transpose(1, 2)
        kv = self.kv_proj(memory).reshape(batch, ctx, 2, self.heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        key, value = kv[0], kv[1]
        query = self.q_norm(query)
        key = self.k_norm(key)
        query, _ = split_rope(
            query, None, height=height, width=width, pt_seq_len=self.pt_seq_len,
        )
        attn_mask = None
        if memory_keep is not None:
            # True keeps the memory token. A missing collage neighbor is False.
            attn_mask = memory_keep[:, None, None, :].to(dtype=torch.bool)
        out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=0.0,
        )
        out = out.transpose(1, 2).reshape(batch, length, dim)
        return self.proj(out)


class BottleneckBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int):
        super().__init__()
        self.norm_sa = TokenRMSNorm(dim)
        self.sa = SelfAttention(dim, heads)
        self.ls_sa = LayerScale(dim)
        self.norm_ca = TokenRMSNorm(dim)
        self.norm_mem = TokenRMSNorm(dim)
        self.ca = CrossAttention(dim, heads)
        self.ls_ca = LayerScale(dim)
        self.norm_ffn = TokenRMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )
        self.ls_ffn = LayerScale(dim)

    def forward(
        self,
        tokens: torch.Tensor,
        memory: torch.Tensor,
        height: int,
        width: int,
        memory_keep: torch.Tensor | None = None,
    ):
        tokens = tokens + self.ls_sa(self.sa(self.norm_sa(tokens), height, width))
        tokens = tokens + self.ls_ca(
            self.ca(
                self.norm_ca(tokens),
                self.norm_mem(memory),
                height,
                width,
                memory_keep,
            )
        )
        tokens = tokens + self.ls_ffn(self.ffn(self.norm_ffn(tokens)))
        return tokens
