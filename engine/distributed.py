"""Single-node DDP setup."""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist


def setup() -> SimpleNamespace:
    if "RANK" not in os.environ:
        return SimpleNamespace(rank=0, world_size=1, local_rank=0, distributed=False)
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return SimpleNamespace(
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        local_rank=local_rank,
        distributed=True,
    )


def cleanup(ctx: SimpleNamespace) -> None:
    if ctx.distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main(ctx: SimpleNamespace) -> bool:
    return ctx.rank == 0
