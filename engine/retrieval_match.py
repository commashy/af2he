"""Local H&E window and the semi-hard negative list for the retrieval table.

The window stays on tiles that exist. An edge field shifts inward instead of
requesting a neighbor past the tissue. Negatives prefer other slides of the
same subtype and tissue bin, then the same benign or malignant class. The same
slide is only the remainder, after the nearest matches have been left out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

GRID = 1024
SHORTLIST = 4
SKIP_NEAREST = 8
SAME_SLIDE_NEGATIVES = 128
TOTAL_NEGATIVES = 256
SMALL_SHIFTS = (128, 256)
EDGE_SHIFTS = (384, 512)


@dataclass(frozen=True)
class Candidate:
    dy: int
    dx: int
    gap: float


def large_offsets(origin: int, coords: set[int], grid: int = GRID) -> list[int]:
    """Tile steps along one axis. A missing side is replaced by one step inward."""
    wanted = [-grid, 0, grid]
    if (origin - grid) not in coords:
        wanted = [value for value in wanted if value != -grid]
        wanted.append(2 * grid)
    if (origin + grid) not in coords:
        wanted = [value for value in wanted if value != grid]
        wanted.append(-2 * grid)
    found: list[int] = []
    for offset in wanted:
        if offset not in found and (origin + offset) in coords:
            found.append(offset)
    return found


def small_offsets(origin: int, coords: set[int], grid: int = GRID) -> list[int]:
    """Sub-tile shifts. A missing neighbor adds 384 and 512 on the interior side."""
    values = [0]
    for delta in SMALL_SHIFTS:
        values.append(delta)
        values.append(-delta)
    if (origin - grid) not in coords:
        values.extend(EDGE_SHIFTS)
    if (origin + grid) not in coords:
        values.extend(-delta for delta in EDGE_SHIFTS)
    found: list[int] = []
    for value in values:
        if value not in found:
            found.append(value)
    return found


def tile_origins_overlapping(start: int, length: int, phase: int, grid: int = GRID) -> list[int]:
    """Grid origins whose tiles intersect ``[start, start + length)``."""
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if grid <= 0:
        raise ValueError(f"grid must be positive, got {grid}")
    phase = phase % grid
    first = phase + ((start - phase) // grid) * grid
    origins = []
    cursor = first
    end = start + length
    while cursor < end:
        origins.append(cursor)
        cursor += grid
    return origins


def crop_covered(
    y: int,
    x: int,
    dy: int,
    dx: int,
    tiles: set[tuple[int, int]],
    grid: int = GRID,
) -> bool:
    """True when the 1024 crop at ``(y + dy, x + dx)`` lies entirely on existing tiles."""
    ys = tile_origins_overlapping(y + dy, grid, y, grid)
    xs = tile_origins_overlapping(x + dx, grid, x, grid)
    return all((ty, tx) in tiles for ty in ys for tx in xs)


def stitch_crop(
    tiles: dict[tuple[int, int], np.ndarray],
    y: int,
    x: int,
    dy: int,
    dx: int,
    grid: int = GRID,
) -> np.ndarray:
    """Cut a grid-sized crop at ``(y + dy, x + dx)`` from uint8 tiles."""
    sample = next(iter(tiles.values()))
    if sample.ndim != 3 or sample.shape[0] != grid or sample.shape[1] != grid:
        raise ValueError(f"tile shape {sample.shape} does not match grid {grid}")
    out = np.empty((grid, grid, sample.shape[2]), dtype=sample.dtype)
    y0 = y + dy
    x0 = x + dx
    filled = np.zeros((grid, grid), dtype=bool)
    for ty in tile_origins_overlapping(y0, grid, y, grid):
        for tx in tile_origins_overlapping(x0, grid, x, grid):
            if (ty, tx) not in tiles:
                raise KeyError(f"missing tile y={ty} x={tx} for crop {y0},{x0}")
            src_y0 = max(0, y0 - ty)
            src_x0 = max(0, x0 - tx)
            src_y1 = min(grid, y0 + grid - ty)
            src_x1 = min(grid, x0 + grid - tx)
            dst_y0 = max(0, ty - y0)
            dst_x0 = max(0, tx - x0)
            dst_y1 = dst_y0 + (src_y1 - src_y0)
            dst_x1 = dst_x0 + (src_x1 - src_x0)
            out[dst_y0:dst_y1, dst_x0:dst_x1] = tiles[(ty, tx)][src_y0:src_y1, src_x0:src_x1]
            filled[dst_y0:dst_y1, dst_x0:dst_x1] = True
    if not filled.all():
        raise RuntimeError(f"crop y={y0} x={x0} was not fully covered")
    return out


def candidate_offsets(
    y: int,
    x: int,
    tiles: set[tuple[int, int]],
    grid: int = GRID,
) -> list[tuple[int, int]]:
    """``(dy, dx)`` crops for one field. ``(0, 0)`` is the coordinate pair.

    Full tiles cover the inward-shifted neighborhood. Small shifts are applied
    only to this field, and only when the crop is covered.
    """
    xs = {tx for ty, tx in tiles if ty == y}
    ys = {ty for ty, tx in tiles if tx == x}
    found: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for dy in large_offsets(y, ys, grid):
        for dx in large_offsets(x, xs, grid):
            offset = (dy, dx)
            if offset not in seen and (y + dy, x + dx) in tiles:
                seen.add(offset)
                found.append(offset)
    for dy in small_offsets(y, ys, grid):
        for dx in small_offsets(x, xs, grid):
            offset = (dy, dx)
            if offset in seen:
                continue
            if crop_covered(y, x, dy, dx, tiles, grid):
                seen.add(offset)
                found.append(offset)
    if (0, 0) not in seen:
        raise RuntimeError(f"field y={y} x={x} has no coordinate tile")
    return found


def rank_shortlist(candidates: list[Candidate], k: int = SHORTLIST) -> list[Candidate]:
    """Lowest tissue gap first. An equal gap keeps the smaller shift."""
    if k < 1:
        raise ValueError(f"shortlist must be positive, got {k}")
    ordered = sorted(
        candidates,
        key=lambda item: (
            item.gap,
            abs(item.dy) + abs(item.dx),
            max(abs(item.dy), abs(item.dx)),
            item.dy,
            item.dx,
        ),
    )
    return ordered[:k]


def choose_positive(scored: list[tuple[Candidate, float]]) -> tuple[Candidate, float]:
    """Highest cosine in the shortlist. An equal cosine keeps the smaller shift."""
    if not scored:
        raise ValueError("positive selection received no shortlist")
    return max(
        scored,
        key=lambda item: (
            item[1],
            -(abs(item[0].dy) + abs(item[0].dx)),
            -max(abs(item[0].dy), abs(item[0].dx)),
            -item[0].dy,
            -item[0].dx,
        ),
    )


def tissue_gap(candidate: float, teacher: float, pseudo: float) -> float:
    """Mean absolute occupancy error against the teacher and the pseudo label."""
    return 0.5 * (abs(candidate - teacher) + abs(candidate - pseudo))


def take_after_skip(ranked: list[int], skip: int, take: int) -> list[int]:
    """Drop the nearest ``skip`` ids, then keep ``take`` of the next-nearest."""
    if skip < 0 or take < 0:
        raise ValueError(f"skip={skip} take={take}")
    if take == 0:
        return []
    return list(ranked[skip:skip + take])


def collect_negatives(
    same_ranked: list[int],
    subtype_ranked: list[int],
    fallback_ranked: list[int] | None = None,
    *,
    skip: int = SKIP_NEAREST,
    same_take: int = SAME_SLIDE_NEGATIVES,
    total: int = TOTAL_NEGATIVES,
) -> list[int]:
    """Nearest-first ids with the window already removed. Length is at most ``total``.

    Order is same-slide semi-hard, then the same subtype, then the wider class.
    """
    if total < 1:
        raise ValueError(f"total must be positive, got {total}")
    chosen: list[int] = []
    seen: set[int] = set()

    def add(ids: list[int], limit: int) -> None:
        kept = 0
        for value in ids:
            if value in seen or value < 0:
                continue
            seen.add(value)
            chosen.append(int(value))
            kept += 1
            if kept >= limit or len(chosen) >= total:
                return

    add(take_after_skip(same_ranked, skip, same_take), same_take)
    add(take_after_skip(subtype_ranked, skip, total), total - len(chosen))
    if fallback_ranked is not None and len(chosen) < total:
        add(take_after_skip(fallback_ranked, skip, total), total - len(chosen))
    return chosen


def extend_negatives(chosen: list[int], ranked: list[int], skip: int, total: int) -> list[int]:
    """Append semi-hard ids from one pool. The nearest ``skip`` stay out."""
    if skip < 0 or total < 1:
        raise ValueError(f"skip={skip} total={total}")
    seen = set(chosen)
    for value in ranked[skip:]:
        if value < 0 or value in seen:
            continue
        seen.add(int(value))
        chosen.append(int(value))
        if len(chosen) >= total:
            break
    return chosen


def collect_binned_negatives(
    subtype_ranked: list[int],
    class_ranked: list[int],
    same_ranked: list[int],
    *,
    skip: int = SKIP_NEAREST,
    total: int = TOTAL_NEGATIVES,
) -> list[int]:
    """Other-slide subtype, then other-slide class, then the same slide.

    Each list is nearest-first and already excludes the positive window.
    The first ``skip`` of each list are the matches left out of that pool.
    """
    chosen = extend_negatives([], subtype_ranked, skip, total)
    if len(chosen) < total:
        extend_negatives(chosen, class_ranked, skip, total)
    if len(chosen) < total:
        extend_negatives(chosen, same_ranked, skip, total)
    return chosen


def pad_negatives(chosen: list[int], total: int = TOTAL_NEGATIVES) -> list[int]:
    """Repeat collected ids so every row has ``total`` entries."""
    if total < 1:
        raise ValueError(f"total must be positive, got {total}")
    if not chosen:
        return [-1] * total
    if len(chosen) >= total:
        return list(chosen[:total])
    base = list(chosen)
    padded = list(chosen)
    index = 0
    while len(padded) < total:
        padded.append(base[index % len(base)])
        index += 1
    return padded


def assemble_negatives(
    same_ranked: list[int],
    subtype_ranked: list[int],
    fallback_ranked: list[int] | None,
    *,
    skip: int = SKIP_NEAREST,
    same_take: int = SAME_SLIDE_NEGATIVES,
    total: int = TOTAL_NEGATIVES,
) -> list[int]:
    """256 ids. A short pool repeats so the row stays dense."""
    return pad_negatives(
        collect_negatives(
            same_ranked,
            subtype_ranked,
            fallback_ranked,
            skip=skip,
            same_take=same_take,
            total=total,
        ),
        total,
    )


def running_topk(
    query: torch.Tensor,
    bank: torch.Tensor,
    k: int,
    query_group: torch.Tensor | None = None,
    bank_group: torch.Tensor | None = None,
    block: int = 4096,
) -> torch.Tensor:
    """Bank indices of the highest dot products, shape ``[B, k]``.

    ``query`` and ``bank`` are L2-normalized. Matching group ids are removed
    before the ranking, so a same-slide patch cannot occupy the other-slide list.
    Unfilled slots are -1.
    """
    if query.ndim != 2 or bank.ndim != 2 or query.shape[1] != bank.shape[1]:
        raise ValueError(f"query {tuple(query.shape)} bank {tuple(bank.shape)}")
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    if block < 1:
        raise ValueError(f"block must be positive, got {block}")
    count = query.shape[0]
    width = bank.shape[0]
    keep = min(k, width)
    best_val = torch.full((count, keep), float("-inf"), device=query.device, dtype=torch.float32)
    best_idx = torch.full((count, keep), -1, device=query.device, dtype=torch.long)
    if width == 0:
        return best_idx
    for start in range(0, width, block):
        stop = min(width, start + block)
        sims = query.float().matmul(bank[start:stop].float().T)
        if query_group is not None:
            if bank_group is None:
                raise ValueError("bank_group is required when query_group is set")
            same = query_group[:, None] == bank_group[start:stop][None, :]
            sims = sims.masked_fill(same, float("-inf"))
        local_k = min(keep, sims.shape[1])
        vals, local = torch.topk(sims, k=local_k, dim=1)
        local = local + start
        cat_val = torch.cat((best_val, vals), dim=1)
        cat_idx = torch.cat((best_idx, local), dim=1)
        best_val, order = torch.topk(cat_val, k=keep, dim=1)
        best_idx = torch.gather(cat_idx, 1, order)
    return best_idx.masked_fill(~torch.isfinite(best_val), -1)
