"""Window, shortlist, and negative-list rules for the retrieval table."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.retrieval_match import (
    Candidate,
    assemble_negatives,
    candidate_offsets,
    choose_positive,
    collect_binned_negatives,
    collect_negatives,
    crop_covered,
    rank_shortlist,
    running_topk,
    stitch_crop,
    tissue_gap,
)


def _block(y0: int, y1: int, x0: int, x1: int, grid: int = 1024) -> set[tuple[int, int]]:
    return {
        (y, x)
        for y in range(y0, y1, grid)
        for x in range(x0, x1, grid)
    }


def test_interior_window_stays_within_one_tile() -> None:
    tiles = _block(0, 4096, 0, 4096)
    offsets = set(candidate_offsets(1024, 1024, tiles))
    if (0, 0) not in offsets or (1024, 1024) not in offsets or (-1024, 0) not in offsets:
        raise AssertionError(f"missing a neighbor: {sorted(offsets)[:12]}")
    if (2048, 0) in offsets or (0, 512) in offsets or (1024, 128) in offsets:
        raise AssertionError(f"interior window reached too far: {sorted(offsets)}")
    if (256, -256) not in offsets:
        raise AssertionError("interior field dropped a covered small shift")


def test_left_edge_shifts_inward() -> None:
    tiles = _block(0, 2048, 0, 3072)
    offsets = set(candidate_offsets(0, 0, tiles))
    if (0, -1024) in offsets or (0, -128) in offsets or (-1024, 0) in offsets:
        raise AssertionError(f"left edge kept a missing side: {sorted(offsets)}")
    for required in ((0, 1024), (0, 2048), (0, 512), (0, 384), (1024, 0)):
        if required not in offsets:
            raise AssertionError(f"left edge missing {required}")
    if (1024, 128) in offsets:
        raise AssertionError("a neighbor tile was small-shifted")


def test_right_edge_shifts_inward() -> None:
    tiles = _block(0, 1024, 0, 3072)
    offsets = set(candidate_offsets(0, 2048, tiles))
    if (0, 1024) in offsets or (0, 512) in offsets:
        raise AssertionError(f"right edge kept the missing side: {sorted(offsets)}")
    for required in ((0, -1024), (0, -2048), (0, -512)):
        if required not in offsets:
            raise AssertionError(f"right edge missing {required}")


def test_single_tile_is_only_itself() -> None:
    offsets = candidate_offsets(0, 0, {(0, 0)})
    if offsets != [(0, 0)]:
        raise AssertionError(offsets)


def test_uncovered_shift_is_rejected() -> None:
    tiles = {(0, 0)}
    if crop_covered(0, 0, 128, 0, tiles):
        raise AssertionError("a shift onto a missing tile was accepted")
    if not crop_covered(0, 0, 0, 0, tiles):
        raise AssertionError("the coordinate tile was rejected")


def test_stitch_uses_both_tiles() -> None:
    grid = 4
    left = np.zeros((grid, grid, 1), dtype=np.uint8)
    right = np.full((grid, grid, 1), 9, dtype=np.uint8)
    left[:] = 3
    tiles = {(0, 0): left, (0, grid): right}
    crop = stitch_crop(tiles, 0, 0, 0, 1, grid)
    if crop.shape != (grid, grid, 1):
        raise AssertionError(crop.shape)
    if not np.all(crop[:, : grid - 1] == 3) or not np.all(crop[:, grid - 1:] == 9):
        raise AssertionError(crop[:, :, 0])


def test_shortlist_uses_both_references_then_cosine() -> None:
    # Teacher and pseudo agree at 0.8. A far crop can be more similar and still lose.
    far = Candidate(0, 2048, tissue_gap(0.0, 0.8, 0.8))
    near = Candidate(0, 0, tissue_gap(0.8, 0.8, 0.8))
    middling = Candidate(128, 0, tissue_gap(0.7, 0.8, 0.8))
    other = Candidate(0, 256, tissue_gap(0.6, 0.8, 0.8))
    fifth = Candidate(0, 512, tissue_gap(0.5, 0.8, 0.8))
    short = rank_shortlist([far, near, middling, other, fifth], k=4)
    if far in short or len(short) != 4:
        raise AssertionError(short)
    winner, cosine = choose_positive([(item, 0.1 if item is not middling else 0.9) for item in short])
    if winner != middling or cosine != 0.9:
        raise AssertionError((winner, cosine))
    tied = choose_positive([(Candidate(0, 256, 0.2), 0.4), (Candidate(0, 0, 0.2), 0.4)])
    if tied[0].dx != 0:
        raise AssertionError(tied)


def test_negatives_skip_eight_then_fill_the_class() -> None:
    same = list(range(20))
    subtype = list(range(100, 110))
    fallback = list(range(200, 400))
    chosen = collect_negatives(same, subtype, None, skip=8, same_take=128, total=256)
    if chosen[:2] != [8, 9]:
        raise AssertionError(chosen[:12])
    if any(value < 8 for value in chosen if value < 20):
        raise AssertionError(chosen)
    # subtype has 10 ids; skip 8 leaves 108 and 109.
    if 100 in chosen or 108 not in chosen:
        raise AssertionError(chosen)
    filled = assemble_negatives(same, subtype, fallback, skip=8, same_take=128, total=32)
    if filled[:4] != [8, 9, 10, 11] or 108 not in filled or 208 not in filled:
        raise AssertionError(filled)
    if len(filled) != 32 or len(set(filled)) != 32:
        raise AssertionError((len(filled), len(set(filled))))
    short = assemble_negatives([1], [], None, skip=0, same_take=128, total=4)
    if short != [1, 1, 1, 1]:
        raise AssertionError(short)


def test_binned_negatives_prefer_other_slides() -> None:
    subtype = list(range(20))
    wider = list(range(40))
    same = list(range(100, 130))
    chosen = collect_binned_negatives(subtype, wider, same, skip=8, total=20)
    if chosen != list(range(8, 20)) + list(range(20, 28)):
        raise AssertionError(chosen)
    if any(value < 8 or value >= 100 for value in chosen):
        raise AssertionError(chosen)
    filled = collect_binned_negatives(subtype[:10], [], same, skip=8, total=6)
    # Two subtype ids survive the skip, then same-slide ids after its own skip.
    if filled != [8, 9, 108, 109, 110, 111]:
        raise AssertionError(filled)


def test_other_slide_ranking_drops_the_same_slide() -> None:
    query = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    bank = torch.tensor([
        [1.0, 0.0],
        [0.8, 0.2],
        [0.0, 1.0],
        [0.3, 0.1],
    ])
    chosen = running_topk(
        query,
        bank,
        k=2,
        query_group=torch.tensor([0, 0]),
        bank_group=torch.tensor([0, 0, 1, 1]),
    )
    if chosen.tolist() != [[3, 2], [3, 2]]:
        raise AssertionError(chosen.tolist())


if __name__ == "__main__":
    test_interior_window_stays_within_one_tile()
    test_left_edge_shifts_inward()
    test_right_edge_shifts_inward()
    test_single_tile_is_only_itself()
    test_uncovered_shift_is_rejected()
    test_stitch_uses_both_tiles()
    test_shortlist_uses_both_references_then_cosine()
    test_negatives_skip_eight_then_fill_the_class()
    test_binned_negatives_prefer_other_slides()
    test_other_slide_ranking_drops_the_same_slide()
    print("ok")
