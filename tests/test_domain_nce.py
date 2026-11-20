"""Domain adapter and debiased PatchNCE. No generator."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.contrastive import DebiasedPatchNCESRC
from models.sample import DomainAdaptedPatchProjector


def test_adapter_keeps_unit_length() -> None:
    projector = DomainAdaptedPatchProjector([8], width=16)
    patches = torch.randn(2, 5, 8)
    projected = projector([patches], "he")
    if projected[0].shape != (2, 5, 16):
        raise AssertionError(tuple(projected[0].shape))
    length = projected[0].norm(dim=-1)
    if not torch.allclose(length, torch.ones_like(length), atol=1e-4):
        raise AssertionError(f"length {length}")


def test_matched_patches_beat_a_shuffle() -> None:
    torch.manual_seed(0)
    loss_fn = DebiasedPatchNCESRC(use_src=True, lambda_src=0.5)
    key = F.normalize(torch.randn(2, 6, 16), dim=-1)
    query = key.detach().clone().requires_grad_(True)
    matched, relation, weight = loss_fn(query, key, key)
    matched.backward()
    if query.grad is None or float(query.grad.abs().sum()) <= 0:
        raise AssertionError("matched query received no gradient")
    shuffled = key.roll(1, dims=1)
    moved, _relation, _weight = loss_fn(shuffled, key, key)
    if float(matched) >= float(moved):
        raise AssertionError(f"matched {float(matched)} shuffled {float(moved)}")
    if float(relation) < -1e-5 or not torch.isfinite(weight):
        raise AssertionError(f"relation {float(relation)} weight {float(weight)}")


if __name__ == "__main__":
    test_adapter_keeps_unit_length()
    test_matched_patches_beat_a_shuffle()
    print("ok")
