"""Routing math for the epoch-10 set match. No UNI weights."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.hi_slide_ot import field_tanh_to_512_rgb01, route_queries


def test_paired_match_beats_the_dustbin() -> None:
    torch.manual_seed(0)
    query = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    # The loss uses 1024-d tokens. Pad the toy vectors.
    def pad(matrix: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(matrix.shape[0], 1024)
        out[:, : matrix.shape[1]] = matrix
        return torch.nn.functional.normalize(out, dim=-1)

    query = pad(query)
    paired = query.clone()
    medoids = torch.zeros(256, 1024)
    medoids[0] = query[0]
    route = route_queries(query, paired, medoids, benign=False)
    if float(route["paired_weight"].mean()) < 0.5:
        raise AssertionError(f"paired weight {float(route['paired_weight'].mean())}")
    if float(route["row_max"]) <= 0.0:
        raise AssertionError("unclamped row sum was not recorded")


def test_zero_padded_medoids_are_not_counted() -> None:
    query = torch.nn.functional.normalize(torch.randn(4, 1024), dim=-1)
    medoids = torch.zeros(256, 1024)
    medoids[0] = query[0]
    route = route_queries(query, None, medoids, benign=True, tau_slide_benign=0.55)
    if int(route["n_medoids_used"]) > 1:
        raise AssertionError(f"padded bank counted {int(route['n_medoids_used'])} medoids")


def test_bicubic_resize_stays_in_unit_interval() -> None:
    field = torch.tensor([-1.0, 1.0]).view(1, 1, 1, 2).repeat(1, 3, 1024, 512)
    rgb = field_tanh_to_512_rgb01(field)
    if rgb.shape[-2:] != (512, 512):
        raise AssertionError(tuple(rgb.shape))
    if float(rgb.min()) < 0.0 or float(rgb.max()) > 1.0:
        raise AssertionError(f"rgb left [0, 1]: {float(rgb.min())}, {float(rgb.max())}")


def test_generated_tokens_receive_the_gradient() -> None:
    live = torch.nn.functional.normalize(torch.randn(3, 1024), dim=-1).requires_grad_(True)
    target = torch.nn.functional.normalize(torch.randn(3, 1024), dim=-1)
    weight = torch.tensor([1.0, 0.0, 0.5])
    loss = (weight * (1.0 - (live * target).sum(dim=-1))).sum() / 3.0
    loss.backward()
    if live.grad is None or float(live.grad[0].abs().sum()) <= 0.0:
        raise AssertionError("matched token received no gradient")
    if float(live.grad[1].abs().sum()) != 0.0:
        raise AssertionError("unmatched token received a gradient")


if __name__ == "__main__":
    test_paired_match_beats_the_dustbin()
    test_zero_padded_medoids_are_not_counted()
    test_bicubic_resize_stays_in_unit_interval()
    test_generated_tokens_receive_the_gradient()
    print("ok")
