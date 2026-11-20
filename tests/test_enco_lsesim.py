"""Local self-similarity loss and the pre-skip decoder tap."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.enco_lsesim import EnCoLSeSimLoss
from models.generator import Generator


def test_uniform_neighborhood_is_identical() -> None:
    loss = EnCoLSeSimLoss(num_layers=1, kernel_size=7)
    feat = torch.ones(1, 4, 8, 8)
    y = torch.tensor([[3]])
    x = torch.tensor([[4]])
    sim = loss._local_self_sim(feat, y, x)
    if sim.shape != (1, 1, 48):
        raise AssertionError(tuple(sim.shape))
    if not torch.allclose(sim, torch.ones_like(sim), atol=1e-5):
        raise AssertionError(float(sim.min()))


def test_matching_maps_score_below_a_mismatch() -> None:
    torch.manual_seed(0)
    loss = EnCoLSeSimLoss(num_layers=1, kernel_size=3, hidden_dim=16, fg_ratio=0.5)
    enc = torch.randn(2, 8, 12, 12)
    matched = loss([enc], [enc.clone()], num_patches=6)
    other = loss([enc], [torch.randn_like(enc)], num_patches=6)
    if float(matched) >= float(other):
        raise AssertionError(f"matched {float(matched)} other {float(other)}")


def test_preskip_matches_encoder_grid_and_trains_the_decoder() -> None:
    torch.manual_seed(0)
    model = Generator()
    image, feats, preskip = model(
        torch.randn(1, 3, 512, 512),
        torch.randn(1, 3, 1024, 1024),
        return_preskip=True,
    )
    if image.shape != (1, 3, 512, 512):
        raise AssertionError(tuple(image.shape))
    for layer in (2, 3, 4):
        if feats[layer].shape[-2:] != preskip[layer].shape[-2:]:
            raise AssertionError(f"layer {layer} {tuple(feats[layer].shape)} {tuple(preskip[layer].shape)}")
    preskip[4].sum().backward()
    if model.up3.conv.weight.grad is None:
        raise AssertionError("pre-skip tap did not reach the decoder")


if __name__ == "__main__":
    test_uniform_neighborhood_is_identical()
    test_matching_maps_score_below_a_mismatch()
    test_preskip_matches_encoder_grid_and_trains_the_decoder()
    print("ok")
