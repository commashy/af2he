"""Local self-similarity distance and the pre-skip decoder tap."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.enco_lsesim import EnCoFSeSimLoss
from models.generator import Generator


def test_module_has_no_parameters() -> None:
    loss = EnCoFSeSimLoss(dilations=(1, 2))
    if any(loss.parameters()):
        raise AssertionError("structure loss should not own parameters")


def test_uniform_encoder_is_left_out() -> None:
    loss = EnCoFSeSimLoss(dilations=(1,), kernel_size=7)
    enc = torch.ones(1, 4, 8, 8)
    dec = torch.randn(1, 4, 8, 8)
    value = loss([enc], [dec], num_patches=4)
    if float(value) != 0.0:
        raise AssertionError(float(value))


def test_identical_and_rescaled_maps_match() -> None:
    torch.manual_seed(0)
    loss = EnCoFSeSimLoss(dilations=(1,), kernel_size=3, fg_ratio=0.5)
    enc = torch.randn(2, 8, 12, 12)
    same = loss([enc], [enc * 4 + 0.7], num_patches=6)
    other = loss([enc], [torch.randn_like(enc)], num_patches=6)
    if float(same) > 1e-4:
        raise AssertionError(f"rescaled match {float(same)}")
    if float(same) >= float(other):
        raise AssertionError(f"matched {float(same)} other {float(other)}")


def test_mismatched_decoder_has_a_finite_gradient() -> None:
    torch.manual_seed(0)
    loss = EnCoFSeSimLoss(dilations=(2,), kernel_size=7, flat_norm=0.05)
    enc = torch.randn(1, 8, 16, 16)
    dec = torch.randn(1, 8, 16, 16, requires_grad=True)
    value = loss([enc], [dec], num_patches=8)
    if not (0.2 < float(value) <= 2.0):
        raise AssertionError(float(value))
    value.backward()
    if dec.grad is None or not torch.isfinite(dec.grad).all():
        raise AssertionError("gradient missing or non-finite")
    if float(dec.grad.abs().max()) > 5:
        raise AssertionError(float(dec.grad.abs().max()))
    if float(dec.grad.abs().sum()) == 0:
        raise AssertionError("decoder received no gradient")


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
    test_module_has_no_parameters()
    test_uniform_encoder_is_left_out()
    test_identical_and_rescaled_maps_match()
    test_mismatched_decoder_has_a_finite_gradient()
    test_preskip_matches_encoder_grid_and_trains_the_decoder()
    print("ok")
