"""Shape, full-image L1, mask fill, tanh range, and contiguous DDP grads."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.mask import mask_patches
from engine.losses import reconstruction_loss
from models.generator import BOTTLENECK_DIM, COLLAGE, TILE, Generator
from models.slide_head import SlideDiagnosisHead


def test_forward_is_tanh_range() -> None:
    torch.manual_seed(0)
    model = Generator().eval()
    tile = torch.randn(1, 3, TILE, TILE)
    collage = torch.randn(1, 3, COLLAGE, COLLAGE)
    with torch.no_grad():
        recon = model(tile, collage)
    if recon.shape != tile.shape:
        raise AssertionError(f"recon {tuple(recon.shape)} != tile {tuple(tile.shape)}")
    if float(recon.min()) < -1.0 - 1e-5 or float(recon.max()) > 1.0 + 1e-5:
        raise AssertionError(f"output left [-1, 1]: {float(recon.min())}, {float(recon.max())}")


def test_l1_uses_the_whole_image() -> None:
    target = torch.zeros(1, 3, 8, 8)
    pred = target.clone()
    pred[:, :, 0, 0] = 1
    loss, _ = reconstruction_loss(pred, target)
    if loss.item() <= 0:
        raise AssertionError("changing one pixel did not change the full-image L1")
    exact, _ = reconstruction_loss(target, target)
    if exact.item() != 0:
        raise AssertionError("identical images do not have zero L1")


def test_mask_fill_is_minus_one() -> None:
    image = torch.ones(2, 3, 64, 64)
    masked = mask_patches(image, patch=64, fraction=1.0, fill=-1.0)
    if not torch.equal(masked, torch.full_like(image, -1.0)):
        raise AssertionError("a fully masked image was not filled with -1")
    kept = mask_patches(image, patch=64, fraction=0.0, fill=-1.0)
    if not torch.equal(kept, image):
        raise AssertionError("fraction 0 changed the image")


def test_film_starts_as_identity() -> None:
    model = Generator().eval()
    feature = torch.randn(2, 96, 8, 8)
    cond = torch.randn(2, 384)
    filmed = model.stem_film(feature, cond)
    if not torch.allclose(filmed, feature):
        raise AssertionError("zero-init FiLM changed the features")


def test_style_grad_is_contiguous() -> None:
    torch.manual_seed(2)
    model = Generator().train()
    tile = torch.randn(2, 3, TILE, TILE)
    collage = torch.randn(2, 3, COLLAGE, COLLAGE)
    recon = model(tile, collage)
    loss, _ = reconstruction_loss(recon, tile)
    loss.backward()
    for name in ("style", "registers", "context_type", "cls"):
        owner = model.bottleneck if name != "cls" else model.collage_encoder
        grad = getattr(owner, name).grad
        if grad is None or not grad.is_contiguous():
            raise AssertionError(f"{name} grad is missing or non-contiguous: {None if grad is None else grad.stride()}")
    pos_grad = model.collage_encoder.pos.grad
    if pos_grad is None or not pos_grad.is_contiguous():
        raise AssertionError("collage position grad is missing or non-contiguous")


def test_collage_encoder_is_in_the_checkpoint_and_the_diagnosis_head_trains_it() -> None:
    torch.manual_seed(0)
    model = Generator()
    state = model.state_dict()
    for key in ("collage_encoder.cls", "collage_encoder.pos", "collage_encoder.proj.weight", "bottleneck.in_proj.weight"):
        if key not in state:
            raise AssertionError(f"missing {key}")
    head = SlideDiagnosisHead(BOTTLENECK_DIM)
    _image, cond = model(
        torch.randn(4, 3, TILE, TILE),
        torch.randn(4, 3, COLLAGE, COLLAGE),
        return_cond=True,
    )
    if cond.shape != (4, BOTTLENECK_DIM):
        raise AssertionError(tuple(cond.shape))
    loss = torch.nn.functional.cross_entropy(head(cond.mean(dim=0, keepdim=True)), torch.tensor([1]))
    loss.backward()
    if model.collage_encoder.cls.grad is None or float(model.collage_encoder.cls.grad.abs().sum()) == 0.0:
        raise AssertionError("diagnosis loss did not reach the collage class token")
    if "norm.weight" not in head.state_dict() or "linear.weight" not in head.state_dict():
        raise AssertionError(sorted(head.state_dict()))


def test_contrastive_encode_accepts_a_1024_field() -> None:
    torch.manual_seed(0)
    model = Generator().train()
    field = torch.randn(1, 3, TILE * 2, TILE * 2, requires_grad=True)
    collage = torch.randn(1, 3, COLLAGE, COLLAGE)
    feats = model.encoder_features(field, collage)
    expected = (
        (96, 1024),
        (96, 1024),
        (192, 512),
        (384, 256),
        (576, 128),
    )
    for feat, (channels, side) in zip(feats, expected):
        if tuple(feat.shape) != (1, channels, side, side):
            raise AssertionError(f"{tuple(feat.shape)} != {(1, channels, side, side)}")
    feats[2].sum().backward()
    left = float(field.grad[:, :, :, :TILE].abs().sum())
    right = float(field.grad[:, :, :, TILE:].abs().sum())
    if left <= 0.0 or right <= 0.0:
        raise AssertionError(f"field gradient left={left} right={right}")
    try:
        model(field.detach(), collage)
    except ValueError:
        return
    raise AssertionError("generate accepted a 1024 field")


def test_local_neighborhood_is_shared_across_the_four_tiles() -> None:
    torch.manual_seed(0)
    model = Generator(local=True).train()
    if model.local_encoder is None or model.local_encoder.cnn[0].in_channels != 4:
        raise AssertionError("local stem should take RGB plus a mask")
    if float(model.local_encoder.cnn[0].weight[:, 3:].abs().sum()) != 0.0:
        raise AssertionError("the mask channel should start at zero")
    tiles = torch.randn(4, 3, TILE, TILE)
    collage = torch.randn(4, 3, COLLAGE, COLLAGE)
    local = torch.randn(1, 4, COLLAGE, COLLAGE)
    image = model(tiles, collage, local)
    if image.shape != tiles.shape:
        raise AssertionError(f"local forward shape {tuple(image.shape)}")
    image.square().mean().backward()
    grad = model.local_encoder.cnn[0].weight.grad
    if grad is None or float(grad[:, :3].abs().sum()) == 0.0 or float(grad[:, 3:].abs().sum()) == 0.0:
        raise AssertionError("local stem did not receive a gradient")
    cls_grad = model.local_encoder.cls.grad
    if cls_grad is None or cls_grad.stride() != (256, 256, 1):
        raise AssertionError(f"class token grad strides {None if cls_grad is None else cls_grad.stride()}")


if __name__ == "__main__":
    test_l1_uses_the_whole_image()
    test_mask_fill_is_minus_one()
    test_forward_is_tanh_range()
    test_film_starts_as_identity()
    test_style_grad_is_contiguous()
    test_collage_encoder_is_in_the_checkpoint_and_the_diagnosis_head_trains_it()
    test_local_neighborhood_is_shared_across_the_four_tiles()
    test_contrastive_encode_accepts_a_1024_field()
    print("ok")
