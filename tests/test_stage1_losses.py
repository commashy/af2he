"""Checks for the stage-1 losses and the encoder feature maps."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch.nn.functional as F

from data.stage1 import exclusion_keys, is_excluded, seam_windows, stitch_quadrants
from engine.contrastive import _drop_diagonal, asp_weights, field_info_nce, mdcl_asp, mdcl_nce, patch_nce, retrieval_pair_weight
from engine.image_losses import (
    GaussianPyramidLoss,
    _RGB_FROM_HED,
    af_brightness,
    bright_af_keep,
    bright_neighbor_fill,
    recon_bright_weight,
    af_presence,
    content_gate,
    hed_recon,
    stain_edge_loss,
    detail_loss,
    occupancy_loss,
    stain_polarity,
    structure_loss,
    recon_epoch_settings,
    recon_loss,
    hed_tissue_mask,
    hed_tissue_prob,
    lsgan,
    masked_l1,
    sliced_wasserstein,
    tissue_probability,
    tone_loss,
)
from models.discriminator import MultiScaleDiscriminator
from models.generator import FEATURE_CHANNELS, Generator


def test_field_info_nce_prefers_the_stored_positive() -> None:
    query = torch.tensor([[1.0, 0.0]], requires_grad=True)
    positive = torch.tensor([[1.0, 0.0]])
    negative = torch.tensor([[[-1.0, 0.0], [0.0, 1.0]]])
    matched = field_info_nce(query, positive, negative, temperature=0.07)
    wrong_positive = negative[:, 0]
    wrong_negative = torch.stack((positive, negative[:, 1]), dim=1)
    swapped = field_info_nce(query, wrong_positive, wrong_negative, temperature=0.07)
    if float(matched) >= float(swapped):
        raise AssertionError(f"matched={float(matched)} swapped={float(swapped)}")
    matched.backward()
    if query.grad is None or float(query.grad.abs().sum()) == 0.0:
        raise AssertionError("query received no gradient")


def test_retrieval_pair_weight_ramps_between_the_cuts() -> None:
    cosine = torch.tensor([0.0, 0.3, 0.45, 0.6, 0.9])
    weight = retrieval_pair_weight(cosine, 0.3, 0.6)
    expected = torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0])
    if not torch.allclose(weight, expected):
        raise AssertionError(f"ramp {weight}")


def test_uni_pair_weight_drops_a_bad_retrieval() -> None:
    query = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
    positive = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    negative = torch.tensor([[[-1.0, 0.0]], [[-1.0, 0.0]]])
    full = field_info_nce(query, positive, negative, pair_weight=torch.tensor([1.0, 1.0]))
    half = field_info_nce(query, positive, negative, pair_weight=torch.tensor([0.0, 1.0]))
    if abs(float(half) - 0.5 * float(full)) > 1e-5:
        raise AssertionError(f"full={float(full)} half={float(half)}")
    dropped = field_info_nce(query, positive, negative, pair_weight=torch.zeros(2))
    dropped.backward()
    if float(dropped) != 0.0 or query.grad is None or float(query.grad.abs().sum()) != 0.0:
        raise AssertionError(f"dropped={float(dropped)} grad={None if query.grad is None else float(query.grad.abs().sum())}")


def test_same_domain_diagonal_is_removed() -> None:
    matrix = torch.tensor([[
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]])
    removed = _drop_diagonal(matrix)
    expected = torch.tensor([[
        [2.0, 3.0],
        [4.0, 6.0],
        [7.0, 8.0],
    ]])
    if not torch.equal(removed, expected):
        raise AssertionError(f"same-domain bank {removed}")


def test_patch_nce_matches_cut_and_prefers_the_same_patch() -> None:
    torch.manual_seed(0)
    batch, count, dim = 2, 8, 16
    key = F.normalize(torch.randn(batch * count, dim), dim=1)
    matched = patch_nce(key.clone().requires_grad_(True), key, batch)
    if matched.shape != (batch * count,) or float(matched.mean()) > 1.0:
        raise AssertionError(f"matched CUT loss {float(matched.mean())}")
    matched.mean().backward()
    random_q = F.normalize(torch.randn(batch * count, dim), dim=1)
    other = float(patch_nce(random_q, key, batch).mean())
    if float(matched.detach().mean()) > 0.05 or other <= float(matched.detach().mean()):
        raise AssertionError(f"matched={float(matched.detach().mean())} random={other}")


def test_mdcl_prefers_the_matched_patch() -> None:
    torch.manual_seed(0)
    batch, count, dim = 2, 8, 16
    key = F.normalize(torch.randn(batch * count, dim), dim=1)
    same = mdcl_nce(key.clone(), key, batch).mean()
    other = mdcl_nce(F.normalize(torch.randn(batch * count, dim), dim=1), key, batch).mean()
    if not same < other:
        raise AssertionError(f"matched {float(same)} is not below random {float(other)}")


def test_asp_is_uniform_before_decay() -> None:
    positive = torch.tensor([-0.4, 0.2, 0.8])
    weights = asp_weights(positive, epoch=1, total_epochs=30, n_epochs_decay=15)
    if not torch.allclose(weights, torch.ones_like(weights)):
        raise AssertionError(f"early ASP weights {weights}")
    query = F.normalize(torch.randn(16, 8), dim=1)
    key = F.normalize(torch.randn(16, 8), dim=1)
    plain = mdcl_nce(query, key, batch_size=2)
    weighted = mdcl_asp(query, key, 2, 0.07, epoch=1, total_epochs=30, n_epochs_decay=15)
    if not torch.allclose(plain, weighted):
        raise AssertionError("ASP changed the loss during the uniform phase")


def test_content_gate_tracks_tissue_amount() -> None:
    pink = torch.tensor([0.85, 0.35, 0.55], dtype=torch.float32).view(1, 3, 1, 1)
    glass = torch.ones(1, 3, 32, 32)
    tissue = glass.clone()
    tissue[:, :, :16, :] = pink
    same = float(content_gate(tissue, tissue))
    mismatch = float(content_gate(tissue, glass))
    if same < 0.99 or mismatch >= same:
        raise AssertionError(f"same={same} mismatch={mismatch}")


def test_masked_l1_uses_only_pseudo_tissue() -> None:
    pink = torch.tensor([0.85, 0.35, 0.55], dtype=torch.float32).view(1, 3, 1, 1)
    pseudo = torch.ones(1, 3, 32, 32)
    pseudo[:, :, :, :16] = pink
    mask = hed_tissue_mask(pseudo)
    covered = float(mask.float().mean())
    if not 0.4 < covered < 0.6:
        raise AssertionError(f"tissue coverage {covered}")
    glass_only = pseudo.clone()
    glass_only[:, :, :, 16:] = pink
    if float(masked_l1(pseudo, glass_only, mask)) > 1e-5:
        raise AssertionError("glass pixels changed the teacher L1")
    tissue_only = pseudo.clone()
    tissue_only[:, :, :, :16] = 0
    if float(masked_l1(pseudo, tissue_only, mask)) <= 0:
        raise AssertionError("tissue pixels did not change the teacher L1")
    if float(masked_l1(pseudo, pseudo, hed_tissue_mask(torch.ones_like(pseudo)))) != 0:
        raise AssertionError("empty tissue mask should be zero")


def test_tissue_vote_requires_af() -> None:
    glass = torch.ones(1, 3, 32, 32)
    af_empty = torch.full((1, 3, 32, 32), -1.0)
    af_present = torch.full((1, 3, 32, 32), -0.6)
    pink = torch.tensor([0.85, 0.35, 0.55]).view(1, 3, 1, 1).expand(1, 3, 32, 32)
    blocked = float(tissue_probability(af_empty, glass, pink).mean())
    allowed = float(tissue_probability(af_present, glass, pink).mean())
    if blocked > 0.05 or allowed < 0.5:
        raise AssertionError(f"blocked={blocked} allowed={allowed}")
    if float(af_presence(af_empty).mean()) > 0.05:
        raise AssertionError("empty AF scored as present")
    if float(hed_tissue_prob(glass).mean()) > 0.05:
        raise AssertionError("white H&E scored as tissue")


def test_tone_uses_pseudo_on_glass_and_teacher_on_tissue() -> None:
    teacher = torch.ones(1, 3, 32, 32)
    pseudo = -torch.ones(1, 3, 32, 32)
    glass = torch.zeros(1, 1, 32, 32)
    tissue = torch.ones(1, 1, 32, 32)
    glass_match = float(tone_loss(pseudo, teacher, pseudo, glass, sigma=1.0))
    tissue_match = float(tone_loss(teacher, teacher, pseudo, tissue, sigma=1.0))
    tissue_miss = float(tone_loss(pseudo, teacher, pseudo, tissue, sigma=1.0))
    if glass_match > 0.05 or tissue_match > 0.05 or tissue_miss < 0.5:
        raise AssertionError(
            f"glass={glass_match} tissue={tissue_match} swapped={tissue_miss}"
        )


def test_stage2_recon_is_sharp_for_three_epochs_then_widens() -> None:
    schedule = [
        recon_epoch_settings(index, 3, 3.0, 0.0, 1.0, [4.0, 8.0, 16.0])
        for index in range(8)
    ]
    expected = [
        (3.0, 0.0), (3.0, 0.0), (3.0, 0.0),
        (1.0, 4.0), (1.0, 8.0),
        (1.0, 16.0), (1.0, 16.0), (1.0, 16.0),
    ]
    if schedule != expected:
        raise AssertionError(schedule)


def test_recon_can_drop_a_benign_field() -> None:
    tissue = _rgb_from_he(0.6, 0.1).expand(2, 3, 16, 16).clone()
    shifted = tissue.clone()
    shifted[0] = _rgb_from_he(0.1, 0.6).expand(1, 3, 16, 16)[0]
    prob = torch.ones(2, 1, 16, 16)
    kept = recon_loss(shifted, tissue, tissue, prob, sigma=0.0, field_weight=torch.tensor([0.0, 1.0]))
    both = recon_loss(shifted, tissue, tissue, prob, sigma=0.0)
    if float(kept) >= float(both):
        raise AssertionError(f"dropping the bad field did not reduce the loss: {float(kept)} vs {float(both)}")


def test_recon_uses_smoothed_teacher_on_tissue_and_pseudo_on_glass() -> None:
    teacher = torch.ones(1, 3, 32, 32)
    pseudo = -torch.ones(1, 3, 32, 32)
    glass = torch.zeros(1, 1, 32, 32)
    tissue = torch.ones(1, 1, 32, 32)
    glass_match = float(recon_loss(pseudo, teacher, pseudo, glass, sigma=8.0))
    tissue_match = float(recon_loss(teacher, teacher, pseudo, tissue, sigma=8.0))
    tissue_miss = float(recon_loss(pseudo, teacher, pseudo, tissue, sigma=8.0))
    if glass_match > 1e-5 or tissue_match > 1e-5 or tissue_miss < 0.2:
        raise AssertionError(
            f"glass={glass_match} tissue={tissue_match} swapped={tissue_miss}"
        )


def test_recon_ignores_detail_finer_than_the_blur() -> None:
    yy = torch.arange(64)
    xx = torch.arange(64)
    checker = (((yy[:, None] + xx[None, :]) % 2).float() * 2.0 - 1.0).view(1, 1, 64, 64)
    base = torch.full((1, 3, 64, 64), 0.2)
    noisy = (base + 0.25 * checker).clamp(-1.0, 1.0)
    tissue = torch.ones(1, 1, 64, 64)
    blurred = float(recon_loss(noisy, base, base, tissue, sigma=8.0))
    sharp = float(recon_loss(noisy, base, base, tissue, sigma=0.0))
    if blurred > sharp * 0.25:
        raise AssertionError(f"blurred={blurred} sharp={sharp}")


def test_recon_trains_the_output_and_not_the_teacher() -> None:
    fake = torch.rand(1, 3, 32, 32, requires_grad=True)
    teacher = torch.rand(1, 3, 32, 32, requires_grad=True)
    pseudo = torch.rand(1, 3, 32, 32)
    tissue = torch.ones(1, 1, 32, 32)
    loss = recon_loss(fake, teacher, pseudo, tissue, sigma=8.0)
    loss.backward()
    if fake.grad is None or float(fake.grad.abs().sum()) == 0.0:
        raise AssertionError("generated image received no gradient")
    if teacher.grad is not None:
        raise AssertionError("teacher received a gradient")


def test_recon_penalizes_a_hematoxylin_eosin_swap() -> None:
    def stain(hematoxylin: float, eosin: float) -> torch.Tensor:
        optical = torch.tensor([hematoxylin, eosin, 0.0], dtype=torch.float32) @ _RGB_FROM_HED.T
        rgb = torch.exp(-optical).clamp(0.0, 1.0)
        return (rgb * 2.0 - 1.0).view(1, 3, 1, 1).expand(1, 3, 16, 16)

    tissue = torch.ones(1, 1, 16, 16)
    full = stain(0.8, 0.1)
    swapped = stain(0.1, 0.8)
    halved = stain(0.4, 0.05)
    same = float(recon_loss(full, full, full, tissue, sigma=0.0))
    hue = float(recon_loss(swapped, full, full, tissue, sigma=0.0))
    amount = float(recon_loss(halved, full, full, tissue, sigma=0.0))
    if same > 1e-5 or hue < 0.2 or hue < amount:
        raise AssertionError(f"same={same} hue={hue} amount={amount}")


def test_recon_adds_weight_on_bright_af() -> None:
    teacher = torch.ones(1, 3, 16, 16)
    pseudo = torch.zeros(1, 3, 16, 16)
    fake = torch.zeros(1, 3, 16, 16)
    tissue = torch.ones(1, 1, 16, 16)
    dark = af_brightness(torch.full((1, 3, 16, 16), -1.0))
    bright = af_brightness(torch.ones(1, 3, 16, 16))
    dark_loss = float(recon_loss(fake, teacher, pseudo, tissue, 0.0, dark, bright_gain=3.0))
    bright_loss = float(recon_loss(fake, teacher, pseudo, tissue, 0.0, bright, bright_gain=3.0))
    if dark_loss <= 0.0 or abs(bright_loss - 3.0 * dark_loss) > 1e-4:
        raise AssertionError(f"dark={dark_loss} bright={bright_loss}")


def _rgb_from_he(hematoxylin: float, eosin: float) -> torch.Tensor:
    optical = torch.tensor([hematoxylin, eosin, 0.0], dtype=torch.float32) @ _RGB_FROM_HED.T
    rgb = torch.exp(-optical).clamp(0.0, 1.0)
    return (rgb * 2.0 - 1.0).view(1, 3, 1, 1)


def _ruifrok_rgb(hematoxylin: float, eosin: float) -> torch.Tensor:
    optical = torch.tensor([hematoxylin, eosin, 0.0], dtype=torch.float32) @ _RGB_FROM_HED
    rgb = torch.exp(-optical).clamp(0.0, 1.0)
    return (rgb * 2.0 - 1.0).view(1, 3, 1, 1)


def test_hed_recon_ignores_stain_added_on_epoch10_glass() -> None:
    eosin = _ruifrok_rgb(0.05, 0.80)
    tissue = eosin.expand(1, 3, 32, 64).clone()
    glass = torch.ones(1, 3, 32, 64)
    target = tissue.clone()
    target[:, :, :, 32:] = 1.0
    added = target.clone()
    added[:, :, :, 32:] = eosin
    missing = glass.clone()
    kept = float(hed_recon(added, target, sigma=4.0, tissue_only=True))
    full = float(hed_recon(added, target, sigma=4.0, tissue_only=False))
    dropped = float(hed_recon(missing, target, sigma=0.0, tissue_only=True))
    if kept > 0.02 or kept > full or dropped < 0.2:
        raise AssertionError(f"kept={kept} full={full} dropped={dropped}")


def _gray_af(gray: float, height: int, width: int) -> torch.Tensor:
    level = float(gray) * 2.0 - 1.0
    return torch.full((1, 3, height, width), level)


def test_bright_fill_borrows_nearby_eosin_when_epoch10_is_pale() -> None:
    pink = _ruifrok_rgb(0.15, 0.80)
    white = torch.ones(1, 3, 32, 32)
    epoch10 = white.clone()
    epoch10[:, :, 0:8, 0:8] = pink
    af = _gray_af(0.05, 32, 32)
    af[:, :, :, 20:] = 0.40  # gray 0.70, above the 0.60 cut
    pale_output = white.clone().requires_grad_(True)
    filled = bright_neighbor_fill(pale_output, epoch10, af, radius=8, dilate=2)
    copied = bright_neighbor_fill(pink.expand(1, 3, 32, 32).contiguous(), epoch10, af, radius=8, dilate=2)
    moderate = af.clone()
    moderate[:, :, :, 20:] = -0.20  # gray 0.40, epoch 10 is still pink there
    below = bright_neighbor_fill(pale_output.detach(), epoch10, moderate, radius=8, dilate=2)
    if float(filled) <= float(copied) + 0.05 or float(below) > 1e-6:
        raise AssertionError(f"filled={float(filled)} copied={float(copied)} below={float(below)}")
    filled.backward()
    if pale_output.grad is None or float(pale_output.grad[:, :, :, 20:].abs().sum()) <= 0.0:
        raise AssertionError("bright fill did not train the bright side")
    if float(pale_output.grad[:, :, :, :16].abs().sum()) > 0.0:
        raise AssertionError("bright fill trained the dark side")


def test_bright_af_keeps_a_light_stain_and_then_stops() -> None:
    white = torch.ones(1, 3, 16, 32, requires_grad=True)
    stained = _ruifrok_rgb(0.10, 0.50).expand(1, 3, 16, 32).contiguous()
    deep = _ruifrok_rgb(0.20, 1.00).expand(1, 3, 16, 32).contiguous()
    bright = _gray_af(0.70, 16, 32)
    dim = _gray_af(0.10, 16, 32)
    blank = bright_af_keep(white, bright)
    held = bright_af_keep(stained, bright)
    darker = bright_af_keep(deep, bright)
    ignored = bright_af_keep(white.detach(), dim)
    if float(blank) <= float(held) + 0.05 or float(held) > 1e-5 or float(darker) > 1e-5 or float(ignored) > 1e-6:
        raise AssertionError(f"blank={float(blank)} held={float(held)} deep={float(darker)} dim={float(ignored)}")
    blank.backward()
    if white.grad is None or float(white.grad.abs().sum()) <= 0.0:
        raise AssertionError("bright AF keep did not train the output")


def test_recon_counts_bright_stained_af_harder() -> None:
    pink = _ruifrok_rgb(0.10, 0.80).expand(1, 3, 16, 32).contiguous()
    white = torch.ones(1, 3, 16, 32)
    af = _gray_af(0.10, 16, 32)
    af[:, :, :, 16:] = 0.20  # gray 0.60
    stained = recon_bright_weight(af, pink)
    empty = pink.clone()
    empty[:, :, :, 16:] = 1.0
    pale = recon_bright_weight(af, empty)
    if float(stained[:, :, :, 16:].mean()) < 2.5 or float(stained[:, :, :, :16].mean()) > 1.05:
        raise AssertionError(f"stained bright={float(stained[:, :, :, 16:].mean())} dim={float(stained[:, :, :, :16].mean())}")
    if float(pale[:, :, :, 16:].mean()) > 1.05:
        raise AssertionError(f"empty epoch 10 was upweighted to {float(pale[:, :, :, 16:].mean())}")
    output = white.clone().requires_grad_(True)
    hed_recon(output, pink, sigma=0.0, weight=stained).backward()
    dim = float(output.grad[:, :, :, :16].abs().mean())
    bright = float(output.grad[:, :, :, 16:].abs().mean())
    if bright < dim * 2.0:
        raise AssertionError(f"bright grad {bright} dim grad {dim}")


def test_hed_recon_can_compare_eosin_only() -> None:
    eosin = _ruifrok_rgb(0.05, 0.80).expand(1, 3, 16, 16).contiguous()
    more_h = _ruifrok_rgb(0.80, 0.80).expand(1, 3, 16, 16).contiguous()
    less_e = _ruifrok_rgb(0.05, 0.20).expand(1, 3, 16, 16).contiguous()
    ignored = float(hed_recon(more_h, eosin, sigma=0.0, eosin_only=True))
    changed = float(hed_recon(less_e, eosin, sigma=0.0, eosin_only=True))
    both = float(hed_recon(more_h, eosin, sigma=0.0, eosin_only=False))
    if ignored > 0.02 or changed < 0.1 or both <= ignored:
        raise AssertionError(f"ignored={ignored} changed={changed} both={both}")


def test_stain_edge_matches_mass_boundaries_and_their_direction() -> None:
    tissue = _ruifrok_rgb(0.20, 0.90)
    glass = torch.ones(1, 3, 48, 64)
    target = glass.clone()
    target[:, :, :, :32] = tissue
    swapped_dye = glass.clone()
    swapped_dye[:, :, :, :32] = _ruifrok_rgb(0.90, 0.20)
    flipped = glass.clone()
    flipped[:, :, :, 32:] = tissue
    same = float(stain_edge_loss(target, target, sigma=4.0))
    dye = float(stain_edge_loss(swapped_dye, target, sigma=4.0))
    reverse = float(stain_edge_loss(flipped, target, sigma=4.0))
    if same > 0.05 or dye > 0.15 or reverse < dye + 0.05:
        raise AssertionError(f"same={same} dye={dye} reverse={reverse}")
    generated = target.detach().clone().requires_grad_(True)
    stain_edge_loss(generated, target, sigma=4.0).backward()
    if generated.grad is None or float(generated.grad.abs().sum()) <= 0.0:
        raise AssertionError("edge loss did not train the output")


def test_polarity_ignores_stain_strength_and_penalizes_a_swap() -> None:
    pale = _rgb_from_he(0.40, 0.05).expand(1, 3, 16, 16)
    dark = _rgb_from_he(0.80, 0.10).expand(1, 3, 16, 16)
    swapped = _rgb_from_he(0.10, 0.80).expand(1, 3, 16, 16)
    pale_p = float(stain_polarity(pale).mean())
    dark_p = float(stain_polarity(dark).mean())
    swap_p = float(stain_polarity(swapped).mean())
    if pale_p < 0.7 or abs(pale_p - dark_p) > 0.05 or swap_p > -0.7:
        raise AssertionError(f"pale={pale_p} dark={dark_p} swap={swap_p}")
    white = torch.ones(1, 3, 16, 16)
    same, matched = structure_loss(dark, pale, sigma=0.0)
    opposite, swapped_loss = structure_loss(swapped, pale, sigma=0.0)
    if float(same) > 0.05 or float(matched) > 0.05:
        raise AssertionError(f"same occupancy={float(same)} polarity={float(matched)}")
    if float(opposite) > 0.05 or float(swapped_loss) < 0.5:
        raise AssertionError(f"swap occupancy={float(opposite)} polarity={float(swapped_loss)}")
    glass = structure_loss(dark, white, sigma=0.0)
    if float(glass[1]) > 1e-5:
        raise AssertionError(f"glass polarity={float(glass[1])}")


def test_occupancy_treats_either_stain_as_tissue() -> None:
    # Optical density 0.8 of hematoxylin, and of eosin. Both are clearly tissue.
    hematoxylin = torch.tensor([0.148, 0.800, 0.566]).view(1, 3, 1, 1).expand(1, 3, 16, 16)
    eosin = torch.tensor([0.106, -0.097, 0.251]).view(1, 3, 1, 1).expand(1, 3, 16, 16)
    white = torch.ones(1, 3, 16, 16)
    target = torch.ones(1, 1, 16, 16)
    hematoxylin_loss = float(occupancy_loss(hematoxylin, target, sigma=0.0))
    eosin_loss = float(occupancy_loss(eosin, target, sigma=0.0))
    white_loss = float(occupancy_loss(white, target, sigma=0.0))
    if hematoxylin_loss > 0.02 or eosin_loss > 0.02 or white_loss < 0.5:
        raise AssertionError(
            f"hematoxylin={hematoxylin_loss} eosin={eosin_loss} white={white_loss}"
        )
    if abs(hematoxylin_loss - eosin_loss) > 0.02:
        raise AssertionError(f"stain color changed occupancy {hematoxylin_loss} {eosin_loss}")


def test_occupancy_saturates_past_a_clear_stain() -> None:
    pale = torch.tensor([0.749, 0.832, 0.850], requires_grad=True).view(1, 3, 1, 1).expand(1, 3, 16, 16).clone()
    dark = torch.tensor([-0.674, -0.451, -0.327], requires_grad=True).view(1, 3, 1, 1).expand(1, 3, 16, 16).clone()
    pale.retain_grad()
    dark.retain_grad()
    target = torch.ones(1, 1, 16, 16)
    pale_loss = occupancy_loss(pale, target, sigma=0.0)
    dark_loss = occupancy_loss(dark, target, sigma=0.0)
    if float(dark_loss) > 0.02 or float(pale_loss) < 0.1:
        raise AssertionError(f"pale={float(pale_loss)} dark={float(dark_loss)}")
    pale_loss.backward()
    dark_loss.backward()
    pale_grad = float(pale.grad.abs().mean())
    dark_grad = float(dark.grad.abs().mean())
    if pale_grad < dark_grad * 20:
        raise AssertionError(f"pale_grad={pale_grad} dark_grad={dark_grad}")


def test_occupancy_ignores_a_boundary_wiggle() -> None:
    hematoxylin = torch.tensor([0.148, 0.800, 0.566]).view(1, 3, 1, 1)
    white = torch.ones(1, 3, 1, 1)
    straight = white.expand(1, 3, 64, 64).clone()
    straight[:, :, :, :32] = hematoxylin
    jagged = white.expand(1, 3, 64, 64).clone()
    color = hematoxylin.view(1, 3, 1, 1)
    for row in range(64):
        cut = 32 + (4 if (row // 2) % 2 == 0 else -4)
        jagged[:, :, row : row + 1, :cut] = color
    target = hed_tissue_prob(straight)
    sharp = float(occupancy_loss(jagged, target, sigma=0.0))
    soft = float(occupancy_loss(jagged, target, sigma=16.0))
    if soft > sharp * 0.5:
        raise AssertionError(f"sharp={sharp} soft={soft}")


def test_occupancy_trains_the_output_only() -> None:
    fake = torch.rand(1, 3, 32, 32, requires_grad=True)
    target = torch.rand(1, 1, 32, 32, requires_grad=True)
    loss = occupancy_loss(fake, target, sigma=4.0)
    loss.backward()
    if fake.grad is None or float(fake.grad.abs().sum()) == 0.0:
        raise AssertionError("generated image received no gradient")
    if target.grad is not None:
        raise AssertionError("tissue map received a gradient")


def test_occupancy_rejects_stain_where_af_is_empty() -> None:
    stained = torch.tensor([0.148, 0.800, 0.566]).view(1, 3, 1, 1).expand(1, 3, 32, 32)
    glass = torch.ones(1, 3, 32, 32)
    empty = tissue_probability(torch.full((1, 3, 32, 32), -1.0), glass, stained)
    present = tissue_probability(torch.full((1, 3, 32, 32), -0.6), glass, stained)
    empty_loss = float(occupancy_loss(stained, empty, sigma=0.0))
    present_loss = float(occupancy_loss(stained, present, sigma=0.0))
    if present_loss > 0.05 or empty_loss < 0.5:
        raise AssertionError(f"present={present_loss} empty={empty_loss}")


def test_exclusion_matches_accession_or_slide() -> None:
    keys = {"F25-06321-1"}
    if not is_excluded("20250514_F25-06321-1", keys):
        raise AssertionError("accession was not excluded")
    if is_excluded("20250514_F25-08325-1", keys):
        raise AssertionError("unlisted slide was excluded")
    if exclusion_keys(None):
        raise AssertionError("missing exclusion file should be empty")


def test_detail_ignores_glass() -> None:
    fake = torch.zeros(1, 3, 32, 32, requires_grad=True)
    teacher = torch.randn(1, 3, 32, 32)
    glass = torch.zeros(1, 1, 32, 32)
    if float(detail_loss(fake, teacher, glass, sigma=1.0)) != 0.0:
        raise AssertionError("glass detail loss should be zero")


def test_pyramid_is_zero_for_identical_images() -> None:
    image = torch.randn(2, 3, 64, 64)
    loss = GaussianPyramidLoss(num_high=5, weights=[0.1, 0.25, 0.5, 1.0, 1.0, 1.0])
    value = loss(image, image)
    if float(value) > 1e-6:
        raise AssertionError(f"pyramid {float(value)}")


def test_lsgan_target() -> None:
    prediction = [torch.ones(2, 1, 4, 4), torch.ones(2, 1, 2, 2)]
    if float(lsgan(prediction, True)) > 1e-6:
        raise AssertionError("LSGAN real target is not zero")
    if float(lsgan([torch.zeros(2, 1, 4, 4)], False)) > 1e-6:
        raise AssertionError("LSGAN fake target is not zero")
    # pix2pixHD adds the scale losses. Two maps of error 1 sum to 2.
    summed = float(lsgan([torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2)], True))
    if abs(summed - 2.0) > 1e-5:
        raise AssertionError(f"scale losses summed to {summed}")


def test_magnitude_wasserstein_ignores_shift() -> None:
    torch.manual_seed(1)
    source = torch.randn(2, 16, 24, 24)
    shifted = torch.roll(source, shifts=5, dims=-1)
    projection = torch.randn(8, 16, 5, 5)
    projection = projection / projection.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
    left = torch.fft.fftn(source, dim=(-2, -1)).abs()
    right = torch.fft.fftn(shifted, dim=(-2, -1)).abs()
    distance = sliced_wasserstein(left, right, projection)
    if float(distance.max()) > 1e-4:
        raise AssertionError(f"shift changed Fourier magnitude distance {float(distance.max())}")


def test_discriminator_returns_three_scales() -> None:
    disc = MultiScaleDiscriminator(num_scales=3)
    outputs = disc(torch.randn(1, 3, 128, 128))
    if len(outputs) != 3 or any(item.shape[1] != 1 for item in outputs):
        raise AssertionError(f"scales {[tuple(item.shape) for item in outputs]}")


def test_seam_windows_sit_on_the_quadrant_cuts() -> None:
    tiles = torch.arange(4, dtype=torch.float32).view(4, 1, 1, 1).expand(4, 1, 512, 512).contiguous()
    field = stitch_quadrants(tiles)
    vertical_top, vertical_bottom, horizontal_left, horizontal_right = seam_windows(field)
    if vertical_top.shape[-2:] != (512, 512):
        raise AssertionError(tuple(vertical_top.shape))
    checks = (
        ("vertical top", vertical_top[0, 0, 0, 0], vertical_top[0, 0, 0, 256], 0, 1),
        ("vertical bottom", vertical_bottom[0, 0, 0, 0], vertical_bottom[0, 0, 0, 256], 2, 3),
        ("horizontal left", horizontal_left[0, 0, 0, 0], horizontal_left[0, 0, 256, 0], 0, 2),
        ("horizontal right", horizontal_right[0, 0, 0, 0], horizontal_right[0, 0, 256, 0], 1, 3),
    )
    for name, left, right, expected_left, expected_right in checks:
        if float(left) != expected_left or float(right) != expected_right:
            raise AssertionError(f"{name} is {float(left)}, {float(right)}")


def test_encoder_feature_channels() -> None:
    model = Generator()
    image, feats = model(torch.randn(1, 3, 512, 512), torch.randn(1, 3, 1024, 1024), return_features=True)
    if image.shape != (1, 3, 512, 512):
        raise AssertionError(tuple(image.shape))
    channels = tuple(feat.shape[1] for feat in feats)
    if channels != FEATURE_CHANNELS:
        raise AssertionError(f"{channels} != {FEATURE_CHANNELS}")


if __name__ == "__main__":
    test_field_info_nce_prefers_the_stored_positive()
    test_retrieval_pair_weight_ramps_between_the_cuts()
    test_uni_pair_weight_drops_a_bad_retrieval()
    test_same_domain_diagonal_is_removed()
    test_patch_nce_matches_cut_and_prefers_the_same_patch()
    test_mdcl_prefers_the_matched_patch()
    test_asp_is_uniform_before_decay()
    test_content_gate_tracks_tissue_amount()
    test_masked_l1_uses_only_pseudo_tissue()
    test_tissue_vote_requires_af()
    test_tone_uses_pseudo_on_glass_and_teacher_on_tissue()
    test_stage2_recon_is_sharp_for_three_epochs_then_widens()
    test_recon_can_drop_a_benign_field()
    test_recon_uses_smoothed_teacher_on_tissue_and_pseudo_on_glass()
    test_recon_ignores_detail_finer_than_the_blur()
    test_recon_trains_the_output_and_not_the_teacher()
    test_recon_adds_weight_on_bright_af()
    test_hed_recon_ignores_stain_added_on_epoch10_glass()
    test_bright_af_keeps_a_light_stain_and_then_stops()
    test_bright_fill_borrows_nearby_eosin_when_epoch10_is_pale()
    test_recon_counts_bright_stained_af_harder()
    test_hed_recon_can_compare_eosin_only()
    test_stain_edge_matches_mass_boundaries_and_their_direction()
    test_polarity_ignores_stain_strength_and_penalizes_a_swap()
    test_occupancy_treats_either_stain_as_tissue()
    test_occupancy_saturates_past_a_clear_stain()
    test_occupancy_ignores_a_boundary_wiggle()
    test_occupancy_trains_the_output_only()
    test_occupancy_rejects_stain_where_af_is_empty()
    test_exclusion_matches_accession_or_slide()
    test_detail_ignores_glass()
    test_pyramid_is_zero_for_identical_images()
    test_lsgan_target()
    test_magnitude_wasserstein_ignores_shift()
    test_discriminator_returns_three_scales()
    test_seam_windows_sit_on_the_quadrant_cuts()
    test_encoder_feature_channels()
    print("ok")
