"""Closed FiLM slide conditioning grafts onto an unconditional discriminator."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.discriminator import (
    MultiScaleDiscriminator,
    backbone_parameters,
    load_discriminator_state,
    slide_classification_loss,
)


def test_closed_gate_matches_the_unconditional_checkpoint() -> None:
    torch.manual_seed(0)
    old = MultiScaleDiscriminator(num_scales=3)
    new = MultiScaleDiscriminator(num_scales=3, slide_count=4)
    grafted = load_discriminator_state(new, old.state_dict())
    if not grafted:
        raise AssertionError("unconditional checkpoint should leave FiLM uninitialized")
    if new.film_alpha() != 0.0:
        raise AssertionError(new.film_alpha())
    image = torch.randn(2, 3, 128, 128)
    ids = torch.tensor([0, 3])
    other = torch.tensor([1, 2])
    old.eval()
    new.eval()
    with torch.no_grad():
        plain = old(image)
        closed = new(image, ids)
        closed_other = new(image, other)
    for left, right, third in zip(plain, closed, closed_other):
        if not torch.allclose(left, right, atol=1e-5):
            raise AssertionError(float((left - right).abs().max()))
        if not torch.allclose(right, third, atol=1e-5):
            raise AssertionError("closed gate changed with the slide id")


def test_open_gate_conditions_only_the_coarsest_scale() -> None:
    torch.manual_seed(1)
    net = MultiScaleDiscriminator(num_scales=3, slide_count=4)
    if list(net.slide_films) != ["2"]:
        raise AssertionError(f"FiLM scales {list(net.slide_films)}")
    with torch.no_grad():
        for films in net.slide_films.values():
            for film in films:
                film.alpha.fill_(1.0)
    image = torch.randn(2, 3, 128, 128)
    net.eval()
    with torch.no_grad():
        first = net(image, torch.tensor([0, 0]))
        second = net(image, torch.tensor([1, 2]))
    for index, (left, right) in enumerate(zip(first, second)):
        changed = not torch.allclose(left, right, atol=1e-4)
        if index < 2 and changed:
            raise AssertionError(f"finer scale {index} read the slide id")
        if index == 2 and not changed:
            raise AssertionError("coarsest scale did not read the slide id")


def test_adam_moments_survive_the_graft() -> None:
    torch.manual_seed(2)
    old = MultiScaleDiscriminator(num_scales=3)
    image = torch.randn(1, 3, 256, 256)
    old.train()
    opt = torch.optim.Adam(old.parameters(), lr=1e-4, betas=(0.5, 0.999))
    old(image)[0].sum().backward()
    opt.step()
    saved = opt.state_dict()
    moment = opt.state[next(old.parameters())]["exp_avg"].clone()

    new = MultiScaleDiscriminator(num_scales=3, slide_count=4)
    load_discriminator_state(new, old.state_dict())
    held = torch.optim.Adam(backbone_parameters(new), lr=1e-4, betas=(0.5, 0.999))
    held.load_state_dict(saved)
    restored = held.state[next(new.parameters())]["exp_avg"]
    if not torch.allclose(restored, moment):
        raise AssertionError("backbone Adam moment was not restored")


def test_slide_classes_graft_and_skip_glass() -> None:
    torch.manual_seed(3)
    old = MultiScaleDiscriminator(num_scales=3)
    new = MultiScaleDiscriminator(num_scales=3, slide_classes=4)
    if not load_discriminator_state(new, old.state_dict()):
        raise AssertionError("class heads should be the only new keys")
    image = torch.randn(2, 3, 128, 128)
    old.eval()
    new.eval()
    with torch.no_grad():
        plain = old(image)
        logits, classes = new(image)
    if len(classes) != 3 or any(item.shape[1] != 4 for item in classes):
        raise AssertionError("expected a 4-way head on every scale")
    for left, right in zip(plain, logits):
        if not torch.allclose(left, right, atol=1e-5):
            raise AssertionError("class head changed the real-versus-fake logit")
    glass = slide_classification_loss(classes, torch.tensor([0, 1]), torch.zeros(2, 1, 128, 128))
    if float(glass) != 0.0:
        raise AssertionError(f"glass loss {float(glass)}")
    field = torch.randn(1, 3, 128, 128, requires_grad=True)
    new.train()
    for parameter in new.parameters():
        parameter.requires_grad_(False)
    _logits, live = new(field)
    loss = slide_classification_loss(live, torch.tensor([2]), torch.ones(1, 1, 128, 128))
    loss.backward()
    if field.grad is None or float(field.grad.abs().sum()) == 0.0:
        raise AssertionError("generated field received no classification gradient")


if __name__ == "__main__":
    test_closed_gate_matches_the_unconditional_checkpoint()
    test_open_gate_conditions_only_the_coarsest_scale()
    test_adam_moments_survive_the_graft()
    test_slide_classes_graft_and_skip_glass()
    print("ok")
