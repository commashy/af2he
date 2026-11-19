"""Epoch-10-routed set match from generated patches to real H&E.

Stage A sends each foreground token to the paired H&E tile or to a dustbin.
Leftover mass goes to 256 real medoids from the same slide. The plan is built
from the epoch-10 field and carries no gradient. Only the generated tokens are
pulled toward the chosen real tokens.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.image_losses import hed_ruifrok, hed_tissue_mask


def field_tanh_to_512_rgb01(field: torch.Tensor) -> torch.Tensor:
    """Map a tanh field to RGB in [0, 1] at 512, clamped again after bicubic."""
    rgb = ((field.float() + 1.0) * 0.5).clamp(0.0, 1.0)
    rgb = F.interpolate(
        rgb, size=(512, 512), mode="bicubic", align_corners=False, antialias=True,
    )
    return rgb.clamp(0.0, 1.0)


def _tanh_from_rgb01(rgb01: torch.Tensor) -> torch.Tensor:
    return rgb01.mul(2.0).sub(1.0)


def tissue_token_mask_32x32(field_tanh: torch.Tensor, min_tissue_frac: float = 0.25) -> torch.Tensor:
    """True for 16-pixel cells on the 512 canvas whose 1024 tissue fraction is high enough.

    ``field_tanh`` is a 1024 field in [-1, 1]. ``hed_tissue_mask`` expects that range.
    """
    if field_tanh.shape[-2:] != (1024, 1024):
        raise ValueError(f"tissue mask expects 1024, got {tuple(field_tanh.shape)}")
    pix_mask = hed_tissue_mask(field_tanh).float()
    patch_frac = F.avg_pool2d(pix_mask, kernel_size=32, stride=32)
    return patch_frac.flatten(1) >= float(min_tissue_frac)


def ruifrok_h_od_tokens_512(rgb01_512: torch.Tensor) -> torch.Tensor:
    """Mean Ruifrok hematoxylin optical density on each 16-pixel cell. Shape (N, 1024)."""
    hed = hed_ruifrok(_tanh_from_rgb01(rgb01_512))
    return F.avg_pool2d(hed[:, 0:1], kernel_size=16, stride=16).flatten(1)


def build_shared_uni(device: torch.device) -> nn.Module:
    """One frozen UNI. It is not wrapped in DDP."""
    import timm

    uni = timm.create_model(
        "hf-hub:MahmoodLab/uni",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True,
        num_classes=0,
    ).eval()
    for parameter in uni.parameters():
        parameter.requires_grad_(False)
    return uni.to(device)


class SlideMedoidBankStore:
    """fp16 banks shaped (256, 1024). A short slide is zero-padded, never repeated."""

    def __init__(self, bank_dir: str | Path) -> None:
        self.bank_dir = Path(bank_dir)
        if not self.bank_dir.is_dir():
            raise FileNotFoundError(f"HiSlideOT bank directory does not exist: {self.bank_dir}")
        self._banks: dict[str, torch.Tensor] = {}
        for path in sorted(self.bank_dir.glob("*.pt")):
            tensor = torch.load(path, map_location="cpu", weights_only=True)
            if tuple(tensor.shape) != (256, 1024):
                raise ValueError(f"{path} has shape {tuple(tensor.shape)}, expected (256, 1024)")
            self._banks[path.stem] = tensor.to(dtype=torch.float16).contiguous()
        if not self._banks:
            raise FileNotFoundError(f"no slide banks in {self.bank_dir}")

    def get(self, slide_id: str, device: torch.device) -> torch.Tensor:
        if slide_id not in self._banks:
            raise KeyError(f"no HiSlideOT bank for {slide_id} in {self.bank_dir}")
        return self._banks[slide_id].to(device=device, dtype=torch.float32, non_blocking=True)


def _sinkhorn_pair(
    cost: torch.Tensor,
    tau: float,
    epsilon: float,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Partial plan from queries to paired tokens. Returns the plan and unclamped row sums."""
    n_g, n_p = cost.shape
    table = cost.new_zeros((n_g + 1, n_p + 1))
    table[:n_g, :n_p] = cost
    table[:n_g, n_p] = float(tau)
    supply = cost.new_ones(n_g + 1)
    supply[n_g] = float(n_p)
    demand = cost.new_ones(n_p + 1)
    demand[n_p] = float(n_g)
    log_supply = supply.log()
    log_demand = demand.log()
    score = -table / float(epsilon)
    row = torch.zeros_like(log_supply)
    col = torch.zeros_like(log_demand)
    for _ in range(int(iterations)):
        row = log_supply - torch.logsumexp(score + col.unsqueeze(0), dim=1)
        col = log_demand - torch.logsumexp(score + row.unsqueeze(1), dim=0)
    plan = torch.exp(score + row.unsqueeze(1) + col.unsqueeze(0))[:n_g, :n_p]
    return plan, plan.sum(dim=1)


def _barycenter(weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return F.normalize(weights @ values, dim=-1, eps=1e-8)


def route_queries(
    query: torch.Tensor,
    paired: torch.Tensor | None,
    medoids: torch.Tensor,
    benign: bool,
    tau_pair: float = 0.28,
    eps_pair: float = 0.05,
    sinkhorn_iters: int = 6,
    tau_slide_malig: float = 0.35,
    tau_slide_benign: float = 0.55,
    eps_slide: float = 0.02,
) -> dict[str, torch.Tensor]:
    """No-gradient route. ``query`` and the targets are L2-normalized."""
    n_g = query.shape[0]
    valid = medoids.norm(dim=-1) > 0.5
    if not bool(valid.any()):
        raise RuntimeError("slide bank has no real medoids")
    bank = F.normalize(medoids, dim=-1, eps=1e-8)
    if paired is not None and paired.shape[0] > 0:
        cost = 1.0 - (query @ paired.t())
        plan, raw_row = _sinkhorn_pair(cost, tau_pair, eps_pair, sinkhorn_iters)
        paired_weight = raw_row.clamp(0.0, 1.0)
        paired_target = _barycenter(plan, paired)
        row_max = raw_row.max()
    else:
        paired_weight = query.new_zeros(n_g)
        paired_target = query.new_zeros((n_g, query.shape[-1]))
        row_max = query.new_zeros(())
    residual = 1.0 - paired_weight
    slide_cost = 1.0 - (query @ bank.t())
    tau = float(tau_slide_benign if benign else tau_slide_malig)
    medoid_logit = (-slide_cost / float(eps_slide)).masked_fill(~valid.unsqueeze(0), -1.0e9)
    dustbin = medoid_logit.new_full((n_g, 1), -tau / float(eps_slide))
    probability = torch.softmax(torch.cat((medoid_logit, dustbin), dim=1), dim=1)
    medoid_probability = probability[:, :-1]
    drop_probability = probability[:, -1]
    slide_weight = (residual * medoid_probability.sum(dim=1)).clamp(0.0, 1.0)
    drop_weight = (residual * drop_probability).clamp(0.0, 1.0)
    slide_target = _barycenter(medoid_probability, bank)
    used = (residual.unsqueeze(1) * medoid_probability).sum(dim=0) > 1.0e-2
    return {
        "paired_weight": paired_weight,
        "paired_target": paired_target,
        "slide_weight": slide_weight,
        "slide_target": slide_target,
        "drop_weight": drop_weight,
        "row_max": row_max,
        "n_medoids_used": used.sum(),
    }


class HiSlideOTLoss(nn.Module):
    """Pull generated UNI tokens toward real H&E tokens chosen from epoch 10."""

    def __init__(
        self,
        uni_model: nn.Module,
        bank_store: SlideMedoidBankStore,
        tau_pair: float = 0.28,
        eps_pair: float = 0.05,
        sinkhorn_iters: int = 6,
        tau_slide_malig: float = 0.35,
        tau_slide_benign: float = 0.55,
        eps_slide: float = 0.02,
    ) -> None:
        super().__init__()
        self.uni = uni_model
        self.bank_store = bank_store
        self.tau_pair = float(tau_pair)
        self.eps_pair = float(eps_pair)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.tau_slide_malig = float(tau_slide_malig)
        self.tau_slide_benign = float(tau_slide_benign)
        self.eps_slide = float(eps_slide)
        self.register_buffer(
            "uni_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False,
        )
        self.register_buffer(
            "uni_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False,
        )

    def _normalize_512(self, field_1024: torch.Tensor) -> torch.Tensor:
        rgb = field_tanh_to_512_rgb01(field_1024)
        return (rgb - self.uni_mean) / self.uni_std

    def _tokens(self, normalized: torch.Tensor, checkpoint: bool) -> torch.Tensor:
        self.uni.eval()
        if hasattr(self.uni, "set_grad_checkpointing"):
            self.uni.set_grad_checkpointing(checkpoint)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=normalized.is_cuda):
            sequence = self.uni.forward_features(normalized)
        if hasattr(self.uni, "set_grad_checkpointing"):
            self.uni.set_grad_checkpointing(False)
        spatial = sequence[:, self.uni.num_prefix_tokens:, :].float()
        if spatial.shape[1] != 1024:
            raise RuntimeError(
                f"UNI returned {spatial.shape[1]} tokens, expected 1024. "
                "Load it with dynamic_img_size=True."
            )
        return F.normalize(spatial, dim=-1, eps=1e-8)

    def forward(
        self,
        generated: torch.Tensor,
        epoch10: torch.Tensor,
        paired: torch.Tensor,
        slide_ids: Sequence[str],
        benign: Sequence[bool],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if generated.shape[0] != len(slide_ids) or len(slide_ids) != len(benign):
            raise ValueError("each 1024 field needs one slide id and one benign flag")
        with torch.no_grad():
            mask_generated = tissue_token_mask_32x32(epoch10)
            mask_paired = tissue_token_mask_32x32(paired)
            reference = self._tokens(
                torch.cat((self._normalize_512(epoch10), self._normalize_512(paired)), dim=0),
                checkpoint=False,
            )
            epoch_tokens, pair_tokens = reference.chunk(2, dim=0)
        live = self._tokens(self._normalize_512(generated), checkpoint=generated.requires_grad)
        losses = []
        totals = {name: [] for name in (
            "pair_frac", "slide_frac", "drop_frac", "cos_dist_gen", "n_medoids_used", "pi_row_max", "n_g",
        )}
        for index in range(generated.shape[0]):
            keep_g = mask_generated[index]
            keep_p = mask_paired[index]
            n_g = int(keep_g.sum().item())
            totals["n_g"].append(float(n_g))
            if n_g == 0:
                losses.append(generated[index].sum() * 0.0)
                continue
            query_live = live[index][keep_g]
            query_epoch = epoch_tokens[index][keep_g]
            destination = pair_tokens[index][keep_p] if int(keep_p.sum().item()) else None
            bank = self.bank_store.get(slide_ids[index], query_epoch.device)
            with torch.no_grad():
                route = route_queries(
                    query_epoch, destination, bank, bool(benign[index]),
                    tau_pair=self.tau_pair, eps_pair=self.eps_pair,
                    sinkhorn_iters=self.sinkhorn_iters,
                    tau_slide_malig=self.tau_slide_malig, tau_slide_benign=self.tau_slide_benign,
                    eps_slide=self.eps_slide,
                )
            distance_pair = 1.0 - (query_live * route["paired_target"]).sum(dim=-1)
            distance_slide = 1.0 - (query_live * route["slide_target"]).sum(dim=-1)
            numerator = (
                route["paired_weight"] * distance_pair + route["slide_weight"] * distance_slide
            ).sum()
            losses.append(numerator / float(n_g))
            matched = (route["paired_weight"] + route["slide_weight"]).sum().clamp(min=1e-6)
            mean_distance = (
                route["paired_weight"] * distance_pair + route["slide_weight"] * distance_slide
            ).sum() / matched
            totals["pair_frac"].append(float(route["paired_weight"].mean()))
            totals["slide_frac"].append(float(route["slide_weight"].mean()))
            totals["drop_frac"].append(float(route["drop_weight"].mean()))
            totals["cos_dist_gen"].append(float(mean_distance.detach()))
            totals["n_medoids_used"].append(float(route["n_medoids_used"]))
            totals["pi_row_max"].append(float(route["row_max"]))
        loss = torch.stack(losses).mean()
        stats = {name: (sum(values) / len(values) if values else 0.0) for name, values in totals.items()}
        return loss, stats
