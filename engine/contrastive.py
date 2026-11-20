"""MDCL PatchNCE and the ASP reweighting of that same loss.

Official Mix-Domain Contrastive Learning. The positive is the patch at the
same location in the other domain. Cross-domain negatives are the other key
patches, with the positive diagonal set to -10. Same-domain negatives are the
other query patches, and that diagonal is removed. The query-query product is
not detached.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _drop_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    """Remove the diagonal of each (N, N) block. Official same-domain bank."""
    batch, count, width = matrix.shape
    if count != width or count < 2:
        raise ValueError(f"diagonal removal expects (B, N, N) with N>=2, got {tuple(matrix.shape)}")
    flat = matrix.reshape(batch, count * count)
    shifted = flat[:, 1:].reshape(batch, count - 1, count + 1)[:, :, :-1]
    return shifted.reshape(batch, count, count - 1)


def field_info_nce(
    query: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    temperature: float = 0.07,
    pair_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """InfoNCE for one UNI vector per field.

    ``query`` and ``positive`` are ``(B, D)``. ``negative`` is ``(B, K, D)``.
    Keys are detached. The similarity is computed in float32.
    """
    if query.ndim != 2 or positive.ndim != 2 or query.shape != positive.shape:
        raise ValueError(f"query {tuple(query.shape)} positive {tuple(positive.shape)}")
    if negative.ndim != 3 or negative.shape[0] != query.shape[0] or negative.shape[-1] != query.shape[-1]:
        raise ValueError(f"negative {tuple(negative.shape)} does not match query {tuple(query.shape)}")
    if float(temperature) <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    q = F.normalize(query.float(), dim=-1)
    k = F.normalize(positive.detach().float(), dim=-1)
    neg = F.normalize(negative.detach().float(), dim=-1)
    positive_logit = (q * k).sum(dim=-1, keepdim=True) / float(temperature)
    negative_logit = torch.matmul(q.unsqueeze(1), neg.transpose(1, 2)).squeeze(1) / float(temperature)
    logits = torch.cat((positive_logit, negative_logit), dim=1)
    target = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
    loss = F.cross_entropy(logits, target, reduction="none")
    if pair_weight is None:
        return loss.mean()
    weight = pair_weight.detach().float().reshape(-1)
    if weight.shape != loss.shape:
        raise ValueError(f"pair weight {tuple(weight.shape)} does not match loss {tuple(loss.shape)}")
    return (loss * weight).mean()


def retrieval_pair_weight(
    cosine: torch.Tensor,
    low: float = 0.3,
    high: float = 0.6,
) -> torch.Tensor:
    """0 at ``low``, 1 at ``high``. A stored teacher–H&E cosine, not the live query."""
    low = float(low)
    high = float(high)
    if high <= low:
        raise ValueError(f"cosine ramp needs high > low, got {low} to {high}")
    return ((cosine.detach().float() - low) / (high - low)).clamp(0.0, 1.0)


class DebiasedPatchNCESRC(nn.Module):
    """PatchNCE with soft same-tissue negatives removed, plus relation consistency.

    The relation term is Jensen-Shannon divergence between the projected
    patch-similarity distributions. It is bounded by ln 2. ``lambda_src``
    already scales that term.
    """

    def __init__(
        self,
        tau_nce: float = 0.07,
        fn_thresh: float = 0.80,
        fn_temp: float = 0.05,
        min_neg_weight: float = 1e-4,
        use_src: bool = True,
        tau_src: float = 1.0,
        lambda_src: float = 0.5,
    ):
        super().__init__()
        self.tau_nce = float(tau_nce)
        self.fn_thresh = float(fn_thresh)
        self.fn_temp = float(fn_temp)
        self.min_neg_weight = float(min_neg_weight)
        self.use_src = bool(use_src)
        self.tau_src = float(tau_src)
        self.lambda_src = float(lambda_src)

    def forward(
        self,
        feat_q: torch.Tensor,
        feat_k: torch.Tensor,
        raw_k: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # The AF encoder input is already detached. The projected key stays
        # connected so the AF scale and shift receive a gradient.
        query = feat_q.float()
        key = feat_k.float()
        if query.ndim != 3 or query.shape != key.shape:
            raise ValueError(f"query {tuple(query.shape)} key {tuple(key.shape)}")
        _batch, count, _dim = query.shape
        eye = torch.eye(count, device=query.device, dtype=torch.bool).unsqueeze(0)
        if raw_k is None:
            ref_raw = key.detach()
        else:
            ref_raw = raw_k.detach().float()
        # Center across the sampled patches so cosine measures a difference
        # inside the tile, not the shared channel mean.
        ref_centered = ref_raw - ref_raw.mean(dim=1, keepdim=True)
        reference = F.normalize(ref_centered, p=2, dim=-1, eps=1e-6)
        affinity = torch.bmm(reference, reference.transpose(1, 2))
        negative_weight = 1.0 - torch.sigmoid((affinity - self.fn_thresh) / self.fn_temp)
        logits = torch.bmm(query, key.transpose(1, 2)) / self.tau_nce
        positive = torch.diagonal(logits, dim1=1, dim2=2)
        weighted = torch.where(
            eye,
            logits,
            logits + torch.log(negative_weight.clamp(min=self.min_neg_weight)),
        )
        nce = (-positive + torch.logsumexp(weighted, dim=-1)).mean()
        off_diagonal = ~eye
        weight_mean = negative_weight.masked_select(off_diagonal.expand_as(negative_weight)).mean()
        if not self.use_src:
            return nce, logits.new_zeros(()), weight_mean.detach()
        # Both sides are uncentered unit-sphere projections. The key side is
        # detached, so this cannot divide by a vanishing centered norm.
        sim_key = torch.bmm(key.detach(), key.detach().transpose(1, 2)).masked_fill(eye, -1e9)
        sim_query = torch.bmm(query, query.transpose(1, 2)).masked_fill(eye, -1e9)
        prob_af = F.softmax(sim_key / self.tau_src, dim=-1)
        prob_he = F.softmax(sim_query / self.tau_src, dim=-1)
        mixture = (0.5 * (prob_af + prob_he)).clamp(min=1e-7)
        kl_af = (prob_af * (prob_af.clamp(min=1e-7).log() - mixture.log())).sum(dim=-1)
        kl_he = (prob_he * (prob_he.clamp(min=1e-7).log() - mixture.log())).sum(dim=-1)
        # Sum over the patch anchors and average over the batch, as in Hneg-SRC.
        relation = (0.5 * (kl_af + kl_he)).sum(dim=-1).mean()
        return nce, self.lambda_src * relation, weight_mean.detach()


def patch_nce(query: torch.Tensor, key: torch.Tensor, batch_size: int, temperature: float = 0.07) -> torch.Tensor:
    """CUT PatchNCE. Negatives are the other key patches only.

    The key is detached. The positive diagonal of the key bank is set to -10
    and kept, matching ``PatchNCELoss`` in the CUT training code. Same-domain
    query patches are not negatives.
    """
    if query.ndim != 2 or query.shape != key.shape:
        raise ValueError(f"query {tuple(query.shape)} key {tuple(key.shape)}")
    if query.shape[0] % int(batch_size) != 0:
        raise ValueError(f"{query.shape[0]} patches do not divide into batch {batch_size}")
    if float(temperature) <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    q = F.normalize(query.float(), dim=1, eps=1e-7)
    k = F.normalize(key.detach().float(), dim=1, eps=1e-7)
    count = q.shape[0] // int(batch_size)
    qv = q.view(int(batch_size), count, -1)
    kv = k.view(int(batch_size), count, -1)
    positive = torch.bmm(qv, kv.transpose(1, 2)).diagonal(dim1=1, dim2=2).reshape(-1, 1)
    eye = torch.eye(count, device=q.device, dtype=torch.bool)[None]
    negative = torch.bmm(qv, kv.transpose(1, 2)).masked_fill(eye, -10.0).reshape(-1, count)
    logits = torch.cat((positive, negative), dim=1) / float(temperature)
    target = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)
    return F.cross_entropy(logits, target, reduction="none")


def mdcl_nce(query: torch.Tensor, key: torch.Tensor, batch_size: int, temperature: float = 0.07) -> torch.Tensor:
    if query.ndim != 2 or query.shape != key.shape:
        raise ValueError(f"query {tuple(query.shape)} key {tuple(key.shape)}")
    if query.shape[0] % int(batch_size) != 0:
        raise ValueError(f"{query.shape[0]} patches do not divide into batch {batch_size}")
    if float(temperature) <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    q = F.normalize(query.float(), dim=1, eps=1e-7)
    k = F.normalize(key.float().detach(), dim=1, eps=1e-7)
    count = q.shape[0] // int(batch_size)
    qv = q.view(int(batch_size), count, -1)
    kv = k.view(int(batch_size), count, -1)
    positive = (qv * kv).sum(dim=-1).reshape(-1, 1)
    eye = torch.eye(count, device=q.device, dtype=torch.bool)[None]
    cross = torch.bmm(qv, kv.transpose(1, 2)).masked_fill(eye, -10.0)
    same = _drop_diagonal(torch.bmm(qv, qv.transpose(1, 2)))
    negative = torch.cat((cross, same), dim=2).reshape(-1, 2 * count - 1)
    logits = torch.cat((positive, negative), dim=1) / float(temperature)
    target = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)
    return F.cross_entropy(logits, target, reduction="none")


def asp_weights(
    positive: torch.Tensor,
    epoch: int,
    total_epochs: int,
    n_epochs_decay: int,
) -> torch.Tensor:
    """Official ASP ``lambda_linear`` weights. ``epoch`` is 1-based."""
    total = max(float(total_epochs), 1.0)
    t = max(0.0, float(epoch) - 1.0) / total
    k = 1.0 - float(n_epochs_decay) / total
    if t >= k:
        slope = 1.0 / max(1.0 - k, 1e-8)
        mix = slope - slope * t
    else:
        mix = 1.0
    learned = torch.relu(positive.detach().float())
    weights = mix + (1.0 - mix) * learned
    return weights / weights.sum().clamp_min(1e-8) * weights.numel()


def mdcl_asp(
    query: torch.Tensor,
    key: torch.Tensor,
    batch_size: int,
    temperature: float,
    epoch: int,
    total_epochs: int,
    n_epochs_decay: int,
) -> torch.Tensor:
    loss = mdcl_nce(query, key, batch_size, temperature)
    q = F.normalize(query.float(), dim=1, eps=1e-7)
    k = F.normalize(key.float().detach(), dim=1, eps=1e-7)
    positive = (q * k).sum(dim=-1)
    return loss * asp_weights(positive, epoch, total_epochs, n_epochs_decay)
