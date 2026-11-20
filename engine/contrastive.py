"""MDCL PatchNCE and the ASP reweighting of that same loss.

Official Mix-Domain Contrastive Learning. The positive is the patch at the
same location in the other domain. Cross-domain negatives are the other key
patches, with the positive diagonal set to -10. Same-domain negatives are the
other query patches, and that diagonal is removed. The query-query product is
not detached.
"""

from __future__ import annotations

import torch
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
