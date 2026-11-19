#!/usr/bin/env python3
"""Build one 256-medoid UNI bank per training slide.

Pass 1 measures the 98.5th percentile of Ruifrok hematoxylin on benign
foreground tokens. Pass 2 drops benign tokens above that cut, then keeps the
nearest real token to each spherical cluster. Short slides are zero-padded to
256. Medoids are not repeated.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import load_native_image
from data.stage1 import Stage1Dataset
from engine.hi_slide_ot import (
    build_shared_uni,
    field_tanh_to_512_rgb01,
    ruifrok_h_od_tokens_512,
    tissue_token_mask_32x32,
)
from train_stage1 import _diagnosis_table


def spherical_medoids(
    tokens: torch.Tensor,
    k: int = 256,
    iterations: int = 25,
    max_pool: int = 50_000,
    seed: int = 42,
) -> tuple[torch.Tensor, int]:
    """Cluster L2-normalized tokens and replace each mean by a real member."""
    count, dim = tokens.shape
    bank = torch.zeros((k, dim), dtype=torch.float16)
    if count == 0:
        return bank, 0
    generator = torch.Generator(device=tokens.device)
    generator.manual_seed(int(seed))
    if count > max_pool:
        choice = torch.randperm(count, generator=generator, device=tokens.device)[:max_pool]
        samples = tokens[choice]
    else:
        samples = tokens
    n = samples.shape[0]
    if n <= k:
        bank[:n] = F.normalize(samples, dim=-1).to(dtype=torch.float16).cpu()
        return bank, n
    centers = torch.empty((k, dim), device=samples.device, dtype=torch.float32)
    first = int(torch.randint(0, n, (1,), generator=generator, device=samples.device).item())
    centers[0] = samples[first]
    nearest = (1.0 - (samples @ centers[0])).clamp_min(0.0)
    for center_index in range(1, k):
        weight = nearest.square()
        total = float(weight.sum())
        if total <= 1e-12:
            pick = int(torch.randint(0, n, (1,), generator=generator, device=samples.device).item())
        else:
            pick = int(torch.multinomial(weight / total, 1).item())
        centers[center_index] = samples[pick]
        nearest = torch.minimum(nearest, (1.0 - (samples @ centers[center_index])).clamp_min(0.0))
    labels = torch.zeros(n, dtype=torch.long, device=samples.device)
    for _ in range(int(iterations)):
        similarity = samples @ centers.t()
        labels = similarity.argmax(dim=1)
        summed = torch.zeros_like(centers)
        summed.index_add_(0, labels, samples)
        counts = torch.bincount(labels, minlength=k)
        empty = counts == 0
        if bool(empty.any()):
            furthest = similarity.max(dim=1).values.argsort()[: int(empty.sum())]
            summed[empty] = samples[furthest]
        centers = F.normalize(summed, dim=-1, eps=1e-8)
    similarity = samples @ centers.t()
    labels = similarity.argmax(dim=1)
    snapped = []
    for cluster in range(k):
        members = samples[labels == cluster]
        if members.shape[0] == 0:
            continue
        mean = F.normalize(members.mean(dim=0), dim=0, eps=1e-8)
        snapped.append(members[int(torch.argmax(members @ mean))])
    if not snapped:
        return bank, 0
    chosen = torch.stack(snapped, dim=0)
    bank[: chosen.shape[0]] = chosen.to(dtype=torch.float16).cpu()
    return bank, int(chosen.shape[0])


def _dataset(config: dict) -> tuple[Stage1Dataset, dict[str, int]]:
    labels = _diagnosis_table(config["slide_labels"])
    dataset = Stage1Dataset(
        config["af_root"], config.get("pseudo_root"), config["he_root"],
        config["alignment"], config.get("collage_index"),
        None, config.get("exclude_slides"), config.get("extra_slides"),
        None, config.get("epoch10_dir"),
        collage_size=int(config.get("collage_size", 1024)),
    )
    return dataset, labels


def _groups(dataset: Stage1Dataset) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for af_path, _pseudo, he_path, _collage in dataset.samples:
        grouped[af_path.name.split("__", 1)[0]].append(he_path)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--pass1_tiles", type=int, default=32)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset, labels = _dataset(config)
    grouped = _groups(dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    benign_values = []
    seen = 0
    for slide, paths in grouped.items():
        if labels.get(slide) != 0:
            continue
        for path in paths[: int(args.pass1_tiles)]:
            field = load_native_image(path).unsqueeze(0).to(device)
            mask = tissue_token_mask_32x32(field)
            optical = ruifrok_h_od_tokens_512(field_tanh_to_512_rgb01(field))
            benign_values.append(optical[mask].detach().cpu())
            seen += 1
    if not benign_values:
        raise RuntimeError("no benign foreground tokens for the hematoxylin cut")
    pooled = torch.cat(benign_values, dim=0).float()
    cutoff = float(torch.quantile(pooled, 0.985))
    print(f"benign hematoxylin p98.5={cutoff:.4f} tokens={pooled.numel()} tiles={seen}", flush=True)

    uni = build_shared_uni(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    meta = {"h_od_p98_5": cutoff, "k": 256, "split": "none", "slides": {}}
    for slide, paths in grouped.items():
        benign = labels.get(slide) == 0
        collected = []
        for start in range(0, len(paths), int(args.batch_size)):
            batch = torch.stack([load_native_image(path) for path in paths[start:start + int(args.batch_size)]])
            batch = batch.to(device)
            keep = tissue_token_mask_32x32(batch)
            rgb = field_tanh_to_512_rgb01(batch)
            if benign:
                keep = keep & (ruifrok_h_od_tokens_512(rgb) <= cutoff)
            if not bool(keep.any()):
                continue
            normalized = (rgb - mean) / std
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                sequence = uni.forward_features(normalized)
            tokens = F.normalize(sequence[:, uni.num_prefix_tokens:, :].float(), dim=-1, eps=1e-8)
            collected.append(tokens[keep].detach())
        if not collected:
            raise RuntimeError(f"{slide} has no foreground tokens after the hematoxylin cut")
        bank, valid = spherical_medoids(torch.cat(collected, dim=0))
        torch.save(bank, out_dir / f"{slide}.pt")
        meta["slides"][slide] = {
            "is_benign": benign,
            "n_tokens": int(sum(item.shape[0] for item in collected)),
            "n_valid_medoids": int(valid),
        }
        print(f"{slide} medoids={valid}", flush=True)
    (out_dir / "bank_meta.json").write_text(json.dumps(meta), encoding="utf-8")


if __name__ == "__main__":
    main()
