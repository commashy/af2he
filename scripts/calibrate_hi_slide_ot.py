#!/usr/bin/env python3
"""Print pair and slide match fractions on epoch-10 fields before a training launch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.stage1 import Stage1Dataset, collate_stage1, stitch_quadrants
from engine.hi_slide_ot import HiSlideOTLoss, SlideMedoidBankStore, build_shared_uni
from torch.utils.data import DataLoader
from train_stage1 import _diagnosis_table


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--bank_dir", required=True)
    parser.add_argument("--n_samples", type=int, default=500)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    labels = _diagnosis_table(config["slide_labels"])
    dataset = Stage1Dataset(
        config["af_root"], config.get("pseudo_root"), config["he_root"],
        config["alignment"], config.get("collage_index"),
        None, config.get("exclude_slides"), config.get("extra_slides"),
        None, config.get("epoch10_dir"),
        collage_size=int(config.get("collage_size", 1024)),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_stage1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss = HiSlideOTLoss(build_shared_uni(device), SlideMedoidBankStore(args.bank_dir)).to(device)
    seen = {"Benign": [], "Malignant": []}
    for index, batch in enumerate(loader):
        if index >= int(args.n_samples):
            break
        slide = batch["slide"][0]
        kind = "Benign" if labels.get(slide) == 0 else "Malignant"
        epoch10 = stitch_quadrants(batch["epoch10"].to(device))
        real = stitch_quadrants(batch["he"].to(device))
        _value, stats = loss(epoch10, epoch10, real, [slide], [kind == "Benign"])
        seen[kind].append(stats)
    for kind, rows in seen.items():
        if not rows:
            continue
        pair = sum(row["pair_frac"] for row in rows) / len(rows)
        slide = sum(row["slide_frac"] for row in rows) / len(rows)
        drop = sum(row["drop_frac"] for row in rows) / len(rows)
        peak = max(row["pi_row_max"] for row in rows)
        print(f"{kind} n={len(rows)} pair={pair:.3f} slide={slide:.3f} drop={drop:.3f} pi_row_max={peak:.3f}")


if __name__ == "__main__":
    main()
