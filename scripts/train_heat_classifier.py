"""Train a malignant-versus-other classifier.

The network and the loss are the MONAI Camelyon tumor detector: ImageNet
ResNet-18, one logit, ``BCEWithLogitsLoss``. The optimizer is the one already used to
fine-tune ResNet-18 here: AdamW, backbone 1e-4, new head 1e-3, weight decay
0.05, cosine over 10 epochs. Ten epochs cover every retained tile; the
thumbnail encoder's 30 epochs were for a few hundred slides, not this many
fields. Flips and quarter-turns are the thumbnail augmentations. Brightness
and the rectangle mask stay off, because those were for AF thumbnails.

The task is malignant versus other. A mapped tile is malignant when its saved
mean is at least 0.60, the start of yellow. Blue, green, and the middle stay
in the other class, as do tissue tiles from a benign slide. That mean is
already in the CSV, so training does not read the heatmap patches again.

Tiles with tissue fraction below 0.15 are dropped. Each epoch draws half
malignant and half other. Sixteen slides of each kind are held out. The
checkpoint is the best balanced accuracy on those slides.

Do not start this while stage 2 is using the four GPUs. One card at batch 4
of 1024 is the fit; four cards are the same launch as the thumbnail encoder.

    torchrun --nproc_per_node=4 --master_port=29538 scripts/train_heat_classifier.py
"""

from __future__ import annotations

import csv
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.heat_classifier import HeatClassifier

TABLE = ROOT / "manifests/heat_patch_targets.csv"
CACHE = ROOT / "manifests/heat_classifier_rows.csv"
PATCHES = Path("/home/taek/af2he_heat_tiles")
MALIGNANT_MIN = 0.60
OUT = ROOT / "checkpoints/heat_classifier"
LOG = ROOT / "logs/heat_classifier.log"
EPOCHS = 10
BACKBONE_LR = 1e-4
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.05
BATCH = 32
LOG_EVERY = 20
SEED = 0
TISSUE_MIN = 0.15
HOLD = 16


def say(text: str) -> None:
    print(text, flush=True)


def load_rows() -> list[dict]:
    rows = []
    with TABLE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            tissue = float(row["tissue"])
            if tissue < TISSUE_MIN:
                continue
            source = row["source"]
            if source == "benign":
                target = 0.0
            else:
                target = 1.0 if float(row["target"]) >= MALIGNANT_MIN else 0.0
            rows.append({
                "name": row["filename"],
                "slide": row["slide"],
                "target": target,
                "source": source,
            })
    if not rows:
        raise RuntimeError(f"{TABLE} has no tissue tiles")
    return rows


def write_cache(rows: list[dict]) -> None:
    with CACHE.open("w", encoding="utf-8") as handle:
        handle.write("name,slide,target,source\n")
        for row in rows:
            handle.write(f"{row['name']},{row['slide']},{row['target']:.6f},{row['source']}\n")


def read_cache() -> list[dict]:
    rows = []
    with CACHE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "name": row["name"],
                "slide": row["slide"],
                "target": float(row["target"]),
                "source": row["source"],
            })
    if not rows:
        raise RuntimeError(f"{CACHE} is empty")
    return rows


def write_cache(rows: list[dict]) -> None:
    with CACHE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("name", "slide", "target", "source"))
        writer.writeheader()
        writer.writerows(rows)


def read_cache() -> list[dict]:
    rows = []
    with CACHE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["target"] = float(row["target"])
            rows.append(row)
    return rows


def holdout(rows: list[dict]) -> set[str]:
    generator = random.Random(SEED)
    benign = sorted({row["slide"] for row in rows if row["source"] == "benign"})
    mapped = sorted({row["slide"] for row in rows if row["source"] == "map"})
    if len(benign) < HOLD or len(mapped) < HOLD:
        raise RuntimeError(f"need {HOLD} benign and {HOLD} mapped slides, got {len(benign)} and {len(mapped)}")
    chosen = generator.sample(benign, HOLD) + generator.sample(mapped, HOLD)
    return set(chosen)


def load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    tensor = torch.from_numpy(array).permute(2, 0, 1).float()
    return tensor.mul_(2.0 / 255.0).sub_(1.0)


class TileSet(Dataset):
    def __init__(self, rows: list[dict], indices: list[int], augment: bool):
        self.rows = rows
        self.indices = indices
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[self.indices[index]]
        image = load_rgb(PATCHES / row["name"])
        if self.augment:
            if random.random() < 0.5:
                image = image.flip(-1)
            if random.random() < 0.5:
                image = image.flip(-2)
            image = torch.rot90(image, random.randrange(4), (1, 2))
        return {"image": image, "target": row["target"]}


def collate(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "target": torch.tensor([item["target"] for item in batch], dtype=torch.float32),
    }


def balanced_indices(rows: list[dict], epoch: int, rank: int, world: int) -> list[int]:
    payload: list[list[int] | None] = [None]
    if rank == 0:
        low = [i for i, row in enumerate(rows) if row["target"] < 0.5]
        high = [i for i, row in enumerate(rows) if row["target"] >= 0.5]
        if not low or not high:
            raise RuntimeError(f"balanced split is empty: other {len(low)} malignant {len(high)}")
        generator = random.Random(SEED + epoch)

        def draw() -> int:
            pool = low if generator.randrange(2) == 0 else high
            return generator.choice(pool)

        chosen = [draw() for _ in range(len(rows))]
        generator.shuffle(chosen)
        while len(chosen) % world:
            chosen.append(draw())
        payload[0] = chosen
    dist.broadcast_object_list(payload, src=0)
    chosen = payload[0]
    if chosen is None:
        raise RuntimeError("the balanced index list was not broadcast")
    return chosen[rank::world]


@torch.no_grad()
def evaluate(model: HeatClassifier, rows: list[dict], device: torch.device) -> dict:
    model.eval()
    loader = DataLoader(
        TileSet(rows, list(range(len(rows))), augment=False),
        batch_size=BATCH,
        shuffle=False,
        num_workers=2,
        collate_fn=collate,
    )
    malignant_correct = 0
    malignant_total = 0
    other_correct = 0
    other_total = 0
    for batch in loader:
        call = torch.sigmoid(model(batch["image"].to(device))).float().cpu() >= 0.5
        target = batch["target"] >= 0.5
        malignant_total += int(target.sum())
        other_total += int((~target).sum())
        malignant_correct += int((call & target).sum())
        other_correct += int((~call & ~target).sum())
    malignant_recall = malignant_correct / malignant_total if malignant_total else 0.0
    other_recall = other_correct / other_total if other_total else 0.0
    return {
        "balanced": 0.5 * (malignant_recall + other_recall),
        "malignant": malignant_correct,
        "malignant_n": malignant_total,
        "other": other_correct,
        "other_n": other_total,
        "n": malignant_total + other_total,
    }


def main() -> None:
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", rank))
    device = torch.device("cuda", rank)
    random.seed(SEED + rank)
    torch.manual_seed(SEED)

    if rank == 0:
        rows = load_rows()
        write_cache(rows)
    dist.barrier()
    rows = read_cache()
    held = holdout(rows)
    train_rows = [row for row in rows if row["slide"] not in held]
    val_rows = [row for row in rows if row["slide"] in held]
    if rank == 0:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        LOG.write_text("")
        say(
            f"gpus {world}  batch/gpu {BATCH}  epochs {EPOCHS}  "
            f"train {len(train_rows)}  val {len(val_rows)}  held slides {len(held)}"
        )
        say(
            "task malignant versus other  mean at least yellow is malignant  "
            "loss BCEWithLogits  adamw 1e-4/1e-3  wd 0.05  cosine  aug flip rot90"
        )

    model = HeatClassifier().to(device)
    raw = model
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(
        [
            {"params": raw.net.fc.parameters(), "lr": HEAD_LR},
            {"params": [p for name, p in raw.net.named_parameters() if not name.startswith("fc")], "lr": BACKBONE_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best = None
    for epoch in range(1, EPOCHS + 1):
        indices = balanced_indices(train_rows, epoch, rank, world)
        loader = DataLoader(
            TileSet(train_rows, indices, augment=True),
            batch_size=BATCH,
            shuffle=False,
            num_workers=4,
            collate_fn=collate,
        )
        model.train()
        running = torch.zeros((), device=device)
        seen = torch.zeros((), device=device)
        window = 0.0
        window_n = 0
        started = time.time()
        for step, batch in enumerate(loader, start=1):
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            loss = F.binary_cross_entropy_with_logits(model(image), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int(image.shape[0])
            value = float(loss.detach())
            running += loss.detach() * count
            seen += count
            window += value * count
            window_n += count
            if rank == 0 and step % LOG_EVERY == 0:
                elapsed = time.time() - started
                say(
                    f"epoch {epoch} step {step}/{len(loader)}  "
                    f"loss {window / window_n:.4f}  "
                    f"{window_n * world / elapsed:.1f} img/s"
                )
                window = 0.0
                window_n = 0
                started = time.time()
        dist.all_reduce(running)
        dist.all_reduce(seen)
        schedule.step()
        if rank == 0:
            metrics = evaluate(raw, val_rows, device)
            say(
                f"epoch {epoch}  train {float(running / seen):.4f}  "
                f"val_balanced {metrics['balanced']:.3f}  "
                f"malignant {metrics['malignant']}/{metrics['malignant_n']}  "
                f"other {metrics['other']}/{metrics['other_n']}"
            )
            OUT.mkdir(parents=True, exist_ok=True)
            state = {"model": raw.state_dict(), "epoch": epoch, "val_balanced": metrics["balanced"]}
            torch.save(state, OUT / "last.pt")
            if best is None or metrics["balanced"] > best:
                best = metrics["balanced"]
                torch.save(state, OUT / "best.pt")
                say(f"saved best  val_balanced {best:.3f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
