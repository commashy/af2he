"""Train the thumbnail encoder on four GPUs.

Every labeled lung thumbnail is used, including poor fixation and poor H&E.
Slides with an unknown label, an empty malignant subtype, or another organ stay out.
The checkpoint is the best benign/malignant balance on 32 held-out slides.

    torchrun --nproc_per_node=4 --master_port=29533 scripts/train_thumb_encoder.py
"""

from __future__ import annotations

import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.thumb_encoder import ThumbEncoder

TRAIN_CSV = ROOT / "manifests/training_wsi.csv"
LABELS = Path("/mnt/10tb_hdd/taek/stain_translation_core/manifests/fresh/train_slide_labels.csv")
EXCLUDE = Path("/mnt/10tb_hdd/taek/stain_translation_enco/configs/exclude_fresh_poor_pairs.txt")
TEST_XLSX = Path("/mnt/10tb_hdd/taek/Lung_VS_Testing Data Label.xlsx")
SIZE = 2048 if "--size" in sys.argv and sys.argv[sys.argv.index("--size") + 1] == "2048" else 1024
if SIZE == 2048:
    THUMB_ROOTS = (Path("/mnt/10tb_hdd/taek/images/fresh_lung/thumbnails_2048"),)
    TEST_THUMBS = Path("/mnt/10tb_hdd/taek/images/Fresh_section/thumbnails/AF2048")
    OUT = ROOT / "checkpoints/thumb_encoder_2048"
    HISTORY = ROOT / "eval_mb/thumb_encoder_2048_history.csv"
    TEST_OUT = ROOT / "eval_mb/thumb_encoder_2048_test.csv"
    LOG = ROOT / "logs/thumb_encoder_2048.log"
    BATCH = 8
else:
    THUMB_ROOTS = (
        Path("/mnt/10tb_hdd/taek/images/fresh_lung/batch1/thumbnails/AF"),
        Path("/mnt/10tb_hdd/taek/images/fresh_lung/batch2/thumbnails/AF"),
    )
    TEST_THUMBS = Path("/mnt/10tb_hdd/taek/images/Fresh_section/thumbnails/AF")
    OUT = ROOT / "checkpoints/thumb_encoder"
    HISTORY = ROOT / "eval_mb/thumb_encoder_history.csv"
    TEST_OUT = ROOT / "eval_mb/thumb_encoder_test.csv"
    LOG = ROOT / "logs/thumb_encoder.log"
    BATCH = 16

EPOCHS = 30
BACKBONE_LR = 1e-4
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.05
SEED = 0
HOLD_PER_CLASS = 16
BIN_NAMES = ("Benign", "Malignant")
SUB_NAMES = ("Benign", "LUAD", "MIA", "LUSC", "SCLC", "Metastatic", "Other")
DROP_REASONS = (
    "unknown",
    "lymph node",
    "breast",
    "thyroid",
)
TEST_SUBTYPE = {
    "Benign": "Benign",
    "Adenocarcinoma (LUAD)": "LUAD",
    "Minimally Invasive Adenocarcinoma (MIA)": "MIA",
    "Squamous cell carcinoma (LUSC)": "LUSC",
    "Small Cell Carcinoma (SCC)": "SCLC",
    "Adenocarcinoma in-situ (AIS)": "LUAD",
    "Mucinous adenocarcinoma": "LUAD",
}


def stem_of(filename: str) -> str:
    name = Path(filename).name
    for suffix in ("-AF-below-R1.tif", "-AF-R1.tif"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def specimen_of(slide: str) -> str:
    if "_" in slide:
        return slide.split("_", 1)[1]
    return slide


def thumb_index() -> dict[str, Path]:
    found = {}
    for root in THUMB_ROOTS:
        for path in root.glob("*.png"):
            found[path.stem] = path
    return found


def exclusion_reasons() -> dict[str, str]:
    reasons = {}
    for line in EXCLUDE.read_text().splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        slide, _, reason = raw.partition("#")
        reasons[slide.strip()] = reason.strip().lower()
    return reasons


def is_dropped(reason: str) -> bool:
    return any(name in reason for name in DROP_REASONS)


def say(text: str) -> None:
    print(text, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as handle:
        handle.write(text + "\n")


def load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    tensor = torch.from_numpy(array).permute(2, 0, 1).float()
    return tensor.mul_(2.0 / 255.0).sub_(1.0)


def training_pool(thumbs: dict[str, Path]) -> tuple[list[dict], list[dict]]:
    labels = {}
    with LABELS.open(newline="") as handle:
        for row in csv.DictReader(handle):
            classes = (row.get("classes1") or "").strip()
            if classes not in BIN_NAMES:
                continue
            subtype = (row.get("subtype_norm") or "").strip()
            if classes == "Benign":
                subtype = "Benign"
            elif subtype == "METASTATIC":
                subtype = "Metastatic"
            elif subtype == "OTHER":
                subtype = "Other"
            elif not subtype or subtype not in SUB_NAMES:
                continue
            labels[row["slide_id"].strip()] = (0 if classes == "Benign" else 1, SUB_NAMES.index(subtype))
    core = set()
    with TRAIN_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            core.add(stem_of(row["af_wsi"]))
    reasons = exclusion_reasons()
    rows = []
    for slide, (binary, subtype) in labels.items():
        path = thumbs.get(slide)
        if path is None:
            continue
        reason = reasons.get(specimen_of(slide), "")
        if is_dropped(reason):
            continue
        rows.append({"slide": slide, "path": path, "bin": binary, "sub": subtype, "core": slide in core})
    if not any(row["core"] for row in rows):
        raise RuntimeError("none of the 425 training slides were found")
    return rows, [row for row in rows if row["core"]]


def holdout(core: list[dict]) -> tuple[list[str], list[str]]:
    by_label = {0: [], 1: []}
    for row in core:
        by_label[row["bin"]].append(row["slide"])
    generator = random.Random(SEED)
    for group in by_label.values():
        generator.shuffle(group)
    val = by_label[0][:HOLD_PER_CLASS] + by_label[1][:HOLD_PER_CLASS]
    if len(val) != 2 * HOLD_PER_CLASS:
        raise RuntimeError("the holdout does not have 16 slides of each class")
    return val, [slide for slide in by_label[0][HOLD_PER_CLASS:] + by_label[1][HOLD_PER_CLASS:]]


def test_rows() -> list[dict]:
    frame = pd.read_excel(TEST_XLSX, header=0)
    rows = []
    for _, row in frame.iterrows():
        slide = str(row["Sample ID"]).strip()
        diagnosis = " ".join(str(row["Diagnosis"]).replace("\n", " ").split())
        classes = "" if pd.isna(row["Classes"]) else str(row["Classes"]).strip()
        if classes in BIN_NAMES:
            binary = 0 if classes == "Benign" else 1
        elif diagnosis == "Benign":
            binary = 0
        else:
            raise RuntimeError(f"{slide} has no benign/malignant label")
        subtype = TEST_SUBTYPE.get(diagnosis)
        if subtype is None:
            raise RuntimeError(f"{slide} diagnosis {diagnosis!r} has no subtype")
        path = TEST_THUMBS / f"{slide}.png"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append({
            "slide": slide,
            "path": path,
            "bin": binary,
            "sub": SUB_NAMES.index(subtype),
            "diagnosis": diagnosis,
            "classes": classes,
        })
    if len(rows) != 76:
        raise RuntimeError(f"expected 76 test slides, found {len(rows)}")
    return rows


def mask_regions(image: torch.Tensor) -> torch.Tensor:
    """Cover one to three rectangles. The fill is mid-gray, not the black border."""
    _, height, width = image.shape
    for _ in range(random.randint(1, 3)):
        box_h = max(1, int(height * random.uniform(0.12, 0.28)))
        box_w = max(1, int(width * random.uniform(0.12, 0.28)))
        top = random.randrange(0, height - box_h + 1)
        left = random.randrange(0, width - box_w + 1)
        image[:, top:top + box_h, left:left + box_w] = 0
    return image


class ThumbSet(Dataset):
    def __init__(self, rows: list[dict], indices: list[int] | None, augment: bool):
        self.rows = rows
        self.indices = list(range(len(rows))) if indices is None else indices
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[self.indices[index]]
        image = load_rgb(row["path"])
        if self.augment:
            if random.random() < 0.5:
                image = image.flip(-1)
            if random.random() < 0.5:
                image = image.flip(-2)
            image = torch.rot90(image, random.randrange(4), (1, 2))
            image = (image * random.uniform(0.75, 1.25) + random.uniform(-0.1, 0.1)).clamp(-1, 1)
            if random.random() < 0.8:
                image = mask_regions(image)
        return {"image": image, "bin": row["bin"], "sub": row["sub"], "slide": row["slide"]}


def collate(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "bin": torch.tensor([item["bin"] for item in batch], dtype=torch.long),
        "sub": torch.tensor([item["sub"] for item in batch], dtype=torch.long),
        "slide": [item["slide"] for item in batch],
    }


def subtype_batch_weight(rows: list[dict]) -> torch.Tensor:
    """Inverse of each subtype's share inside a half-benign batch."""
    malignant = max(sum(row["bin"] == 1 for row in rows), 1)
    counts = [sum(row["sub"] == i for row in rows) for i in range(len(SUB_NAMES))]
    share = []
    for index, count in enumerate(counts):
        if index == 0:
            share.append(0.5 if count else 0.0)
        else:
            share.append(0.5 * count / malignant if count else 0.0)
    present = [1.0 / value for value in share if value > 0]
    cap = 8.0 * min(present)
    weight = [min(cap, 1.0 / value) if value > 0 else 0.0 for value in share]
    tensor = torch.tensor(weight, dtype=torch.float32)
    return tensor / tensor.sum().clamp(min=1e-8) * len(SUB_NAMES)


def balanced_indices(rows: list[dict], epoch: int, rank: int, world: int) -> list[int]:
    payload: list[list[int] | None] = [None]
    if rank == 0:
        benign = [i for i, row in enumerate(rows) if row["bin"] == 0]
        malignant = [i for i, row in enumerate(rows) if row["bin"] == 1]
        generator = random.Random(SEED + epoch)
        chosen = [
            generator.choice(benign if i % 2 == 0 else malignant)
            for i in range(len(rows))
        ]
        generator.shuffle(chosen)
        while len(chosen) % world:
            chosen.append(generator.choice(benign if len(chosen) % 2 == 0 else malignant))
        payload[0] = chosen
    dist.broadcast_object_list(payload, src=0)
    chosen = payload[0]
    if chosen is None:
        raise RuntimeError("the balanced index list was not broadcast")
    return chosen[rank::world]


@torch.no_grad()
def evaluate(model: ThumbEncoder, rows: list[dict], device: torch.device) -> dict:
    model.eval()
    loader = DataLoader(
        ThumbSet(rows, None, augment=False),
        batch_size=BATCH,
        shuffle=False,
        num_workers=2,
        collate_fn=collate,
    )
    bin_correct = [0, 0]
    bin_total = [0, 0]
    sub_correct = [0] * len(SUB_NAMES)
    sub_total = [0] * len(SUB_NAMES)
    predictions = []
    for batch in loader:
        binary, subtype, _pooled = model(batch["image"].to(device))
        bin_call = binary.argmax(dim=1).cpu()
        sub_call = subtype.argmax(dim=1).cpu()
        for slide, truth, call, sub_truth, sub_pred, logit in zip(
            batch["slide"], batch["bin"], bin_call, batch["sub"], sub_call, binary.cpu(),
        ):
            truth = int(truth)
            call = int(call)
            sub_truth = int(sub_truth)
            sub_pred = int(sub_pred)
            bin_total[truth] += 1
            bin_correct[truth] += int(call == truth)
            sub_total[sub_truth] += 1
            sub_correct[sub_truth] += int(sub_pred == sub_truth)
            probability = torch.softmax(logit, dim=0)
            predictions.append({
                "slide": slide,
                "truth": BIN_NAMES[truth],
                "pred": BIN_NAMES[call],
                "subtype_truth": SUB_NAMES[sub_truth],
                "subtype_pred": SUB_NAMES[sub_pred],
                "p_malignant": float(probability[1]),
                "correct": int(call == truth),
            })
    recalls = [bin_correct[i] / bin_total[i] if bin_total[i] else 0.0 for i in (0, 1)]
    seen = sum(bin_total)
    return {
        "acc": sum(bin_correct) / seen if seen else 0.0,
        "balanced": sum(recalls) / 2.0,
        "correct": sum(bin_correct),
        "n": seen,
        "benign": bin_correct[0],
        "benign_n": bin_total[0],
        "malignant": bin_correct[1],
        "malignant_n": bin_total[1],
        "sub_correct": sub_correct,
        "sub_total": sub_total,
        "predictions": predictions,
    }


def subtype_text(metrics: dict) -> str:
    parts = []
    for name, correct, total in zip(SUB_NAMES, metrics["sub_correct"], metrics["sub_total"]):
        if total:
            parts.append(f"{name} {correct}/{total}")
    return "  ".join(parts)


def main() -> None:
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", rank)
    random.seed(SEED + rank)
    torch.manual_seed(SEED)

    rows, core = training_pool(thumb_index())
    val_ids, _train_core = holdout(core)
    val_set = set(val_ids)
    train_rows = [row for row in rows if row["slide"] not in val_set]
    val_rows = [row for row in rows if row["slide"] in val_set]
    test = test_rows()
    if rank == 0:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        LOG.write_text("")
        bins = [sum(row["bin"] == i for row in train_rows) for i in (0, 1)]
        say(
            f"size {SIZE}  gpus {world}  batch/gpu {BATCH}  global {BATCH * world}  "
            f"train {len(train_rows)} benign {bins[0]} malignant {bins[1]}  "
            f"val {len(val_rows)}  test {len(test)}",
        )
        say(
            "aug  flip  rot90  brightness 0.75-1.25  shift 0.1  "
            "mask 1-3 rectangles on 80% of thumbnails",
        )

    model = ThumbEncoder(len(SUB_NAMES), size=SIZE).to(device)
    model = DistributedDataParallel(model, device_ids=[rank])
    raw = model.module
    optimizer = torch.optim.AdamW(
        [
            {"params": raw.backbone.parameters(), "lr": BACKBONE_LR},
            {"params": [p for name, p in raw.named_parameters() if not name.startswith("backbone")], "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    # Batches are already half benign and half malignant. Weighting the
    # subtype loss by the raw slide counts makes a benign error several
    # times more expensive than a LUAD error, so the subtype head calls
    # every slide benign. Weight by how often each subtype appears in
    # these balanced batches, and cap the rare classes.
    sub_weight = subtype_batch_weight(train_rows).to(device)

    history = []
    best_key = None
    for epoch in range(1, EPOCHS + 1):
        indices = balanced_indices(train_rows, epoch, rank, world)
        loader = DataLoader(
            ThumbSet(train_rows, indices, augment=True),
            batch_size=BATCH,
            shuffle=False,
            num_workers=2,
            collate_fn=collate,
        )
        model.train()
        running = torch.zeros((), device=device)
        seen = torch.zeros((), device=device)
        for batch in loader:
            image = batch["image"].to(device)
            binary, subtype, _pooled = model(image)
            loss = F.cross_entropy(binary, batch["bin"].to(device))
            loss = loss + 0.5 * F.cross_entropy(subtype, batch["sub"].to(device), weight=sub_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.detach() * image.shape[0]
            seen += image.shape[0]
        dist.all_reduce(running)
        dist.all_reduce(seen)
        schedule.step()
        if rank == 0:
            train_metrics = evaluate(raw, train_rows, device)
            val_metrics = evaluate(raw, val_rows, device)
            test_metrics = evaluate(raw, test, device)
            record = {
                "epoch": epoch,
                "loss": float(running / seen.clamp(min=1)),
                "train_acc": train_metrics["acc"],
                "test_acc": test_metrics["acc"],
                "test_benign": f"{test_metrics['benign']}/{test_metrics['benign_n']}",
                "test_malignant": f"{test_metrics['malignant']}/{test_metrics['malignant_n']}",
                "val_balanced": val_metrics["balanced"],
            }
            history.append(record)
            key = (val_metrics["balanced"], val_metrics["acc"], -epoch)
            marker = ""
            if best_key is None or key > best_key:
                best_key = key
                marker = "  saved"
                OUT.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model": raw.state_dict(),
                        "sub_names": SUB_NAMES,
                        "test": {
                            name: test_metrics[name]
                            for name in ("acc", "balanced", "correct", "n", "benign", "benign_n", "malignant", "malignant_n")
                        },
                        "test_predictions": test_metrics["predictions"],
                    },
                    OUT / "best.pt",
                )
            say(
                f"epoch {epoch:02d}  loss {record['loss']:.3f}  "
                f"train {100 * train_metrics['acc']:.1f}% ({train_metrics['correct']}/{train_metrics['n']})  "
                f"test {100 * test_metrics['acc']:.1f}% ({test_metrics['correct']}/{test_metrics['n']})  "
                f"benign {test_metrics['benign']}/{test_metrics['benign_n']}  "
                f"malignant {test_metrics['malignant']}/{test_metrics['malignant_n']}{marker}",
            )
            say(f"         subtype test  {subtype_text(test_metrics)}")
        dist.barrier()

    if rank == 0:
        with HISTORY.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        payload = torch.load(OUT / "best.pt", map_location="cpu", weights_only=False)
        fields = ["slide", "truth", "pred", "subtype_truth", "subtype_pred", "p_malignant", "correct"]
        with TEST_OUT.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(payload["test_predictions"])
        chosen = payload["test"]
        say(
            f"best epoch {payload['epoch']}  "
            f"test {chosen['correct']}/{chosen['n']}  "
            f"benign {chosen['benign']}/{chosen['benign_n']}  "
            f"malignant {chosen['malignant']}/{chosen['malignant_n']}  "
            f"balanced {chosen['balanced']:.3f}",
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
