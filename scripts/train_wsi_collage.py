"""Train the collage encoder once the 8x8 sheets exist.

Training slides draw a random collage version. Validation and test use
version 0. The log is logs/wsi_collage_encoder.log.

    torchrun --nproc_per_node=4 --master_port=29535 scripts/train_wsi_collage.py
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_thumb_encoder as thumb
from models.wsi_collage_encoder import WSICollageEncoder

TRAIN_ROOT = ROOT / "collages_grid" / "train"
TEST_ROOT = ROOT / "collages_grid" / "test"
OUT = ROOT / "checkpoints/wsi_collage_encoder"
HISTORY = ROOT / "eval_mb/wsi_collage_encoder_history.csv"
TEST_OUT = ROOT / "eval_mb/wsi_collage_encoder_test.csv"
LOG = ROOT / "logs/wsi_collage_encoder.log"
VERSIONS = 8
EPOCHS = 100
BATCH = 8
BACKBONE_LR = 1e-4
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.05
SEED = 0

thumb.LOG = LOG


def say(text: str) -> None:
    print(text, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as handle:
        handle.write(text + "\n")


def latest_epoch() -> int:
    numbers = []
    for path in OUT.glob("epoch_*.pt"):
        try:
            numbers.append(int(path.stem.split("_")[1]))
        except ValueError:
            continue
    return max(numbers) if numbers else 0


def read_history() -> list[dict]:
    import csv
    if not HISTORY.is_file():
        return []
    with HISTORY.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_history(rows: list[dict]) -> None:
    import csv
    if not rows:
        return
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def with_collage(rows: list[dict], root: Path) -> list[dict]:
    kept = []
    missing = []
    for row in rows:
        folder = root / row["slide"]
        if not (folder / "v0.png").is_file():
            missing.append(row["slide"])
            continue
        item = dict(row)
        item["folder"] = folder
        item["path"] = folder / "v0.png"
        kept.append(item)
    if missing and len(missing) == len(rows):
        raise RuntimeError(f"no collages under {root}")
    return kept


class CollageSet(thumb.ThumbSet):
    def __getitem__(self, index: int) -> dict:
        row = self.rows[self.indices[index]]
        version = 0 if not self.augment else random.randrange(VERSIONS)
        path = row["folder"] / f"v{version}.png"
        if not path.is_file():
            path = row["path"]
        image = thumb.load_rgb(path)
        if self.augment:
            if random.random() < 0.5:
                image = image.flip(-1)
            if random.random() < 0.5:
                image = image.flip(-2)
            image = torch.rot90(image, random.randrange(4), (1, 2))
            image = (image * random.uniform(0.75, 1.25) + random.uniform(-0.1, 0.1)).clamp(-1, 1)
            if random.random() < 0.8:
                image = thumb.mask_regions(image)
        return {"image": image, "bin": row["bin"], "sub": row["sub"], "slide": row["slide"]}


def main() -> None:
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", rank)
    random.seed(SEED + rank)
    torch.manual_seed(SEED)

    thumb.BATCH = BATCH
    thumb.LOG = LOG
    rows, core = thumb.training_pool(thumb.thumb_index())
    val_ids, _core_train = thumb.holdout(core)
    val_set = set(val_ids)
    train_rows = with_collage([row for row in rows if row["slide"] not in val_set], TRAIN_ROOT)
    val_rows = with_collage([row for row in rows if row["slide"] in val_set], TRAIN_ROOT)
    test = with_collage(thumb.test_rows(), TEST_ROOT)
    completed = latest_epoch()
    start_epoch = completed + 1
    if rank == 0:
        if start_epoch == 1:
            LOG.write_text("")
        bins = [sum(row["bin"] == index for row in train_rows) for index in (0, 1)]
        if start_epoch == 1:
            say(
                f"collage 2048  versions {VERSIONS}  gpus {world}  batch/gpu {BATCH}  "
                f"global {BATCH * world}  train {len(train_rows)} benign {bins[0]} malignant {bins[1]}  "
                f"val {len(val_rows)}  test {len(test)}"
            )
            say("aug  flip  rot90  brightness 0.75-1.25  shift 0.1  mask 1-3 rectangles  random train version")
        else:
            say(
                f"resume epoch {start_epoch:02d} of {EPOCHS}  "
                f"loaded epoch_{completed:04d}  train {len(train_rows)}  val {len(val_rows)}  test {len(test)}"
            )

    model = WSICollageEncoder(len(thumb.SUB_NAMES)).to(device)
    model = DistributedDataParallel(model, device_ids=[rank])
    raw = model.module
    optimizer = torch.optim.AdamW(
        [
            {"params": raw.backbone.parameters(), "lr": BACKBONE_LR},
            {
                "params": [parameter for name, parameter in raw.named_parameters() if not name.startswith("backbone")],
                "lr": HEAD_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    if completed:
        state = torch.load(OUT / f"epoch_{completed:04d}.pt", map_location=device, weights_only=False)
        raw.load_state_dict(state["model"])
        for _ in range(completed):
            schedule.step()
    sub_weight = thumb.subtype_batch_weight(train_rows).to(device)
    history = []
    if rank == 0 and completed:
        history = [row for row in read_history() if int(row["epoch"]) < start_epoch]
    best_key = None
    if rank == 0 and completed and (OUT / "best.pt").is_file():
        held = {key: value.detach().cpu().clone() for key, value in raw.state_dict().items()}
        best_state = torch.load(OUT / "best.pt", map_location=device, weights_only=False)
        raw.load_state_dict(best_state["model"])
        recovered = thumb.evaluate(raw, val_rows, device)
        best_key = (recovered["balanced"], recovered["acc"], -int(best_state["epoch"]))
        raw.load_state_dict(held)
        say(
            f"kept best epoch {int(best_state['epoch'])}  "
            f"val balanced {recovered['balanced']:.3f}"
        )
    dist.barrier()
    for epoch in range(start_epoch, EPOCHS + 1):
        indices = thumb.balanced_indices(train_rows, epoch, rank, world)
        loader = DataLoader(
            CollageSet(train_rows, indices, augment=True),
            batch_size=BATCH,
            shuffle=False,
            num_workers=2,
            collate_fn=thumb.collate,
        )
        model.train()
        running = torch.zeros((), device=device)
        seen = torch.zeros((), device=device)
        for batch in loader:
            image = batch["image"].to(device)
            binary, subtype, _tokens = model(image)
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
            train_metrics = thumb.evaluate(raw, train_rows, device)
            val_metrics = thumb.evaluate(raw, val_rows, device)
            test_metrics = thumb.evaluate(raw, test, device)
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
            write_history(history)
            key = (val_metrics["balanced"], val_metrics["acc"], -epoch)
            payload = {
                "epoch": epoch,
                "model": raw.state_dict(),
                "sub_names": thumb.SUB_NAMES,
                "test": {
                    name: test_metrics[name]
                    for name in ("acc", "balanced", "correct", "n", "benign", "benign_n", "malignant", "malignant_n")
                },
                "test_predictions": test_metrics["predictions"],
            }
            OUT.mkdir(parents=True, exist_ok=True)
            torch.save(payload, OUT / f"epoch_{epoch:04d}.pt")
            marker = ""
            if best_key is None or key > best_key:
                best_key = key
                marker = "  best"
                torch.save(payload, OUT / "best.pt")
            say(
                f"epoch {epoch:02d}  loss {record['loss']:.3f}  "
                f"train {100 * train_metrics['acc']:.1f}% ({train_metrics['correct']}/{train_metrics['n']})  "
                f"test {100 * test_metrics['acc']:.1f}% ({test_metrics['correct']}/{test_metrics['n']})  "
                f"benign {test_metrics['benign']}/{test_metrics['benign_n']}  "
                f"malignant {test_metrics['malignant']}/{test_metrics['malignant_n']}  "
                f"val {val_metrics['balanced']:.3f}{marker}"
            )
            say(f"         subtype test  {thumb.subtype_text(test_metrics)}")
        dist.barrier()

    if rank == 0:
        import csv
        write_history(history)
        payload = torch.load(OUT / "best.pt", map_location="cpu", weights_only=False)
        fields = ["slide", "truth", "pred", "subtype_truth", "subtype_pred", "p_malignant", "correct"]
        with TEST_OUT.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(payload["test_predictions"])
        chosen = payload["test"]
        say(
            f"best epoch {payload['epoch']}  test {chosen['correct']}/{chosen['n']}  "
            f"benign {chosen['benign']}/{chosen['benign_n']}  "
            f"malignant {chosen['malignant']}/{chosen['malignant_n']}  "
            f"balanced {chosen['balanced']:.3f}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
