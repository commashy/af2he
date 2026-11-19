"""Score the epoch-9 collage encoder on the 76 fresh test slides.

The head is the two-class benign/malignant head from stage-1 epoch 9.
Class 0 is benign and class 1 is malignant. The collage is version 0 of
the 1024 rule used by ``scripts/infer_one_slide.py`` for the epoch-9
virtual-stain run.

    python scripts/score_epoch9_mb.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CLEAN_SCRIPTS = Path("/mnt/10tb_hdd/taek/stain_translation_clean/scripts")
for path in (ROOT, CLEAN_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.collage import PATCH, build_collage_from_tiles
from fresh_processing_utils import af_to_rgb_uint8
from models.generator import CollageEncoder
from models.slide_head import SlideDiagnosisHead

RAW = Path("/mnt/10tb_hdd/taek/images/Fresh_section/16bit-raw-af")
LABELS = Path("/mnt/10tb_hdd/taek/Lung_VS_Testing Data Label.xlsx")
CHECKPOINT = ROOT / "checkpoints/stage1/epoch_0009.pt"
OUT = ROOT / "eval_mb"
COLLAGE_DIR = OUT / "collages"
NAMES = ("Benign", "Malignant")


def grid_tiles(path: Path) -> dict[tuple[int, int], np.ndarray]:
    """Decode the full-resolution page in parallel, then cut the 1024 grid."""
    image = tifffile.imread(path, maxworkers=8)
    if image.ndim != 2:
        raise RuntimeError(f"{path.name} is {image.shape}, expected a 2D page")
    height, width = image.shape
    ys = range(0, height - PATCH + 1, PATCH)
    xs = list(range(0, width - PATCH + 1, PATCH))
    if not xs:
        raise RuntimeError(f"{path.name} is {height}x{width}, smaller than {PATCH}")
    tiles = {}
    for y in ys:
        strip = af_to_rgb_uint8(image[y:y + PATCH])
        for x in xs:
            tiles[(y, x)] = np.ascontiguousarray(strip[:, x:x + PATCH])
    return tiles


def ensure_collage(slide: str) -> Path:
    """Version 0 only, the collage the epoch-9 slide inference used."""
    folder = COLLAGE_DIR / slide
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "v0.png"
    if path.is_file() and path.stat().st_size > 0:
        return path
    started = time.time()
    tiles = grid_tiles(RAW / f"{slide}-AF-R1.tif")
    image, records = build_collage_from_tiles(tiles, version=0)
    kinds = {}
    for record in records:
        kinds[record["kind"]] = kinds.get(record["kind"], 0) + 1
    Image.fromarray(image, mode="RGB").save(path)
    print(f"{slide} tiles {len(tiles)} collage {kinds} in {time.time() - started:.0f}s", flush=True)
    return path


def to_model(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float()
    return tensor.mul_(2.0 / 255.0).sub_(1.0)


def load_models() -> tuple[CollageEncoder, SlideDiagnosisHead]:
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    generator = payload["generator"]
    head_state = payload["slide_head"]
    if head_state is None:
        raise RuntimeError(f"{CHECKPOINT} has no slide head")
    weight = head_state["linear.weight"]
    if tuple(weight.shape) != (2, 384):
        raise RuntimeError(f"expected a 2-class head, got {tuple(weight.shape)}")
    encoder = CollageEncoder(1024).eval()
    encoder_state = {
        key.split("collage_encoder.", 1)[1]: value
        for key, value in generator.items()
        if key.startswith("collage_encoder.")
    }
    encoder.load_state_dict(encoder_state)
    head = SlideDiagnosisHead(384, n_classes=2).eval()
    head.load_state_dict(head_state)
    print(
        f"checkpoint epoch field {payload['epoch']} step {payload['global_step']} "
        f"head {tuple(weight.shape)}",
        flush=True,
    )
    del payload
    return encoder, head


def labels() -> list[dict]:
    frame = pd.read_excel(LABELS, header=0)
    rows = []
    for _, row in frame.iterrows():
        diagnosis = " ".join(str(row["Diagnosis"]).replace("\n", " ").split())
        classes = row["Classes"]
        classes_text = "" if pd.isna(classes) else str(classes).strip()
        truth = classes_text if classes_text in NAMES else ("Benign" if diagnosis == "Benign" else "")
        rows.append({
            "slide": str(row["Sample ID"]).strip(),
            "diagnosis": diagnosis,
            "classes": classes_text,
            "truth": truth,
        })
    return rows


def score_slide(encoder: CollageEncoder, head: SlideDiagnosisHead, path: Path) -> dict:
    with torch.no_grad():
        _memory, cond = encoder(to_model(path).unsqueeze(0))
        logit = head(cond.float())[0]
    probability = torch.softmax(logit, dim=0)
    return {
        "p_benign": float(probability[0]),
        "p_malignant": float(probability[1]),
        "pred": NAMES[int(logit.argmax())],
        "logit_benign": float(logit[0]),
        "logit_malignant": float(logit[1]),
    }


def main() -> None:
    Image.MAX_IMAGE_PIXELS = None
    OUT.mkdir(parents=True, exist_ok=True)
    encoder, head = load_models()
    rows = labels()
    raw_ids = {path.name[: -len("-AF-R1.tif")] for path in RAW.glob("*-AF-R1.tif")}
    label_ids = {row["slide"] for row in rows}
    if label_ids != raw_ids:
        raise RuntimeError(f"label/raw mismatch only-label={sorted(label_ids - raw_ids)} only-raw={sorted(raw_ids - label_ids)}")
    destination = OUT / "epoch9_test.csv"
    done = {}
    if destination.is_file():
        with destination.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("pred"):
                    done[row["slide"]] = row
    fields = [
        "slide", "diagnosis", "classes", "truth", "pred",
        "p_benign", "p_malignant", "logit_benign", "logit_malignant", "correct",
    ]
    pending = [row for row in rows if row["slide"] not in done]
    print(f"slides {len(rows)} done {len(done)} pending {len(pending)}", flush=True)
    records = [done[row["slide"]] for row in rows if row["slide"] in done]
    for row in pending:
        path = ensure_collage(row["slide"])
        scored = score_slide(encoder, head, path)
        correct = scored["pred"] == row["truth"]
        record = {**row, **scored, "correct": int(correct)}
        records.append(record)
        with destination.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        print(
            f"{row['slide']} truth={row['truth']} pred={scored['pred']} "
            f"p_mal={scored['p_malignant']:.3f}",
            flush=True,
        )
    frame = pd.DataFrame(records)
    print("\noverall", f"{int(frame['correct'].astype(int).sum())}/{len(frame)}", flush=True)
    for name in NAMES:
        part = frame[frame["truth"] == name]
        if len(part):
            print(f"{name} {int(part['correct'].astype(int).sum())}/{len(part)}", flush=True)
    print(frame.groupby(["diagnosis", "truth", "pred"]).size().to_string(), flush=True)


if __name__ == "__main__":
    main()
