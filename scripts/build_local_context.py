"""Prebuild the 1024 neighborhood the local encoder reads.

Each training field gets one RGBA image. RGB is a 2048 AF crop centered on
that field, downsampled to 1024. Alpha is the fraction of each pixel that
came from a real tile; missing neighbors stay black, which the loader maps
to -1.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

AF = Path("/mnt/10tb_hdd/taek/images/Fresh_section/patches_1024/trainA")
INDEX = Path("/mnt/10tb_hdd/taek/af2he/retrieval/index.csv")
OUT = Path("/mnt/10tb_hdd/taek/af2he/local_1024")
PATCH = 1024
CANVAS = 3072
CROP = 2048
LOW = 1024


def load_fields() -> dict[str, list[tuple[str, int, int]]]:
    grouped: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    with INDEX.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            grouped[record["slide"]].append((record["filename"], int(record["y"]), int(record["x"])))
    if not grouped:
        raise RuntimeError(f"{INDEX} has no fields")
    return grouped


def _read(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.shape != (PATCH, PATCH, 3):
        raise RuntimeError(f"bad AF tile {path}")
    return image


def build_slide(task: dict) -> dict:
    cv2.setNumThreads(1)
    import time

    slide = task["slide"]
    fields: list[tuple[str, int, int]] = task["fields"]
    out = Path(task["out"])
    done_path = out / "done" / f"{slide}.ok"
    if done_path.is_file():
        return {"slide": slide, "fields": len(fields), "skipped": 1}
    started = time.perf_counter()
    tiles: dict[tuple[int, int], np.ndarray] = {}
    for filename, y, x in fields:
        tiles[(y, x)] = _read(AF / filename)
    image_dir = out / "images"
    written = 0
    for index, (filename, y, x) in enumerate(fields):
        destination = image_dir / filename
        if destination.is_file() and destination.stat().st_size > 0:
            continue
        canvas = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)
        mask = np.zeros((CANVAS, CANVAS), dtype=np.float32)
        for dy in (-PATCH, 0, PATCH):
            for dx in (-PATCH, 0, PATCH):
                tile = tiles.get((y + dy, x + dx))
                if tile is None:
                    continue
                top = dy + PATCH
                left = dx + PATCH
                canvas[top:top + PATCH, left:left + PATCH] = tile
                mask[top:top + PATCH, left:left + PATCH] = 1.0
        crop = canvas[512:512 + CROP, 512:512 + CROP]
        crop_mask = mask[512:512 + CROP, 512:512 + CROP]
        small = cv2.resize(crop, (LOW, LOW), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(crop_mask, (LOW, LOW), interpolation=cv2.INTER_AREA)
        center = small[256:768, 256:768]
        reference = cv2.resize(tiles[(y, x)], (512, 512), interpolation=cv2.INTER_AREA)
        if int(np.max(np.abs(center.astype(np.int16) - reference.astype(np.int16)))) > 1:
            raise RuntimeError(f"{filename} neighborhood center does not match the field")
        if float(small_mask[256:768, 256:768].min()) < 0.99:
            raise RuntimeError(f"{filename} center mask is not covered")
        alpha = np.clip(np.round(small_mask * 255.0), 0, 255).astype(np.uint8)
        bgra = np.dstack((small, alpha))
        temporary = destination.with_suffix(".tmp.png")
        if not cv2.imwrite(str(temporary), bgra, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise RuntimeError(f"failed to write {temporary}")
        temporary.replace(destination)
        written += 1
        if index % 200 == 0:
            print(f"{slide} {index}/{len(fields)}", flush=True)
    done_path.parent.mkdir(parents=True, exist_ok=True)
    done_path.write_text("ok\n", encoding="utf-8")
    elapsed = time.perf_counter() - started
    print(f"done {slide} fields={len(fields)} wrote={written} seconds={elapsed:.1f}", flush=True)
    return {"slide": slide, "fields": len(fields), "seconds": elapsed, "skipped": 0, "wrote": written}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "done").mkdir(parents=True, exist_ok=True)
    (OUT / "meta.json").write_text(
        json.dumps(
            {
                "source": str(AF),
                "window_px": CROP,
                "stored_px": LOW,
                "channels": "BGRA, alpha is covered fraction",
                "fill": "black, loader maps 0 to -1",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    grouped = load_fields()
    tasks = [
        {"slide": slide, "fields": fields, "out": str(OUT)}
        for slide, fields in grouped.items()
    ]
    tasks.sort(key=lambda item: len(item["fields"]), reverse=True)
    print(f"slides={len(tasks)} fields={sum(len(item['fields']) for item in tasks)} workers={args.workers}", flush=True)
    if args.workers == 1:
        for task in tasks:
            build_slide(task)
        return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(build_slide, task) for task in tasks]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
