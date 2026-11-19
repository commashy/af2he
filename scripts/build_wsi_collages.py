"""8x8 collages of 1024 fields for the WSI encoder.

The tissue box is split into eight by eight. Each cell keeps one 1024
field, resized to 256. Version 0 is the field nearest the cell center.
Later versions walk outward through the other tissue fields in that cell.
Test slides get version 0 only.

    python scripts/build_wsi_collages.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CORE_SCRIPTS = Path("/mnt/10tb_hdd/taek/stain_translation_core/scripts")
CLEAN_SCRIPTS = Path("/mnt/10tb_hdd/taek/stain_translation_clean/scripts")
for path in (ROOT, CORE_SCRIPTS, CLEAN_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.groups import slide_and_coord
from fresh_processing_utils import af_to_rgb_uint8

PATCH_ROOT = Path("/mnt/10tb_hdd/taek/images/Fresh_section/patches_1024/trainA")
RAW = Path("/mnt/10tb_hdd/taek/images/Fresh_section/16bit-raw-af")
OUT = ROOT / "collages_grid"
TRAIN_OUT = OUT / "train"
TEST_OUT = OUT / "test"
CELL = 256
GRID = 8
CANVAS = CELL * GRID
PATCH = 1024
VERSIONS = 8
GLASS_STD = 5.0
WORKERS = 8


def tissue_box(patches: list[dict]) -> tuple[int, int, int, int]:
    tissue = [item for item in patches if item["tissue"]]
    if not tissue:
        tissue = patches
    y0 = min(item["y"] for item in tissue)
    x0 = min(item["x"] for item in tissue)
    y1 = max(item["y"] + PATCH for item in tissue)
    x1 = max(item["x"] + PATCH for item in tissue)
    return y0, x0, y1, x1


def choose_cells(patches: list[dict], versions: int) -> tuple[tuple[int, int, int, int], list[list]]:
    y0, x0, y1, x1 = tissue_box(patches)
    cell_h = max((y1 - y0) / GRID, 1.0)
    cell_w = max((x1 - x0) / GRID, 1.0)
    chosen = []
    for row in range(GRID):
        for col in range(GRID):
            top = y0 + row * cell_h
            left = x0 + col * cell_w
            center_y = top + cell_h / 2.0
            center_x = left + cell_w / 2.0
            inside = []
            for item in patches:
                if not item["tissue"]:
                    continue
                cy = item["y"] + PATCH / 2.0
                cx = item["x"] + PATCH / 2.0
                if top <= cy < top + cell_h and left <= cx < left + cell_w:
                    distance = abs(cy - center_y) + abs(cx - center_x)
                    inside.append((distance, item))
            inside.sort(key=lambda pair: pair[0])
            cells = []
            for version in range(versions):
                if not inside:
                    cells.append(None)
                else:
                    cells.append(inside[version % len(inside)][1])
            chosen.append(cells)
    return (y0, x0, y1, x1), chosen


def paint(chosen: list[list], version: int, images: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    canvas = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)
    for index, versions in enumerate(chosen):
        item = versions[version]
        if item is None:
            continue
        row, col = divmod(index, GRID)
        image = images[(item["y"], item["x"])]
        canvas[
            row * CELL:(row + 1) * CELL,
            col * CELL:(col + 1) * CELL,
        ] = image
    return canvas


def load_png(path: Path, y: int, x: int) -> dict | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    small = cv2.resize(image, (CELL, CELL), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return {
        "y": y,
        "x": x,
        "tissue": float(gray.std()) >= GLASS_STD,
        "image": rgb,
    }


def build_from_patches(slide: str, paths: list[tuple[int, int, str]]) -> str:
    destination = TRAIN_OUT / slide
    targets = [destination / f"v{version}.png" for version in range(VERSIONS)]
    if all(path.is_file() and path.stat().st_size > 0 for path in targets) and (destination / "meta.json").is_file():
        return f"skip {slide}"
    loaded = []
    images = {}
    for y, x, path in paths:
        item = load_png(Path(path), y, x)
        if item is None:
            continue
        images[(y, x)] = item.pop("image")
        loaded.append(item)
    if not loaded:
        return f"empty {slide}"
    box, chosen = choose_cells(loaded, VERSIONS)
    destination.mkdir(parents=True, exist_ok=True)
    for version, path in enumerate(targets):
        image = paint(chosen, version, images)
        temporary = path.with_suffix(".partial")
        Image.fromarray(image, mode="RGB").save(temporary, format="PNG", compress_level=1)
        temporary.replace(path)
    meta = {
        "slide": slide,
        "grid": GRID,
        "cell": CELL,
        "patch": PATCH,
        "versions": VERSIONS,
        "box": list(box),
        "cells": [
            [
                None if item is None else [item["y"], item["x"]]
                for item in versions
            ]
            for versions in chosen
        ],
    }
    (destination / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    tissue = sum(item["tissue"] for item in loaded)
    return f"{slide} patches {len(loaded)} tissue {tissue}"


def build_one_train(payload: tuple[str, list[tuple[int, int, str]]]) -> str:
    slide, paths = payload
    started = time.time()
    note = build_from_patches(slide, paths)
    return f"{note} in {time.time() - started:.0f}s"


def index_training_patches() -> dict[str, list[tuple[int, int, str]]]:
    grouped: dict[str, list[tuple[int, int, str]]] = {}
    with os.scandir(PATCH_ROOT) as entries:
        for entry in entries:
            if not entry.name.endswith(".png") or "__" not in entry.name:
                continue
            slide, (y, x) = slide_and_coord(Path(entry.name))
            grouped.setdefault(slide, []).append((y, x, entry.path))
    return grouped


def read_wsi(path: Path) -> np.ndarray:
    image = tifffile.imread(path, maxworkers=4)
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.moveaxis(image, 0, -1)
    return image


def build_test_slide(path: Path) -> str:
    slide = path.name[: -len("-AF-R1.tif")]
    destination = TEST_OUT / slide
    target = destination / "v0.png"
    if target.is_file() and target.stat().st_size > 0 and (destination / "meta.json").is_file():
        return f"skip {slide}"
    started = time.time()
    image = read_wsi(path)
    if image.ndim == 2:
        height, width = image.shape
    else:
        height, width = image.shape[:2]
    loaded = []
    images = {}
    for y in range(0, height - PATCH + 1, PATCH):
        for x in range(0, width - PATCH + 1, PATCH):
            crop = image[y:y + PATCH, x:x + PATCH]
            rgb = af_to_rgb_uint8(crop)
            small = cv2.resize(rgb, (CELL, CELL), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
            loaded.append({"y": y, "x": x, "tissue": float(gray.std()) >= GLASS_STD})
            images[(y, x)] = small
    if not loaded:
        return f"empty {slide}"
    box, chosen = choose_cells(loaded, versions=1)
    destination.mkdir(parents=True, exist_ok=True)
    painted = paint(chosen, 0, images)
    temporary = target.with_suffix(".partial")
    Image.fromarray(painted, mode="RGB").save(temporary, format="PNG", compress_level=1)
    temporary.replace(target)
    meta = {
        "slide": slide,
        "grid": GRID,
        "cell": CELL,
        "patch": PATCH,
        "versions": 1,
        "box": list(box),
        "cells": [
            [None if item is None else [item["y"], item["x"]] for item in versions]
            for versions in chosen
        ],
    }
    (destination / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return f"{slide} tiles {len(loaded)} in {time.time() - started:.0f}s"


def main() -> None:
    TRAIN_OUT.mkdir(parents=True, exist_ok=True)
    TEST_OUT.mkdir(parents=True, exist_ok=True)
    grouped = index_training_patches()
    print(f"training slides {len(grouped)} versions {VERSIONS}", flush=True)
    done = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(build_one_train, item) for item in grouped.items()]
        for future in as_completed(futures):
            done += 1
            try:
                note = future.result()
            except Exception as exc:
                failed += 1
                note = f"FAILED {exc}"
            print(f"train {done}/{len(futures)} {note}", flush=True)
    if failed:
        raise SystemExit(f"train failures {failed}")
    raws = sorted(RAW.glob("*-AF-R1.tif"))
    print(f"test slides {len(raws)}", flush=True)
    for index, path in enumerate(raws, start=1):
        print(f"test {index}/{len(raws)} {build_test_slide(path)}", flush=True)
    print(
        f"done train {len(list(TRAIN_OUT.glob('*')))} test {len(list(TEST_OUT.glob('*')))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
