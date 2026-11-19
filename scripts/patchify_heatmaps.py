"""Cut one heatmap patch for every real H&E training tile.

The heatmap JPEG has the same pixel size as that slide's H&E image. A tile
``slide__y*_x*`` is the 1024 window at that corner. Each pixel is the
continuous position from dark blue ``(0, 0, 128)`` to dark red ``(159, 0, 0)``.
The saved patch is that grayscale field. The CSV target is its mean.

Benign training slides with no heatmap are rows with target 0. Malignant
slides with no heatmap are left out.

    python scripts/patchify_heatmaps.py
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.heat_label import jet_score_cube, score_image
from engine.image_losses import hed_tissue_fraction

Image.MAX_IMAGE_PIXELS = None

HEAT_ROOT = Path("/mnt/10tb_hdd/taek/images/taek-HE_svs_trainingdataset-heatmaps")
HE_ROOTS = (
    Path("/mnt/10tb_hdd/taek/images/Fresh_section/HE-WSI"),
    Path("/mnt/10tb_hdd/taek/images/Fresh_section/Training-WSI/HE-WSI"),
)
PATCH_ROOT = Path("/mnt/10tb_hdd/taek/images/Fresh_section/patches_1024/trainB")
LABELS = Path("/mnt/10tb_hdd/taek/stain_translation_core/manifests/fresh/train_slide_labels.csv")
TRAIN_CSV = ROOT / "manifests/training_wsi.csv"
OUT = ROOT / "heat_patches"
TABLE = ROOT / "manifests/heat_patch_targets.csv"
DONE = ROOT / "manifests/heat_patch_done.txt"
TILE = 1024
COORD = re.compile(r"__y(?P<y>\d+)_x(?P<x>\d+)\.png$")


def say(text: str) -> None:
    print(text, flush=True)


def accession(name: str) -> str:
    return name.split("-HE_icc-", 1)[0]


def labels_by_slide() -> dict[str, str]:
    found = {}
    with LABELS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            slide = (row.get("slide_id") or "").strip()
            found[slide] = (row.get("classes1") or "").strip()
    return found


def training_slides() -> list[str]:
    slides = []
    with TRAIN_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            slides.append(row["af_wsi"].split("-AF-", 1)[0])
    return slides


def heat_index() -> dict[str, Path]:
    found = {}
    for path in HEAT_ROOT.glob("*_heatmap.jpg"):
        found[accession(path.name)] = path
    return found


def he_index() -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    for root in HE_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or "-HE_icc-" not in path.name:
                continue
            if path.suffix.lower() not in {".jpg", ".jpeg", ".tif", ".tiff"}:
                continue
            found.setdefault(accession(path.name), []).append(path)
    return found


def patch_index() -> dict[str, list[tuple[str, int, int]]]:
    found: dict[str, list[tuple[str, int, int]]] = {}
    for path in PATCH_ROOT.iterdir():
        match = COORD.search(path.name)
        if match is None:
            continue
        slide = path.name[: match.start()]
        found.setdefault(slide, []).append(
            (path.name, int(match.group("y")), int(match.group("x")))
        )
    return found


def tissue_fraction(path: Path) -> float:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.mul(2.0 / 255.0).sub(1.0)
    return float(hed_tissue_fraction(tensor)[0])


def write_row(handle, filename: str, slide: str, target: float, tissue: float, source: str) -> None:
    handle.write(f"{filename},{slide},{target:.6f},{tissue:.6f},{source}\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    classes = labels_by_slide()
    slides = training_slides()
    heats = heat_index()
    hes = he_index()
    patches = patch_index()
    done = set(DONE.read_text().split()) if DONE.is_file() else set()
    new_table = not TABLE.is_file() or TABLE.stat().st_size == 0
    cube = jet_score_cube()
    say(
        f"slides {len(slides)}  heatmaps {len(heats)}  "
        f"patch slides {len(patches)}  already done {len(done)}"
    )
    with TABLE.open("a", encoding="utf-8") as handle:
        if new_table:
            handle.write("filename,slide,target,tissue,source\n")
        for index, slide in enumerate(slides, start=1):
            if slide in done:
                continue
            kind = classes.get(slide, "")
            acc = slide.split("_", 1)[1] if "_" in slide else slide
            tiles = patches.get(slide, [])
            if not tiles:
                say(f"{index}/{len(slides)} {slide} no patches")
                done.add(slide)
                continue
            heat_path = heats.get(acc)
            if heat_path is None:
                if kind != "Benign":
                    say(f"{index}/{len(slides)} {slide} malignant without a heatmap, skipped")
                    continue
                buffered = []
                for name, _y, _x in tiles:
                    tissue = tissue_fraction(PATCH_ROOT / name)
                    buffered.append((name, slide, 0.0, tissue, "benign"))
                for item in buffered:
                    write_row(handle, *item)
                handle.flush()
                done.add(slide)
                DONE.write_text("\n".join(sorted(done)) + "\n")
                say(f"{index}/{len(slides)} {slide} benign target 0  tiles {len(tiles)}")
                continue
            he_paths = hes.get(acc)
            if not he_paths:
                say(f"{index}/{len(slides)} {slide} heatmap without an H&E image, skipped")
                done.add(slide)
                DONE.write_text("\n".join(sorted(done)) + "\n")
                continue
            with Image.open(heat_path) as image:
                rgb = np.asarray(image.convert("RGB"))
            score = score_image(rgb, cube)
            del rgb
            height, width = score.shape
            written = 0
            outside = 0
            buffered = []
            for name, y, x in tiles:
                if y < 0 or x < 0 or y + TILE > height or x + TILE > width:
                    outside += 1
                    continue
                window = score[y:y + TILE, x:x + TILE]
                target = float(window.mean())
                mask = (window * 255.0).round().astype(np.uint8)
                Image.fromarray(mask, mode="L").save(OUT / name, compress_level=1)
                tissue = tissue_fraction(PATCH_ROOT / name)
                buffered.append((name, slide, target, tissue, "map"))
                written += 1
            for item in buffered:
                write_row(handle, *item)
            handle.flush()
            del score
            done.add(slide)
            DONE.write_text("\n".join(sorted(done)) + "\n")
            say(
                f"{index}/{len(slides)} {slide} map {heat_path.name} "
                f"tiles {written} outside {outside} canvas {width}x{height}"
            )
    say("done")


if __name__ == "__main__":
    main()
