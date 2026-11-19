"""2048 thumbnails for the training slides and the 76 test slides.

Training slides use the same AF crop as the 1024 thumbnails. Test slides use
the whole raw file. Both are letterboxed on a black 2048 canvas.

    python scripts/build_thumbnails_2048.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CORE_SCRIPTS = Path("/mnt/10tb_hdd/taek/stain_translation_core/scripts")
CLEAN_SCRIPTS = Path("/mnt/10tb_hdd/taek/stain_translation_clean/scripts")
for path in (ROOT, CORE_SCRIPTS, CLEAN_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fresh_processing_utils import af_to_rgb_uint8, parse_box, thumb_canvas_geometry

CROP_MANIFEST = Path("/mnt/10tb_hdd/taek/stain_translation_core/manifests/fresh/crop_manifest_for_thumbs.csv")
RAW = Path("/mnt/10tb_hdd/taek/images/Fresh_section/16bit-raw-af")
TRAIN_OUT = Path("/mnt/10tb_hdd/taek/images/fresh_lung/thumbnails_2048")
TEST_OUT = Path("/mnt/10tb_hdd/taek/images/Fresh_section/thumbnails/AF2048")
THUMB = 2048


def read_slide(path: Path) -> np.ndarray:
    image = tifffile.imread(path, maxworkers=4)
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.moveaxis(image, 0, -1)
    return image


def letterbox(image: np.ndarray, box: tuple[int, int, int, int] | None) -> Image.Image:
    if image.ndim == 2:
        height, width = image.shape
    else:
        height, width = image.shape[:2]
    if box is None:
        x1, y1, x2, y2 = 0, 0, width, height
    else:
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
    crop = image[y1:y2, x1:x2]
    crop_h, crop_w = crop.shape[:2]
    _canvas, xoff, yoff, thumb_w, thumb_h = thumb_canvas_geometry(crop_w, crop_h, THUMB)
    small = cv2.resize(crop, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
    rgb = af_to_rgb_uint8(small)
    canvas = Image.new("RGB", (THUMB, THUMB), (0, 0, 0))
    canvas.paste(Image.fromarray(rgb, mode="RGB"), (xoff, yoff))
    return canvas


def resolve_af(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_file():
        return path
    alt = Path(str(path).replace("/AF-WSI/", "/AF_batch1/"))
    if alt.is_file():
        return alt
    raise FileNotFoundError(path_str)


def save(canvas: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, compress_level=1)


def build_training() -> None:
    rows = list(csv.DictReader(CROP_MANIFEST.open(newline="")))
    pending = []
    for row in rows:
        destination = TRAIN_OUT / f"{row['pair_id']}.png"
        if destination.is_file() and destination.stat().st_size > 0:
            continue
        pending.append(row)
    print(f"training {len(rows)} pending {len(pending)}", flush=True)
    for index, row in enumerate(pending, start=1):
        started = time.time()
        image = read_slide(resolve_af(row["af_path"]))
        canvas = letterbox(image, parse_box(row["af_crop_box"]))
        destination = TRAIN_OUT / f"{row['pair_id']}.png"
        save(canvas, destination)
        print(f"train {index}/{len(pending)} {row['pair_id']} in {time.time() - started:.0f}s", flush=True)
        del image


def build_test() -> None:
    names = sorted(path.name[: -len("-AF-R1.tif")] for path in RAW.glob("*-AF-R1.tif"))
    if len(names) != 76:
        raise RuntimeError(f"expected 76 test slides, found {len(names)}")
    pending = [name for name in names if not (TEST_OUT / f"{name}.png").is_file()]
    print(f"test {len(names)} pending {len(pending)}", flush=True)
    for index, name in enumerate(pending, start=1):
        started = time.time()
        image = read_slide(RAW / f"{name}-AF-R1.tif")
        canvas = letterbox(image, None)
        save(canvas, TEST_OUT / f"{name}.png")
        print(f"test {index}/{len(pending)} {name} in {time.time() - started:.0f}s", flush=True)
        del image


def main() -> None:
    build_test()
    build_training()
    print(
        f"done test {len(list(TEST_OUT.glob('F*.png')))} "
        f"train {len(list(TRAIN_OUT.glob('*.png')))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
