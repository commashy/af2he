"""1024 thumbnails of the 76 test AF slides.

The whole raw slide is used, with no crop. The long side fits in 1024 and
the image is letterboxed on a black canvas, the same geometry as
``thumb_canvas_geometry``.

    python scripts/build_test_thumbnails.py
"""

from __future__ import annotations

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

from fresh_processing_utils import af_to_rgb_uint8, thumb_canvas_geometry

RAW = Path("/mnt/10tb_hdd/taek/images/Fresh_section/16bit-raw-af")
OUT = Path("/mnt/10tb_hdd/taek/images/Fresh_section/thumbnails/AF")
THUMB = 1024


def slides() -> list[str]:
    suffix = "-AF-R1.tif"
    return sorted(path.name[: -len(suffix)] for path in RAW.glob(f"*{suffix}"))


def build_one(slide: str) -> None:
    destination = OUT / f"{slide}.png"
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"skip {slide}", flush=True)
        return
    started = time.time()
    image = tifffile.imread(RAW / f"{slide}-AF-R1.tif", maxworkers=8)
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.moveaxis(image, 0, -1)
    height, width = image.shape[:2]
    _canvas, xoff, yoff, thumb_w, thumb_h = thumb_canvas_geometry(width, height, THUMB)
    small = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
    rgb = af_to_rgb_uint8(small)
    canvas = Image.new("RGB", (THUMB, THUMB), (0, 0, 0))
    canvas.paste(Image.fromarray(rgb, mode="RGB"), (xoff, yoff))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, compress_level=1)
    print(
        f"{slide} {height}x{width} -> {thumb_w}x{thumb_h} at {xoff},{yoff} "
        f"in {time.time() - started:.0f}s",
        flush=True,
    )


def main() -> None:
    names = slides()
    if len(names) != 76:
        raise RuntimeError(f"expected 76 test slides, found {len(names)}")
    print(f"slides {len(names)} out {OUT}", flush=True)
    for slide in names:
        build_one(slide)
    written = sorted(OUT.glob("F*.png"))
    print(f"done {len(written)}", flush=True)


if __name__ == "__main__":
    main()
