"""Crop batch-1 pseudo-label WSIs into the training patch grid.

Only rows with ``use=crop_absolute`` are written. The box is the AF crop
origin plus the patch ``y``/``x`` from the filename. Output names match
``patches_1024/trainA``.

    python scripts/extract_pseudo_patches.py
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ALIGNMENT = Path("/mnt/10tb_hdd/taek/af2he/manifests/pseudo_alignment.csv")
CROP_MANIFEST = Path(
    "/mnt/10tb_hdd/taek/stain_translation_clean/manifests/fresh/crop_manifest_for_thumbs.csv"
)
PSEUDO_ROOT = Path("/mnt/10tb_hdd/taek/images/202610_inference_training_set")
PATCH_ROOT = Path("/mnt/10tb_hdd/taek/images/Fresh_section/patches_1024/trainA")
OUT_ROOT = Path("/mnt/10tb_hdd/taek/images/Fresh_section/patches_1024/pseudo")
PATCH = 1024
NAME_RE = re.compile(r"__y(\d+)_x(\d+)\.png$")


def specimen_key(name: str) -> str:
    stem = Path(name).stem
    for token in ("-AF-below-R1", "-AF-R1-1", "-AF-R1"):
        if token in stem:
            return stem.split(token)[0]
    return stem


def main() -> None:
    allowed = {
        row["specimen_key"]
        for row in csv.DictReader(ALIGNMENT.open(newline=""))
        if row["use"] == "crop_absolute"
    }
    crops = {
        row["specimen_key"]: row
        for row in csv.DictReader(CROP_MANIFEST.open(newline=""))
        if row["specimen_key"] in allowed
    }
    pseudo = {
        specimen_key(path.name): path
        for path in PSEUDO_ROOT.iterdir()
        if path.suffix.lower() == ".jpg" and specimen_key(path.name) in allowed
    }
    missing = sorted(set(crops) - set(pseudo))
    if missing:
        raise FileNotFoundError(f"pseudo JPEG missing for {missing[:8]}")
    by_slide: dict[str, list[Path]] = {}
    for path in PATCH_ROOT.glob("*.png"):
        slide_id = path.name.split("__", 1)[0]
        specimen = slide_id.split("_", 1)[1]
        if specimen in crops:
            by_slide.setdefault(slide_id, []).append(path)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"slides {len(by_slide)} patches {sum(len(v) for v in by_slide.values())}", flush=True)
    written = skipped = 0
    for index, (slide_id, paths) in enumerate(sorted(by_slide.items()), start=1):
        specimen = slide_id.split("_", 1)[1]
        row = crops[specimen]
        if row["pair_id"] != slide_id:
            raise RuntimeError(f"{slide_id} does not match manifest pair {row['pair_id']}")
        x1, y1, _x2, _y2 = [int(value) for value in row["af_crop_box"].split(",")]
        with Image.open(pseudo[specimen]) as image:
            width, height = image.size
            for path in paths:
                match = NAME_RE.search(path.name)
                if match is None:
                    raise ValueError(f"bad patch name {path.name}")
                rel_y, rel_x = int(match.group(1)), int(match.group(2))
                box = (x1 + rel_x, y1 + rel_y, x1 + rel_x + PATCH, y1 + rel_y + PATCH)
                if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
                    raise RuntimeError(f"{path.name} box {box} outside JPEG {width}x{height}")
                out = OUT_ROOT / path.name
                if out.is_file() and out.stat().st_size > 0:
                    skipped += 1
                    continue
                image.crop(box).convert("RGB").save(out, compress_level=1)
                written += 1
        print(
            f"{index}/{len(by_slide)} {slide_id} patches={len(paths)} "
            f"written={written} skipped={skipped}",
            flush=True,
        )
    print(f"done written={written} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
