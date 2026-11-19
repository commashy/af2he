"""Audit batch-1 pseudo-label WSIs against the fresh AF crop contract.

Training patches are named ``{date}_{specimen}__y{rel_y}_x{rel_x}``.
``rel_x`` and ``rel_y`` are inside the AF crop box from the crop manifest,
not raw WSI coordinates. The absolute AF box is

    af_crop_x1 + rel_x, af_crop_y1 + rel_y, plus the patch size.

The pseudo JPEG is a full-slide image on that AF canvas, often trimmed by
at most 64 px on the right and bottom. Crop it at the absolute box.
Do not resize it to the AF slide.

    python scripts/prepare_pseudo_wsi.py
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

AF_ROOT = Path("/mnt/10tb_hdd/taek/images/Fresh_section/Training-WSI/AF_batch1")
HE_ROOT = Path("/mnt/10tb_hdd/taek/images/Fresh_section/HE-WSI")
PSEUDO_ROOT = Path("/mnt/10tb_hdd/taek/images/202610_inference_training_set")
PATCH_ROOT = Path("/mnt/10tb_hdd/taek/images/Fresh_section/patches_1024/trainA")
CROP_MANIFEST = Path(
    "/mnt/10tb_hdd/taek/stain_translation_clean/manifests/fresh/crop_manifest_for_thumbs.csv"
)
OUT = Path("/mnt/10tb_hdd/taek/af2he/manifests/pseudo_alignment.csv")
PATCH = 1024
TRIM = 64
SPECIMEN_RE = re.compile(r"__y(\d+)_x(\d+)")


def specimen_key(name: str) -> str:
    stem = Path(name).stem
    for token in ("-HE_icc-", "-AF-below-", "-AF-"):
        if token in stem:
            return stem.split(token)[0]
    return stem


def index_files(root: Path, suffixes: set[str]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            found[specimen_key(path.name)] = path
    return found


def patch_boxes(root: Path) -> dict[str, list[tuple[int, int]]]:
    boxes: dict[str, list[tuple[int, int]]] = {}
    for path in root.glob("*.png"):
        specimen = path.name.split("_", 1)[1].split("__", 1)[0]
        match = SPECIMEN_RE.search(path.name)
        if match is None:
            raise ValueError(f"patch name has no y/x: {path.name}")
        boxes.setdefault(specimen, []).append((int(match.group(1)), int(match.group(2))))
    return boxes


def fits(coords: list[tuple[int, int]], origin_x: int, origin_y: int, width: int, height: int) -> int:
    outside = 0
    for rel_y, rel_x in coords:
        x1 = origin_x + rel_x
        y1 = origin_y + rel_y
        if x1 < 0 or y1 < 0 or x1 + PATCH > width or y1 + PATCH > height:
            outside += 1
    return outside


def main() -> None:
    af = index_files(AF_ROOT, {".tif", ".tiff"})
    he = index_files(HE_ROOT, {".jpg", ".jpeg", ".tif", ".tiff"})
    pseudo = index_files(PSEUDO_ROOT, {".jpg", ".jpeg"})
    boxes = patch_boxes(PATCH_ROOT)
    crops = {
        row["specimen_key"]: row
        for row in csv.DictReader(CROP_MANIFEST.open(newline=""))
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in sorted(set(af) | set(pseudo)):
        row = crops.get(key)
        pseudo_path = pseudo.get(key)
        af_path = af.get(key)
        he_path = he.get(key)
        record = {
            "specimen_key": key,
            "in_crop_manifest": bool(row),
            "has_af": af_path is not None,
            "has_he": he_path is not None,
            "has_pseudo": pseudo_path is not None,
            "af_width": "",
            "af_height": "",
            "he_width": "",
            "he_height": "",
            "pseudo_width": "",
            "pseudo_height": "",
            "crop_box": "" if row is None else row["af_crop_box"],
            "n_patches": len(boxes.get(key, [])),
            "patches_outside_absolute": "",
            "use": "",
        }
        if af_path is not None:
            with Image.open(af_path) as image:
                record["af_width"], record["af_height"] = image.size
        if he_path is not None:
            with Image.open(he_path) as image:
                record["he_width"], record["he_height"] = image.size
        if pseudo_path is not None:
            with Image.open(pseudo_path) as image:
                record["pseudo_width"], record["pseudo_height"] = image.size
        if row is not None and pseudo_path is not None and key in boxes:
            x1, y1, _x2, _y2 = [int(v) for v in row["af_crop_box"].split(",")]
            outside = fits(
                boxes[key], x1, y1,
                int(record["pseudo_width"]), int(record["pseudo_height"]),
            )
            record["patches_outside_absolute"] = outside
            aw, ah = int(record["af_width"]), int(record["af_height"])
            pw, ph = int(record["pseudo_width"]), int(record["pseudo_height"])
            trimmed = 0 <= aw - pw <= TRIM and 0 <= ah - ph <= TRIM
            if outside == 0 and trimmed:
                record["use"] = "crop_absolute"
            else:
                record["use"] = "hold"
        elif pseudo_path is None:
            record["use"] = "no_pseudo"
        else:
            record["use"] = "hold"
        rows.append(record)
    fields = list(rows[0].keys())
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for record in rows:
        counts[record["use"]] = counts.get(record["use"], 0) + 1
    print(f"wrote {OUT}")
    print(counts)


if __name__ == "__main__":
    main()
