"""Side-by-side AF patch and pseudo-label crop, plus offset search for held slides."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

CROP_MANIFEST = Path(
    "/mnt/10tb_hdd/taek/stain_translation_clean/manifests/fresh/crop_manifest_for_thumbs.csv"
)
PSEUDO_ROOT = Path("/mnt/10tb_hdd/taek/images/202610_inference_training_set")
PATCH_ROOT = Path("/mnt/10tb_hdd/taek/images/Fresh_section/patches_1024/trainA")
THUMB_ROOTS = (
    Path("/mnt/10tb_hdd/taek/images/fresh_lung/batch1/thumbnails/AF"),
    Path("/mnt/10tb_hdd/taek/images/fresh_lung/batch2/thumbnails/AF"),
)
OUT = Path("/mnt/10tb_hdd/taek/af2he/collages/verify/pseudo_align")
PATCH = 1024
SHOW = 512

EXAMPLES = (
    "F25-06301-1",
    "F25-06321-1",
    "F25-12033-1",
    "F25-11651-1",
)


def specimen_key(name: str) -> str:
    stem = Path(name).stem
    for token in ("-AF-below-R1", "-AF-R1-1", "-AF-R1"):
        if token in stem:
            return stem.split(token)[0]
    return stem


def pseudo_paths() -> dict[str, Path]:
    return {
        specimen_key(path.name): path
        for path in PSEUDO_ROOT.iterdir()
        if path.suffix.lower() == ".jpg"
    }


def thumb_path(slide_id: str) -> Path:
    for root in THUMB_ROOTS:
        for suffix in (".png", ".jpg"):
            path = root / f"{slide_id}{suffix}"
            if path.is_file():
                return path
    raise FileNotFoundError(slide_id)


def thumb_geometry(width: int, height: int, thumb_size: int = 1024):
    scale = thumb_size / max(width, height)
    thumb_w = max(1, int(round(width * scale)))
    thumb_h = max(1, int(round(height * scale)))
    xoff = (thumb_size - thumb_w) // 2
    yoff = (thumb_size - thumb_h) // 2
    return xoff, yoff, thumb_w, thumb_h


def patch_list(slide_id: str) -> list[tuple[int, int, Path]]:
    found = []
    for path in PATCH_ROOT.glob(f"{slide_id}__*.png"):
        match = re.search(r"__y(\d+)_x(\d+)", path.name)
        found.append((int(match.group(1)), int(match.group(2)), path))
    return found


def pick_patches(items: list[tuple[int, int, Path]], count: int = 2):
    scored = []
    for rel_y, rel_x, path in items:
        with Image.open(path) as image:
            gray = np.asarray(image.convert("L").resize((128, 128)), dtype=np.float32)
        scored.append((float(gray.std()), rel_y, rel_x, path))
    scored.sort()
    # Upper-middle tissue, not the single brightest edge.
    start = int(len(scored) * 0.75)
    return scored[start:start + count]


def crop_pseudo(image: Image.Image, x1: int, y1: int) -> Image.Image:
    return image.crop((x1, y1, x1 + PATCH, y1 + PATCH)).convert("RGB")


def panel(images: list[Image.Image], title: str) -> Image.Image:
    tiles = [image.resize((SHOW, SHOW), Image.Resampling.BOX) for image in images]
    canvas = Image.new("RGB", (SHOW * len(tiles), SHOW + 28), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(255, 255, 255))
    for index, tile in enumerate(tiles):
        canvas.paste(tile, (index * SHOW, 28))
    return canvas


def visualize(crops: dict[str, dict], jpgs: dict[str, Path]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sheets = []
    for key in EXAMPLES:
        row = crops[key]
        slide_id = row["pair_id"]
        x1, y1, _x2, _y2 = [int(value) for value in row["af_crop_box"].split(",")]
        chosen = pick_patches(patch_list(slide_id))
        with Image.open(jpgs[key]) as pseudo:
            for std, rel_y, rel_x, path in chosen:
                af = Image.open(path).convert("RGB")
                absolute = crop_pseudo(pseudo, x1 + rel_x, y1 + rel_y)
                images = [af, absolute]
                label = f"{slide_id} y{rel_y} x{rel_x}  AF | pseudo absolute ({x1 + rel_x},{y1 + rel_y})"
                if x1 or y1:
                    images.append(crop_pseudo(pseudo, rel_x, rel_y))
                    label += " | pseudo relative"
                sheet = panel(images, label)
                out = OUT / f"{slide_id}__y{rel_y}_x{rel_x}.jpg"
                sheet.save(out, quality=90)
                sheets.append(np.asarray(sheet))
                print("wrote", out, "af_std", round(std, 1))
                af.close()
    if sheets:
        contact = np.concatenate(sheets, axis=0)
        Image.fromarray(contact).save(OUT / "contact.jpg", quality=85)
        print("wrote", OUT / "contact.jpg")


def match_hold(crops: dict[str, dict], jpgs: dict[str, Path]) -> None:
    print("\nheld slides")
    for key, path in sorted(jpgs.items()):
        row = crops.get(key)
        if row is None:
            print(key, "no crop manifest; not extractable from patch coordinates")
            continue
        with Image.open(path) as image:
            jw, jh = image.size
        aw, ah = int(row["af_width"]), int(row["af_height"])
        cw, ch = int(row["crop_width"]), int(row["crop_height"])
        if abs(aw - jw) <= 64 and abs(ah - jh) <= 64:
            continue
        x1, y1, _x2, _y2 = [int(value) for value in row["af_crop_box"].split(",")]
        slide_id = row["pair_id"]
        xoff, yoff, thumb_w, thumb_h = thumb_geometry(cw, ch)
        thumb = np.asarray(Image.open(thumb_path(slide_id)).convert("L"))
        content = thumb[yoff:yoff + thumb_h, xoff:xoff + thumb_w]
        with Image.open(path) as image:
            image.draft("L", (500, 500))
            draft = image.convert("L")
        template_w = max(8, int(round(jw * thumb_w / cw)))
        template_h = max(8, int(round(jh * thumb_h / ch)))
        template = cv2.resize(np.asarray(draft), (template_w, template_h), interpolation=cv2.INTER_AREA)
        if template.shape[0] >= content.shape[0] or template.shape[1] >= content.shape[1]:
            print(key, "pseudo larger than crop thumbnail; no contained offset")
            continue
        score = cv2.matchTemplate(content, template, cv2.TM_CCOEFF_NORMED)
        _, peak, _, where = cv2.minMaxLoc(score)
        match_x, match_y = where
        origin_x = x1 + int(round(match_x * cw / thumb_w))
        origin_y = y1 + int(round(match_y * ch / thumb_h))
        coords = [(y, x) for y, x, _path in patch_list(slide_id)]
        outside = 0
        for rel_y, rel_x in coords:
            ax, ay = x1 + rel_x - origin_x, y1 + rel_y - origin_y
            if ax < 0 or ay < 0 or ax + PATCH > jw or ay + PATCH > jh:
                outside += 1
        inside = len(coords) - outside
        print(
            f"{key} peak={peak:.3f} jpeg_origin_on_AF=({origin_x},{origin_y}) "
            f"patches_inside={inside}/{len(coords)}"
        )


def main() -> None:
    crops = {
        row["specimen_key"]: row
        for row in csv.DictReader(CROP_MANIFEST.open(newline=""))
    }
    jpgs = pseudo_paths()
    visualize(crops, jpgs)
    match_hold(crops, jpgs)


if __name__ == "__main__":
    import csv
    main()
