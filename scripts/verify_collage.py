"""Build a few fresh-lung collages and write a labeled contact sheet."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.collage import CELL, build_collage

ROOT = Path("/mnt/10tb_hdd/taek/images/Fresh_section/patches_1024/trainA")
THUMBS = Path("/mnt/10tb_hdd/taek/thumbnails")
OUT = Path("/mnt/10tb_hdd/taek/af2he/collages/verify")


def label(collage: np.ndarray, records: list[dict]) -> np.ndarray:
    image = Image.fromarray(collage, mode="RGB")
    draw = ImageDraw.Draw(image)
    names = ["T"] + [f"L{i}" for i in range(3)] + [f"M{i}" for i in range(4)] + [f"N{i}" for i in range(8)]
    for index, (name, record) in enumerate(zip(names, records)):
        row, col = divmod(index, 4)
        text = f"{name} {record['mean']:.0f}/{record['std']:.0f}"
        draw.rectangle((col * CELL, row * CELL, col * CELL + 118, row * CELL + 16), fill=(0, 0, 0))
        draw.text((col * CELL + 2, row * CELL + 2), text, fill=(255, 255, 255))
    return np.asarray(image)


def main() -> None:
    slides = [
        "20250918_F25-13053-1",
        "20250603_F25-07342-1",
        "20250520_F25-06675-1",
    ]
    missing = None
    thumbs = {path.stem for path in THUMBS.glob("*.png")}
    for path in sorted(ROOT.glob("*.png")):
        slide = path.name.split("__", 1)[0]
        if slide not in thumbs:
            missing = slide
            break
    if missing is None:
        raise RuntimeError("every slide has a thumbnail")
    slides.append(missing)
    OUT.mkdir(parents=True, exist_ok=True)
    sheets = []
    for slide in slides:
        collage, records = build_collage(ROOT, slide, THUMBS, version=0)
        native = [item for item in records if item["kind"] == "native"]
        means = [item["mean"] for item in native]
        stds = [item["std"] for item in native]
        places = {(item["y"], item["x"]) for item in native}
        print(
            f"{slide} thumb={records[0]['mean']:.1f}/{records[0]['std']:.1f} "
            f"native_mean {min(means):.1f}-{max(means):.1f} "
            f"native_std {min(stds):.1f}-{max(stds):.1f} "
            f"native_places {len(places)}/8"
        )
        for record in records:
            extra = ""
            if "y" in record:
                extra = f" y={record['y']} x={record['x']}"
            print(f"  {record['kind']:6} mean={record['mean']:.1f} std={record['std']:.1f}{extra}")
        sheet = label(collage, records)
        Image.fromarray(sheet, mode="RGB").save(OUT / f"{slide}.jpg", quality=90)
        sheets.append(sheet)
    contact = np.concatenate(sheets, axis=0)
    Image.fromarray(contact, mode="RGB").save(OUT / "contact.jpg", quality=85)
    print(f"wrote {OUT / 'contact.jpg'}")


if __name__ == "__main__":
    main()
