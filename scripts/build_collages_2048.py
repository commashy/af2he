"""Six 2048 AF collages for each of the 425 training slides.

Same 4x4 layout as the 1024 collages, with 512-pixel cells. Fields are taken
from different quadrants, at or above the median texture. Existing files are
left in place, so the job can be started again.

    python scripts/build_collages_2048.py
"""

from __future__ import annotations

import csv
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.collage import COLLAGE_VERSIONS, build_spread_collage, index_patches
from data.groups import group_key

AF_ROOT = Path("/mnt/10tb_hdd/taek/images/Fresh_section/patches_1024/trainA")
THUMBS = Path("/mnt/10tb_hdd/taek/images/fresh_lung/batch1/thumbnails/AF")
INDEX = ROOT / "retrieval" / "index.csv"
OUT = ROOT / "collages_2048"
WORKERS = 8


def training_slides() -> list[str]:
    slides: set[str] = set()
    with INDEX.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            slides.add(row["slide"])
    if len(slides) != 425:
        raise RuntimeError(f"expected 425 training slides, found {len(slides)}")
    return sorted(slides)


def build_one(slide: str) -> tuple[str, list[str] | None, str | None]:
    destination = OUT / "af" / slide
    destination.mkdir(parents=True, exist_ok=True)
    paths = [destination / f"v{version}.png" for version in range(COLLAGE_VERSIONS)]
    if all(path.is_file() and path.stat().st_size > 0 for path in paths):
        return slide, [str(path) for path in paths], None
    try:
        patches = index_patches(AF_ROOT, slide)
        excluded: set = set()
        for version, path in enumerate(paths):
            if path.is_file() and path.stat().st_size > 0:
                continue
            try:
                image, records = build_spread_collage(
                    AF_ROOT, slide, THUMBS, version=version, patches=patches, native_exclude=excluded,
                )
            except RuntimeError:
                image, records = build_spread_collage(
                    AF_ROOT, slide, THUMBS, version=version, patches=patches, native_exclude=set(),
                )
            if image.shape != (2048, 2048, 3):
                raise RuntimeError(f"{slide} v{version} is {image.shape}")
            temporary = path.with_suffix(".png.partial")
            Image.fromarray(image, mode="RGB").save(temporary, format="PNG", compress_level=1)
            temporary.replace(path)
            for record in records:
                key = record.get("key")
                if key is not None:
                    excluded.add(tuple(key))
        return slide, [str(path) for path in paths], None
    except Exception as exc:
        return slide, None, f"{type(exc).__name__}: {exc}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = training_slides()
    print(f"slides {len(jobs)} versions {COLLAGE_VERSIONS} out {OUT}", flush=True)
    index: dict[str, list[str]] = {}
    failed: list[str] = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(build_one, slide) for slide in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            slide, paths, error = future.result()
            if error is None and paths is not None:
                index[group_key(AF_ROOT, slide)] = paths
            else:
                failed.append(f"{slide} {error}")
            if done % 10 == 0 or error is not None or done == len(futures):
                print(
                    f"collages {done}/{len(futures)} {slide} ok={len(index)} failed={len(failed)}",
                    flush=True,
                )
    (OUT / "index.json").write_text(json.dumps({"groups": index}, indent=0), encoding="utf-8")
    if failed:
        (OUT / "failed.txt").write_text("\n".join(failed) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'index.json'} groups={len(index)} failed={len(failed)}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
