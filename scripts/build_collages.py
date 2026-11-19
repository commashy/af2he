"""Build three fresh-lung AF collages per slide.

    python scripts/build_collages.py
"""

from __future__ import annotations

import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.collage import build_collage, index_patches
from data.groups import group_key

ROOT = Path("/mnt/10tb_hdd/taek/images/Fresh_section/patches_1024/trainA")
THUMBS = Path("/mnt/10tb_hdd/taek/images/fresh_lung/batch1/thumbnails/AF")
OUT = Path("/mnt/10tb_hdd/taek/af2he/collages")
WORKERS = 8


def slides() -> list[str]:
    found = set()
    for path in ROOT.glob("*__*.png"):
        found.add(path.name.split("__", 1)[0])
    return sorted(found)


def build_one(slide: str) -> tuple[str, list[str] | None, str | None, str]:
    destination = OUT / "af" / slide
    destination.mkdir(parents=True, exist_ok=True)
    paths = [destination / f"v{version}.png" for version in range(3)]
    if all(path.is_file() and path.stat().st_size > 0 for path in paths):
        return slide, [str(path) for path in paths], None, ""
    try:
        patches = index_patches(ROOT, slide)
        excluded: set[tuple[str, int, int]] = set()
        written: list[str] = []
        kinds: dict[str, int] = {}
        for version, path in enumerate(paths):
            image, records = build_collage(
                ROOT, slide, THUMBS, version=version,
                patches=patches, native_exclude=excluded,
            )
            Image.fromarray(image, mode="RGB").save(path, compress_level=1)
            written.append(str(path))
            for record in records:
                kinds[record["kind"]] = kinds.get(record["kind"], 0) + 1
                if record["kind"] == "native":
                    excluded.add((record["path"], record["row"], record["col"]))
        note = ""
        if kinds.get("2048", 0) < 9 or kinds.get("512") or kinds.get("256"):
            note = " " + " ".join(f"{name}={kinds[name]}" for name in ("2048", "1024", "512", "256", "native") if kinds.get(name))
        return slide, written, None, note
    except Exception as exc:
        return slide, None, f"{type(exc).__name__}: {exc}", ""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = slides()
    print(f"slides {len(jobs)}", flush=True)
    index: dict[str, list[str]] = {}
    failed: list[str] = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(build_one, slide) for slide in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            slide, paths, error, note = future.result()
            if error is None and paths is not None:
                index[group_key(ROOT, slide)] = paths
            else:
                failed.append(f"{slide} {error}")
            print(f"collages {done}/{len(futures)} {slide} ok={len(index)} failed={len(failed)}{note}", flush=True)
    (OUT / "index.json").write_text(json.dumps({"groups": index}, indent=0), encoding="utf-8")
    if failed:
        (OUT / "failed.txt").write_text("\n".join(failed) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'index.json'} groups={len(index)} failed={len(failed)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
