"""Match each epoch-10 field to a nearby real H&E crop.

The search stays on the same slide, within 2048 px. Tissue amount is the
hematoxylin-plus-eosin fraction. Morphology is normalized correlation of the
downsampled stain density, so stain strength is not the target. The chosen
crop is written under he_match_e10/images with the epoch-10 filename.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/mnt/10tb_hdd/taek/af2he")
E10 = ROOT / "e10_overlap"
HE = Path("/mnt/10tb_hdd/taek/images/Fresh_section/patches_1024/trainB")
INDEX = ROOT / "retrieval" / "index.csv"
OUT = ROOT / "he_match_e10"

PATCH = 1024
SCALE = 8
LOW = PATCH // SCALE
RADIUS = 2048 // SCALE
TISSUE_GATE = 0.15
GRID = 1024

_RGB_FROM_HED = np.array(
    [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]],
    dtype=np.float64,
)
_HED_FROM_RGB = np.linalg.inv(_RGB_FROM_HED).astype(np.float32)


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    if bgr.shape != (PATCH, PATCH, 3):
        raise ValueError(f"{path} has shape {bgr.shape}")
    return np.ascontiguousarray(bgr[:, :, ::-1])


def stain_density(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Block density, block tissue fraction, and the full-resolution fraction."""
    rgb01 = rgb.astype(np.float32)
    rgb01 *= 1.0 / 255.0
    np.clip(rgb01, 1e-6, 1.0, out=rgb01)
    optical = np.log(rgb01)
    optical *= -1.0
    hematoxylin = optical @ _HED_FROM_RGB[0]
    eosin = optical @ _HED_FROM_RGB[1]
    np.clip(hematoxylin, 0.0, None, out=hematoxylin)
    np.clip(eosin, 0.0, None, out=eosin)
    stain = hematoxylin
    stain += eosin
    tissue = float((stain > TISSUE_GATE).mean())
    blocks = stain.reshape(LOW, SCALE, LOW, SCALE)
    density = blocks.mean(axis=(1, 3)).astype(np.float32)
    covered = (blocks > TISSUE_GATE).mean(axis=(1, 3)).astype(np.float32)
    return density, covered, tissue


def window_sums(image: np.ndarray, size: int) -> np.ndarray:
    integral = np.pad(image.astype(np.float64, copy=False), ((1, 0), (1, 0)))
    integral = integral.cumsum(0).cumsum(1)
    summed = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return summed.astype(np.float32)


def tile_origins(start: int, length: int, phase: int, grid: int = GRID) -> list[int]:
    phase = phase % grid
    cursor = phase + ((start - phase) // grid) * grid
    end = start + length
    origins = []
    while cursor < end:
        origins.append(cursor)
        cursor += grid
    return origins


def stitch_crop(
    tiles: dict[tuple[int, int], np.ndarray],
    y: int,
    x: int,
    dy: int,
    dx: int,
) -> np.ndarray:
    y0 = y + dy
    x0 = x + dx
    out = np.empty((PATCH, PATCH, 3), dtype=np.uint8)
    filled = np.zeros((PATCH, PATCH), dtype=bool)
    for ty in tile_origins(y0, PATCH, y):
        for tx in tile_origins(x0, PATCH, x):
            src_y0 = max(0, y0 - ty)
            src_x0 = max(0, x0 - tx)
            src_y1 = min(PATCH, y0 + PATCH - ty)
            src_x1 = min(PATCH, x0 + PATCH - tx)
            dst_y0 = max(0, ty - y0)
            dst_x0 = max(0, tx - x0)
            dst_y1 = dst_y0 + (src_y1 - src_y0)
            dst_x1 = dst_x0 + (src_x1 - src_x0)
            out[dst_y0:dst_y1, dst_x0:dst_x1] = tiles[(ty, tx)][src_y0:src_y1, src_x0:src_x1]
            filled[dst_y0:dst_y1, dst_x0:dst_x1] = True
    if not filled.all():
        raise RuntimeError(f"crop y={y0} x={x0} was not fully covered")
    return out


def choose_offset(
    ncc: np.ndarray,
    tissue: np.ndarray,
    valid: np.ndarray,
    origin_y: int,
    origin_x: int,
    query_tissue: float,
) -> tuple[int, int, float, float, int]:
    """Return low-res (row, col), correlation, tissue, and whether the gate passed."""
    if ncc.shape != tissue.shape or ncc.shape != valid.shape:
        raise ValueError(f"maps {ncc.shape} {tissue.shape} {valid.shape} differ")
    shift_y = np.arange(ncc.shape[0], dtype=np.int32)[:, None] - int(origin_y)
    shift_x = np.arange(ncc.shape[1], dtype=np.int32)[None, :] - int(origin_x)
    shift = np.abs(shift_y) + np.abs(shift_x)
    usable = valid & np.isfinite(ncc)
    if not bool(usable.any()):
        raise RuntimeError("no fully covered H&E crop in the search window")
    correlation = np.where(usable, ncc, -np.inf)
    gap = np.abs(tissue - float(query_tissue))
    eligible = usable & (gap <= TISSUE_GATE)
    if bool(eligible.any()):
        flat = np.flatnonzero(eligible.ravel())
        picked = flat[np.lexsort((shift.ravel()[flat], -correlation.ravel()[flat]))][0]
        accepted = 1
    else:
        ranked_gap = np.where(usable, gap, np.inf)
        flat = np.flatnonzero(usable.ravel())
        picked = flat[np.lexsort((shift.ravel()[flat], ranked_gap.ravel()[flat]))][0]
        accepted = 0
    row, col = np.unravel_index(int(picked), ncc.shape)
    return int(row), int(col), float(ncc[row, col]), float(tissue[row, col]), accepted


def _check_choose() -> None:
    ncc = np.array([[0.2, 0.9], [0.85, 0.1]], dtype=np.float32)
    tissue = np.array([[0.50, 0.90], [0.52, 0.10]], dtype=np.float32)
    valid = np.ones((2, 2), dtype=bool)
    row, col, score, amount, accepted = choose_offset(ncc, tissue, valid, 0, 0, 0.50)
    if (row, col) != (1, 0) or accepted != 1 or abs(score - 0.85) > 1e-6:
        raise AssertionError(f"expected the similar-tissue neighbor, got {(row, col, score, amount, accepted)}")
    farther = np.array([[0.50, 0.90], [0.52, 0.40]], dtype=np.float32)
    row, col, _score, amount, accepted = choose_offset(ncc, farther, valid, 0, 0, 0.0)
    if (row, col) != (1, 1) or accepted != 0 or abs(amount - 0.40) > 1e-6:
        raise AssertionError(f"expected the closest tissue when none pass, got {(row, col, amount, accepted)}")


def load_fields(index_path: Path) -> dict[str, list[tuple[int, str, int, int]]]:
    grouped: dict[str, list[tuple[int, str, int, int]]] = {}
    with index_path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            slide = record["slide"]
            grouped.setdefault(slide, []).append(
                (int(record["row"]), record["filename"], int(record["y"]), int(record["x"]))
            )
    if not grouped:
        raise RuntimeError(f"{index_path} has no fields")
    return grouped


def match_slide(task: dict) -> dict:
    cv2.setNumThreads(1)
    slide = task["slide"]
    fields: list[tuple[int, str, int, int]] = task["fields"]
    e10_root = Path(task["e10"])
    he_root = Path(task["he"])
    out = Path(task["out"])
    image_dir = out / "images"
    preview_dir = out / "preview"
    done_path = out / "done" / f"{slide}.ok"
    csv_path = out / "slides" / f"{slide}.csv"
    if done_path.is_file():
        return {"slide": slide, "fields": len(fields), "skipped": 1}
    started = time.time()
    print(f"start {slide} fields={len(fields)}", flush=True)
    tiles: dict[tuple[int, int], np.ndarray] = {}
    ys = []
    xs = []
    for _row, filename, y, x in fields:
        if y % GRID or x % GRID:
            raise RuntimeError(f"{filename} is not on the {GRID} grid")
        tiles[(y, x)] = read_rgb(he_root / filename)
        ys.append(y)
        xs.append(x)
    ymin, xmin = min(ys), min(xs)
    height = (max(ys) - ymin) // SCALE + LOW
    width = (max(xs) - xmin) // SCALE + LOW
    density = np.zeros((height, width), dtype=np.float32)
    covered = np.zeros((height, width), dtype=np.float32)
    valid = np.zeros((height, width), dtype=np.uint8)
    for (y, x), rgb in tiles.items():
        tile_density, tile_tissue, _fraction = stain_density(rgb)
        oy = (y - ymin) // SCALE
        ox = (x - xmin) // SCALE
        density[oy:oy + LOW, ox:ox + LOW] = tile_density
        covered[oy:oy + LOW, ox:ox + LOW] = tile_tissue
        valid[oy:oy + LOW, ox:ox + LOW] = 1
    tissue_window = window_sums(covered, LOW) / float(LOW * LOW)
    valid_window = window_sums(valid, LOW)
    queries: dict[tuple[int, int], tuple[np.ndarray, float]] = {}
    for _row, filename, y, x in fields:
        query_density, _blocks, query_tissue = stain_density(read_rgb(e10_root / filename))
        queries[(y, x)] = (query_density, query_tissue)
    rows = []
    previews = []
    tissue_error = 0.0
    checked = 0
    for index, (row, filename, y, x) in enumerate(fields):
        oy = (y - ymin) // SCALE
        ox = (x - xmin) // SCALE
        y0 = max(0, oy - RADIUS)
        x0 = max(0, ox - RADIUS)
        y1 = min(tissue_window.shape[0] - 1, oy + RADIUS)
        x1 = min(tissue_window.shape[1] - 1, ox + RADIUS)
        search = density[y0:y1 + LOW, x0:x1 + LOW]
        template, query_tissue = queries[(y, x)]
        ncc = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED).astype(np.float32)
        local_tissue = tissue_window[y0:y1 + 1, x0:x1 + 1]
        local_valid = valid_window[y0:y1 + 1, x0:x1 + 1] >= float(LOW * LOW) - 0.5
        if ncc.shape != local_tissue.shape:
            raise RuntimeError(f"{filename} correlation {ncc.shape} != tissue {local_tissue.shape}")
        local_row, local_col, score, amount, accepted = choose_offset(
            ncc, local_tissue, local_valid, oy - y0, ox - x0, query_tissue,
        )
        dy = (y0 + local_row - oy) * SCALE
        dx = (x0 + local_col - ox) * SCALE
        zero = ncc[oy - y0, ox - x0]
        gap0 = abs(float(local_tissue[oy - y0, ox - x0]) - query_tissue)
        crop = tiles[(y, x)] if dy == 0 and dx == 0 else stitch_crop(tiles, y, x, dy, dx)
        if checked < 8:
            _density, _blocks, full = stain_density(crop)
            tissue_error = max(tissue_error, abs(full - amount))
            checked += 1
        destination = image_dir / filename
        if not cv2.imwrite(
            str(destination),
            np.ascontiguousarray(crop[:, :, ::-1]),
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        ):
            raise RuntimeError(f"failed to write {destination}")
        gap = abs(amount - query_tissue)
        rows.append((
            row, filename, slide, y, x, dy, dx,
            score, float(zero), query_tissue, amount, gap, gap0, accepted,
        ))
        if task["preview"] and (accepted == 1) and (len(previews) < int(task["preview"])):
            if abs(dy) + abs(dx) >= 256 or len(previews) < 2:
                previews.append((filename, y, x, dy, dx, score, gap, accepted, queries[(y, x)][0]))
    if tissue_error > 0.02:
        raise RuntimeError(f"{slide} low-res tissue disagreed with the crop by {tissue_error:.3f}")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "row", "filename", "slide", "y", "x", "dy", "dx",
            "ncc", "ncc0", "tissue_e10", "tissue_he", "gap", "gap0", "accepted",
        ])
        writer.writerows(rows)
    if previews:
        preview_dir.mkdir(parents=True, exist_ok=True)
        for filename, y, x, dy, dx, score, gap, accepted, _template in previews[: int(task["preview"])]:
            query = read_rgb(e10_root / filename)
            nominal = tiles[(y, x)]
            matched = tiles[(y, x)] if dy == 0 and dx == 0 else stitch_crop(tiles, y, x, dy, dx)
            panels = [query, nominal, matched]
            shown = [
                cv2.resize(panel, (512, 512), interpolation=cv2.INTER_AREA)
                for panel in panels
            ]
            sheet = np.concatenate(shown, axis=1)
            text = f"{filename} shift=({dy:+d},{dx:+d}) ncc={score:.3f} gap={gap:.3f} accepted={accepted}"
            cv2.putText(sheet, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            preview_path = preview_dir / f"{Path(filename).stem}.jpg"
            cv2.imwrite(str(preview_path), np.ascontiguousarray(sheet[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    done_path.parent.mkdir(parents=True, exist_ok=True)
    done_path.write_text("ok\n", encoding="utf-8")
    accepted_count = sum(item[-1] for item in rows)
    shifts = [abs(item[5]) + abs(item[6]) for item in rows]
    elapsed = time.time() - started
    print(
        f"done {slide} fields={len(fields)} accepted={accepted_count}/{len(fields)} "
        f"median_shift={int(np.median(shifts))} seconds={elapsed:.1f}",
        flush=True,
    )
    return {
        "slide": slide,
        "fields": len(fields),
        "accepted": accepted_count,
        "seconds": elapsed,
        "skipped": 0,
        "bytes": sum((image_dir / item[1]).stat().st_size for item in rows),
    }


def merge_matches(out: Path, index_path: Path) -> dict[str, float]:
    grouped = load_fields(index_path)
    total = sum(len(items) for items in grouped.values())
    dy = np.full(total, np.iinfo(np.int16).min, dtype=np.int16)
    dx = np.full(total, np.iinfo(np.int16).min, dtype=np.int16)
    ncc = np.full(total, np.nan, dtype=np.float32)
    gap = np.full(total, np.nan, dtype=np.float32)
    accepted = np.zeros(total, dtype=np.uint8)
    filled = np.zeros(total, dtype=bool)
    columns = [
        "row", "filename", "slide", "y", "x", "dy", "dx",
        "ncc", "ncc0", "tissue_e10", "tissue_he", "gap", "gap0", "accepted",
    ]
    table = out / "index.csv"
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for slide in sorted(grouped):
            csv_path = out / "slides" / f"{slide}.csv"
            if not csv_path.is_file():
                raise RuntimeError(f"missing {csv_path}")
            with csv_path.open(newline="", encoding="utf-8") as source:
                for record in csv.DictReader(source):
                    row = int(record["row"])
                    if row < 0 or row >= total or filled[row]:
                        raise RuntimeError(f"bad or repeated row {row} in {csv_path}")
                    writer.writerow([record[name] for name in columns])
                    dy[row] = np.int16(int(record["dy"]))
                    dx[row] = np.int16(int(record["dx"]))
                    ncc[row] = float(record["ncc"])
                    gap[row] = float(record["gap"])
                    accepted[row] = np.uint8(int(record["accepted"]))
                    filled[row] = True
    missing = int((~filled).sum())
    if missing:
        raise RuntimeError(f"{missing} fields have no match row")
    np.save(out / "dy.npy", dy)
    np.save(out / "dx.npy", dx)
    np.save(out / "ncc.npy", ncc)
    np.save(out / "gap.npy", gap)
    np.save(out / "accepted.npy", accepted)
    return {
        "fields": float(total),
        "accepted": float(accepted.mean()),
        "median_gap": float(np.median(gap)),
        "median_shift": float(np.median(np.abs(dy.astype(np.int32)) + np.abs(dx.astype(np.int32)))),
    }


def check_against_torch(e10_root: Path, filename: str) -> None:
    import sys
    import torch

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from data.dataset import load_native_image
    from engine.image_losses import hed_tissue_fraction

    rgb = read_rgb(e10_root / filename)
    _density, _blocks, tissue = stain_density(rgb)
    image = load_native_image(e10_root / filename).unsqueeze(0)
    reference = float(hed_tissue_fraction(image)[0])
    if abs(tissue - reference) > 1e-4:
        raise RuntimeError(f"tissue {tissue:.6f} != training definition {reference:.6f} for {filename}")
    del torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--slide", action="append", default=[])
    parser.add_argument("--preview", type=int, default=0)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    _check_choose()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "slides").mkdir(parents=True, exist_ok=True)
    (OUT / "done").mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        print(merge_matches(OUT, INDEX), flush=True)
        return
    free = shutil.disk_usage(OUT).free
    if free < 700 * 1024 ** 3:
        raise RuntimeError(f"only {free / 1024 ** 3:.0f} GiB free; the crops need several hundred")
    grouped = load_fields(INDEX)
    first = next(iter(grouped.values()))[0][1]
    check_against_torch(E10, first)
    meta = {
        "e10": str(E10),
        "he": str(HE),
        "radius_px": RADIUS * SCALE,
        "scale": SCALE,
        "tissue_gate": TISSUE_GATE,
        "score": "max density correlation among crops with tissue gap <= 0.15; otherwise closest tissue",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    wanted = set(args.slide) if args.slide else set(grouped)
    missing = sorted(wanted - set(grouped))
    if missing:
        raise RuntimeError(f"unknown slide {missing[:3]}")
    tasks = []
    for slide, fields in grouped.items():
        if slide not in wanted:
            continue
        tasks.append({
            "slide": slide,
            "fields": fields,
            "e10": str(E10),
            "he": str(HE),
            "out": str(OUT),
            "preview": args.preview if len(tasks) == 0 else 0,
        })
    tasks.sort(key=lambda item: len(item["fields"]), reverse=True)
    print(f"slides={len(tasks)} workers={args.workers}", flush=True)
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.workers == 1:
        results = [match_slide(task) for task in tasks]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
            futures = [pool.submit(match_slide, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
    done = [item for item in results if not item.get("skipped")]
    if done:
        total_fields = sum(item["fields"] for item in done)
        total_bytes = sum(item["bytes"] for item in done)
        print(
            f"wrote {total_fields} crops, {total_bytes / 1024 ** 3:.1f} GiB, "
            f"{total_bytes / max(total_fields, 1) / 1024:.0f} KiB each",
            flush=True,
        )
    if not args.slide:
        summary = merge_matches(OUT, INDEX)
        print(f"merged {summary}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
