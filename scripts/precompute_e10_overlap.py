"""Overlapping epoch-10 stains for every training patch.

Same 512 window and stride 384 as the teacher cache. The kept centers meet
with a 12-pixel linear blend instead of a hard edge. Saved files are the
1024 training-grid patches. The original teacher cache is not overwritten.

    python scripts/precompute_e10_overlap.py
"""

from __future__ import annotations

import csv
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.generator import Generator

_SPEC = importlib.util.spec_from_file_location("precompute_teacher", ROOT / "scripts" / "precompute_teacher.py")
_TEACHER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TEACHER)
CROP = _TEACHER.CROP
PATCH = _TEACHER.PATCH
EDGE_MARGIN = _TEACHER.EDGE_MARGIN
collect_slides = _TEACHER.collect_slides
context_origin = _TEACHER.context_origin
crop_positions = _TEACHER.crop_positions
load_af_index = _TEACHER.load_af_index
read_row = _TEACHER.read_row
save_png = _TEACHER.save_png
balance = _TEACHER.balance

from fresh_processing_utils import TiffRegionReader

OUT_ROOT = ROOT / "e10_overlap"
COLLAGE_ROOT = ROOT / "collages" / "af"
CHECKPOINT = ROOT / "checkpoints/stage1/epoch_0010.pt"
INDEX = ROOT / "retrieval" / "index.csv"
FEATHER = 12


def _axis_weight() -> np.ndarray:
    """Positive on the kept center plus 12 px past each side. The outer rim stays zero."""
    weight = np.zeros(CROP, dtype=np.float32)
    lo = EDGE_MARGIN - FEATHER
    rise = EDGE_MARGIN + FEATHER
    fall = CROP - EDGE_MARGIN - FEATHER
    hi = CROP - EDGE_MARGIN + FEATHER
    count = rise - lo
    weight[lo:rise] = np.linspace(1.0 / count, 1.0, count)
    weight[rise:fall] = 1.0
    weight[fall:hi] = np.linspace(1.0, 1.0 / count, count)
    return weight


_AXIS = _axis_weight()
_WEIGHT = _AXIS[:, None] * _AXIS[None, :]
_RIM = _WEIGHT <= 0


def blend_crop(
    image: np.ndarray,
    crop_y: int,
    crop_x: int,
    accum: dict[tuple[int, int], np.ndarray],
    cover: dict[tuple[int, int], np.ndarray],
) -> None:
    """Add one 512 prediction with a 12-pixel blend where kept centers meet."""
    y0 = (crop_y // PATCH) * PATCH
    x0 = (crop_x // PATCH) * PATCH
    for patch_y in (y0 - PATCH, y0, y0 + PATCH):
        for patch_x in (x0 - PATCH, x0, x0 + PATCH):
            if (patch_y, patch_x) not in accum:
                continue
            if crop_y >= patch_y + PATCH or crop_y + CROP <= patch_y:
                continue
            if crop_x >= patch_x + PATCH or crop_x + CROP <= patch_x:
                continue
            src_y0 = max(crop_y, patch_y) - crop_y
            src_x0 = max(crop_x, patch_x) - crop_x
            dst_y0 = max(crop_y, patch_y) - patch_y
            dst_x0 = max(crop_x, patch_x) - patch_x
            height = min(crop_y + CROP, patch_y + PATCH) - crop_y - src_y0
            width = min(crop_x + CROP, patch_x + PATCH) - crop_x - src_x0
            weight = _WEIGHT[src_y0:src_y0 + height, src_x0:src_x0 + width]
            take = weight > 0
            if not np.any(take):
                continue
            dst = accum[(patch_y, patch_x)][dst_y0:dst_y0 + height, dst_x0:dst_x0 + width]
            dst_w = cover[(patch_y, patch_x)][dst_y0:dst_y0 + height, dst_x0:dst_x0 + width]
            src = image[src_y0:src_y0 + height, src_x0:src_x0 + width]
            dst[take] += src[take] * weight[take, None]
            dst_w[take] += weight[take]


def blend_border(
    image: np.ndarray,
    crop_y: int,
    crop_x: int,
    accum: dict[tuple[int, int], np.ndarray],
    cover: dict[tuple[int, int], np.ndarray],
    holes: dict[tuple[int, int], np.ndarray],
) -> None:
    """Fill pixels the blend never reached, using the crop rim those pixels came from."""
    y0 = (crop_y // PATCH) * PATCH
    x0 = (crop_x // PATCH) * PATCH
    for patch_y in (y0 - PATCH, y0, y0 + PATCH):
        for patch_x in (x0 - PATCH, x0, x0 + PATCH):
            hole = holes.get((patch_y, patch_x))
            if hole is None:
                continue
            if crop_y >= patch_y + PATCH or crop_y + CROP <= patch_y:
                continue
            if crop_x >= patch_x + PATCH or crop_x + CROP <= patch_x:
                continue
            src_y0 = max(crop_y, patch_y) - crop_y
            src_x0 = max(crop_x, patch_x) - crop_x
            dst_y0 = max(crop_y, patch_y) - patch_y
            dst_x0 = max(crop_x, patch_x) - patch_x
            height = min(crop_y + CROP, patch_y + PATCH) - crop_y - src_y0
            width = min(crop_x + CROP, patch_x + PATCH) - crop_x - src_x0
            frame = _RIM[src_y0:src_y0 + height, src_x0:src_x0 + width]
            dest_hole = hole[dst_y0:dst_y0 + height, dst_x0:dst_x0 + width]
            take = frame & dest_hole
            if not np.any(take):
                continue
            dst = accum[(patch_y, patch_x)][dst_y0:dst_y0 + height, dst_x0:dst_x0 + width]
            dst_w = cover[(patch_y, patch_x)][dst_y0:dst_y0 + height, dst_x0:dst_x0 + width]
            src = image[src_y0:src_y0 + height, src_x0:src_x0 + width]
            dst[take] += src[take]
            dst_w[take] += 1.0


def crops_touching_holes(
    by_row: dict[int, list[int]],
    holes: dict[tuple[int, int], np.ndarray],
) -> list[tuple[int, int]]:
    needed: set[tuple[int, int]] = set()
    for (patch_y, patch_x), hole in holes.items():
        ys, xs = np.nonzero(hole)
        if len(ys) == 0:
            continue
        global_y = ys + patch_y
        global_x = xs + patch_x
        y_min = int(global_y.min())
        y_max = int(global_y.max()) + 1
        x_min = int(global_x.min())
        x_max = int(global_x.max()) + 1
        for crop_y, row in by_row.items():
            if crop_y >= y_max or crop_y + CROP <= y_min:
                continue
            for crop_x in row:
                if crop_x >= x_max or crop_x + CROP <= x_min:
                    continue
                local_y = global_y - crop_y
                local_x = global_x - crop_x
                inside = (local_y >= 0) & (local_y < CROP) & (local_x >= 0) & (local_x < CROP)
                if np.any(inside) and np.any(_RIM[local_y[inside], local_x[inside]]):
                    needed.add((crop_y, crop_x))
    return sorted(needed)


def paint_coverage(keys: set[tuple[int, int]], height: int, width: int) -> dict[tuple[int, int], np.ndarray]:
    positions = crop_positions(keys, height, width)
    by_row: dict[int, list[int]] = {}
    for y, x in positions:
        by_row.setdefault(y, []).append(x)
    accum = {key: np.zeros((PATCH, PATCH, 1), dtype=np.float32) for key in keys}
    cover = {key: np.zeros((PATCH, PATCH), dtype=np.float32) for key in keys}
    image = np.zeros((CROP, CROP, 1), dtype=np.float32)
    for y, x in positions:
        blend_crop(image, y, x, accum, cover)
    holes = {key: mask <= 0 for key, mask in cover.items() if float(mask.min()) <= 0}
    for y, x in crops_touching_holes(by_row, holes):
        blend_border(image, y, x, accum, cover, holes)
    return cover


def self_check() -> None:
    keys = {(y, x) for y in (0, PATCH) for x in range(0, PATCH * 3, PATCH)}
    cover = paint_coverage(keys, PATCH * 2, PATCH * 3)
    for key, mask in cover.items():
        if float(mask.min()) <= 0:
            missing = np.argwhere(mask <= 0)
            raise AssertionError(f"feathered paste leaves a hole in {key}, first at {missing[0].tolist()}")


def training_slides() -> set[str]:
    slides: set[str] = set()
    with INDEX.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            slides.add(row["slide"])
    if len(slides) != 425:
        raise RuntimeError(f"expected 425 training slides, found {len(slides)}")
    return slides


def slide_done(items: list[tuple[int, int, str]]) -> bool:
    for _, _, name in items:
        path = OUT_ROOT / name
        if not path.is_file() or path.stat().st_size == 0:
            return False
    return True


def load_collage(slide: str, device: torch.device) -> torch.Tensor:
    path = COLLAGE_ROOT / slide / "v0.png"
    if not path.is_file():
        raise FileNotFoundError(f"missing collage {path}")
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    if image.shape != (PATCH, PATCH, 3):
        raise RuntimeError(f"{path} is {image.shape}")
    tensor = torch.from_numpy(image).permute(2, 0, 1).mul_(2.0 / 255.0).sub_(1.0)
    return tensor.unsqueeze(0).to(device)


def run_slide(
    model: Generator,
    collage: torch.Tensor,
    slide: str,
    items: list[tuple[int, int, str]],
    source: tuple[Path, int, int, int, int],
    batch: int,
    rank: int,
) -> int:
    af_path, crop_x, crop_y, crop_h, crop_w = source
    keys = {(y, x) for y, x, _name in items}
    by_row: dict[int, list[int]] = {}
    for origin_y, origin_x in crop_positions(keys, crop_h, crop_w):
        by_row.setdefault(origin_y, []).append(origin_x)
    if not by_row:
        raise RuntimeError(f"{slide} produced no crops")
    accum = {key: np.zeros((PATCH, PATCH, 3), dtype=np.float32) for key in keys}
    cover = {key: np.zeros((PATCH, PATCH), dtype=np.float32) for key in keys}
    names = {(y, x): name for y, x, name in items}
    done = 0
    total = sum(len(xs) for xs in by_row.values())

    with TiffRegionReader(af_path) as reader:
        for origin_y in sorted(by_row):
            ctx_y = context_origin(origin_y, crop_h)
            xs = sorted(by_row[origin_y])
            x_start, strip = read_row(reader, crop_x, crop_y, ctx_y, [context_origin(x, crop_w) for x in xs])
            cursor = 0
            while cursor < len(xs):
                count = min(batch, len(xs) - cursor)
                chunk_x = xs[cursor:cursor + count]
                crops = []
                for x in chunk_x:
                    ctx_x = context_origin(x, crop_w)
                    field = strip[:, ctx_x - x_start:ctx_x - x_start + PATCH]
                    if field.shape != (PATCH, PATCH, 3):
                        raise RuntimeError(f"{slide} context at {ctx_y},{ctx_x} is {field.shape}")
                    crops.append(field[origin_y - ctx_y:origin_y - ctx_y + CROP, x - ctx_x:x - ctx_x + CROP])
                crop_tensor = torch.from_numpy(np.ascontiguousarray(np.stack(crops))).permute(0, 3, 1, 2)
                crop_tensor = crop_tensor.to(device=collage.device, dtype=torch.float32).mul_(2.0 / 255.0).sub_(1.0)
                try:
                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        predicted = model(crop_tensor, collage.expand(count, -1, -1, -1))
                    predicted = predicted.float().cpu().numpy()
                except torch.cuda.OutOfMemoryError:
                    if batch == 1:
                        raise
                    batch -= 1
                    torch.cuda.empty_cache()
                    print(f"gpu {rank} reduced batch to {batch}", flush=True)
                    continue
                for index, origin_x in enumerate(chunk_x):
                    blend_crop(np.transpose(predicted[index], (1, 2, 0)), origin_y, origin_x, accum, cover)
                cursor += count
                done += count
            if done == total or done % 200 < count:
                print(f"gpu {rank} {slide} crops {done}/{total}", flush=True)
        holes = {key: mask <= 0 for key, mask in cover.items() if float(mask.min()) <= 0}
        needed = crops_touching_holes(by_row, holes)
        if needed:
            print(f"gpu {rank} {slide} hole-fill crops {len(needed)}", flush=True)
            fill_rows: dict[int, list[int]] = {}
            for origin_y, origin_x in needed:
                fill_rows.setdefault(origin_y, []).append(origin_x)
            for origin_y in sorted(fill_rows):
                ctx_y = context_origin(origin_y, crop_h)
                xs = sorted(fill_rows[origin_y])
                x_start, strip = read_row(reader, crop_x, crop_y, ctx_y, [context_origin(x, crop_w) for x in xs])
                cursor = 0
                while cursor < len(xs):
                    count = min(batch, len(xs) - cursor)
                    chunk_x = xs[cursor:cursor + count]
                    crops = []
                    for x in chunk_x:
                        ctx_x = context_origin(x, crop_w)
                        field = strip[:, ctx_x - x_start:ctx_x - x_start + PATCH]
                        if field.shape != (PATCH, PATCH, 3):
                            raise RuntimeError(f"{slide} context at {ctx_y},{ctx_x} is {field.shape}")
                        crops.append(field[origin_y - ctx_y:origin_y - ctx_y + CROP, x - ctx_x:x - ctx_x + CROP])
                    crop_tensor = torch.from_numpy(np.ascontiguousarray(np.stack(crops))).permute(0, 3, 1, 2)
                    crop_tensor = crop_tensor.to(device=collage.device, dtype=torch.float32).mul_(2.0 / 255.0).sub_(1.0)
                    try:
                        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            predicted = model(crop_tensor, collage.expand(count, -1, -1, -1))
                        predicted = predicted.float().cpu().numpy()
                    except torch.cuda.OutOfMemoryError:
                        if batch == 1:
                            raise
                        batch -= 1
                        torch.cuda.empty_cache()
                        print(f"gpu {rank} reduced batch to {batch}", flush=True)
                        continue
                    for index, origin_x in enumerate(chunk_x):
                        blend_border(
                            np.transpose(predicted[index], (1, 2, 0)),
                            origin_y,
                            origin_x,
                            accum,
                            cover,
                            holes,
                        )
                    cursor += count
    written = 0
    for key, pixels in accum.items():
        weights = cover[key]
        if float(weights.min()) <= 0:
            missing = np.argwhere(weights <= 0)
            raise RuntimeError(
                f"{slide} {names[key]} has {len(missing)} pixels with no coverage, first at {missing[0].tolist()}"
            )
        merged = pixels / weights[..., None]
        uint8 = np.clip(np.rint((merged + 1.0) * 127.5), 0, 255).astype(np.uint8)
        save_png(uint8, OUT_ROOT / names[key])
        written += 1
    return written


def worker(rank: int, slides: list[str], grouped: dict[str, list], af_index: dict, checkpoint: str) -> None:
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    torch.backends.cudnn.benchmark = True
    pending = [slide for slide in slides if not slide_done(grouped[slide])]
    print(
        f"gpu {rank} slides {len(slides)} pending {len(pending)} patches {sum(len(grouped[s]) for s in slides)}",
        flush=True,
    )
    if not pending:
        return
    model = Generator().to(device).eval()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["generator"])
    del payload
    for index, slide in enumerate(pending, start=1):
        collage = load_collage(slide, device)
        count = run_slide(model, collage, slide, grouped[slide], af_index[slide], batch=8, rank=rank)
        print(f"gpu {rank} done {index}/{len(pending)} {slide} patches={count}", flush=True)


def main() -> None:
    self_check()
    world = torch.cuda.device_count()
    if world < 1:
        raise RuntimeError("epoch-10 overlap precompute needs a GPU")
    grouped = collect_slides(training_slides())
    af_index = load_af_index()
    missing = sorted(set(grouped) - set(af_index))
    if missing:
        raise RuntimeError(f"{len(missing)} slides have no AF TIFF, for example {missing[:5]}")
    buckets = balance(grouped, world)
    total = sum(len(items) for items in grouped.values())
    print(f"patches={total} slides={len(grouped)} gpus={world} feather={FEATHER} out={OUT_ROOT}", flush=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    mp.set_start_method("spawn")
    processes = []
    for rank, slides in enumerate(buckets):
        process = mp.Process(target=worker, args=(rank, slides, grouped, af_index, str(CHECKPOINT)))
        process.start()
        processes.append(process)
    failed = False
    for process in processes:
        process.join()
        if process.exitcode != 0:
            failed = True
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
