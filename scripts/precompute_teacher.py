"""Precompute teacher H&E for every stage-1 AF patch.

AF pixels come from the whole-slide TIFF, one strip at a time. Inside each
1024 window the generator runs four 512 crops at stride 384. The center
384 of each crop is kept and the outer 64 is discarded, so the kept squares
meet without blending. Large windows are spaced 768 apart.

    python scripts/precompute_teacher.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CLEAN_SCRIPTS = Path("/mnt/10tb_hdd/taek/stain_translation_clean/scripts")
for path in (ROOT, CLEAN_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from engine.teacher import CleanTeacher
from fresh_processing_utils import TiffRegionReader, af_to_rgb_uint8

PATCH = 1024
CROP = 512
CROP_STRIDE = 384
LARGE_STRIDE = 768
EDGE_MARGIN = 64
CROP_OFFSETS = (0, CROP_STRIDE)
AF_ROOT = Path("/mnt/10tb_hdd/taek/images/Fresh_section/patches_1024/trainA")
ALIGNMENT = ROOT / "manifests" / "pseudo_alignment.csv"
CROP_MANIFEST = Path("/mnt/10tb_hdd/taek/stain_translation_clean/manifests/fresh/crop_manifest_for_thumbs.csv")
OUT_ROOT = ROOT / "teacher_cache"
CONFIG = "/mnt/10tb_hdd/taek/stain_translation_clean/checkpoints/clean_fresh_wsi_fdl/config.yaml"
CHECKPOINT = "/mnt/10tb_hdd/taek/stain_translation_clean/checkpoints/clean_fresh_wsi_fdl/epoch_8.pth"
THUMBNAILS = "/mnt/10tb_hdd/taek/stain_translation_clean/manifests/fresh/thumbnails"
BATCH_CANDIDATES = (24, 16, 12, 8, 4)


def parse_yx(name: str) -> tuple[int, int]:
    body = name.split("__", 1)[1]
    if body.lower().endswith(".png"):
        body = body[:-4]
    y_text, x_text = body.split("_")
    if not y_text.startswith("y") or not x_text.startswith("x"):
        raise ValueError(f"cannot read coordinates from {name}")
    return int(y_text[1:]), int(x_text[1:])


def cells_of_window(origin_y: int, origin_x: int) -> list[tuple[int, int]]:
    """1024-grid patches that a 1024 window touches."""
    ys = {(origin_y // PATCH) * PATCH, ((origin_y + PATCH - 1) // PATCH) * PATCH}
    xs = {(origin_x // PATCH) * PATCH, ((origin_x + PATCH - 1) // PATCH) * PATCH}
    return [(y, x) for y in sorted(ys) for x in sorted(xs)]


def _axis_origins(coords: list[int], limit: int) -> list[int]:
    start = min(coords)
    end = max(coords) + PATCH
    origins = list(range(start, max(end - PATCH, start) + 1, LARGE_STRIDE))
    last = min(end, limit) - PATCH
    if last >= 0 and (not origins or origins[-1] < last):
        origins.append(last)
    return [value for value in sorted(set(origins)) if value >= 0 and value + PATCH <= limit]


def large_origins(keys: set[tuple[int, int]], height: int, width: int) -> list[tuple[int, int]]:
    """1024 windows on a 768 grid that touch at least one training patch."""
    ys = _axis_origins([y for y, _x in keys], height)
    xs = _axis_origins([x for _y, x in keys], width)
    origins = []
    for y in ys:
        for x in xs:
            if any(cell in keys for cell in cells_of_window(y, x)):
                origins.append((y, x))
    return origins


def paste_crop(
    image: np.ndarray,
    crop_y: int,
    crop_x: int,
    accum: dict[tuple[int, int], np.ndarray],
    cover: dict[tuple[int, int], np.ndarray],
) -> None:
    """Write one 512 prediction. The center overwrites; the border only fills holes."""
    local_y = np.arange(CROP)[:, None]
    local_x = np.arange(CROP)[None, :]
    center = (
        (local_y >= EDGE_MARGIN)
        & (local_y < CROP - EDGE_MARGIN)
        & (local_x >= EDGE_MARGIN)
        & (local_x < CROP - EDGE_MARGIN)
    )
    priority = np.where(center, np.uint8(2), np.uint8(1))
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
            pri = priority[src_y0:src_y0 + height, src_x0:src_x0 + width]
            dst_cover = cover[(patch_y, patch_x)][dst_y0:dst_y0 + height, dst_x0:dst_x0 + width]
            take = pri > dst_cover
            if not np.any(take):
                continue
            dst = accum[(patch_y, patch_x)][dst_y0:dst_y0 + height, dst_x0:dst_x0 + width]
            src = image[src_y0:src_y0 + height, src_x0:src_x0 + width]
            dst[take] = src[take]
            dst_cover[take] = pri[take]


def context_origin(crop: int, limit: int) -> int:
    """1024 context that contains ``crop`` and is centered on it when it fits."""
    if limit < PATCH:
        raise ValueError(f"crop extent {limit} is smaller than {PATCH}")
    origin = crop - (PATCH - CROP) // 2
    origin = max(0, min(origin, limit - PATCH))
    if crop < origin or crop + CROP > origin + PATCH:
        raise ValueError(f"crop {crop} does not fit in context {origin} within {limit}")
    return origin


def crop_axis(coords: list[int], limit: int) -> list[int]:
    """512-crop origins whose 384 centers tile the content, plus a tail crop if needed."""
    start = min(coords)
    end = min(max(coords) + PATCH, limit)
    origins: list[int] = []
    origin = start
    while origin + EDGE_MARGIN < end - EDGE_MARGIN and origin + CROP <= limit:
        origins.append(origin)
        origin += CROP_STRIDE
    need = end - EDGE_MARGIN
    if origins and origins[-1] + (CROP - EDGE_MARGIN) < need:
        final = min(end, limit) - CROP
        if final >= 0 and final not in origins:
            origins.append(final)
    if not origins:
        raise RuntimeError("no crop origins")
    return origins


def crop_positions(keys: set[tuple[int, int]], height: int, width: int) -> list[tuple[int, int]]:
    ys = crop_axis([y for y, _x in keys], height)
    xs = crop_axis([x for _y, x in keys], width)

    def overlaps(crop_y: int, crop_x: int) -> bool:
        y0 = (crop_y // PATCH) * PATCH
        x0 = (crop_x // PATCH) * PATCH
        for patch_y in (y0 - PATCH, y0, y0 + PATCH):
            for patch_x in (x0 - PATCH, x0, x0 + PATCH):
                if (patch_y, patch_x) not in keys:
                    continue
                if crop_y < patch_y + PATCH and crop_y + CROP > patch_y and crop_x < patch_x + PATCH and crop_x + CROP > patch_x:
                    return True
        return False

    return [(y, x) for y in ys for x in xs if overlaps(y, x)]


def self_check() -> None:
    image = np.zeros((CROP, CROP, 3), dtype=np.float16)
    keys = {(0, x) for x in range(0, PATCH * 4, PATCH)}
    accum = {key: np.zeros((PATCH, PATCH, 3), dtype=np.float16) for key in keys}
    masks = {key: np.zeros((PATCH, PATCH), dtype=np.uint8) for key in keys}
    for y, x in crop_positions(keys, PATCH, PATCH * 4):
        paste_crop(image, y, x, accum, masks)
    full = np.zeros((PATCH, PATCH * 4), dtype=np.uint8)
    for (_y, x), mask in masks.items():
        full[:, x:x + PATCH] = np.maximum(full[:, x:x + PATCH], mask)
    interior = full[EDGE_MARGIN:PATCH - EDGE_MARGIN, EDGE_MARGIN:PATCH * 4 - EDGE_MARGIN]
    if int(interior.min()) < 2:
        missing = np.argwhere(interior < 2)
        raise AssertionError(f"center cut leaves a hole, first at {missing[0].tolist()}")


def allowed_slides() -> set[str]:
    slides = set()
    with ALIGNMENT.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["use"] == "crop_absolute":
                slides.add(row["specimen_key"])
    if not slides:
        raise RuntimeError(f"{ALIGNMENT} has no crop_absolute slides")
    return slides


def collect_slides(only: set[str] | None = None) -> dict[str, list[tuple[int, int, str]]]:
    allowed = None if only is not None else allowed_slides()
    grouped: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for path in AF_ROOT.iterdir():
        if not path.is_file() or path.suffix.lower() != ".png" or "__" not in path.name:
            continue
        if path.stat().st_size == 0:
            raise RuntimeError(f"empty AF patch {path}")
        slide = path.name.split("__", 1)[0]
        if only is not None:
            if slide not in only:
                continue
        else:
            specimen = slide.split("_", 1)[1]
            if specimen not in allowed:
                continue
        y, x = parse_yx(path.name)
        grouped[slide].append((y, x, path.name))
    if only is not None:
        missing = sorted(only - set(grouped))
        if missing:
            raise RuntimeError(f"{len(missing)} requested slides have no AF patches, for example {missing[:5]}")
    if not grouped:
        raise RuntimeError("no stage-1 AF patches")
    return grouped


def balance(grouped: dict[str, list], world: int) -> list[list[str]]:
    order = sorted(grouped, key=lambda slide: len(grouped[slide]), reverse=True)
    buckets: list[list[str]] = [[] for _ in range(world)]
    loads = [0] * world
    for slide in order:
        index = min(range(world), key=loads.__getitem__)
        buckets[index].append(slide)
        loads[index] += len(grouped[slide])
    return buckets


def load_af_index() -> dict[str, tuple[Path, int, int, int, int]]:
    """Slide id -> AF TIFF, crop origin, and crop height/width."""
    index = {}
    with CROP_MANIFEST.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            x0, y0, x1, y1 = (int(value) for value in row["af_crop_box"].split(","))
            path = Path(row["af_path"])
            if not path.is_file():
                raise FileNotFoundError(f"missing AF slide {path}")
            index[row["pair_id"]] = (path, x0, y0, y1 - y0, x1 - x0)
    return index


def read_row(reader: TiffRegionReader, crop_x: int, crop_y: int, origin_y: int, xs: list[int]) -> tuple[int, np.ndarray]:
    """One full-height strip covering every window on this row."""
    x_start = min(xs)
    width = max(xs) + PATCH - x_start
    box = (
        crop_x + x_start,
        crop_y + origin_y,
        crop_x + x_start + width,
        crop_y + origin_y + PATCH,
    )
    if box[0] < 0 or box[1] < 0 or box[2] > reader.width or box[3] > reader.height:
        raise RuntimeError(f"window {box} is outside {reader.path} ({reader.width}x{reader.height})")
    image = af_to_rgb_uint8(reader.read(box), reader.axes)
    if image.shape != (PATCH, width, 3):
        raise RuntimeError(f"strip {image.shape} != {(PATCH, width, 3)} from {reader.path}")
    return x_start, image


def choose_batch(teacher: CleanTeacher, slide: str, device: torch.device) -> int:
    total = torch.cuda.get_device_properties(device).total_memory
    for batch in BATCH_CANDIDATES:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            fields = torch.empty(batch, 3, PATCH, PATCH, device=device).uniform_(-1, 1)
            teacher.predict_pairs(fields, fields[:, :, :CROP, :CROP].contiguous(), slide)
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
            print(
                f"gpu {device.index} batch {batch} fits, peak {peak / 1024**3:.1f} / {total / 1024**3:.1f} GiB",
                flush=True,
            )
            del fields
            torch.cuda.empty_cache()
            return batch
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    raise RuntimeError(f"teacher does not fit on {device}")


def save_png(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".png.partial")
    Image.fromarray(array, mode="RGB").save(temporary, format="PNG", compress_level=1)
    temporary.replace(path)


def slide_done(items: list[tuple[int, int, str]]) -> bool:
    for _, _, name in items:
        path = OUT_ROOT / name
        if not path.is_file() or path.stat().st_size == 0:
            return False
    return True


def run_slide(
    teacher: CleanTeacher,
    slide: str,
    items: list[tuple[int, int, str]],
    source: tuple[Path, int, int, int, int],
    batch: int,
    rank: int,
) -> int:
    af_path, crop_x, crop_y, crop_h, crop_w = source
    keys = {(y, x) for y, x, _name in items}
    by_row: dict[int, list[int]] = defaultdict(list)
    for origin_y, origin_x in crop_positions(keys, crop_h, crop_w):
        by_row[origin_y].append(origin_x)
    if not by_row:
        raise RuntimeError(f"{slide} produced no crops")
    row_ys = sorted(by_row)
    accum = {key: np.zeros((PATCH, PATCH, 3), dtype=np.float16) for key in keys}
    cover = {key: np.zeros((PATCH, PATCH), dtype=np.uint8) for key in keys}
    names = {(y, x): name for y, x, name in items}
    done = 0
    total = sum(len(xs) for xs in by_row.values())

    def row_contexts(origin_y: int) -> tuple[int, np.ndarray]:
        ctx_y = context_origin(origin_y, crop_h)
        ctx_xs = [context_origin(x, crop_w) for x in by_row[origin_y]]
        return read_row(reader, crop_x, crop_y, ctx_y, ctx_xs)

    with TiffRegionReader(af_path) as reader, ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(row_contexts, row_ys[0])
        for row_index, origin_y in enumerate(row_ys):
            x_start, strip = pending.result()
            if row_index + 1 < len(row_ys):
                pending = pool.submit(row_contexts, row_ys[row_index + 1])
            ctx_y = context_origin(origin_y, crop_h)
            xs = sorted(by_row[origin_y])
            cursor = 0
            while cursor < len(xs):
                count = min(batch, len(xs) - cursor)
                chunk_x = xs[cursor:cursor + count]
                try:
                    contexts = []
                    crops = []
                    for x in chunk_x:
                        ctx_x = context_origin(x, crop_w)
                        field = strip[:, ctx_x - x_start:ctx_x - x_start + PATCH]
                        if field.shape != (PATCH, PATCH, 3):
                            raise RuntimeError(f"{slide} context at {ctx_y},{ctx_x} is {field.shape}")
                        contexts.append(field)
                        crops.append(field[origin_y - ctx_y:origin_y - ctx_y + CROP, x - ctx_x:x - ctx_x + CROP])
                    context_tensor = torch.from_numpy(np.ascontiguousarray(np.stack(contexts))).permute(0, 3, 1, 2)
                    crop_tensor = torch.from_numpy(np.ascontiguousarray(np.stack(crops))).permute(0, 3, 1, 2)
                    context_tensor = context_tensor.to(dtype=torch.float32).mul_(2.0 / 255.0).sub_(1.0)
                    crop_tensor = crop_tensor.to(dtype=torch.float32).mul_(2.0 / 255.0).sub_(1.0)
                    predicted = teacher.predict_pairs(context_tensor, crop_tensor, slide).numpy()
                except torch.cuda.OutOfMemoryError:
                    if batch == 1:
                        raise
                    batch -= 1
                    torch.cuda.empty_cache()
                    print(f"gpu {rank} reduced batch to {batch}", flush=True)
                    continue
                for index, origin_x in enumerate(chunk_x):
                    paste_crop(np.transpose(predicted[index], (1, 2, 0)), origin_y, origin_x, accum, cover)
                cursor += count
                done += count
            if done == total or done % 200 < count:
                print(f"gpu {rank} {slide} crops {done}/{total}", flush=True)
    written = 0
    for key, pixels in accum.items():
        if int(cover[key].min()) < 1:
            raise RuntimeError(f"{slide} {names[key]} has pixels with no teacher coverage")
        uint8 = np.clip(np.rint((pixels.astype(np.float32) + 1.0) * 127.5), 0, 255).astype(np.uint8)
        save_png(uint8, OUT_ROOT / names[key])
        written += 1
    return written


def worker(
    rank: int,
    slides: list[str],
    grouped: dict[str, list],
    af_index: dict,
    thumbnail_dir: str = THUMBNAILS,
) -> None:
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    pending = [slide for slide in slides if not slide_done(grouped[slide])]
    print(
        f"gpu {rank} slides {len(slides)} pending {len(pending)} patches {sum(len(grouped[s]) for s in slides)}",
        flush=True,
    )
    if not pending:
        return
    teacher = CleanTeacher(CONFIG, CHECKPOINT, thumbnail_dir, device)
    teacher.model.netG = torch.compile(teacher.model.netG, mode="default", dynamic=False)
    print(f"gpu {rank} compiling generator", flush=True)
    batch = choose_batch(teacher, pending[0], device)
    for index, slide in enumerate(pending, start=1):
        count = run_slide(teacher, slide, grouped[slide], af_index[slide], batch, rank)
        print(f"gpu {rank} done {index}/{len(pending)} {slide} patches={count}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slides-file", default=None, help="Full slide ids, one per line. Default is every crop_absolute slide.")
    parser.add_argument("--thumbnail-dir", default=THUMBNAILS)
    args = parser.parse_args()
    self_check()
    world = torch.cuda.device_count()
    if world < 1:
        raise RuntimeError("precompute needs a GPU")
    only = None
    if args.slides_file:
        only = {
            line.split("#", 1)[0].strip()
            for line in Path(args.slides_file).read_text(encoding="utf-8").splitlines()
            if line.split("#", 1)[0].strip()
        }
    grouped = collect_slides(only)
    af_index = load_af_index()
    missing = sorted(set(grouped) - set(af_index))
    if missing:
        raise RuntimeError(f"{len(missing)} slides have no AF TIFF, for example {missing[:5]}")
    buckets = balance(grouped, world)
    total = sum(len(items) for items in grouped.values())
    print(f"patches={total} slides={len(grouped)} gpus={world} out={OUT_ROOT}", flush=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    mp.set_start_method("spawn")
    processes = []
    for rank, slides in enumerate(buckets):
        process = mp.Process(
            target=worker,
            args=(rank, slides, grouped, af_index, args.thumbnail_dir),
        )
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
