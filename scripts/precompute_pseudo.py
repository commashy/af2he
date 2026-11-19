"""Write pseudo-labels for the added slides from the epoch-7 generator.

AF pixels come from the whole-slide TIFF, with the same 512 window, stride
384, and center-overwrite paste as the teacher cache. The saved files are
the 1024 training-grid patches. Each window uses that slide's version-0
collage.

    python scripts/precompute_pseudo.py --slides-file expand/slides.txt
"""

from __future__ import annotations

import argparse
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
collect_slides = _TEACHER.collect_slides
context_origin = _TEACHER.context_origin
crop_positions = _TEACHER.crop_positions
load_af_index = _TEACHER.load_af_index
paste_crop = _TEACHER.paste_crop
read_row = _TEACHER.read_row
save_png = _TEACHER.save_png
self_check = _TEACHER.self_check
balance = _TEACHER.balance

from fresh_processing_utils import TiffRegionReader

OUT_ROOT = Path("/mnt/10tb_hdd/taek/images/Fresh_section/patches_1024/pseudo")
COLLAGE_ROOT = ROOT / "collages" / "af"
CHECKPOINT = ROOT / "checkpoints/stage1/epoch_0007.pt"


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
    row_ys = sorted(by_row)
    accum = {key: np.zeros((PATCH, PATCH, 3), dtype=np.float16) for key in keys}
    cover = {key: np.zeros((PATCH, PATCH), dtype=np.uint8) for key in keys}
    names = {(y, x): name for y, x, name in items}
    done = 0
    total = sum(len(xs) for xs in by_row.values())

    with TiffRegionReader(af_path) as reader:
        for origin_y in row_ys:
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
                    paste_crop(np.transpose(predicted[index], (1, 2, 0)), origin_y, origin_x, accum, cover)
                cursor += count
                done += count
            if done == total or done % 200 < count:
                print(f"gpu {rank} {slide} crops {done}/{total}", flush=True)
    written = 0
    for key, pixels in accum.items():
        if int(cover[key].min()) < 1:
            raise RuntimeError(f"{slide} {names[key]} has pixels with no pseudo coverage")
        uint8 = np.clip(np.rint((pixels.astype(np.float32) + 1.0) * 127.5), 0, 255).astype(np.uint8)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slides-file", required=True)
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    args = parser.parse_args()
    self_check()
    world = torch.cuda.device_count()
    if world < 1:
        raise RuntimeError("pseudo precompute needs a GPU")
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
        process = mp.Process(target=worker, args=(rank, slides, grouped, af_index, args.checkpoint))
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
