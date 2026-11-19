"""AF to H&E inference with the stage-1 epoch-9 generator.

The generator receives a native 512 tile and one 1024 AF collage for the
whole slide. Tiles do not overlap. 16-bit AF is scaled by the full uint16
range, then mapped to [-1, 1]. There is no contrast stretch.

The collage is version 0: thumbnail mosaic, three 2048 fields, four 1024
fields, and eight 256 windows.

    python infer.py --af /path/to/slide-AF-R1.tif --out /path/to/out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from af_io import TiffRegionReader, af_to_rgb_uint8
from data.collage import PATCH, build_collage_from_tiles
from models.generator import TILE, Generator

CHECKPOINT = ROOT / "checkpoints" / "epoch_0009.pt"
BATCH = 8


def to_model(image: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()
    return tensor.mul_(2.0 / 255.0).sub_(1.0)


def from_model(image: torch.Tensor) -> np.ndarray:
    array = (image.detach().float().clamp(-1.0, 1.0).add(1.0).mul(127.5)).round()
    return array.byte().permute(1, 2, 0).cpu().numpy()


def read_rgb(reader: TiffRegionReader, y: int, x: int, size: int) -> np.ndarray:
    y2 = min(reader.height, y + size)
    x2 = min(reader.width, x + size)
    raw = reader.read((x, y, x2, y2))
    rgb = af_to_rgb_uint8(raw)
    if rgb.shape[0] == size and rgb.shape[1] == size:
        return rgb
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[: rgb.shape[0], : rgb.shape[1]] = rgb
    return canvas


def grid_tiles(reader: TiffRegionReader) -> dict[tuple[int, int], np.ndarray]:
    tiles = {}
    ys = range(0, reader.height - PATCH + 1, PATCH)
    xs = range(0, reader.width - PATCH + 1, PATCH)
    for y in ys:
        for x in xs:
            tiles[(y, x)] = read_rgb(reader, y, x, PATCH)
    print(f"collage tiles {len(tiles)}", flush=True)
    return tiles


def infer(reader: TiffRegionReader, model: Generator, collage: torch.Tensor, device: torch.device, batch: int) -> np.ndarray:
    ys = list(range(0, reader.height - TILE + 1, TILE))
    xs = list(range(0, reader.width - TILE + 1, TILE))
    if not ys or not xs:
        raise RuntimeError(f"slide {reader.height}x{reader.width} is smaller than a {TILE} tile")
    canvas = np.zeros((ys[-1] + TILE, xs[-1] + TILE, 3), dtype=np.uint8)
    jobs = [(y, x) for y in ys for x in xs]
    print(f"tiles {len(jobs)} grid {len(ys)}x{len(xs)}", flush=True)
    for start in range(0, len(jobs), batch):
        chunk = jobs[start:start + batch]
        tiles = torch.stack([to_model(read_rgb(reader, y, x, TILE)) for y, x in chunk]).to(device)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            fake = model(tiles, collage.repeat(tiles.shape[0], 1, 1, 1))
        for (y, x), image in zip(chunk, fake):
            canvas[y:y + TILE, x:x + TILE] = from_model(image)
        if start % (batch * 20) == 0:
            print(f"inferred {start + len(chunk)}/{len(jobs)}", flush=True)
    return canvas


def _crop_box(image: np.ndarray, y: int, x: int, height: int, width: int) -> np.ndarray:
    y = max(0, min(int(y), max(image.shape[0] - height, 0)))
    x = max(0, min(int(x), max(image.shape[1] - width, 0)))
    y2, x2 = min(image.shape[0], y + height), min(image.shape[1], x + width)
    out = np.zeros((height, width, 3), dtype=np.uint8)
    out[: y2 - y, : x2 - x] = image[y:y2, x:x2]
    return out


def _read_box(reader: TiffRegionReader, y: int, x: int, height: int, width: int) -> np.ndarray:
    y2 = min(reader.height, y + height)
    x2 = min(reader.width, x + width)
    rgb = af_to_rgb_uint8(reader.read((x, y, x2, y2)))
    if rgb.shape[0] == height and rgb.shape[1] == width:
        return rgb
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[: rgb.shape[0], : rgb.shape[1]] = rgb
    return canvas


def _contain(image: np.ndarray, box_w: int, box_h: int) -> Image.Image:
    height, width = image.shape[:2]
    scale = min(box_w / max(width, 1), box_h / max(height, 1))
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    resized = Image.fromarray(image, mode="RGB").resize((resized_w, resized_h), Image.Resampling.BOX)
    canvas = Image.new("RGB", (box_w, box_h), (255, 255, 255))
    canvas.paste(resized, ((box_w - resized_w) // 2, (box_h - resized_h) // 2))
    return canvas


def diverse_fields(prediction: np.ndarray, size: int = 3072, count: int = 6) -> list[tuple[int, int]]:
    height, width = prediction.shape[:2]
    if min(height, width) < size:
        size = 2048
    step = 32
    tissue = prediction[::step, ::step].min(axis=2) < 235
    stride = max(size // 2, 1024)
    candidates = []
    for y in range(0, height - size + 1, stride):
        for x in range(0, width - size + 1, stride):
            view = tissue[y // step:(y + size) // step, x // step:(x + size) // step]
            if view.size == 0:
                continue
            candidates.append((float(view.mean()), y, x))
    candidates.sort(reverse=True)
    min_dist = max(size, int(0.28 * max(height, width)))
    chosen: list[tuple[int, int]] = []
    for _pass in range(3):
        for fraction, y, x in candidates:
            if fraction < 0.12:
                continue
            if any(abs(y - oy) + abs(x - ox) < min_dist for oy, ox in chosen):
                continue
            chosen.append((y, x))
            if len(chosen) == count:
                break
        if len(chosen) == count:
            break
        min_dist //= 2
    chosen.sort()
    return [(y + size // 2, x + size // 2) for y, x in chosen[:count]]


def save_views(af_reader: TiffRegionReader, prediction: np.ndarray, out_dir: Path, slide: str) -> None:
    width, height = 5760, 3240
    gutter = 6
    header = 32
    pairs_per_row = 3
    rows = 2
    full_h = int((height - header) * 0.38)
    zoom_top = header + full_h + gutter
    zoom_h = height - zoom_top
    images_per_row = pairs_per_row * 2
    cell_w = (width - gutter * (pairs_per_row + 1)) // images_per_row
    cell_h = (zoom_h - gutter * (rows + 1)) // rows
    full_w = (width - gutter * 3) // 2
    pred_h, pred_w = prediction.shape[:2]
    aspect = cell_w / max(cell_h, 1)
    if aspect >= 1:
        crop_h = 2560
        crop_w = max(2560, int(round(2560 * aspect)))
    else:
        crop_w = 2560
        crop_h = max(2560, int(round(2560 / aspect)))
    crop_w = min(crop_w, pred_w)
    crop_h = min(crop_h, pred_h)

    board = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(board)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    draw.text(
        (gutter, 6),
        f"{slide}    full slide, then {crop_w}x{crop_h}, then 1024 detail    AF | prediction",
        fill=(0, 0, 0),
        font=font,
    )
    covered = af_to_rgb_uint8(af_reader.read((0, 0, pred_w, pred_h)))
    board.paste(_contain(covered, full_w, full_h - gutter), (gutter, header))
    board.paste(_contain(prediction, full_w, full_h - gutter), (gutter * 2 + full_w, header))

    centers = diverse_fields(prediction, count=pairs_per_row)
    for pair, (center_y, center_x) in enumerate(centers):
        for row, (box_h, box_w) in enumerate(((crop_h, crop_w), (1024, 1024))):
            origin_y = int(center_y - box_h / 2)
            origin_x = int(center_x - box_w / 2)
            af_panel = Image.fromarray(_read_box(af_reader, origin_y, origin_x, box_h, box_w), mode="RGB")
            pred_panel = Image.fromarray(_crop_box(prediction, origin_y, origin_x, box_h, box_w), mode="RGB")
            af_panel = af_panel.resize((cell_w, cell_h), Image.Resampling.BOX)
            pred_panel = pred_panel.resize((cell_w, cell_h), Image.Resampling.BOX)
            left = gutter + pair * (cell_w * 2 + gutter * 2)
            top = zoom_top + gutter + row * (cell_h + gutter)
            board.paste(af_panel, (left, top))
            board.paste(pred_panel, (left + cell_w, top))
    board.save(out_dir / "views.png")
    print(f"wrote {out_dir / 'views.png'}", flush=True)


def slide_name(path: Path) -> str:
    stem = path.stem
    suffix = "-AF-R1"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def slide_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    paths = sorted(path for path in source.iterdir() if path.suffix.lower() in {".tif", ".tiff"})
    if not paths:
        raise FileNotFoundError(f"no TIFF files in {source}")
    return paths


def run_slide(
    path: Path,
    model: Generator,
    device: torch.device,
    out_root: Path,
    batch: int,
    views: bool,
) -> None:
    slide = slide_name(path)
    destination = out_root / slide / "fake_he.jpg"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"skip {destination}", flush=True)
        return
    with TiffRegionReader(path) as reader:
        print(f"slide {slide} {reader.height}x{reader.width}", flush=True)
        collage_u8, records = build_collage_from_tiles(grid_tiles(reader), version=0)
        kinds = {}
        for record in records:
            kinds[record["kind"]] = kinds.get(record["kind"], 0) + 1
        print(f"collage {kinds}", flush=True)
        Image.fromarray(collage_u8, mode="RGB").save(destination.parent / "collage.png")
        collage = to_model(collage_u8).unsqueeze(0).to(device)
        prediction = infer(reader, model, collage, device, batch)
        Image.fromarray(prediction, mode="RGB").save(destination, quality=90)
        print(f"wrote {destination} {prediction.shape}", flush=True)
        if views:
            save_views(reader, prediction, destination.parent, slide)


def main() -> None:
    parser = argparse.ArgumentParser(description="Epoch-9 AF to H&E inference")
    parser.add_argument("--af", type=Path, required=True, help="one AF TIFF, or a directory of TIFFs")
    parser.add_argument("--out", type=Path, required=True, help="directory for collage.png, fake_he.jpg, and views.png")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-views", action="store_true")
    args = parser.parse_args()
    Image.MAX_IMAGE_PIXELS = None
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    paths = slide_paths(args.af)
    device = torch.device(args.device)
    model = Generator().to(device).eval()
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["generator"])
    print(
        f"checkpoint {args.checkpoint} epoch {payload['epoch']} step {payload['global_step']} slides={len(paths)}",
        flush=True,
    )
    del payload
    for path in paths:
        run_slide(path, model, device, args.out, args.batch, views=not args.no_views)


if __name__ == "__main__":
    main()
