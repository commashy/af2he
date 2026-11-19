"""Infer fresh AF slides with the stage-1 generator.

The generator receives a native 512 tile and one 1024 AF collage for the
whole slide. Tiles do not overlap. 16-bit AF is scaled by the full uint16
range, the same conversion used to build the training patches, then mapped
to [-1, 1] with no contrast stretch.

The collage is version 0 of the training collage rule: thumbnail mosaic,
three 2048 fields, four 1024 fields, and eight 256 windows, chosen by the
same tissue scores and spacing as ``data.collage``.

    python scripts/infer_one_slide.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
CLEAN_SCRIPTS = Path("/mnt/10tb_hdd/taek/stain_translation_clean/scripts")
for path in (ROOT, CLEAN_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.collage import PATCH, build_collage_from_tiles
from fresh_processing_utils import TiffRegionReader, af_to_rgb_uint8
from models.generator import TILE, Generator

RAW = Path("/mnt/10tb_hdd/taek/images/Fresh_section/16bit-raw-af")
SLIDES = ("F25-13716-1", "F25-10674-1", "F25-18103-2")
CHECKPOINT = ROOT / "checkpoints/stage1/epoch_0006.pt"
OUT = ROOT / "inference"
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


def infer(reader: TiffRegionReader, model: Generator, collage: torch.Tensor, device: torch.device) -> np.ndarray:
    ys = list(range(0, reader.height - TILE + 1, TILE))
    xs = list(range(0, reader.width - TILE + 1, TILE))
    canvas = np.zeros((ys[-1] + TILE, xs[-1] + TILE, 3), dtype=np.uint8)
    jobs = [(y, x) for y in ys for x in xs]
    print(f"tiles {len(jobs)} grid {len(ys)}x{len(xs)}", flush=True)
    for start in range(0, len(jobs), BATCH):
        chunk = jobs[start:start + BATCH]
        tiles = torch.stack([to_model(read_rgb(reader, y, x, TILE)) for y, x in chunk]).to(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            fake = model(tiles, collage.repeat(tiles.shape[0], 1, 1, 1))
        for (y, x), image in zip(chunk, fake):
            canvas[y:y + TILE, x:x + TILE] = from_model(image)
        if start % (BATCH * 20) == 0:
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
    """Fit inside the box without stretching. Empty margin stays white."""
    height, width = image.shape[:2]
    scale = min(box_w / max(width, 1), box_h / max(height, 1))
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    resized = Image.fromarray(image, mode="RGB").resize((resized_w, resized_h), Image.Resampling.BOX)
    canvas = Image.new("RGB", (box_w, box_h), (255, 255, 255))
    canvas.paste(resized, ((box_w - resized_w) // 2, (box_h - resized_h) // 2))
    return canvas


def diverse_fields(prediction: np.ndarray, size: int = 3072, count: int = 6) -> list[tuple[int, int, int]]:
    """Tissue windows of ``size``, spread across the slide. Returns (y, x, size)."""
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


def save_views(
    af_reader: TiffRegionReader,
    prediction: np.ndarray,
    out_dir: Path,
    slide: str,
    centers: list[tuple[int, int]] | None = None,
) -> None:
    """Wide 16:9 sheet. The six fields together take more room than the full slide."""
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
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    draw.text(
        (gutter, 6),
        f"{slide}    full slide, then {crop_w}x{crop_h}, then 1024 detail    AF | prediction",
        fill=(0, 0, 0),
        font=font,
    )
    covered = af_to_rgb_uint8(af_reader.read((0, 0, pred_w, pred_h)))
    board.paste(_contain(covered, full_w, full_h - gutter), (gutter, header))
    board.paste(_contain(prediction, full_w, full_h - gutter), (gutter * 2 + full_w, header))

    if centers is None:
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
    full_area = full_w * 2 * full_h
    zoom_area = cell_w * cell_h * images_per_row * rows
    board.save(out_dir / "views.png")
    print(
        f"wrote {out_dir / 'views.png'} {width}x{height} full_h={full_h} "
        f"cell={cell_w}x{cell_h} crop={crop_w}x{crop_h} zoom/full={zoom_area / max(full_area, 1):.2f}",
        flush=True,
    )


def test_slides() -> list[str]:
    suffix = "-AF-R1.tif"
    return sorted(path.name[: -len(suffix)] for path in RAW.glob(f"*{suffix}"))


def run_slide(
    slide: str,
    model: Generator,
    device: torch.device,
    out_root: Path = OUT,
    match_root: Path | None = None,
    flat: bool = False,
) -> None:
    path = RAW / f"{slide}-AF-R1.tif"
    if flat:
        out_root.mkdir(parents=True, exist_ok=True)
        destination = out_root / f"{slide}.jpg"
    else:
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
        if not flat:
            Image.fromarray(collage_u8, mode="RGB").save(destination.parent / "collage.png")
        collage = to_model(collage_u8).unsqueeze(0).to(device)
        prediction = infer(reader, model, collage, device)
        Image.fromarray(prediction, mode="RGB").save(destination, quality=90)
        print(f"wrote {destination} {prediction.shape}", flush=True)
        if flat:
            return
        centers = None
        if match_root is not None:
            Image.MAX_IMAGE_PIXELS = None
            previous = np.asarray(Image.open(match_root / slide / "fake_he.jpg").convert("RGB"))
            centers = diverse_fields(previous, count=3)
        save_views(reader, prediction, destination.parent, slide, centers)


def rebuild_views(slide: str) -> None:
    """Rewrite views.png from the saved prediction."""
    out_dir = OUT / slide
    Image.MAX_IMAGE_PIXELS = None
    prediction = np.asarray(Image.open(out_dir / "fake_he.jpg").convert("RGB"))
    with TiffRegionReader(RAW / f"{slide}-AF-R1.tif") as reader:
        save_views(reader, prediction, out_dir, slide)


def _option(flag: str, default: Path) -> Path:
    if flag not in sys.argv:
        return default
    return Path(sys.argv[sys.argv.index(flag) + 1])


def _int_option(flag: str, default: int) -> int:
    if flag not in sys.argv:
        return default
    return int(sys.argv[sys.argv.index(flag) + 1])


def main() -> None:
    Image.MAX_IMAGE_PIXELS = None
    if "--views-only" in sys.argv:
        for slide in SLIDES:
            rebuild_views(slide)
        return
    checkpoint = _option("--checkpoint", CHECKPOINT)
    out_root = _option("--out", OUT)
    match_root = _option("--match-views", OUT) if "--match-views" in sys.argv else None
    flat = "--flat" in sys.argv
    slides = test_slides() if "--all-raw" in sys.argv else list(SLIDES)
    shard = _int_option("--shard", 0)
    shards = _int_option("--shards", 1)
    if shards < 1 or not 0 <= shard < shards:
        raise ValueError(f"shard {shard} of {shards} is invalid")
    slides = [slide for index, slide in enumerate(slides) if index % shards == shard]
    device = torch.device("cuda:0")
    model = Generator().to(device).eval()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["generator"])
    print(
        f"checkpoint {checkpoint} epoch field {payload['epoch']} step {payload['global_step']} "
        f"slides={len(slides)} shard={shard}/{shards}",
        flush=True,
    )
    del payload
    for slide in slides:
        run_slide(slide, model, device, out_root, match_root, flat)


if __name__ == "__main__":
    main()
