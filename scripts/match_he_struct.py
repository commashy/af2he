"""CPU structural match of a pseudo patch to nearby real H&E.

Reads one pseudo 1024 and the 3x3 real H&E neighborhood, downsamples by 8,
and runs normalized correlation. Writes a sheet per example.

    python scripts/match_he_struct.py
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

PSEUDO = Path("/mnt/10tb_hdd/taek/images/Fresh_section/patches_1024/pseudo")
HE = Path("/mnt/10tb_hdd/taek/images/Fresh_section/patches_1024/trainB")
OUT = Path("/mnt/10tb_hdd/taek/af2he/collages/verify/he_struct_match")
PATCH = 1024
SCALE = 8
RADIUS = 1
NAME = re.compile(r"^(?P<slide>.+)__y(?P<y>\d+)_x(?P<x>\d+)\.png$")


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return array.mean(axis=2).astype(np.float32)


def edges(gray: np.ndarray) -> np.ndarray:
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    return gx + gy


def down(gray: np.ndarray) -> np.ndarray:
    image = Image.fromarray(gray.astype(np.uint8), mode="L")
    small = image.resize((gray.shape[1] // SCALE, gray.shape[0] // SCALE), Image.Resampling.BOX)
    return np.asarray(small, dtype=np.float32)


def stitch(slide: str, y: int, x: int) -> tuple[np.ndarray, int]:
    span = (2 * RADIUS + 1) * PATCH
    canvas = np.zeros((span, span), dtype=np.float32)
    present = 0
    for row in range(-RADIUS, RADIUS + 1):
        for col in range(-RADIUS, RADIUS + 1):
            path = HE / f"{slide}__y{y + row * PATCH}_x{x + col * PATCH}.png"
            if not path.is_file():
                continue
            top = (row + RADIUS) * PATCH
            left = (col + RADIUS) * PATCH
            canvas[top:top + PATCH, left:left + PATCH] = load_gray(path)
            present += 1
    return canvas, present


def correlate(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    score = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    return np.nan_to_num(score, nan=-1.0, posinf=-1.0, neginf=-1.0)


def peak_and_second(score: np.ndarray, suppress: int = 12) -> tuple[tuple[int, int], float, float]:
    flat = int(np.argmax(score))
    peak_y, peak_x = np.unravel_index(flat, score.shape)
    best = float(score[peak_y, peak_x])
    masked = score.copy()
    y0 = max(0, peak_y - suppress)
    y1 = min(score.shape[0], peak_y + suppress + 1)
    x0 = max(0, peak_x - suppress)
    x1 = min(score.shape[1], peak_x + suppress + 1)
    masked[y0:y1, x0:x1] = -1.0
    second = float(masked.max())
    return (int(peak_y), int(peak_x)), best, second


def crop(canvas: np.ndarray, top: int, left: int) -> np.ndarray:
    tile = canvas[top:top + PATCH, left:left + PATCH]
    if tile.shape != (PATCH, PATCH):
        raise RuntimeError(f"crop {tile.shape} at {(top, left)}")
    return tile


def colorize(score: np.ndarray) -> np.ndarray:
    shown = np.clip((score + 1.0) * 0.5, 0.0, 1.0)
    gray = (shown * 255).astype(np.uint8)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def sheet(pseudo: np.ndarray, nominal: np.ndarray, matched: np.ndarray, score: np.ndarray, peak, text: str) -> Image.Image:
    panels = [
        cv2.cvtColor(pseudo.astype(np.uint8), cv2.COLOR_GRAY2RGB),
        cv2.cvtColor(nominal.astype(np.uint8), cv2.COLOR_GRAY2RGB),
        cv2.cvtColor(matched.astype(np.uint8), cv2.COLOR_GRAY2RGB),
        colorize(score),
    ]
    size = 512
    resized = [
        np.asarray(Image.fromarray(panel).resize((size, size), Image.Resampling.BOX))
        for panel in panels
    ]
    score_panel = np.array(resized[3], copy=True)
    scale_y = size / score.shape[0]
    scale_x = size / score.shape[1]
    cv2.drawMarker(
        score_panel,
        (int(peak[1] * scale_x), int(peak[0] * scale_y)),
        (255, 80, 80),
        markerType=cv2.MARKER_CROSS,
        markerSize=18,
        thickness=2,
    )
    resized[3] = score_panel
    gap = 8
    canvas = np.full((size + 36, size * 4 + gap * 3, 3), 255, dtype=np.uint8)
    for index, panel in enumerate(resized):
        left = index * (size + gap)
        canvas[36:36 + size, left:left + size] = panel
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    draw.text((4, 8), text, fill=(0, 0, 0))
    labels = ["pseudo", "nominal H&E", "matched H&E", "correlation"]
    for index, label in enumerate(labels):
        draw.text((index * (size + gap) + 4, 22), label, fill=(0, 0, 0))
    return image


def match_one(name: str) -> dict | None:
    found = NAME.match(name)
    if found is None:
        return None
    slide = found.group("slide")
    y = int(found.group("y"))
    x = int(found.group("x"))
    pseudo_path = PSEUDO / name
    if not pseudo_path.is_file() or not (HE / name).is_file():
        return None
    canvas, present = stitch(slide, y, x)
    if present < 9:
        return None
    pseudo = load_gray(pseudo_path)
    template = down(edges(pseudo))
    search = down(edges(canvas))
    score = correlate(search, template)
    nominal = (RADIUS * PATCH // SCALE, RADIUS * PATCH // SCALE)
    # matchTemplate coordinate is the template's top-left.
    peak, best, second = peak_and_second(score)
    nominal_score = float(score[nominal[0], nominal[1]])
    shift_y = (peak[0] - nominal[0]) * SCALE
    shift_x = (peak[1] - nominal[1]) * SCALE
    matched = crop(canvas, peak[0] * SCALE, peak[1] * SCALE)
    nominal_tile = crop(canvas, RADIUS * PATCH, RADIUS * PATCH)
    text = (
        f"{slide}  y{y} x{x}  shift=({shift_y:+d},{shift_x:+d})px  "
        f"peak={best:.3f} nominal={nominal_score:.3f} second={second:.3f}"
    )
    image = sheet(pseudo, nominal_tile, matched, score, peak, text)
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / f"{slide}__y{y}_x{x}.jpg"
    image.save(destination, quality=90)
    return {
        "path": destination,
        "shift": (shift_y, shift_x),
        "peak": best,
        "nominal": nominal_score,
        "second": second,
        "text": text,
    }


def _std(path: Path) -> float:
    with Image.open(path) as image:
        small = np.asarray(image.convert("L").resize((64, 64), Image.Resampling.BOX), dtype=np.float32)
    return float(small.std())


def interior_examples(limit: int) -> list[str]:
    slides = []
    seen = set()
    for path in PSEUDO.iterdir():
        found = NAME.match(path.name)
        if found is None or found.group("slide") in seen:
            continue
        seen.add(found.group("slide"))
        slides.append(found.group("slide"))
        if len(slides) >= 16:
            break
    wanted = set(slides)
    he_coords: dict[str, set[tuple[int, int]]] = {slide: set() for slide in slides}
    pseudo_coords: dict[str, set[tuple[int, int]]] = {slide: set() for slide in slides}
    for folder, store in ((HE, he_coords), (PSEUDO, pseudo_coords)):
        for path in folder.iterdir():
            found = NAME.match(path.name)
            if found is None or found.group("slide") not in wanted:
                continue
            store[found.group("slide")].add((int(found.group("y")), int(found.group("x"))))
    names = []
    for slide in slides:
        he_set = he_coords[slide]
        candidates = []
        for y, x in pseudo_coords[slide] & he_set:
            neighbors = {
                (y + row * PATCH, x + col * PATCH)
                for row in range(-RADIUS, RADIUS + 1)
                for col in range(-RADIUS, RADIUS + 1)
            }
            if neighbors <= he_set:
                candidates.append((y, x))
        candidates.sort()
        best_name = None
        best_score = -1.0
        for y, x in candidates[:: max(1, len(candidates) // 12)]:
            name = f"{slide}__y{y}_x{x}.png"
            score = min(_std(PSEUDO / name), _std(HE / name))
            if score > best_score:
                best_score = score
                best_name = name
        if best_name is not None and best_score >= 8.0:
            names.append(best_name)
        if len(names) >= limit:
            break
    return names


def main() -> None:
    chosen = []
    for name in interior_examples(8):
        result = match_one(name)
        if result is None:
            continue
        chosen.append(result)
        print(result["text"], flush=True)
    if not chosen:
        raise RuntimeError("no full 3x3 neighborhoods matched")
    thumbs = []
    for result in chosen:
        with Image.open(result["path"]) as image:
            thumbs.append(image.resize((image.width // 2, image.height // 2), Image.Resampling.BOX))
    width = max(item.width for item in thumbs)
    contact = Image.new("RGB", (width, sum(item.height for item in thumbs)), (255, 255, 255))
    top = 0
    for item in thumbs:
        contact.paste(item, (0, top))
        top += item.height
    contact_path = OUT / "contact_tissue.jpg"
    contact.save(contact_path, quality=85)
    print(f"wrote {contact_path}", flush=True)


if __name__ == "__main__":
    main()
