"""One 4x4 collage per slide, three scales plus the whole-slide thumbnail.

Cell order, row-major:
    0                 whole-slide thumbnail -> 256
    1, 2, 3           2048 fields -> 256
    4, 5, 6, 7        1024 fields -> 256
    8 .. 15           native 256 windows, textured and spread in intensity

A missing larger field is filled from the next real crop that exists:
2048, then 1024, then a 512 quadrant, then a 256 window. The cell still
sits in its slot; the record kind is the crop that was actually used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from data.groups import slide_and_coord

PATCH = 1024
CELL = 256
COLLAGE = 1024
WINDOWS = 4
THUMB_ROOTS = (
    Path("/mnt/10tb_hdd/taek/images/fresh_lung/batch1/thumbnails/AF"),
    Path("/mnt/10tb_hdd/taek/images/fresh_lung/batch2/thumbnails/AF"),
)


@dataclass(frozen=True)
class Window:
    path: Path
    y: int
    x: int
    row: int
    col: int
    mean: float
    std: float
    p5: float
    p50: float
    p99: float
    fine: float
    coarse: float


@dataclass(frozen=True)
class PatchInfo:
    path: Path
    y: int
    x: int
    windows: tuple[Window, ...]

    @property
    def scratched(self) -> bool:
        return any(item.p99 > 170 for item in self.windows)

    def tissue_score(self, kept: set[int]) -> float:
        values = [item.std for item in self.windows if id(item) in kept]
        # The resized field shows the whole patch, so one textured corner is not enough.
        if len(values) < 8:
            return 0.0
        return float(np.median(np.asarray(values)))


def _luminance(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32).mean(axis=2)


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode not in ("RGB", "L"):
            raise ValueError(f"{path} has mode {image.mode}")
        array = np.asarray(image, dtype=np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.shape[0] < PATCH or array.shape[1] < PATCH:
        raise ValueError(f"{path} is {array.shape[:2]}, expected at least {PATCH}")
    return array[:PATCH, :PATCH]


def _resize(image: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(
        Image.fromarray(image, mode="RGB").resize((size, size), Image.Resampling.BOX),
        dtype=np.uint8,
    )


def index_patches(root: Path, slide: str) -> list[PatchInfo]:
    patches: list[PatchInfo] = []
    for path in sorted(root.glob(f"{slide}__*.png")):
        _, (y, x) = slide_and_coord(path)
        image = _load_rgb(path)
        gray = _luminance(image)
        windows = []
        for row in range(WINDOWS):
            for col in range(WINDOWS):
                view = gray[row * CELL:(row + 1) * CELL, col * CELL:(col + 1) * CELL]
                percentiles = np.percentile(view, (5, 50, 99))
                fine = float(np.abs(np.diff(view, axis=0)).mean() + np.abs(np.diff(view, axis=1)).mean())
                coarse = float(view[::16, ::16].std())
                windows.append(Window(
                    path, y, x, row, col,
                    float(view.mean()), float(view.std()),
                    float(percentiles[0]), float(percentiles[1]), float(percentiles[2]),
                    fine, coarse,
                ))
        patches.append(PatchInfo(path, y, x, tuple(windows)))
    if not patches:
        raise FileNotFoundError(f"no patches for {slide} in {root}")
    return patches


def _greedy(items: list[tuple[float, int, int, object]], count: int, min_dist: int):
    ordered = sorted(items, key=lambda item: item[0], reverse=True)
    chosen = []
    distance = int(min_dist)
    while len(chosen) < count and distance > 0:
        added = False
        for score, y, x, payload in ordered:
            if any(payload is existing for _, _, _, existing in chosen):
                continue
            if any(abs(y - cy) + abs(x - cx) < distance for _, cy, cx, _ in chosen):
                continue
            chosen.append((score, y, x, payload))
            added = True
            if len(chosen) == count:
                return chosen
        if not added:
            distance //= 2
    return chosen


def _background(patches: list[PatchInfo]) -> float:
    flat = [window.p50 for patch in patches for window in patch.windows if window.std < 5]
    if not flat:
        return 15.0
    return float(np.median(np.asarray(flat)))


def _is_tissue(window: Window, background: float) -> bool:
    if window.std < 5 or window.p99 > 160:
        return False
    if window.p50 < background + 12 or window.p5 < background + 6:
        return False
    # A hole edge has its median on the dark side and a long bright tail.
    above = window.p99 - window.p50
    below = max(window.p50 - window.p5, 1.0)
    if above > 30 and above > 2.0 * below:
        return False
    return True


def _tissue_windows(patches: list[PatchInfo]) -> list[Window]:
    background = _background(patches)
    return [
        window for patch in patches for window in patch.windows
        if _is_tissue(window, background)
    ]


def _blocks(patches: list[PatchInfo], tissue: set[int]) -> list[tuple[float, int, int, tuple]]:
    by_coord = {(item.y, item.x): item for item in patches}
    found = []
    for (y, x), top_left in by_coord.items():
        mates = (
            top_left,
            by_coord.get((y, x + PATCH)),
            by_coord.get((y + PATCH, x)),
            by_coord.get((y + PATCH, x + PATCH)),
        )
        if any(item is None for item in mates):
            continue
        if any(item.tissue_score(tissue) <= 0 for item in mates):
            continue
        score = sum(item.tissue_score(tissue) for item in mates) / 4.0
        found.append((score, y + PATCH, x + PATCH, mates))
    return found


def _separated(window: Window, chosen: list[Window], distance: int) -> bool:
    center_y = window.y + window.row * CELL + CELL // 2
    center_x = window.x + window.col * CELL + CELL // 2
    for other in chosen:
        other_y = other.y + other.row * CELL + CELL // 2
        other_x = other.x + other.col * CELL + CELL // 2
        if abs(center_y - other_y) + abs(center_x - other_x) < distance:
            return False
    return True


def _pick(pool: list[Window], chosen: list[Window]) -> Window:
    for distance in (PATCH, PATCH // 2, 0):
        for window in pool:
            if window in chosen:
                continue
            if _separated(window, chosen, distance):
                return window
    return pool[0]


def _native(patches: list[PatchInfo], exclude: set[tuple[str, int, int]] | None = None) -> list[Window]:
    """Eight tissue windows spread in brightness, fine detail, and coarse structure."""
    banned = exclude or set()
    eligible = [
        window for window in _tissue_windows(patches)
        if (str(window.path), window.row, window.col) not in banned
    ]
    if len(eligible) < 8:
        return sorted(eligible, key=lambda item: item.mean)
    columns = np.stack([
        np.array([item.mean for item in eligible]),
        np.array([item.fine for item in eligible]),
        np.array([item.coarse for item in eligible]),
    ], axis=1)
    center = columns.mean(axis=0)
    scale = columns.std(axis=0)
    scale[scale < 1e-6] = 1.0
    features = (columns - center) / scale
    start = int(np.argmax(columns[:, 0]))
    chosen_index = [start]
    while len(chosen_index) < 8:
        distance = np.linalg.norm(features[:, None, :] - features[chosen_index][None, :, :], axis=2)
        distance = distance.min(axis=1)
        chosen = [eligible[index] for index in chosen_index]
        for index, window in enumerate(eligible):
            if index in chosen_index or not _separated(window, chosen, PATCH):
                distance[index] = -1.0
        next_index = int(np.argmax(distance))
        if distance[next_index] < 0:
            distance = np.linalg.norm(
                features[:, None, :] - features[chosen_index][None, :, :], axis=2,
            ).min(axis=1)
            distance[chosen_index] = -1.0
            next_index = int(np.argmax(distance))
        chosen_index.append(next_index)
    chosen = [eligible[index] for index in chosen_index]
    return sorted(chosen, key=lambda item: item.mean)


def _mosaic_thumbnail(
    patches: list[PatchInfo],
    images: dict[tuple[int, int], np.ndarray] | None = None,
    size: int = CELL,
) -> np.ndarray:
    """Patch mosaic used when a slide has no saved thumbnail."""
    ys = [item.y for item in patches]
    xs = [item.x for item in patches]
    y0, x0 = min(ys), min(xs)
    height = max(ys) + PATCH - y0
    width = max(xs) + PATCH - x0
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    for patch in patches:
        image = _stored_rgb(patch, images)
        top = int((patch.y - y0) / height * size)
        left = int((patch.x - x0) / width * size)
        bottom = max(top + 1, int((patch.y + PATCH - y0) / height * size))
        right = max(left + 1, int((patch.x + PATCH - x0) / width * size))
        small = np.asarray(
            Image.fromarray(image, mode="RGB").resize((right - left, bottom - top), Image.Resampling.BOX),
            dtype=np.uint8,
        )
        canvas[top:bottom, left:right] = small
    return canvas


def _thumbnail(slide: str, thumb_dir: Path, patches: list[PatchInfo], size: int = CELL) -> np.ndarray:
    roots = [thumb_dir, *THUMB_ROOTS]
    seen = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        for suffix in (".png", ".jpg", ".jpeg"):
            path = root / f"{slide}{suffix}"
            if path.is_file():
                with Image.open(path) as image:
                    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
                return _resize(array, size)
    return _mosaic_thumbnail(patches, size=size)


def _paint(cells: list[np.ndarray]) -> np.ndarray:
    if len(cells) != 16:
        raise RuntimeError(f"collage needs 16 cells, got {len(cells)}")
    canvas = np.zeros((COLLAGE, COLLAGE, 3), dtype=np.uint8)
    for index, cell in enumerate(cells):
        if cell.shape != (CELL, CELL, 3):
            raise RuntimeError(f"cell {index} is {cell.shape}")
        row, col = divmod(index, 4)
        canvas[row * CELL:(row + 1) * CELL, col * CELL:(col + 1) * CELL] = cell
    return canvas


def _block_patch(patch: PatchInfo, blocked: set) -> None:
    blocked.add(("1024", str(patch.path)))
    for qy in (0, 1):
        for qx in (0, 1):
            blocked.add(("512", str(patch.path), qy, qx))


def _occupy(item, blocked: set) -> None:
    kind, payload = item[3], item[4]
    blocked.add(item[5])
    if kind == "2048":
        for mate in payload:
            _block_patch(mate, blocked)
    elif kind == "1024":
        _block_patch(payload, blocked)


def _draw(pool: list, need: int, version: int, blocked: set) -> list:
    free = [item for item in pool if item[5] not in blocked]
    if need <= 0 or not free:
        return []
    if len(free) > need:
        start = (version * need) % len(free)
        free = free[start:] + free[:start]
    return free[:need]


def _choose(pools: list[list], count: int, version: int, blocked: set) -> list:
    chosen = []
    for pool in pools:
        for item in _draw(pool, count - len(chosen), version, blocked):
            chosen.append(item)
            _occupy(item, blocked)
            if len(chosen) == count:
                return chosen
    source = [item for pool in pools for item in pool]
    index = 0
    while len(chosen) < count and source:
        chosen.append(source[index % len(source)])
        index += 1
    return chosen


def _field_pools(patches: list[PatchInfo], tissue: set[int]):
    large = [
        (score, y, x, "2048", mates, ("2048", y, x))
        for score, y, x, mates in _greedy(_blocks(patches, tissue), 9, min_dist=2 * PATCH)
    ]
    mid_src = [
        (item.tissue_score(tissue), item.y, item.x, item)
        for item in patches if item.tissue_score(tissue) >= 5 and not item.scratched
    ]
    mids = [
        (score, y, x, "1024", patch, ("1024", str(patch.path)))
        for score, y, x, patch in _greedy(mid_src, 12, min_dist=PATCH)
    ]
    quad_src = []
    for patch in patches:
        if patch.scratched:
            continue
        for qy in (0, 1):
            for qx in (0, 1):
                group = [window for window in patch.windows if window.row // 2 == qy and window.col // 2 == qx]
                kept = [window for window in group if id(window) in tissue]
                if len(kept) < 2:
                    continue
                score = float(np.median(np.asarray([window.std for window in kept])))
                quad_src.append((score, patch.y + qy * 512, patch.x + qx * 512, (patch, qy, qx)))
    quads = [
        (score, y, x, "512", payload, ("512", str(payload[0].path), payload[1], payload[2]))
        for score, y, x, payload in _greedy(quad_src, 16, min_dist=PATCH)
    ]
    small_src = [
        (window.std, window.y + window.row * CELL, window.x + window.col * CELL, window)
        for window in _tissue_windows(patches)
    ]
    if not small_src:
        small_src = [
            (window.std, window.y + window.row * CELL, window.x + window.col * CELL, window)
            for patch in patches for window in patch.windows
            if window.std >= 5 and window.p99 <= 170
        ]
    if not small_src:
        small_src = [
            (window.std, window.y + window.row * CELL, window.x + window.col * CELL, window)
            for patch in patches for window in patch.windows
        ]
    smalls = [
        (score, y, x, "256", window, ("256", str(window.path), window.row, window.col))
        for score, y, x, window in _greedy(small_src, 24, min_dist=CELL)
    ]
    return large, mids, quads, smalls


def _pad_native(
    patches: list[PatchInfo],
    natives: list[Window],
    exclude: set[tuple[str, int, int]] | None,
    tissue: set[int],
) -> list[Window]:
    chosen = list(natives)
    seen = {(str(window.path), window.row, window.col) for window in chosen}
    banned = exclude or set()
    extras = [
        window
        for patch in patches for window in patch.windows
        if (str(window.path), window.row, window.col) not in seen
        and (str(window.path), window.row, window.col) not in banned
        and (id(window) in tissue or (window.std >= 5 and window.p99 <= 170))
    ]
    extras.sort(key=lambda window: window.std, reverse=True)
    for window in extras:
        if len(chosen) >= 8:
            break
        chosen.append(window)
    if not chosen:
        chosen = [window for patch in patches for window in patch.windows]
    if not chosen:
        raise RuntimeError("no windows")
    base = list(chosen)
    index = 0
    while len(chosen) < 8:
        chosen.append(base[index % len(base)])
        index += 1
    return sorted(chosen[:8], key=lambda window: window.mean)


def _stored_rgb(owner, images: dict[tuple[int, int], np.ndarray] | None) -> np.ndarray:
    if images is None:
        return _load_rgb(owner.path)
    image = images[(owner.y, owner.x)]
    if image.shape[0] < PATCH or image.shape[1] < PATCH:
        raise ValueError(f"tile {(owner.y, owner.x)} is {image.shape[:2]}")
    return image[:PATCH, :PATCH]


def _cell(kind: str, payload, images: dict[tuple[int, int], np.ndarray] | None = None) -> np.ndarray:
    if kind == "2048":
        parts = [_stored_rgb(item, images) for item in payload]
        field = np.zeros((2 * PATCH, 2 * PATCH, 3), dtype=np.uint8)
        field[0:PATCH, 0:PATCH] = parts[0]
        field[0:PATCH, PATCH:2 * PATCH] = parts[1]
        field[PATCH:2 * PATCH, 0:PATCH] = parts[2]
        field[PATCH:2 * PATCH, PATCH:2 * PATCH] = parts[3]
        return _resize(field, CELL)
    if kind == "1024":
        return _resize(_stored_rgb(payload, images), CELL)
    if kind == "512":
        patch, qy, qx = payload
        image = _stored_rgb(patch, images)
        return _resize(image[qy * 512:(qy + 1) * 512, qx * 512:(qx + 1) * 512], CELL)
    window = payload
    image = _stored_rgb(window, images)
    return image[
        window.row * CELL:(window.row + 1) * CELL,
        window.col * CELL:(window.col + 1) * CELL,
    ].copy()


def _assemble(
    slide: str,
    patches: list[PatchInfo],
    thumbnail: np.ndarray,
    version: int,
    native_exclude: set[tuple[str, int, int]] | None,
    images: dict[tuple[int, int], np.ndarray] | None,
) -> tuple[np.ndarray, list[dict]]:
    tissue_windows = _tissue_windows(patches)
    tissue = {id(window) for window in tissue_windows}
    large_pool, mid_pool, quad_pool, small_pool = _field_pools(patches, tissue)
    blocked: set = set()
    large = _choose([large_pool, mid_pool, quad_pool, small_pool], 3, version, blocked)
    mids = _choose([mid_pool, quad_pool, small_pool], 4, version, blocked)
    natives = _pad_native(patches, _native(patches, native_exclude), native_exclude, tissue)
    if len(large) < 3 or len(mids) < 4 or len(natives) < 8:
        raise RuntimeError(
            f"{slide} version {version}: large={len(large)} mid={len(mids)} native={len(natives)}"
        )

    cells = [thumbnail]
    records = [{"kind": "thumb", "mean": float(_luminance(cells[0]).mean()), "std": float(_luminance(cells[0]).std())}]
    for score, y, x, kind, payload, _key in (*large, *mids):
        cell = _cell(kind, payload, images)
        cells.append(cell)
        gray = _luminance(cell)
        records.append({
            "kind": kind, "y": y, "x": x, "score": score,
            "mean": float(gray.mean()), "std": float(gray.std()),
        })
    for window in natives:
        cell = _cell("256", window, images)
        cells.append(cell)
        records.append({
            "kind": "native", "path": str(window.path),
            "y": window.y, "x": window.x,
            "row": window.row, "col": window.col,
            "mean": window.mean, "std": window.std,
        })
    return _paint(cells), records


def build_collage(
    root: Path,
    slide: str,
    thumb_dir: Path,
    version: int = 0,
    patches: list[PatchInfo] | None = None,
    native_exclude: set[tuple[str, int, int]] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Build collage version ``version`` (0, 1, or 2) for one slide."""
    if patches is None:
        patches = index_patches(root, slide)
    thumbnail = _thumbnail(slide, thumb_dir, patches)
    return _assemble(slide, patches, thumbnail, version, native_exclude, None)


def patch_from_array(image: np.ndarray, y: int, x: int) -> PatchInfo:
    """Same window statistics as ``index_patches``, from an in-memory 1024 tile."""
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] < PATCH or image.shape[1] < PATCH:
        raise ValueError(f"tile {(y, x)} is {getattr(image, 'shape', None)}")
    image = np.asarray(image[:PATCH, :PATCH], dtype=np.uint8)
    path = Path(f"y{y}_x{x}.png")
    gray = _luminance(image)
    windows = []
    for row in range(WINDOWS):
        for col in range(WINDOWS):
            view = gray[row * CELL:(row + 1) * CELL, col * CELL:(col + 1) * CELL]
            percentiles = np.percentile(view, (5, 50, 99))
            fine = float(np.abs(np.diff(view, axis=0)).mean() + np.abs(np.diff(view, axis=1)).mean())
            coarse = float(view[::16, ::16].std())
            windows.append(Window(
                path, y, x, row, col,
                float(view.mean()), float(view.std()),
                float(percentiles[0]), float(percentiles[1]), float(percentiles[2]),
                fine, coarse,
            ))
    return PatchInfo(path, y, x, tuple(windows))


def build_collage_from_tiles(
    tiles: dict[tuple[int, int], np.ndarray],
    version: int = 0,
) -> tuple[np.ndarray, list[dict]]:
    """Training collage rule on a 1024 grid that was read from a slide.

    Version 0 is the first of the three collages saved for a training slide.
    There is no saved thumbnail for these slides, so the corner cell is the
    same patch mosaic the training builder uses in that case.
    """
    if not tiles:
        raise ValueError("collage needs at least one 1024 tile")
    patches = [patch_from_array(tiles[key], key[0], key[1]) for key in sorted(tiles)]
    thumbnail = _mosaic_thumbnail(patches, tiles)
    return _assemble("wsi", patches, thumbnail, version, None, tiles)


CELL_HI = 512
COLLAGE_HI = 2048
COLLAGE_VERSIONS = 6


def _quadrant(y: int, x: int, y_mid: float, x_mid: float) -> int:
    return (0 if y < y_mid else 2) + (0 if x < x_mid else 1)


def _upper_half(items: list) -> list:
    """Candidates at or above the median score. The brightest fold is not preferred."""
    if not items:
        return []
    cut = float(np.median([item[0] for item in items]))
    kept = [item for item in items if item[0] >= cut]
    return kept or list(items)


def _take(pool: list, version: int, used: set):
    free = [item for item in pool if item[5] not in used]
    if not free:
        return None
    ranked = sorted(_upper_half(free), key=lambda item: (item[0], item[1], item[2]))
    start = max(0, len(ranked) // 2 - 1)
    return ranked[(start + int(version)) % len(ranked)]


def _spread_pools(patches: list[PatchInfo], tissue: set[int]):
    y_mid, x_mid = (
        float(np.median([item.y + PATCH // 2 for item in patches])),
        float(np.median([item.x + PATCH // 2 for item in patches])),
    )
    large = [[], [], [], []]
    for score, y, x, mates in _blocks(patches, tissue):
        large[_quadrant(y, x, y_mid, x_mid)].append((score, y, x, "2048", mates, ("2048", y, x)))
    mids = [[], [], [], []]
    halves = [[], [], [], []]
    for patch in patches:
        if patch.scratched:
            continue
        score = patch.tissue_score(tissue)
        if score > 0:
            cy, cx = patch.y + PATCH // 2, patch.x + PATCH // 2
            mids[_quadrant(cy, cx, y_mid, x_mid)].append(
                (score, patch.y, patch.x, "1024", patch, ("1024", str(patch.path)))
            )
        for qy in (0, 1):
            for qx in (0, 1):
                group = [window for window in patch.windows if window.row // 2 == qy and window.col // 2 == qx]
                kept = [window for window in group if id(window) in tissue]
                if len(kept) < 2:
                    continue
                score = float(np.median([window.std for window in kept]))
                cy = patch.y + qy * 512 + 256
                cx = patch.x + qx * 512 + 256
                halves[_quadrant(cy, cx, y_mid, x_mid)].append(
                    (score, cy, cx, "512", (patch, qy, qx), ("512", str(patch.path), qy, qx))
                )
    return large, mids, halves


def _from_quadrants(groups: list[list], quadrants: list[int], version: int, used: set, fallbacks: list[list]):
    chosen = []
    for quad in quadrants:
        item = _take(groups[quad], version, used)
        if item is None:
            for pool in fallbacks:
                item = _take(pool, version, used)
                if item is not None:
                    break
        if item is None:
            continue
        chosen.append(item)
        _occupy(item, used)
    return chosen


def _repeat(chosen: list, count: int) -> list:
    if not chosen:
        return []
    out = list(chosen)
    index = 0
    while len(out) < count:
        out.append(chosen[index % len(chosen)])
        index += 1
    return out[:count]


def _cell_at(kind: str, payload, images, size: int) -> np.ndarray:
    if kind == "2048":
        parts = [_stored_rgb(item, images) for item in payload]
        field = np.zeros((2 * PATCH, 2 * PATCH, 3), dtype=np.uint8)
        field[0:PATCH, 0:PATCH] = parts[0]
        field[0:PATCH, PATCH:2 * PATCH] = parts[1]
        field[PATCH:2 * PATCH, 0:PATCH] = parts[2]
        field[PATCH:2 * PATCH, PATCH:2 * PATCH] = parts[3]
        return _resize(field, size)
    if kind == "1024":
        return _resize(_stored_rgb(payload, images), size)
    if kind == "512":
        patch, qy, qx = payload
        image = _stored_rgb(patch, images)
        return _resize(image[qy * 512:(qy + 1) * 512, qx * 512:(qx + 1) * 512], size)
    raise ValueError(f"unknown collage cell {kind}")


def _paint_at(cells: list[np.ndarray], cell: int, canvas: int) -> np.ndarray:
    if len(cells) != 16:
        raise RuntimeError(f"collage needs 16 cells, got {len(cells)}")
    image = np.zeros((canvas, canvas, 3), dtype=np.uint8)
    for index, tile in enumerate(cells):
        if tile.shape != (cell, cell, 3):
            raise RuntimeError(f"cell {index} is {tile.shape}")
        row, col = divmod(index, 4)
        image[row * cell:(row + 1) * cell, col * cell:(col + 1) * cell] = tile
    return image


def build_spread_collage(
    root: Path,
    slide: str,
    thumb_dir: Path,
    version: int = 0,
    patches: list[PatchInfo] | None = None,
    native_exclude: set | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """4x4 collage at 2048. Cells are 512.

    Three 2048 fields, four 1024 fields, and eight native 512 fields. Each
    scale prefers a different quadrant, from tissue at or above the median
    texture. ``version`` walks to another field in that same pool.
    """
    if patches is None:
        patches = index_patches(root, slide)
    tissue = {id(window) for window in _tissue_windows(patches)}
    large, mids, halves = _spread_pools(patches, tissue)
    used: set = set(native_exclude or ())
    flat_large = [item for group in large for item in group]
    flat_mids = [item for group in mids for item in group]
    flat_halves = [item for group in halves for item in group]
    picked_large = _from_quadrants(
        large, [(version + offset) % 4 for offset in range(3)], version, used, [flat_large, flat_mids, flat_halves],
    )
    picked_mids = _from_quadrants(
        mids, list(range(4)), version, used, [flat_mids, flat_halves, flat_large],
    )
    picked_halves = []
    for _ in range(2):
        picked_halves.extend(_from_quadrants(
            halves, list(range(4)), version + len(picked_halves), used, [flat_halves, flat_mids],
        ))
    picked_large = _repeat(picked_large, 3)
    picked_mids = _repeat(picked_mids, 4)
    picked_halves = _repeat(picked_halves, 8)
    if len(picked_large) < 3 or len(picked_mids) < 4 or len(picked_halves) < 8:
        raise RuntimeError(
            f"{slide} version {version}: large={len(picked_large)} mid={len(picked_mids)} half={len(picked_halves)}"
        )
    thumbnail = _thumbnail(slide, thumb_dir, patches, size=CELL_HI)
    cells = [thumbnail]
    records = [{"kind": "thumb"}]
    for score, y, x, kind, payload, key in (*picked_large, *picked_mids, *picked_halves):
        cells.append(_cell_at(kind, payload, None, CELL_HI))
        record = {"kind": kind, "y": int(y), "x": int(x), "score": float(score), "key": key}
        if kind == "512":
            patch, qy, qx = payload
            record.update({"path": str(patch.path), "qy": qy, "qx": qx})
        records.append(record)
    return _paint_at(cells, CELL_HI, COLLAGE_HI), records


GRID = 8
GRID_CELL = 256
GRID_CANVAS = GRID * GRID_CELL
GRID_VERSIONS = 6
# 4 quadrant overviews, 12 large fields, 16 mid fields, 32 native windows.
GRID_LARGE_PER_QUADRANT = 3
GRID_MID_PER_QUADRANT = 4
GRID_NATIVE_PER_QUADRANT = 8


def _glass_level(windows: list[Window]) -> float:
    """Gray level of empty glass: the darkest 15% of windows, not the flattest."""
    levels = np.array([window.p50 for window in windows], dtype=np.float32)
    dark_cut = float(np.percentile(levels, 15))
    dark = levels[levels <= dark_cut]
    if len(dark) == 0:
        return float(np.min(levels))
    return float(np.median(dark))


def _window_is_content(window: Window, glass: float) -> bool:
    """Tissue includes a flat dark sheet. Glass is the dark mode, not low texture."""
    if window.p99 > 170:
        return False
    above = window.p99 - window.p50
    below = max(window.p50 - window.p5, 1.0)
    # A hole edge sits on the glass and has a long bright tail.
    if window.p50 < glass + 12 and above > 30 and above > 2.0 * below:
        return False
    return window.p50 >= glass + 12


def _content_windows(patches: list[PatchInfo]) -> tuple[float, list[Window]]:
    windows = [window for patch in patches for window in patch.windows]
    if not windows:
        return 0.0, []
    glass = _glass_level(windows)
    kept = [window for window in windows if _window_is_content(window, glass)]
    if len(kept) < max(8, int(0.05 * len(windows))):
        # No separate glass mode. The slide is one material, so keep the non-scratch windows.
        kept = [
            window for window in windows
            if window.p99 <= 170
        ]
    return glass, kept


def _window_center(window: Window) -> tuple[int, int]:
    return window.y + window.row * GRID_CELL + GRID_CELL // 2, window.x + window.col * GRID_CELL + GRID_CELL // 2


def _quadrant_split(items: list, y_of, x_of, y_mid: float, x_mid: float) -> list[list]:
    groups = [[], [], [], []]
    for item in items:
        groups[_quadrant(y_of(item), x_of(item), y_mid, x_mid)].append(item)
    return groups


def _pick_spread(pool: list, count: int, version: int, used: set) -> list:
    """``count`` fields spread from dark to bright. Reuse the pool when it is short."""
    if count <= 0 or not pool:
        return []
    fresh = [item for item in pool if item[5] not in used]
    source = fresh if fresh else list(pool)
    source.sort(key=lambda item: (item[0], item[1], item[2]))
    total = len(source)
    stride = max(total // count, 1)
    start = (int(version) * stride) % total
    chosen = []
    for index in range(count):
        chosen.append(source[(start + index * stride) % total])
    for item in chosen:
        used.add(item[5])
    return chosen


def _fill_quadrants(groups: list[list], per_quadrant: int, version: int, used: set) -> list:
    chosen = []
    flat = [item for group in groups for item in group]
    for quad in range(4):
        picked = _pick_spread(groups[quad], per_quadrant, version, used)
        if len(picked) < per_quadrant:
            picked.extend(_pick_spread(flat, per_quadrant - len(picked), version + len(picked), used))
        chosen.extend(picked[:per_quadrant])
    if len(chosen) < 4 * per_quadrant and flat:
        chosen.extend(_pick_spread(flat, 4 * per_quadrant - len(chosen), version, used))
    if len(chosen) < 4 * per_quadrant and chosen:
        index = 0
        while len(chosen) < 4 * per_quadrant:
            chosen.append(chosen[index % len(chosen)])
            index += 1
    return chosen[: 4 * per_quadrant]


def _grid_pools(patches: list[PatchInfo], content: set[tuple[str, int, int]], y_mid: float, x_mid: float):
    by_coord = {(item.y, item.x): item for item in patches}

    def patch_content(patch: PatchInfo) -> int:
        return sum((str(patch.path), window.row, window.col) in content for window in patch.windows)

    def patch_gray(patch: PatchInfo) -> float:
        values = [
            window.p50 for window in patch.windows
            if (str(patch.path), window.row, window.col) in content
        ]
        if not values:
            values = [window.p50 for window in patch.windows]
        return float(np.median(values))

    large = []
    for (y, x), top_left in by_coord.items():
        mates = (
            top_left,
            by_coord.get((y, x + PATCH)),
            by_coord.get((y + PATCH, x)),
            by_coord.get((y + PATCH, x + PATCH)),
        )
        if any(item is None for item in mates):
            continue
        eligible = sum(patch_content(item) >= 4 for item in mates)
        if eligible < 2:
            continue
        gray = float(np.median([patch_gray(item) for item in mates]))
        center_y, center_x = y + PATCH, x + PATCH
        large.append((gray, center_y, center_x, "2048", mates, ("2048", y, x)))
    mids = []
    natives = []
    for patch in patches:
        count = patch_content(patch)
        if count >= 4:
            gray = patch_gray(patch)
            mids.append((gray, patch.y + PATCH // 2, patch.x + PATCH // 2, "1024", patch, ("1024", str(patch.path))))
        for window in patch.windows:
            key = (str(patch.path), window.row, window.col)
            if key not in content:
                continue
            cy, cx = _window_center(window)
            natives.append((window.p50, cy, cx, "256", window, ("256", str(window.path), window.row, window.col)))
    return (
        _quadrant_split(large, lambda item: item[1], lambda item: item[2], y_mid, x_mid),
        _quadrant_split(mids, lambda item: item[1], lambda item: item[2], y_mid, x_mid),
        _quadrant_split(natives, lambda item: item[1], lambda item: item[2], y_mid, x_mid),
    )


def _quadrant_overviews(patches: list[PatchInfo], y_mid: float, x_mid: float) -> list[np.ndarray]:
    groups = _quadrant_split(
        patches,
        lambda patch: patch.y + PATCH // 2,
        lambda patch: patch.x + PATCH // 2,
        y_mid, x_mid,
    )
    views = []
    for group in groups:
        source = group or patches
        views.append(_mosaic_thumbnail(source, size=GRID_CELL))
    return views


def _pad_fields(chosen: list, count: int, patches: list[PatchInfo]) -> list:
    """Repeat a chosen field, or any patch, until ``count`` cells exist."""
    out = list(chosen)
    backup = []
    for patch in patches:
        gray = float(np.median([window.p50 for window in patch.windows]))
        backup.append((gray, patch.y, patch.x, "1024", patch, ("pad", str(patch.path), len(backup))))
    index = 0
    while len(out) < count:
        source = out or backup
        if not source:
            raise RuntimeError("collage quadrant has no patches to repeat")
        out.append(source[index % len(source)])
        index += 1
    return out[:count]


def _paint_grid(cells: list[np.ndarray]) -> np.ndarray:
    if len(cells) != GRID * GRID:
        raise RuntimeError(f"collage needs {GRID * GRID} cells, got {len(cells)}")
    canvas = np.zeros((GRID_CANVAS, GRID_CANVAS, 3), dtype=np.uint8)
    for index, cell in enumerate(cells):
        if cell.shape != (GRID_CELL, GRID_CELL, 3):
            raise RuntimeError(f"cell {index} is {cell.shape}")
        row, col = divmod(index, GRID)
        canvas[
            row * GRID_CELL:(row + 1) * GRID_CELL,
            col * GRID_CELL:(col + 1) * GRID_CELL,
        ] = cell
    return canvas


def build_grid64(
    root: Path,
    slide: str,
    thumb_dir: Path,
    version: int = 0,
    patches: list[PatchInfo] | None = None,
    used: set | None = None,
    overviews: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """8x8 collage, 256-pixel cells, 2048 canvas.

    Four quadrant overviews, then large, mid, and native fields. Inside each
    quadrant the fields run from darker tissue to brighter tissue. A large
    field is repeated when that quadrant has no further block.
    """
    if patches is None:
        patches = index_patches(root, slide)
    if not patches:
        raise RuntimeError(f"{slide} has no patches")
    _glass, content_list = _content_windows(patches)
    content = {(str(window.path), window.row, window.col) for window in content_list}
    y_mid = float(np.median([patch.y + PATCH // 2 for patch in patches]))
    x_mid = float(np.median([patch.x + PATCH // 2 for patch in patches]))
    if overviews is None:
        overviews = _quadrant_overviews(patches, y_mid, x_mid)
    if used is None:
        used = set()
    large, mids, natives = _grid_pools(patches, content, y_mid, x_mid)
    # Large fields claim their 1024 patches first, so mid cells prefer other tissue.
    picked_large = _pad_fields(
        _fill_quadrants(large, GRID_LARGE_PER_QUADRANT, version, used),
        4 * GRID_LARGE_PER_QUADRANT, patches,
    )
    for _gray, _y, _x, _kind, mates, _key in picked_large:
        if _kind == "2048":
            for mate in mates:
                used.add(("1024", str(mate.path)))
    picked_mids = _pad_fields(
        _fill_quadrants(mids, GRID_MID_PER_QUADRANT, version, used),
        4 * GRID_MID_PER_QUADRANT, patches,
    )
    for _gray, _y, _x, _kind, patch, _key in picked_mids:
        if _kind == "1024":
            for window in patch.windows:
                used.add(("256", str(window.path), window.row, window.col))
    picked_native = _pad_fields(
        _fill_quadrants(natives, GRID_NATIVE_PER_QUADRANT, version, used),
        4 * GRID_NATIVE_PER_QUADRANT, patches,
    )
    fields = [*picked_large, *picked_mids, *picked_native]
    if len(overviews) != 4 or len(fields) != 60:
        raise RuntimeError(
            f"{slide} version {version}: overviews={len(overviews)} "
            f"large={len(picked_large)} mid={len(picked_mids)} native={len(picked_native)}"
        )
    cells = list(overviews)
    records = [{"kind": "overview", "quadrant": index} for index in range(4)]
    for gray, y, x, kind, payload, key in fields:
        cells.append(_cell_at(kind, payload, None, GRID_CELL) if kind != "256" else _cell_at_native(payload))
        records.append({
            "kind": kind, "y": int(y), "x": int(x),
            "gray": float(gray), "key": key,
        })
    return _paint_grid(cells), records


def _cell_at_native(window: Window) -> np.ndarray:
    image = _load_rgb(window.path)
    return image[
        window.row * GRID_CELL:(window.row + 1) * GRID_CELL,
        window.col * GRID_CELL:(window.col + 1) * GRID_CELL,
    ].copy()
