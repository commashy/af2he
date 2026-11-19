"""Tiles plus that slide's AF collage. No percentile stretch.

H&E tiles use the same AF collage. The collage encoder only ever sees AF,
which is also what inference has.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.groups import slide_and_coord

TILE = 512
SOURCE = 1024


def load_native_image(path: Path) -> torch.Tensor:
    """uint8 image -> float CHW in [-1, 1]. Full stored range, no contrast stretch."""
    with Image.open(path) as image:
        if image.mode not in ("RGB", "L"):
            raise ValueError(f"{path} has mode {image.mode}; expected RGB or L")
        array = np.array(image, dtype=np.uint8, copy=True)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{path} has shape {array.shape}")
    if min(array.shape[0], array.shape[1]) < TILE:
        raise ValueError(f"{path} is {array.shape[:2]}, smaller than {TILE}")
    tensor = torch.tensor(array, dtype=torch.float32).permute(2, 0, 1).contiguous()
    return tensor.mul_(2.0 / 255.0).sub_(1.0)


def load_local_context(path: Path) -> torch.Tensor:
    """RGBA neighborhood. RGB is [-1, 1]. Alpha is the covered fraction in [0, 1]."""
    with Image.open(path) as image:
        if image.mode != "RGBA":
            raise ValueError(f"{path} has mode {image.mode}; expected RGBA")
        array = np.array(image, dtype=np.uint8, copy=True)
    if array.shape != (SOURCE, SOURCE, 4):
        raise ValueError(f"{path} has shape {array.shape}, expected {(SOURCE, SOURCE, 4)}")
    rgb = torch.tensor(array[:, :, :3], dtype=torch.float32).permute(2, 0, 1).contiguous()
    rgb.mul_(2.0 / 255.0).sub_(1.0)
    mask = torch.tensor(array[:, :, 3], dtype=torch.float32).mul_(1.0 / 255.0).unsqueeze(0)
    return torch.cat((rgb, mask), dim=0)


def tissue_cell(box: list[int] | tuple[int, int, int, int], y: int, x: int) -> tuple[int, int]:
    """Row and column of the 1024 field inside the tissue box. Outside is (-1, -1)."""
    y0, x0, y1, x1 = box
    cy = y + SOURCE / 2.0
    cx = x + SOURCE / 2.0
    if cy < y0 or cy >= y1 or cx < x0 or cx >= x1:
        return -1, -1
    cell_h = max((y1 - y0) / 8.0, 1.0)
    cell_w = max((x1 - x0) / 8.0, 1.0)
    row = min(7, int((cy - y0) / cell_h))
    col = min(7, int((cx - x0) / cell_w))
    return row, col


def black_cell(cells: list, version: int) -> int:
    """First glass cell in this collage version, or -1 when every cell has tissue."""
    for index, cell in enumerate(cells):
        if cell[version] is None:
            return index
    return -1


def _af_collages(groups: dict) -> dict[str, list[str]]:
    """Map a slide id to its AF collage paths."""
    found: dict[str, list[str]] = {}
    for key, paths in groups.items():
        if not isinstance(key, str) or "|" not in key:
            raise RuntimeError(f"collage index key {key!r} has no slide id")
        slide = key.rsplit("|", 1)[-1]
        collages = [str(path) for path in paths]
        if len(collages) < 1:
            raise RuntimeError(f"{slide} has no collages")
        previous = found.get(slide)
        if previous is not None and previous != collages:
            raise RuntimeError(f"{slide} is listed with two different AF collages")
        found[slide] = collages
    return found


class PretrainDataset(Dataset):
    """Four 512 quadrants of a 1024 patch, plus one AF collage for that slide."""

    def __init__(
        self,
        roots: list[str],
        index_path: str,
        exclude: set[str] | None = None,
        collage_root: str | None = None,
        labels: dict[str, tuple[int, int]] | None = None,
    ):
        self.collage_root = None if collage_root is None else Path(collage_root)
        self.labels = labels or {}
        self.meta: dict[str, dict] = {}
        self.pairs: list[tuple[int, int]] = []
        by_slide: dict[str, list[str]] | None = None
        if self.collage_root is None:
            index_file = Path(index_path)
            if not index_file.is_file():
                raise FileNotFoundError(
                    f"missing collage index {index_file}. Run scripts/build_collages.py first."
                )
            payload = json.loads(index_file.read_text(encoding="utf-8"))
            groups = payload.get("groups")
            if not isinstance(groups, dict) or not groups:
                raise RuntimeError(f"{index_file} has no collage groups")
            by_slide = _af_collages(groups)
        self.targets: list[tuple] = []
        self.counts: dict[str, int] = {}
        self.excluded = 0
        self.unlabeled = 0
        blocked = exclude or set()
        seen: set[str] = set()
        missing = 0
        examples: list[str] = []
        for root in roots:
            directory = Path(root).resolve()
            if not directory.is_dir():
                raise FileNotFoundError(f"patch root does not exist: {directory}")
            count = 0
            for path in sorted(directory.iterdir()):
                if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                slide, (y, x) = slide_and_coord(path)
                specimen = slide.split("_", 1)[1] if "_" in slide else slide
                if specimen in blocked:
                    self.excluded += 1
                    continue
                if self.collage_root is not None:
                    if slide not in self.labels:
                        self.unlabeled += 1
                        continue
                    collage = self.collage_root / slide / "v0.png"
                    if not collage.is_file():
                        missing += 1
                        if len(examples) < 5:
                            examples.append(slide)
                        continue
                    if slide not in seen:
                        seen.add(slide)
                        self.meta[slide] = json.loads(
                            (self.collage_root / slide / "meta.json").read_text(encoding="utf-8")
                        )
                        self.pairs.append(self.labels[slide])
                    self.targets.append((path, slide, y, x))
                else:
                    collages = by_slide.get(slide) if by_slide is not None else None
                    if not collages:
                        missing += 1
                        if len(examples) < 5:
                            examples.append(slide)
                        continue
                    self.targets.append((path, collages))
                count += 1
            self.counts[str(directory)] = count
        if missing:
            raise RuntimeError(
                f"{missing} tiles have no AF collage, for example {examples}"
            )
        if not self.targets:
            raise RuntimeError("no tiles matched the collage set")

    def _cell(self, slide: str, y: int, x: int) -> tuple[int, int]:
        # A field whose center is outside the tissue box has no collage cell.
        return tissue_cell(self.meta[slide]["box"], y, x)

    def _black(self, slide: str, version: int) -> int:
        return black_cell(self.meta[slide]["cells"], version)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.targets[index]
        path = item[0]
        image = load_native_image(path)
        if image.shape[1] < SOURCE or image.shape[2] < SOURCE:
            raise ValueError(f"{path} is {tuple(image.shape[-2:])}, smaller than {SOURCE}")
        image = image[:, :SOURCE, :SOURCE]
        tiles = []
        for row in (0, TILE):
            for col in (0, TILE):
                tile = image[:, row:row + TILE, col:col + TILE]
                if random.random() < 0.5:
                    tile = torch.flip(tile, dims=(-1,))
                tiles.append(tile)
        stacked = torch.stack(tiles, dim=0)
        if self.collage_root is None:
            collages = item[1]
            collage = load_native_image(Path(random.choice(collages)))
            if collage.shape[-2:] != (SOURCE, SOURCE):
                raise ValueError(f"collage {collages[0]} is {tuple(collage.shape)}")
            return {"tile": stacked, "collage": collage}
        _path, slide, y, x = item
        version = random.randrange(8)
        collage_path = self.collage_root / slide / f"v{version}.png"
        if not collage_path.is_file():
            version = 0
            collage_path = self.collage_root / slide / "v0.png"
        collage = load_native_image(collage_path)
        binary, subtype = self.labels[slide]
        row, col = self._cell(slide, y, x)
        return {
            "tile": stacked,
            "collage": collage,
            "cell": (row, col),
            "black": self._black(slide, version),
            "bin": binary,
            "sub": subtype,
        }


def collate_quadrants(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Four quadrants from one file share that file's collage."""
    tiles = torch.cat([item["tile"] for item in batch], dim=0)
    collages = torch.stack([item["collage"] for item in batch], dim=0).repeat_interleave(4, dim=0)
    if tiles.shape[0] != collages.shape[0]:
        raise RuntimeError(f"tiles {tiles.shape[0]} != collages {collages.shape[0]}")
    packed = {"tile": tiles, "collage": collages}
    if "cell" not in batch[0]:
        return packed
    packed["cell"] = torch.tensor([item["cell"] for item in batch], dtype=torch.long).repeat_interleave(4, dim=0)
    packed["black"] = torch.tensor([item["black"] for item in batch], dtype=torch.long).repeat_interleave(4, dim=0)
    packed["bin"] = torch.tensor([item["bin"] for item in batch], dtype=torch.long).repeat_interleave(4, dim=0)
    packed["sub"] = torch.tensor([item["sub"] for item in batch], dtype=torch.long).repeat_interleave(4, dim=0)
    return packed
