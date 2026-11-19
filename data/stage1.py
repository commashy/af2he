"""AF, aligned pseudo label, and the same-coordinate real H&E.

The four quadrants of one 1024 file are one sample. AF, pseudo, and real H&E
share one horizontal flip. The H&E section is misaligned, so it is a GAN and
FDL target, not a pixel target.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from data.dataset import (
    SOURCE,
    TILE,
    _af_collages,
    black_cell,
    load_local_context,
    load_native_image,
    tissue_cell,
)
from data.groups import slide_and_coord


def _quadrants(image: torch.Tensor, flip: bool) -> torch.Tensor:
    """Split a 1024 into TL, TR, BL, BR. One flip is applied to the whole field."""
    if image.shape[-2] < SOURCE or image.shape[-1] < SOURCE:
        raise ValueError(f"image is {tuple(image.shape[-2:])}, smaller than {SOURCE}")
    image = image[:, :SOURCE, :SOURCE]
    if flip:
        image = torch.flip(image, dims=(-1,))
    tiles = []
    for row in (0, TILE):
        for col in (0, TILE):
            tiles.append(image[:, row:row + TILE, col:col + TILE])
    return torch.stack(tiles, dim=0)


# 512 windows centered on the cuts of a 1024 field: vertical top, vertical
# bottom, horizontal left, horizontal right.
SEAM_WINDOWS = ((0, 256), (512, 256), (256, 0), (256, 512))


def seam_windows(field: torch.Tensor) -> list[torch.Tensor]:
    """Four 512 crops of a stitched 1024, each centered on one half of a cut."""
    if field.ndim != 4 or field.shape[-2:] != (SOURCE, SOURCE):
        raise ValueError(f"expected N x C x {SOURCE} x {SOURCE}, got {tuple(field.shape)}")
    return [field[:, :, y:y + TILE, x:x + TILE] for y, x in SEAM_WINDOWS]


def stitch_quadrants(tiles: torch.Tensor) -> torch.Tensor:
    """Invert ``_quadrants`` for a batch of 4-tile groups. Returns N x 3 x 1024 x 1024."""
    if tiles.ndim != 4 or tiles.shape[0] % 4 != 0 or tiles.shape[-2:] != (TILE, TILE):
        raise ValueError(f"expected groups of four {TILE} tiles, got {tuple(tiles.shape)}")
    groups = tiles.view(tiles.shape[0] // 4, 4, tiles.shape[1], TILE, TILE)
    top = torch.cat((groups[:, 0], groups[:, 1]), dim=-1)
    bottom = torch.cat((groups[:, 2], groups[:, 3]), dim=-1)
    return torch.cat((top, bottom), dim=-2)


def _specimen(slide: str) -> str:
    return slide.split("_", 1)[1]


def exclusion_keys(path: str | None) -> set[str]:
    """Accession ids from an exclusion list. Blank lines and comments are ignored."""
    if not path:
        return set()
    keys = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        key = line.split("#", 1)[0].strip()
        if key:
            keys.add(key)
    return keys


def is_excluded(slide: str, keys: set[str]) -> bool:
    return slide in keys or _specimen(slide) in keys


class Stage1Dataset(Dataset):
    def __init__(
        self,
        af_root: str,
        pseudo_root: str | None,
        he_root: str,
        alignment: str,
        index_path: str | None,
        teacher_root: str | None = None,
        exclude_path: str | None = None,
        extra_slides: str | None = None,
        pair_index: str | None = None,
        epoch10_root: str | None = None,
        collage_size: int = SOURCE,
        local_root: str | None = None,
        collage_root: str | None = None,
        heat_targets: dict[str, tuple[float, bool]] | None = None,
    ):
        af_dir = Path(af_root).resolve()
        he_dir = Path(he_root).resolve()
        # Stage 2 passes the epoch-10 cache and does not read the old pseudo labels.
        use_pseudo = epoch10_root is None
        pseudo_dir = Path(pseudo_root).resolve() if use_pseudo else None
        if use_pseudo and pseudo_dir is None:
            raise ValueError("pseudo_root is required when epoch10_root is not set")
        for directory in (af_dir, he_dir, pseudo_dir):
            if directory is not None and not directory.is_dir():
                raise FileNotFoundError(f"patch root does not exist: {directory}")
        allowed = {
            row["specimen_key"]
            for row in csv.DictReader(Path(alignment).open(newline="", encoding="utf-8"))
            if row["use"] == "crop_absolute"
        }
        if not allowed:
            raise RuntimeError(f"{alignment} has no crop_absolute slides")
        excluded = exclusion_keys(exclude_path)
        allowed -= {key for key in excluded if key in allowed}
        if not allowed:
            raise RuntimeError(f"every crop_absolute slide is excluded by {exclude_path}")
        added = exclusion_keys(extra_slides)
        self.collage_root = Path(collage_root) if collage_root else None
        self.meta: dict[str, dict] = {}
        if self.collage_root is None:
            collages = _af_collages(json.loads(Path(index_path).read_text(encoding="utf-8"))["groups"])
        else:
            if not self.collage_root.is_dir():
                raise FileNotFoundError(f"collage root does not exist: {self.collage_root}")
            collages = None
        self.samples: list[tuple[Path, Path, Path, list[str]]] = []
        missing = 0
        missing_he = 0
        examples: list[str] = []
        he_examples: list[str] = []
        # The retrieval index is the training set. Reading it avoids a cold
        # walk of every file in the patch directory, which stalls four ranks
        # on the hard drive.
        if pair_index:
            names = [
                row["filename"]
                for row in csv.DictReader(Path(pair_index).open(newline="", encoding="utf-8"))
            ]
            candidates = [af_dir / name for name in names]
            check_files = False
        else:
            candidates = sorted(af_dir.iterdir())
            check_files = True
        for path in candidates:
            if not path.suffix.lower() == ".png" or "__" not in path.name:
                continue
            slide = path.name.split("__", 1)[0]
            if is_excluded(slide, excluded):
                continue
            if _specimen(slide) not in allowed and slide not in added:
                continue
            pseudo = None if pseudo_dir is None else pseudo_dir / path.name
            he = he_dir / path.name
            if check_files and pseudo is not None and (not pseudo.is_file() or pseudo.stat().st_size == 0):
                missing += 1
                if len(examples) < 5:
                    examples.append(path.name)
                continue
            if check_files and (not he.is_file() or he.stat().st_size == 0):
                missing_he += 1
                if len(he_examples) < 5:
                    he_examples.append(path.name)
                continue
            if self.collage_root is not None:
                if slide not in self.meta:
                    meta_path = self.collage_root / slide / "meta.json"
                    grid = self.collage_root / slide / "v0.png"
                    if not meta_path.is_file() or not grid.is_file():
                        raise RuntimeError(f"{slide} has no AF collage grid")
                    self.meta[slide] = json.loads(meta_path.read_text(encoding="utf-8"))
                self.samples.append((path, pseudo, he, slide))
                continue
            slide_collages = collages.get(slide) if collages is not None else None
            if not slide_collages:
                raise RuntimeError(f"{slide} has no AF collage")
            self.samples.append((path, pseudo, he, slide_collages))
        if missing:
            raise RuntimeError(f"{missing} AF patches have no pseudo label, for example {examples}")
        if missing_he:
            raise RuntimeError(f"{missing_he} AF patches have no paired H&E, for example {he_examples}")
        if not self.samples:
            raise RuntimeError("no stage-1 pairs")
        self.teacher_root = Path(teacher_root) if teacher_root else None
        if self.teacher_root is not None and not self.teacher_root.is_dir():
            raise FileNotFoundError(f"teacher cache does not exist: {self.teacher_root}")
        self.epoch10_root = Path(epoch10_root) if epoch10_root else None
        if self.epoch10_root is not None and not self.epoch10_root.is_dir():
            raise FileNotFoundError(f"epoch-10 cache does not exist: {self.epoch10_root}")
        # The added slides use the epoch-10 image as their teacher. The
        # original slides keep the teacher cache and use epoch 10 as the gate.
        self.epoch10_as_teacher = added if self.epoch10_root is not None else set()
        self.local_root = Path(local_root) if local_root else None
        if self.local_root is not None and not self.local_root.is_dir():
            raise FileNotFoundError(f"local neighborhood cache does not exist: {self.local_root}")
        self.collage_size = int(collage_size)
        self.heat_targets = heat_targets
        self.slides = len({path.name.split('__', 1)[0] for path, _, _, _ in self.samples})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        af_path, pseudo_path, he_path, collage_ref = self.samples[index]
        paired_flip = random.random() < 0.5
        version = 0
        if self.collage_root is None:
            collage = load_native_image(Path(random.choice(collage_ref)))
        else:
            slide = collage_ref
            versions = int(self.meta[slide].get("versions", 8))
            version = random.randrange(versions)
            collage_path = self.collage_root / slide / f"v{version}.png"
            if not collage_path.is_file():
                version = 0
                collage_path = self.collage_root / slide / "v0.png"
            collage = load_native_image(collage_path)
        if collage.shape[-2:] != (self.collage_size, self.collage_size):
            raise ValueError(f"collage is {tuple(collage.shape)}, expected {self.collage_size}")
        sample = {
            "af": _quadrants(load_native_image(af_path), paired_flip),
            "he": _quadrants(load_native_image(he_path), paired_flip),
            "collage": collage,
            "slide": af_path.name.split("__", 1)[0],
            "name": af_path.name,
            "flip": paired_flip,
        }
        if self.collage_root is not None:
            parsed, (y, x) = slide_and_coord(af_path)
            if parsed != collage_ref:
                raise RuntimeError(f"{af_path.name} belongs to {parsed}, not {collage_ref}")
            sample["cell"] = tissue_cell(self.meta[collage_ref]["box"], y, x)
            sample["black"] = black_cell(self.meta[collage_ref]["cells"], version)
        if self.heat_targets is not None:
            known = af_path.name in self.heat_targets
            target, from_map = self.heat_targets.get(af_path.name, (0.0, False))
            sample["heat_t"] = float(target)
            sample["heat_map"] = bool(from_map)
            sample["heat_known"] = bool(known)
        if self.local_root is not None:
            local = load_local_context(self.local_root / af_path.name)
            if paired_flip:
                local = torch.flip(local, dims=(-1,))
            sample["local"] = local
        if pseudo_path is not None:
            sample["pseudo"] = _quadrants(load_native_image(pseudo_path), paired_flip)
        epoch10 = None
        if self.epoch10_root is not None:
            epoch10 = _quadrants(load_native_image(self.epoch10_root / af_path.name), paired_flip)
            sample["epoch10"] = epoch10
        if self.teacher_root is not None:
            slide = af_path.name.split("__", 1)[0]
            if epoch10 is not None and slide in self.epoch10_as_teacher:
                sample["teacher"] = epoch10.clone()
            else:
                teacher_path = self.teacher_root / af_path.name
                sample["teacher"] = _quadrants(load_native_image(teacher_path), paired_flip)
        return sample


def collate_stage1(batch: list[dict]) -> dict:
    def cat(key: str) -> torch.Tensor:
        return torch.cat([item[key] for item in batch], dim=0)

    collage = torch.stack([item["collage"] for item in batch], dim=0).repeat_interleave(4, dim=0)
    packed = {
        "af": cat("af"),
        "he": cat("he"),
        "collage": collage,
        "slide": [item["slide"] for item in batch],
        "name": [item["name"] for item in batch],
        "flip": [bool(item["flip"]) for item in batch],
    }
    if "pseudo" in batch[0]:
        packed["pseudo"] = cat("pseudo")
    if "teacher" in batch[0]:
        packed["teacher"] = cat("teacher")
    if "epoch10" in batch[0]:
        packed["epoch10"] = cat("epoch10")
    if "local" in batch[0]:
        packed["local"] = torch.stack([item["local"] for item in batch], dim=0)
    if "cell" in batch[0]:
        packed["cell"] = torch.tensor(
            [item["cell"] for item in batch], dtype=torch.long,
        ).repeat_interleave(4, dim=0)
        packed["black"] = torch.tensor(
            [item["black"] for item in batch], dtype=torch.long,
        ).repeat_interleave(4, dim=0)
    if "heat_t" in batch[0]:
        packed["heat_t"] = torch.tensor([item["heat_t"] for item in batch], dtype=torch.float32)
        packed["heat_map"] = torch.tensor([item["heat_map"] for item in batch], dtype=torch.bool)
        packed["heat_known"] = torch.tensor([item["heat_known"] for item in batch], dtype=torch.bool)
    return packed
