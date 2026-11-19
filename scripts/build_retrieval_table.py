"""Build the UNI retrieval table for stage-1.

One GPU encodes. Two reader processes decode the next slides into shared
memory. Run this beside epoch 5, on the GPU the trainer is not using:

    CUDA_VISIBLE_DEVICES=3 python scripts/build_retrieval_table.py --gpu 0

The stored positive is the UNI class token plus the crop offset. The chosen
pixels stay in the existing H&E tiles; ``dy`` and ``dx`` cut them later.
A slide is marked done only after its rows are flushed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.stage1 import Stage1Dataset, exclusion_keys
from engine.image_losses import hed_tissue_fraction
from engine.retrieval_match import (
    GRID,
    SHORTLIST,
    SKIP_NEAREST,
    TOTAL_NEGATIVES,
    Candidate,
    candidate_offsets,
    choose_positive,
    crop_covered,
    extend_negatives,
    pad_negatives,
    rank_shortlist,
    running_topk,
    stitch_crop,
    tile_origins_overlapping,
    tissue_gap,
)

DIM = 1024
ENCODER = "MahmoodLab/uni"
LABELS = Path("/mnt/10tb_hdd/taek/stain_translation_core/manifests/fresh/train_slide_labels.csv")
CONFIG = ROOT / "configs" / "stage1_pseudo.yaml"
TISSUE_BATCH = 8
TOPK_BLOCK = 4096


@dataclass
class Field:
    row: int
    filename: str
    slide: str
    specimen: str
    y: int
    x: int
    classes1: str
    subtype: str
    pseudo: str
    he: str
    teacher: str


def parse_yx(name: str) -> tuple[int, int]:
    body = name.split("__", 1)[1]
    if body.lower().endswith(".png"):
        body = body[:-4]
    y_text, x_text = body.split("_")
    if not y_text.startswith("y") or not x_text.startswith("x"):
        raise ValueError(f"cannot read coordinates from {name}")
    return int(y_text[1:]), int(x_text[1:])


def load_labels(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    by_slide: dict[str, dict[str, str]] = {}
    by_specimen: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            slide = (row.get("slide_id") or "").strip()
            if not slide:
                continue
            by_slide[slide] = row
            if "_" in slide:
                by_specimen[slide.split("_", 1)[1]] = row
    return by_slide, by_specimen


def diagnosis(row: dict[str, str] | None, slide: str) -> tuple[str, str]:
    if row is None:
        raise RuntimeError(f"{slide} has no benign/malignant label")
    classes1 = (row.get("classes1") or "").strip()
    subtype = (row.get("subtype_norm") or "").strip()
    if classes1 not in {"Benign", "Malignant"}:
        raise RuntimeError(f"{slide} has classes1 {classes1!r}")
    return classes1, subtype


def group_name(classes1: str, subtype: str) -> str:
    if classes1 == "Benign":
        return "Benign"
    return subtype or "Malignant"


def build_fields(config: dict, labels_path: Path) -> list[Field]:
    dataset = Stage1Dataset(
        config["af_root"],
        config["pseudo_root"],
        config["he_root"],
        config["alignment"],
        config["collage_index"],
        config.get("teacher_dir"),
        config.get("exclude_slides"),
    )
    teacher_root = Path(config["teacher_dir"])
    by_slide, by_specimen = load_labels(labels_path)
    fields: list[Field] = []
    missing: list[str] = []
    for af_path, pseudo_path, he_path, _collages in dataset.samples:
        teacher = teacher_root / af_path.name
        if not teacher.is_file() or teacher.stat().st_size == 0:
            if len(missing) < 5:
                missing.append(af_path.name)
            continue
        slide = af_path.name.split("__", 1)[0]
        y, x = parse_yx(af_path.name)
        label = by_slide.get(slide) or by_specimen.get(slide.split("_", 1)[1])
        classes1, subtype = diagnosis(label, slide)
        fields.append(
            Field(
                row=len(fields),
                filename=af_path.name,
                slide=slide,
                specimen=slide.split("_", 1)[1],
                y=y,
                x=x,
                classes1=classes1,
                subtype=subtype,
                pseudo=str(pseudo_path),
                he=str(he_path),
                teacher=str(teacher),
            )
        )
    if missing:
        raise RuntimeError(f"{len(missing)} training patches have no teacher, for example {missing}")
    if not fields:
        raise RuntimeError("no retrieval fields")
    return fields


def write_index(path: Path, fields: list[Field]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row", "filename", "slide", "specimen", "y", "x", "classes1", "subtype_norm",
            ],
        )
        writer.writeheader()
        for field in fields:
            writer.writerow(
                {
                    "row": field.row,
                    "filename": field.filename,
                    "slide": field.slide,
                    "specimen": field.specimen,
                    "y": field.y,
                    "x": field.x,
                    "classes1": field.classes1,
                    "subtype_norm": field.subtype,
                }
            )


def read_index(path: Path) -> list[tuple[str, str, str, str]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append((row["filename"], row["slide"], row["classes1"], row["subtype_norm"]))
    return rows


def open_array(path: Path, shape: tuple[int, ...], dtype: np.dtype, create: bool) -> np.memmap:
    dtype = np.dtype(dtype)
    if create or not path.exists():
        array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
        array.flush()
        return array
    array = np.lib.format.open_memmap(path, mode="r+")
    if array.shape != shape or array.dtype != dtype:
        raise RuntimeError(f"{path} is {array.dtype} {array.shape}, expected {dtype} {shape}")
    return array


def read_done(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def mark_done(path: Path, slide: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(slide + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _crop_uint8(path: str) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3 or array.shape[0] < GRID or array.shape[1] < GRID:
        raise ValueError(f"{path} has shape {array.shape}")
    return np.ascontiguousarray(array[:GRID, :GRID])


def _fill_shared(n: int, names: dict[str, str], paths: dict[str, list[str]]) -> None:
    attached = []
    try:
        for kind, name in names.items():
            shm = shared_memory.SharedMemory(name=name)
            attached.append(shm)
            array = np.ndarray((n, GRID, GRID, 3), dtype=np.uint8, buffer=shm.buf)
            for index, path in enumerate(paths[kind]):
                array[index] = _crop_uint8(path)
    finally:
        for shm in attached:
            shm.close()


def _reader(in_queue, out_queue) -> None:
    while True:
        job = in_queue.get()
        if job is None:
            return
        slide, n_items, names, paths = job
        try:
            _fill_shared(n_items, names, paths)
        except Exception:
            out_queue.put((slide, traceback.format_exc()))
        else:
            out_queue.put((slide, None))


def _attach(name: str, n_items: int) -> tuple[shared_memory.SharedMemory, np.ndarray]:
    shm = shared_memory.SharedMemory(name=name)
    array = np.ndarray((n_items, GRID, GRID, 3), dtype=np.uint8, buffer=shm.buf)
    return shm, array


def tissue_values(images: np.ndarray, device: torch.device) -> np.ndarray:
    """Hematoxylin plus eosin fraction at the FDL threshold, one float per image."""
    values = np.empty((images.shape[0],), dtype=np.float32)
    for start in range(0, images.shape[0], TISSUE_BATCH):
        stop = min(images.shape[0], start + TISSUE_BATCH)
        batch = torch.from_numpy(np.ascontiguousarray(images[start:stop]))
        batch = batch.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
        batch = batch.mul_(2.0 / 255.0).sub_(1.0)
        with torch.inference_mode():
            values[start:stop] = hed_tissue_fraction(batch).float().cpu().numpy()
    return values


def load_encoder(device: torch.device):
    """UNI ViT-L/16. The class token is the 1024-d tile embedding."""
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    model = timm.create_model(
        "hf-hub:MahmoodLab/uni",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True,
        num_classes=0,
    )
    model.eval().to(device)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, transform


def _class_token(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim == 2 and tokens.shape[-1] == DIM:
        return tokens.float()
    if tokens.ndim == 3 and tokens.shape[-1] == DIM:
        return tokens[:, 0].float()
    raise RuntimeError(f"UNI output {tuple(tokens.shape)} is not a {DIM}-d class token")


def embed_uint8(model, transform, images: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    count = images.shape[0]
    out = np.empty((count, DIM), dtype=np.float32)
    cursor = 0
    current = max(1, batch_size)
    while cursor < count:
        stop = min(count, cursor + current)
        tensors = [transform(Image.fromarray(images[index], mode="RGB")) for index in range(cursor, stop)]
        batch = torch.stack(tensors, dim=0).to(device, non_blocking=True)
        try:
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16):
                tokens = model(batch)
        except torch.cuda.OutOfMemoryError:
            del batch, tensors
            torch.cuda.empty_cache()
            if current == 1:
                raise
            current = max(1, current // 2)
            print(f"UNI batch reduced to {current}", flush=True)
            continue
        out[cursor:stop] = _class_token(tokens).cpu().numpy()
        cursor = stop
    if not np.isfinite(out).all():
        raise RuntimeError("UNI produced a non-finite embedding")
    return out


def _unit(matrix: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norm == 0):
        raise RuntimeError("an embedding has zero length")
    return matrix / norm


def process_slide(fields: list[Field], images: dict[str, np.ndarray], model, transform, tables: dict, device, batch_size: int) -> None:
    n_items = len(fields)
    start = fields[0].row
    if [field.row for field in fields] != list(range(start, start + n_items)):
        raise RuntimeError(f"{fields[0].slide} rows are not contiguous")
    tissue_he = tissue_values(images["he"], device)
    tissue_teacher = tissue_values(images["teacher"], device)
    tissue_pseudo = tissue_values(images["pseudo"], device)
    tables["tissue_he"][start:start + n_items] = tissue_he
    tables["tissue_teacher"][start:start + n_items] = tissue_teacher
    tables["tissue_pseudo"][start:start + n_items] = tissue_pseudo

    he_emb = embed_uint8(model, transform, images["he"], device, batch_size)
    teacher_emb = embed_uint8(model, transform, images["teacher"], device, batch_size)
    tables["he"][start:start + n_items] = he_emb.astype(np.float16)
    tables["teacher"][start:start + n_items] = teacher_emb.astype(np.float16)
    he_unit = _unit(he_emb)
    teacher_unit = _unit(teacher_emb)

    tiles = {(field.y, field.x): index for index, field in enumerate(fields)}
    tile_set = set(tiles)
    he_tiles = {key: images["he"][index] for key, index in tiles.items()}
    tissue_at = {key: float(tissue_he[index]) for key, index in tiles.items()}
    field_offsets: list[list[tuple[int, int]]] = []
    pending: list[tuple[tuple[int, int], int, int, int, int]] = []
    seen = set(tissue_at)
    for field in fields:
        offsets = candidate_offsets(field.y, field.x, tile_set)
        field_offsets.append(offsets)
        for dy, dx in offsets:
            key = (field.y + dy, field.x + dx)
            if key in seen:
                continue
            if not crop_covered(field.y, field.x, dy, dx, tile_set):
                raise RuntimeError(f"{field.filename} shift {(dy, dx)} is not covered")
            seen.add(key)
            pending.append((key, field.y, field.x, dy, dx))
    for cursor in range(0, len(pending), TISSUE_BATCH):
        chunk = pending[cursor:cursor + TISSUE_BATCH]
        crops = np.stack([stitch_crop(he_tiles, y, x, dy, dx) for _key, y, x, dy, dx in chunk])
        fractions = tissue_values(crops, device)
        for (key, _y, _x, _dy, _dx), fraction in zip(chunk, fractions):
            tissue_at[key] = float(fraction)

    shortlists: list[list[Candidate]] = []
    shift_jobs: list[tuple[tuple[int, int], int, int, int, int]] = []
    shift_seen: set[tuple[int, int]] = set()
    for index, field in enumerate(fields):
        scored = []
        for dy, dx in field_offsets[index]:
            gap = tissue_gap(
                tissue_at[(field.y + dy, field.x + dx)],
                float(tissue_teacher[index]),
                float(tissue_pseudo[index]),
            )
            scored.append(Candidate(dy, dx, gap))
        short = rank_shortlist(scored, SHORTLIST)
        shortlists.append(short)
        for item in short:
            key = (field.y + item.dy, field.x + item.dx)
            if key in tiles or key in shift_seen:
                continue
            shift_seen.add(key)
            shift_jobs.append((key, field.y, field.x, item.dy, item.dx))
    shift_unit: dict[tuple[int, int], np.ndarray] = {}
    shift_raw: dict[tuple[int, int], np.ndarray] = {}
    for cursor in range(0, len(shift_jobs), 32):
        chunk = shift_jobs[cursor:cursor + 32]
        crops = np.stack([
            stitch_crop(he_tiles, y, x, dy, dx) for _key, y, x, dy, dx in chunk
        ])
        embedded = embed_uint8(model, transform, crops, device, batch_size)
        for (key, _y, _x, _dy, _dx), vector in zip(chunk, embedded):
            shift_raw[key] = vector
            shift_unit[key] = _unit(vector[None, :])[0]

    positive = np.empty((n_items, DIM), dtype=np.float16)
    positive_row = np.full((n_items,), -1, dtype=np.int32)
    positive_dy = np.zeros((n_items,), dtype=np.int16)
    positive_dx = np.zeros((n_items,), dtype=np.int16)
    positive_cosine = np.empty((n_items,), dtype=np.float32)
    positive_gap = np.empty((n_items,), dtype=np.float32)
    short_row = np.full((n_items, SHORTLIST), -1, dtype=np.int32)
    short_dy = np.zeros((n_items, SHORTLIST), dtype=np.int16)
    short_dx = np.zeros((n_items, SHORTLIST), dtype=np.int16)
    short_cosine = np.full((n_items, SHORTLIST), np.nan, dtype=np.float32)
    for index, field in enumerate(fields):
        scored = []
        vectors = []
        for slot, item in enumerate(shortlists[index]):
            key = (field.y + item.dy, field.x + item.dx)
            if key in tiles:
                local = tiles[key]
                vector = he_unit[local]
                raw = he_emb[local]
                short_row[index, slot] = fields[local].row
            else:
                vector = shift_unit[key]
                raw = shift_raw[key]
            cosine = float(np.dot(teacher_unit[index], vector))
            scored.append((item, cosine))
            vectors.append(raw)
            short_dy[index, slot] = item.dy
            short_dx[index, slot] = item.dx
            short_cosine[index, slot] = cosine
        winner, cosine = choose_positive(scored)
        winner_index = shortlists[index].index(winner)
        positive[index] = vectors[winner_index].astype(np.float16)
        positive_dy[index] = winner.dy
        positive_dx[index] = winner.dx
        positive_cosine[index] = cosine
        positive_gap[index] = winner.gap
        key = (field.y + winner.dy, field.x + winner.dx)
        if key in tiles:
            positive_row[index] = fields[tiles[key]].row
    tables["positive"][start:start + n_items] = positive
    tables["positive_row"][start:start + n_items] = positive_row
    tables["positive_dy"][start:start + n_items] = positive_dy
    tables["positive_dx"][start:start + n_items] = positive_dx
    tables["positive_cosine"][start:start + n_items] = positive_cosine
    tables["positive_gap"][start:start + n_items] = positive_gap
    tables["shortlist_row"][start:start + n_items] = short_row
    tables["shortlist_dy"][start:start + n_items] = short_dy
    tables["shortlist_dx"][start:start + n_items] = short_dx
    tables["shortlist_cosine"][start:start + n_items] = short_cosine
    for array in tables.values():
        array.flush()


def _normalize_memmap(array: np.memmap, device: torch.device) -> torch.Tensor:
    out = torch.empty((array.shape[0], array.shape[1]), dtype=torch.float16, device=device)
    for start in range(0, array.shape[0], 8192):
        stop = min(array.shape[0], start + 8192)
        chunk = torch.from_numpy(np.asarray(array[start:stop])).to(device=device, dtype=torch.float32)
        out[start:stop] = torch.nn.functional.normalize(chunk, dim=1).half()
    return out


def _ranked_global(local_indices: torch.Tensor, bank_rows: np.ndarray) -> list[list[int]]:
    ranked = []
    for row in local_indices.tolist():
        ranked.append([int(bank_rows[item]) for item in row if item >= 0])
    return ranked


def _same_slide_ranked(
    positive: torch.Tensor,
    he: torch.Tensor,
    rows: np.ndarray,
    excluded: list[set[int]],
    device: torch.device,
) -> list[list[int]]:
    if len(rows) == 0:
        return []
    rows_t = torch.as_tensor(rows, dtype=torch.long, device=device)
    query = positive.index_select(0, rows_t)
    bank = he.index_select(0, rows_t)
    sims = query.float().matmul(bank.float().T)
    local_of = {int(global_row): local for local, global_row in enumerate(rows.tolist())}
    mask = torch.zeros((len(rows), len(rows)), dtype=torch.bool, device=device)
    mask.fill_diagonal_(True)
    for local, blocked in enumerate(excluded):
        columns = [local_of[item] for item in blocked if item in local_of]
        if columns:
            mask[local, columns] = True
    sims = sims.masked_fill(mask, float("-inf"))
    keep = min(SKIP_NEAREST + TOTAL_NEGATIVES, sims.shape[1])
    _vals, local = torch.topk(sims, k=keep, dim=1)
    local = local.masked_fill(~torch.isfinite(_vals), -1)
    return _ranked_global(local, rows)


def _other_ranked(
    positive: torch.Tensor,
    he: torch.Tensor,
    query_rows: np.ndarray,
    query_slide: torch.Tensor,
    bank_rows: np.ndarray,
    bank_slide: torch.Tensor,
    device: torch.device,
) -> list[list[int]]:
    if len(query_rows) == 0 or len(bank_rows) == 0:
        return [[] for _ in query_rows]
    keep = SKIP_NEAREST + TOTAL_NEGATIVES
    pieces = []
    bank = he.index_select(0, torch.as_tensor(bank_rows, dtype=torch.long, device=device))
    query_rows_t = torch.as_tensor(query_rows, dtype=torch.long, device=device)
    for start in range(0, len(query_rows), 1024):
        stop = min(len(query_rows), start + 1024)
        chosen = running_topk(
            positive.index_select(0, query_rows_t[start:stop]),
            bank,
            keep,
            query_group=query_slide[start:stop],
            bank_group=bank_slide,
            block=TOPK_BLOCK,
        )
        pieces.extend(_ranked_global(chosen, bank_rows))
    return pieces


def tissue_slide_job(payload: tuple) -> tuple[str, int]:
    """Tissue fraction of one slide's shifted positive crops. Writes the shared table."""
    slide, he_root, tissue_path, tiles, crops = payload
    images: dict[tuple[int, int], np.ndarray] = {}
    root = Path(he_root)
    for y, x, filename in tiles:
        images[(int(y), int(x))] = _crop_uint8(str(root / filename))
    tissue = np.load(tissue_path, mmap_mode="r+")
    device = torch.device("cpu")
    batch_rows: list[int] = []
    batch_crops: list[np.ndarray] = []

    def flush() -> None:
        if not batch_rows:
            return
        values = tissue_values(np.stack(batch_crops), device)
        tissue[np.asarray(batch_rows, dtype=np.int64)] = values
        batch_rows.clear()
        batch_crops.clear()

    for row, y, x, dy, dx in crops:
        batch_crops.append(stitch_crop(images, int(y), int(x), int(dy), int(dx)))
        batch_rows.append(int(row))
        if len(batch_rows) == TISSUE_BATCH:
            flush()
    flush()
    tissue.flush()
    del tissue
    return slide, len(crops)


def ensure_positive_tissue(
    fields: list[Field],
    tables: dict,
    he_root: Path,
    out_dir: Path,
    workers: int,
) -> np.ndarray:
    """H+E fraction of the chosen positive crop. Grid crops reuse the stored tile fraction."""
    path = out_dir / "positive_tissue.npy"
    n_items = len(fields)
    prow = np.asarray(tables["positive_row"])
    tile_tissue = np.asarray(tables["tissue_he"], dtype=np.float32)
    if path.is_file():
        current = np.load(path, mmap_mode="r")
        if current.shape != (n_items,):
            raise RuntimeError(f"{path} has shape {current.shape}, expected {(n_items,)}")
        if np.isfinite(current).all():
            print(f"positive tissue already stored n={n_items}", flush=True)
            return np.asarray(current, dtype=np.float32)
        del current
        tissue = np.load(path, mmap_mode="r+")
    else:
        tissue = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n_items,))
        tissue[:] = np.nan
    on_grid = prow >= 0
    on_rows = np.flatnonzero(on_grid)
    tissue[on_rows] = tile_tissue[prow[on_rows]]
    tissue.flush()
    sample = on_rows[:4]
    if not np.allclose(np.asarray(tissue[sample]), tile_tissue[prow[sample]]):
        raise RuntimeError("on-grid positive tissue was not stored")
    check = next(field for field in fields if int(prow[field.row]) == field.row)
    checked = float(tissue_values(_crop_uint8(str(he_root / check.filename))[None], torch.device("cpu"))[0])
    stored = float(tile_tissue[check.row])
    if abs(checked - stored) > 1e-4:
        raise RuntimeError(f"{check.filename} tissue {checked:.6f} does not match the table {stored:.6f}")
    pending = ~np.isfinite(np.asarray(tissue))
    by_slide: dict[str, list[Field]] = {}
    for field in fields:
        by_slide.setdefault(field.slide, []).append(field)
    dy = np.asarray(tables["positive_dy"])
    dx = np.asarray(tables["positive_dx"])
    jobs = []
    for slide, rows in by_slide.items():
        crops = []
        needed: set[tuple[int, int]] = set()
        names = {(field.y, field.x): field.filename for field in rows}
        for field in rows:
            if not pending[field.row]:
                continue
            shift_y = int(dy[field.row])
            shift_x = int(dx[field.row])
            crops.append((field.row, field.y, field.x, shift_y, shift_x))
            y0 = field.y + shift_y
            x0 = field.x + shift_x
            for ty in tile_origins_overlapping(y0, GRID, field.y, GRID):
                for tx in tile_origins_overlapping(x0, GRID, field.x, GRID):
                    needed.add((ty, tx))
        if not crops:
            continue
        tiles = [(y, x, names[(y, x)]) for y, x in sorted(needed)]
        jobs.append((slide, str(he_root), str(path), tiles, crops))
    del tissue
    print(f"positive tissue off-grid slides={len(jobs)} crops={int(pending.sum())}", flush=True)
    if jobs:
        if workers == 1:
            for index, job in enumerate(jobs, start=1):
                slide, count = tissue_slide_job(job)
                print(f"tissue {index}/{len(jobs)} {slide} crops={count}", flush=True)
        else:
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                futures = [pool.submit(tissue_slide_job, job) for job in jobs]
                finished = 0
                for future in as_completed(futures):
                    slide, count = future.result()
                    finished += 1
                    print(f"tissue {finished}/{len(jobs)} {slide} crops={count}", flush=True)
    stored_tissue = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
    if not np.isfinite(stored_tissue).all():
        missing = int((~np.isfinite(stored_tissue)).sum())
        raise RuntimeError(f"positive tissue left {missing} crops empty")
    return stored_tissue


def assign_negatives(
    fields: list[Field],
    tables: dict,
    device: torch.device,
    out_dir: Path,
    he_root: Path,
    workers: int,
) -> None:
    """256 negatives. Other slides of the same subtype and tissue bin come first.

    A short subtype pool continues with other slides of the same benign or
    malignant class and the same bin. The same slide fills only what is still
    missing, after the positive window and the 8 nearest patches are left out.
    """
    n_items = len(fields)
    positive_tissue = ensure_positive_tissue(fields, tables, he_root, out_dir, workers)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(f"loading embeddings for {n_items} negatives", flush=True)
    he = _normalize_memmap(tables["he"], device)
    positive = _normalize_memmap(tables["positive"], device)
    query_tissue = positive_tissue >= 0.15
    bank_tissue = np.asarray(tables["tissue_he"]) >= 0.15
    subtype = [group_name(field.classes1, field.subtype) for field in fields]
    classes = [field.classes1 for field in fields]
    slides: list[str] = []
    slide_code_of: dict[str, int] = {}
    for field in fields:
        if field.slide not in slide_code_of:
            slide_code_of[field.slide] = len(slides)
            slides.append(field.slide)
    slide_codes_np = np.array([slide_code_of[field.slide] for field in fields], dtype=np.int64)
    slide_codes = torch.from_numpy(slide_codes_np).to(device)
    chosen: list[list[int]] = [[] for _ in range(n_items)]
    for name in sorted(set(subtype)):
        members = np.array([row for row, value in enumerate(subtype) if value == name], dtype=np.int64)
        for is_tissue in (False, True):
            label = "tissue" if is_tissue else "background"
            query_rows = members[query_tissue[members] == is_tissue]
            bank_rows = members[bank_tissue[members] == is_tissue]
            print(
                f"negatives {name} {label}: queries={len(query_rows)} bank={len(bank_rows)}",
                flush=True,
            )
            if len(query_rows) == 0:
                continue
            ranked = _other_ranked(
                positive,
                he,
                query_rows,
                slide_codes[torch.from_numpy(query_rows).to(device)],
                bank_rows,
                slide_codes[torch.from_numpy(bank_rows).to(device)],
                device,
            )
            for row, ids in zip(query_rows.tolist(), ranked):
                extend_negatives(chosen[row], ids, SKIP_NEAREST, TOTAL_NEGATIVES)
    short = [row for row, ids in enumerate(chosen) if len(ids) < TOTAL_NEGATIVES]
    print(f"short after subtype {len(short)}", flush=True)
    class_groups: dict[tuple[str, bool], list[int]] = {}
    for row in short:
        class_groups.setdefault((classes[row], bool(query_tissue[row])), []).append(row)
    for (class_name, is_tissue), rows in class_groups.items():
        label = "tissue" if is_tissue else "background"
        query_rows = np.array(rows, dtype=np.int64)
        bank_rows = np.array(
            [
                row for row, value in enumerate(classes)
                if value == class_name and bool(bank_tissue[row]) == is_tissue
            ],
            dtype=np.int64,
        )
        print(
            f"negatives {class_name} class {label}: queries={len(query_rows)} bank={len(bank_rows)}",
            flush=True,
        )
        ranked = _other_ranked(
            positive,
            he,
            query_rows,
            slide_codes[torch.from_numpy(query_rows).to(device)],
            bank_rows,
            slide_codes[torch.from_numpy(bank_rows).to(device)],
            device,
        )
        for row, ids in zip(query_rows.tolist(), ranked):
            extend_negatives(chosen[row], ids, SKIP_NEAREST, TOTAL_NEGATIVES)
    short = [row for row in short if len(chosen[row]) < TOTAL_NEGATIVES]
    print(f"short after class {len(short)}", flush=True)
    if short:
        by_slide_rows: dict[str, list[Field]] = {}
        for field in fields:
            by_slide_rows.setdefault(field.slide, []).append(field)
        tile_sets = {
            slide: {(item.y, item.x) for item in rows}
            for slide, rows in by_slide_rows.items()
        }
        yx_to_row = {(field.slide, field.y, field.x): field.row for field in fields}
        short_by_slide: dict[str, list[int]] = {}
        for row in short:
            short_by_slide.setdefault(fields[row].slide, []).append(row)
        for slide, needed_rows in short_by_slide.items():
            slide_rows = np.array([field.row for field in by_slide_rows[slide]], dtype=np.int64)
            excluded: list[set[int]] = []
            for row in slide_rows.tolist():
                field = fields[row]
                blocked = {row}
                for shift_y, shift_x in candidate_offsets(field.y, field.x, tile_sets[slide]):
                    key = (slide, field.y + shift_y, field.x + shift_x)
                    if key in yx_to_row:
                        blocked.add(yx_to_row[key])
                for short_row in tables["shortlist_row"][row].tolist():
                    if int(short_row) >= 0:
                        blocked.add(int(short_row))
                if int(tables["positive_row"][row]) >= 0:
                    blocked.add(int(tables["positive_row"][row]))
                excluded.append(blocked)
            ranked = _same_slide_ranked(positive, he, slide_rows, excluded, device)
            local = {int(row): index for index, row in enumerate(slide_rows.tolist())}
            for row in needed_rows:
                extend_negatives(chosen[row], ranked[local[row]], SKIP_NEAREST, TOTAL_NEGATIVES)
    temporary = out_dir / "negatives.i32.npy.growing"
    if temporary.exists():
        temporary.unlink()
    negatives = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.int32, shape=(n_items, TOTAL_NEGATIVES),
    )
    padded_rows = 0
    empty_rows = 0
    for row, ids in enumerate(chosen):
        if not ids:
            empty_rows += 1
            negatives[row] = np.int32(-1)
            continue
        final = pad_negatives(ids)
        if len(set(final)) < TOTAL_NEGATIVES:
            padded_rows += 1
        negatives[row] = np.array(final, dtype=np.int32)
    negatives.flush()
    del negatives
    if empty_rows:
        temporary.unlink()
        raise RuntimeError(f"{empty_rows} rows have no negative candidates")
    written = np.load(temporary, mmap_mode="r")
    identity = np.arange(n_items, dtype=np.int32)[:, None]
    if np.any(written == identity):
        raise RuntimeError("a negative list contains its own row")
    other = slide_codes_np[written] != slide_codes_np[:, None]
    mismatch = other & (bank_tissue[written] != query_tissue[:, None])
    if np.any(mismatch):
        raise RuntimeError(f"{int(mismatch.sum())} other-slide negatives are in the wrong tissue bin")
    same_slide_rows = int(np.any(~other, axis=1).sum())
    del written
    os.replace(temporary, out_dir / "negatives.i32.npy")
    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["n"] = n_items
    meta["negatives_complete"] = True
    meta["negative_rule"] = "other_subtype_bin, other_class_bin, same_slide"
    meta["same_slide_negatives"] = "remainder"
    meta["negatives_padded_rows"] = padded_rows
    meta["negatives_empty_rows"] = empty_rows
    meta["negatives_same_slide_rows"] = same_slide_rows
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(
        f"negatives written, padded_rows={padded_rows} same_slide_rows={same_slide_rows}",
        flush=True,
    )


def write_positive_csv(path: Path, fields: list[Field], tables: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filename", "slide", "dy", "dx", "cosine", "tissue_gap", "positive_row",
                "classes1", "subtype_norm",
            ],
        )
        writer.writeheader()
        dy = np.asarray(tables["positive_dy"])
        dx = np.asarray(tables["positive_dx"])
        cosine = np.asarray(tables["positive_cosine"])
        gap = np.asarray(tables["positive_gap"])
        positive_row = np.asarray(tables["positive_row"])
        for field in fields:
            row = field.row
            writer.writerow(
                {
                    "filename": field.filename,
                    "slide": field.slide,
                    "dy": int(dy[row]),
                    "dx": int(dx[row]),
                    "cosine": float(cosine[row]),
                    "tissue_gap": float(gap[row]),
                    "positive_row": int(positive_row[row]),
                    "classes1": field.classes1,
                    "subtype_norm": field.subtype,
                }
            )


def fields_from_index(path: Path) -> list[Field]:
    fields = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            fields.append(
                Field(
                    row=int(row["row"]),
                    filename=row["filename"],
                    slide=row["slide"],
                    specimen=row["specimen"],
                    y=int(row["y"]),
                    x=int(row["x"]),
                    classes1=row["classes1"],
                    subtype=row["subtype_norm"],
                    pseudo="",
                    he="",
                    teacher="",
                )
            )
    if [field.row for field in fields] != list(range(len(fields))):
        raise RuntimeError(f"{path} rows are not 0..{len(fields) - 1}")
    return fields


def discover_added_fields(config: dict, labels_path: Path, slides: set[str], start_row: int) -> list[Field]:
    af_root = Path(config["af_root"])
    he_root = Path(config["he_root"])
    pseudo_root = Path(config["pseudo_root"])
    teacher_root = Path(config["teacher_dir"])
    by_slide, by_specimen = load_labels(labels_path)
    grouped: dict[str, list[Path]] = {}
    for path in af_root.iterdir():
        if not path.is_file() or path.suffix.lower() != ".png" or "__" not in path.name:
            continue
        slide = path.name.split("__", 1)[0]
        if slide not in slides:
            continue
        grouped.setdefault(slide, []).append(path)
    missing = sorted(slides - set(grouped))
    if missing:
        raise RuntimeError(f"{len(missing)} added slides have no AF patches, for example {missing[:3]}")
    fields: list[Field] = []
    absent: list[str] = []
    for slide in sorted(grouped):
        label = by_slide.get(slide) or by_specimen.get(slide.split("_", 1)[1])
        classes1, subtype = diagnosis(label, slide)
        for path in sorted(grouped[slide], key=lambda item: parse_yx(item.name)):
            he = he_root / path.name
            pseudo = pseudo_root / path.name
            teacher = teacher_root / path.name
            if not (he.is_file() and pseudo.is_file() and teacher.is_file()):
                if len(absent) < 5:
                    absent.append(path.name)
                continue
            y, x = parse_yx(path.name)
            fields.append(
                Field(
                    row=start_row + len(fields),
                    filename=path.name,
                    slide=slide,
                    specimen=slide.split("_", 1)[1],
                    y=y,
                    x=x,
                    classes1=classes1,
                    subtype=subtype,
                    pseudo=str(pseudo),
                    he=str(he),
                    teacher=str(teacher),
                )
            )
    if absent:
        raise RuntimeError(f"{len(absent)} added patches are missing he, teacher, or pseudo, for example {absent}")
    # Rows were assigned before filtering. Reassign contiguous rows per slide.
    cursor = start_row
    by_slide_fields: dict[str, list[Field]] = {}
    for field in fields:
        by_slide_fields.setdefault(field.slide, []).append(field)
    ordered: list[Field] = []
    for slide in sorted(by_slide_fields):
        block = by_slide_fields[slide]
        for offset, field in enumerate(block):
            ordered.append(
                Field(
                    row=cursor + offset,
                    filename=field.filename,
                    slide=field.slide,
                    specimen=field.specimen,
                    y=field.y,
                    x=field.x,
                    classes1=field.classes1,
                    subtype=field.subtype,
                    pseudo=field.pseudo,
                    he=field.he,
                    teacher=field.teacher,
                )
            )
        cursor += len(block)
    return ordered


def append_index(path: Path, fields: list[Field]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row", "filename", "slide", "specimen", "y", "x", "classes1", "subtype_norm",
            ],
        )
        for field in fields:
            writer.writerow(
                {
                    "row": field.row,
                    "filename": field.filename,
                    "slide": field.slide,
                    "specimen": field.specimen,
                    "y": field.y,
                    "x": field.x,
                    "classes1": field.classes1,
                    "subtype_norm": field.subtype,
                }
            )


def grow_array(path: Path, new_shape: tuple[int, ...]) -> None:
    old = np.load(path, mmap_mode="r")
    if old.shape == new_shape:
        return
    if old.shape[0] > new_shape[0] or old.shape[1:] != new_shape[1:]:
        raise RuntimeError(f"{path} is {old.shape}, cannot grow to {new_shape}")
    temporary = path.with_name(path.name + ".growing")
    grown = np.lib.format.open_memmap(temporary, mode="w+", dtype=old.dtype, shape=new_shape)
    grown[: old.shape[0]] = np.asarray(old)
    if new_shape[0] > old.shape[0]:
        grown[old.shape[0]:] = 0
    grown.flush()
    del grown
    del old
    os.replace(temporary, path)


def open_tables(out_dir: Path, n_items: int) -> dict[str, np.memmap]:
    spec = {
        "he": ((n_items, DIM), np.float16),
        "teacher": ((n_items, DIM), np.float16),
        "positive": ((n_items, DIM), np.float16),
        "tissue_he": ((n_items,), np.float32),
        "tissue_teacher": ((n_items,), np.float32),
        "tissue_pseudo": ((n_items,), np.float32),
        "positive_row": ((n_items,), np.int32),
        "positive_dy": ((n_items,), np.int16),
        "positive_dx": ((n_items,), np.int16),
        "positive_cosine": ((n_items,), np.float32),
        "positive_gap": ((n_items,), np.float32),
        "shortlist_row": ((n_items, SHORTLIST), np.int32),
        "shortlist_dy": ((n_items, SHORTLIST), np.int16),
        "shortlist_dx": ((n_items, SHORTLIST), np.int16),
        "shortlist_cosine": ((n_items, SHORTLIST), np.float32),
    }
    tables = {}
    for name, (shape, dtype) in spec.items():
        path = out_dir / f"{name}.npy"
        tables[name] = open_array(path, shape, dtype, not path.exists())
    return tables


def _submit(in_queue, fields: list[Field], names: dict[str, str]) -> None:
    paths = {
        "he": [field.he for field in fields],
        "teacher": [field.teacher for field in fields],
        "pseudo": [field.pseudo for field in fields],
    }
    in_queue.put((fields[0].slide, len(fields), names, paths))


def run_embed(
    fields: list[Field],
    tables: dict,
    out_dir: Path,
    device: torch.device,
    workers: int,
    batch_size: int,
    shard: int = 0,
    shards: int = 1,
) -> None:
    done = read_done(out_dir / "done_slides.txt")
    grouped: list[tuple[str, list[Field]]] = []
    current: list[Field] = []
    for field in fields:
        if current and field.slide != current[0].slide:
            grouped.append((current[0].slide, current))
            current = []
        current.append(field)
    if current:
        grouped.append((current[0].slide, current))
    pending = [(slide, rows) for slide, rows in grouped if slide not in done]
    if shards > 1:
        pending = [item for item in pending if zlib.crc32(item[0].encode()) % shards == shard]
    print(
        f"embed {len(pending)} slides remaining on shard {shard}/{shards}, {len(done)} already done",
        flush=True,
    )
    if not pending:
        return
    model, transform = load_encoder(device)
    print(
        f"UNI weights {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GiB",
        flush=True,
    )
    ctx = mp.get_context("spawn")
    in_queue = ctx.Queue()
    out_queue = ctx.Queue()
    procs = [ctx.Process(target=_reader, args=(in_queue, out_queue), daemon=True) for _ in range(workers)]
    for proc in procs:
        proc.start()
    live: dict[str, dict] = {}
    cursor = 0
    try:
        while cursor < len(pending) or live:
            while cursor < len(pending) and len(live) < workers:
                slide, rows = pending[cursor]
                cursor += 1
                names = {}
                blocks = {}
                nbytes = len(rows) * GRID * GRID * 3
                for kind in ("he", "teacher", "pseudo"):
                    shm = shared_memory.SharedMemory(create=True, size=nbytes)
                    names[kind] = shm.name
                    blocks[kind] = shm
                live[slide] = {"rows": rows, "names": names, "blocks": blocks}
                _submit(in_queue, rows, names)
            slide, error = out_queue.get()
            record = live.pop(slide)
            attached: dict[str, tuple[shared_memory.SharedMemory, np.ndarray]] = {}
            try:
                if error:
                    raise RuntimeError(f"{slide} failed to read\n{error}")
                attached = {
                    kind: _attach(record["names"][kind], len(record["rows"]))
                    for kind in record["names"]
                }
                images = {kind: array for kind, (_shm, array) in attached.items()}
                started = time.time()
                process_slide(record["rows"], images, model, transform, tables, device, batch_size)
                mark_done(out_dir / "done_slides.txt", slide)
                print(
                    f"slide {slide} patches={len(record['rows'])} "
                    f"seconds={time.time() - started:.1f} left={len(pending) - cursor + len(live)}",
                    flush=True,
                )
            finally:
                images = None
                handles = [shm for shm, _array in attached.values()]
                attached.clear()
                for shm in handles:
                    shm.close()
                for shm in record["blocks"].values():
                    try:
                        shm.close()
                        shm.unlink()
                    except FileNotFoundError:
                        pass
    finally:
        for proc in procs:
            in_queue.put(None)
        for proc in procs:
            proc.join(timeout=5)
        for record in live.values():
            for shm in record["blocks"].values():
                try:
                    shm.close()
                    shm.unlink()
                except FileNotFoundError:
                    pass


def _window_tiles(y: int, x: int, tiles: set[tuple[int, int]]) -> set[tuple[int, int]]:
    needed: set[tuple[int, int]] = set()
    for dy, dx in candidate_offsets(y, x, tiles):
        for ty in tile_origins_overlapping(y + dy, GRID, y, GRID):
            for tx in tile_origins_overlapping(x + dx, GRID, x, GRID):
                needed.add((ty, tx))
    return needed


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def select_smoke(config: dict, labels_path: Path) -> tuple[list[Field], dict[str, str]]:
    """One edge field, one interior field, and every tile their windows touch."""
    excluded = exclusion_keys(config.get("exclude_slides"))
    by_slide, by_specimen = load_labels(labels_path)
    he_root = Path(config["he_root"])
    pseudo_root = Path(config["pseudo_root"])
    teacher_root = Path(config["teacher_dir"])
    with Path(config["alignment"]).open(newline="", encoding="utf-8") as handle:
        specimens = [
            row["specimen_key"]
            for row in csv.DictReader(handle)
            if row["use"] == "crop_absolute"
            and row["specimen_key"] not in excluded
            and int(row["n_patches"] or 0) >= 40
        ]
    for specimen in specimens:
        paths = sorted(he_root.glob(f"*_{specimen}__*.png"))
        if len(paths) < 9:
            continue
        slide = paths[0].name.split("__", 1)[0]
        paths = [path for path in paths if path.name.startswith(slide + "__")]
        coords = [parse_yx(path.name) for path in paths]
        tiles = set(coords)
        ys = sorted({y for y, _x in tiles})
        xs = sorted(x for y, x in tiles if y == ys[0])
        corner = (ys[0], xs[0])
        interior = None
        for y, x in sorted(tiles):
            if all((y + dy, x + dx) in tiles for dy in (-GRID, 0, GRID) for dx in (-GRID, 0, GRID)):
                interior = (y, x)
                break
        if interior is None or interior == corner:
            continue
        needed: set[tuple[int, int]] = set()
        queries = {"edge": corner, "interior": interior}
        for origin in queries.values():
            needed |= _window_tiles(origin[0], origin[1], tiles)
        label = by_slide.get(slide) or by_specimen.get(specimen)
        classes1, subtype = diagnosis(label, slide)
        fields: list[Field] = []
        missing = False
        for y, x in sorted(needed):
            name = f"{slide}__y{y}_x{x}.png"
            pseudo = pseudo_root / name
            teacher = teacher_root / name
            he = he_root / name
            if not (pseudo.is_file() and teacher.is_file() and he.is_file()):
                missing = True
                break
            fields.append(
                Field(
                    row=len(fields),
                    filename=name,
                    slide=slide,
                    specimen=specimen,
                    y=y,
                    x=x,
                    classes1=classes1,
                    subtype=subtype,
                    pseudo=str(pseudo),
                    he=str(he),
                    teacher=str(teacher),
                )
            )
        if missing or not fields:
            continue
        names = {
            role: f"{slide}__y{origin[0]}_x{origin[1]}.png"
            for role, origin in queries.items()
        }
        return fields, names
    raise RuntimeError("no slide had both an edge field and a full interior window")


def run_smoke(config: dict, labels_path: Path, out_dir: Path, device: torch.device, workers: int, batch_size: int) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    fields, queries = select_smoke(config, labels_path)
    print(
        f"smoke slide {fields[0].slide} {fields[0].classes1} {fields[0].subtype or 'benign'} "
        f"tiles={len(fields)} queries={queries}",
        flush=True,
    )
    write_index(out_dir / "index.csv", fields)
    (out_dir / "meta.json").write_text(
        json.dumps({"n": len(fields), "dim": DIM, "model": ENCODER, "negatives_complete": False}) + "\n",
        encoding="utf-8",
    )
    tables = open_tables(out_dir, len(fields))
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    run_embed(fields, tables, out_dir, device, workers, batch_size)
    by_name = {field.filename: field for field in fields}
    print(
        f"peak allocated {torch.cuda.max_memory_allocated(device) / 1024 ** 3:.2f} GiB, "
        f"peak reserved {torch.cuda.max_memory_reserved(device) / 1024 ** 3:.2f} GiB",
        flush=True,
    )
    he = np.asarray(tables["he"])
    teacher = np.asarray(tables["teacher"])
    if he.shape != (len(fields), DIM) or not np.isfinite(he.astype(np.float32)).all():
        raise RuntimeError(f"H&E embeddings are {he.shape}, finite={np.isfinite(he.astype(np.float32)).all()}")
    if teacher.shape != he.shape or not np.isfinite(teacher.astype(np.float32)).all():
        raise RuntimeError("teacher embeddings are missing or non-finite")
    for role, name in queries.items():
        field = by_name[name]
        row = field.row
        paired = _cosine(teacher[row], he[row])
        winner_row = int(tables["positive_row"][row])
        if winner_row >= 0:
            recomputed = _cosine(teacher[row], he[winner_row])
        else:
            recomputed = _cosine(teacher[row], tables["positive"][row])
        stored = float(tables["positive_cosine"][row])
        if abs(stored - recomputed) > 0.02:
            raise RuntimeError(f"{name} stored cosine {stored:.4f} != recomputed {recomputed:.4f}")
        print(
            f"{role} {name} y={field.y} x={field.x} "
            f"tissue teacher={float(tables['tissue_teacher'][row]):.3f} "
            f"pseudo={float(tables['tissue_pseudo'][row]):.3f} "
            f"paired_he={float(tables['tissue_he'][row]):.3f}",
            flush=True,
        )
        print(
            f"  paired cosine {paired:.4f}  winner dy={int(tables['positive_dy'][row])} "
            f"dx={int(tables['positive_dx'][row])} gap={float(tables['positive_gap'][row]):.3f} "
            f"cosine={stored:.4f}",
            flush=True,
        )
        for slot in range(SHORTLIST):
            cosine = float(tables["shortlist_cosine"][row, slot])
            if not np.isfinite(cosine):
                continue
            print(
                f"  shortlist dy={int(tables['shortlist_dy'][row, slot])} "
                f"dx={int(tables['shortlist_dx'][row, slot])} cosine={cosine:.4f}",
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the UNI positive and negative table.")
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--labels", default=str(LABELS))
    parser.add_argument("--out", default=str(ROOT / "retrieval"))
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--embed-batch", type=int, default=16)
    parser.add_argument("--phase", choices=("all", "embed", "negatives"), default="all")
    parser.add_argument("--slides-file", default=None, help="Append these slide ids. Existing rows stay.")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError(f"shard {args.shard} is outside 0..{args.shards - 1}")
    out_dir = Path(args.out)
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.smoke:
        if out_dir.resolve() == (ROOT / "retrieval").resolve():
            out_dir = ROOT / "retrieval_smoke"
        run_smoke(config, Path(args.labels), out_dir, torch.device(f"cuda:{args.gpu}"), args.workers, args.embed_batch)
        return
    if args.phase == "negatives":
        index_path = out_dir / "index.csv"
        if not index_path.is_file():
            raise FileNotFoundError(index_path)
        fields = fields_from_index(index_path)
        meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
        if int(meta["n"]) != len(fields):
            raise RuntimeError(f"meta n={meta['n']} does not match index {len(fields)}")
        done = read_done(out_dir / "done_slides.txt")
        slide_count = len({field.slide for field in fields})
        if len(done) != slide_count or {field.slide for field in fields} - done:
            raise RuntimeError(f"embeddings are unfinished: {len(done)}/{slide_count} slides")
        negative_path = out_dir / "negatives.i32.npy"
        if (
            meta.get("negatives_complete")
            and meta.get("negative_rule") == "other_subtype_bin, other_class_bin, same_slide"
            and negative_path.is_file()
            and tuple(np.load(negative_path, mmap_mode="r").shape) == (len(fields), TOTAL_NEGATIVES)
        ):
            print("negatives already complete", flush=True)
            return
        tables = open_tables(out_dir, len(fields))
        assign_negatives(
            fields,
            tables,
            torch.device(f"cuda:{args.gpu}"),
            out_dir,
            Path(config["he_root"]),
            args.workers,
        )
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print("indexing training patches", flush=True)
    index_path = out_dir / "index.csv"
    if args.slides_file:
        if not index_path.is_file():
            raise FileNotFoundError(f"cannot append without {index_path}")
        slides = {
            line.split("#", 1)[0].strip()
            for line in Path(args.slides_file).read_text(encoding="utf-8").splitlines()
            if line.split("#", 1)[0].strip()
        }
        existing_fields = fields_from_index(index_path)
        known = {field.filename for field in existing_fields}
        added = [
            field for field in discover_added_fields(config, Path(args.labels), slides, len(existing_fields))
            if field.filename not in known
        ]
        # Reassign rows against the index length, not against files already recorded.
        cursor = len(existing_fields)
        by_slide_added: dict[str, list[Field]] = {}
        for field in added:
            by_slide_added.setdefault(field.slide, []).append(field)
        added = []
        for slide in sorted(by_slide_added):
            for offset, field in enumerate(by_slide_added[slide]):
                added.append(Field(
                    row=cursor + offset,
                    filename=field.filename,
                    slide=field.slide,
                    specimen=field.specimen,
                    y=field.y,
                    x=field.x,
                    classes1=field.classes1,
                    subtype=field.subtype,
                    pseudo=field.pseudo,
                    he=field.he,
                    teacher=field.teacher,
                ))
            cursor += len(by_slide_added[slide])
        target = len(existing_fields) + len(added)
        spec = {
            "he": ((target, DIM), np.float16),
            "teacher": ((target, DIM), np.float16),
            "positive": ((target, DIM), np.float16),
            "tissue_he": ((target,), np.float32),
            "tissue_teacher": ((target,), np.float32),
            "tissue_pseudo": ((target,), np.float32),
            "positive_row": ((target,), np.int32),
            "positive_dy": ((target,), np.int16),
            "positive_dx": ((target,), np.int16),
            "positive_cosine": ((target,), np.float32),
            "positive_gap": ((target,), np.float32),
            "shortlist_row": ((target, SHORTLIST), np.int32),
            "shortlist_dy": ((target, SHORTLIST), np.int16),
            "shortlist_dx": ((target, SHORTLIST), np.int16),
            "shortlist_cosine": ((target, SHORTLIST), np.float32),
        }
        for name, (shape, _dtype) in spec.items():
            grow_array(out_dir / f"{name}.npy", shape)
        if added:
            append_index(index_path, added)
            print(f"appended {len(added)} rows, total {target}", flush=True)
        fields = fields_from_index(index_path)
        paths = {
            field.filename: field
            for field in discover_added_fields(config, Path(args.labels), slides, 0)
        }
        fields = [
            paths[field.filename] if field.filename in paths else field
            for field in fields
        ]
        # discover_added_fields row numbers are not the index rows. Restore index rows.
        indexed = {field.filename: field.row for field in fields_from_index(index_path)}
        fields = [
            Field(
                row=indexed[field.filename],
                filename=field.filename,
                slide=field.slide,
                specimen=field.specimen,
                y=field.y,
                x=field.x,
                classes1=field.classes1,
                subtype=field.subtype,
                pseudo=field.pseudo,
                he=field.he,
                teacher=field.teacher,
            )
            for field in fields
        ]
    else:
        fields = build_fields(config, Path(args.labels))
        if index_path.exists():
            existing = read_index(index_path)
            current = [(field.filename, field.slide, field.classes1, field.subtype) for field in fields]
            if existing != current:
                raise RuntimeError(f"{index_path} does not match the current training set")
        else:
            write_index(index_path, fields)
    meta_path = out_dir / "meta.json"
    meta = {
        "n": len(fields),
        "dim": DIM,
        "model": ENCODER,
        "grid": GRID,
        "shortlist": SHORTLIST,
        "skip_nearest": SKIP_NEAREST,
        "same_slide_negatives": 128,
        "total_negatives": TOTAL_NEGATIVES,
        "tissue_threshold": 0.15,
        "negatives_complete": False,
    }
    if meta_path.exists():
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        if int(previous["dim"]) != DIM:
            raise RuntimeError(f"{meta_path} records dim={previous['dim']}")
        if args.slides_file:
            if int(previous["n"]) > len(fields):
                raise RuntimeError(f"{meta_path} records n={previous['n']}, larger than {len(fields)}")
            meta["negatives_complete"] = False
        else:
            if int(previous["n"]) != len(fields):
                raise RuntimeError(f"{meta_path} records n={previous['n']} dim={previous['dim']}")
            meta["negatives_complete"] = bool(previous.get("negatives_complete", False))
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    tables = open_tables(out_dir, len(fields))
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    if args.phase in {"all", "embed"}:
        run_embed(fields, tables, out_dir, device, args.workers, args.embed_batch, args.shard, args.shards)
        done = read_done(out_dir / "done_slides.txt")
        slide_count = len({field.slide for field in fields})
        if len(done) == slide_count:
            lock = out_dir / ".positive_csv.lock"
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                print("positive csv already being written", flush=True)
            else:
                os.close(fd)
                write_positive_csv(out_dir / "positive.csv", fields, tables)
                print(f"wrote {out_dir / 'positive.csv'}", flush=True)
    if args.phase in {"all", "negatives"}:
        done = read_done(out_dir / "done_slides.txt")
        slide_count = len({field.slide for field in fields})
        if len(done) != slide_count:
            raise RuntimeError(f"embeddings are unfinished: {len(done)}/{slide_count} slides")
        if meta.get("negatives_complete") and args.phase == "all":
            print("negatives already complete", flush=True)
            return
        assign_negatives(fields, tables, device, out_dir, Path(config["he_root"]), args.workers)


if __name__ == "__main__":
    main()
