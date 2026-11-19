"""Stored UNI keys for the field contrastive loss.

The positive and the 256 negatives are rows in the precomputed table.
No H&E image is read here.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch


class RetrievalTable:
    def __init__(self, directory: str | Path):
        directory = Path(directory)
        index_path = directory / "index.csv"
        if not index_path.is_file():
            raise FileNotFoundError(f"retrieval index is missing: {index_path}")
        self.row_of: dict[str, int] = {}
        with index_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                self.row_of[row["filename"]] = int(row["row"])
        if not self.row_of:
            raise RuntimeError(f"{index_path} has no rows")
        self.positive = np.load(directory / "positive.npy", mmap_mode="r")
        self.he = np.load(directory / "he.npy", mmap_mode="r")
        self.negative = np.load(directory / "negatives.i32.npy", mmap_mode="r")
        cosine_path = directory / "positive_cosine.npy"
        if not cosine_path.is_file():
            raise FileNotFoundError(f"retrieval cosine is missing: {cosine_path}")
        self.pair_cosine_values = np.load(cosine_path, mmap_mode="r")
        if (
            self.positive.shape[0] != len(self.row_of)
            or self.negative.shape[0] != len(self.row_of)
            or self.pair_cosine_values.shape[0] != len(self.row_of)
        ):
            raise RuntimeError(
                f"retrieval table length {self.positive.shape[0]} does not match index {len(self.row_of)}"
            )

    def keys(self, names: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            rows = [self.row_of[name] for name in names]
        except KeyError as exc:
            raise KeyError(f"retrieval table has no embedding for {exc}") from exc
        row_index = np.asarray(rows, dtype=np.int64)
        positive = torch.tensor(np.array(self.positive[row_index], copy=True), device=device)
        negative_rows = np.array(self.negative[row_index], copy=True)
        negative = torch.tensor(np.array(self.he[negative_rows], copy=True), device=device)
        return positive, negative

    def pair_cosine(self, names: list[str], device: torch.device) -> torch.Tensor:
        """Teacher-to-retrieved-H&E cosine stored when the table was built."""
        try:
            rows = [self.row_of[name] for name in names]
        except KeyError as exc:
            raise KeyError(f"retrieval table has no embedding for {exc}") from exc
        row_index = np.asarray(rows, dtype=np.int64)
        return torch.tensor(np.array(self.pair_cosine_values[row_index], copy=True), device=device)
