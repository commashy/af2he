"""Slide identity and patch coordinates for collage building."""

from __future__ import annotations

import re
from pathlib import Path

_FRESH_XY = re.compile(r"__y(?P<y>\d+)_x(?P<x>\d+)$")


def slide_and_coord(path: Path) -> tuple[str, tuple[int, int]]:
    """Return (slide id, (row, col)).

    Fresh and thin-lung patches use ``slide__y*_x*``.
    MIST H&E uses ``slide_row_col``.
    IHC4BC H&E uses ``case_slide_*_row_col_*``; the slide is the first two fields.
    """
    stem = path.stem
    fresh = _FRESH_XY.search(stem)
    if fresh is not None:
        slide = stem[: fresh.start()]
        return slide, (int(fresh.group("y")), int(fresh.group("x")))
    parts = stem.split("_")
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        return parts[0], (int(parts[1]), int(parts[2]))
    if len(parts) >= 5 and parts[-3].isdigit() and parts[-2].isdigit():
        return f"{parts[0]}_{parts[1]}", (int(parts[-3]), int(parts[-2]))
    raise ValueError(f"no slide coordinate in {path.name}")


def group_key(root: Path, slide: str) -> str:
    return f"{root.resolve()}|{slide}"
