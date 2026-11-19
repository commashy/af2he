"""Copy the classifier tiles from the hard disk onto the NVMe.

Training was reading these 1024 PNGs in random order, and the disk fell to
about 50 MB/s. The copy walks each slide in grid order so the read is mostly
sequential. Files already present with the same size are skipped.
"""

from __future__ import annotations

import csv
import shutil
import time
from pathlib import Path

SRC = Path("/mnt/10tb_hdd/taek/images/Fresh_section/patches_1024/trainB")
DST = Path("/home/taek/af2he_heat_tiles")
CACHE = Path("/mnt/10tb_hdd/taek/af2he/manifests/heat_classifier_rows.csv")


def order(name: str) -> tuple:
    slide, rest = name.split("__y", 1)
    y, x = rest[: -len(".png")].split("_x")
    return slide, int(y), int(x)


def main() -> None:
    with CACHE.open(newline="", encoding="utf-8") as handle:
        names = sorted({row["name"] for row in csv.DictReader(handle)}, key=order)
    DST.mkdir(parents=True, exist_ok=True)
    started = time.time()
    copied = 0
    nbytes = 0
    for index, name in enumerate(names, start=1):
        source = SRC / name
        target = DST / name
        size = source.stat().st_size
        if not target.is_file() or target.stat().st_size != size:
            shutil.copyfile(source, target)
            copied += 1
        nbytes += size
        if index % 2000 == 0 or index == len(names):
            elapsed = time.time() - started
            print(
                f"{index}/{len(names)}  {nbytes / 1e9:.1f} GB  "
                f"{nbytes / 1e6 / elapsed:.0f} MB/s  new {copied}",
                flush=True,
            )
    print(f"done  {copied} new files  {nbytes / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
