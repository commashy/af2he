"""Read a region of a 16-bit AF TIFF and scale it to RGB.

This is the conversion used for the epoch-9 slide inference. A uint16 page
is divided by 65535. There is no percentile stretch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def infer_axes(shape: tuple[int, ...]) -> str:
    if len(shape) == 2:
        return "YX"
    if len(shape) == 3:
        if shape[-1] in (1, 2, 3, 4):
            return "YXS"
        if shape[0] in (1, 2, 3, 4):
            return "SYX"
    return "".join(chr(ord("A") + i) for i in range(len(shape)))


def yx_from_shape(shape: tuple[int, ...], axes: str) -> tuple[int, int]:
    if "Y" in axes and "X" in axes:
        return int(shape[axes.index("Y")]), int(shape[axes.index("X")])
    if len(shape) >= 2:
        return int(shape[-2]), int(shape[-1])
    raise ValueError(f"Cannot infer YX from shape={shape}, axes={axes}")


def to_hwc(arr: np.ndarray, axes: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    axes = axes or infer_axes(tuple(arr.shape))
    if "Y" in axes and "X" in axes:
        y_axis = axes.index("Y")
        x_axis = axes.index("X")
        channel_axes = [i for i, (dim, ax) in enumerate(zip(arr.shape, axes)) if i not in (y_axis, x_axis) and (ax in {"S", "C"} or dim <= 4)]
        if channel_axes:
            c_axis = channel_axes[0]
            return np.moveaxis(arr, (y_axis, x_axis, c_axis), (0, 1, 2))
        return np.moveaxis(arr, (y_axis, x_axis), (0, 1))
    if arr.ndim == 3 and arr.shape[-1] in (1, 2, 3, 4):
        return arr
    if arr.ndim == 3 and arr.shape[0] in (1, 2, 3, 4):
        return np.moveaxis(arr, 0, -1)
    raise ValueError(f"Cannot convert array with shape={arr.shape}, axes={axes} to HWC")


def scale_integer_to_uint8(arr: np.ndarray, fixed_uint16_full_range: bool = True) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.integer):
        if fixed_uint16_full_range and arr.dtype == np.uint16:
            denom = 65535.0
            offset = 0.0
        else:
            info = np.iinfo(arr.dtype)
            denom = float(info.max - info.min) if info.max > info.min else 1.0
            offset = float(info.min)
        return ((arr.astype(np.float32) - offset) / denom * 255.0).clip(0, 255).astype(np.uint8)
    return np.nan_to_num(arr.astype(np.float32)).clip(0, 255).astype(np.uint8)


def af_to_rgb_uint8(arr: np.ndarray, axes: str = "") -> np.ndarray:
    arr = to_hwc(arr, axes)
    arr = scale_integer_to_uint8(arr)
    if arr.ndim == 2:
        return np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[2] == 1:
        return np.repeat(arr, 3, axis=2)
    if arr.shape[2] >= 3:
        return arr[:, :, :3]
    return np.repeat(arr[:, :, :1], 3, axis=2)


def first_zarr_array(obj, expected_shape=None):
    """Return the first array from a tifffile/zarr object or group."""
    expected_shape = tuple(int(v) for v in expected_shape) if expected_shape is not None else None
    if hasattr(obj, "shape") and hasattr(obj, "__getitem__"):
        if expected_shape is None or tuple(int(v) for v in obj.shape) == expected_shape:
            return obj
        raise TypeError(f"Zarr array shape {tuple(obj.shape)} does not match expected {expected_shape}")
    fallback = None
    candidate_keys = ["0", "/0"]
    if hasattr(obj, "array_keys"):
        try:
            candidate_keys = list(obj.array_keys()) + candidate_keys
        except Exception:
            pass
    if hasattr(obj, "keys"):
        try:
            candidate_keys = list(obj.keys()) + candidate_keys
        except Exception:
            pass
    seen = set()
    for key in candidate_keys:
        if key in seen:
            continue
        seen.add(key)
        try:
            child = obj[key]
        except Exception:
            continue
        if hasattr(child, "shape") and hasattr(child, "__getitem__"):
            if expected_shape is None or tuple(int(v) for v in child.shape) == expected_shape:
                return child
            if fallback is None:
                fallback = child
            continue
        try:
            return first_zarr_array(child, expected_shape=expected_shape)
        except Exception:
            continue
    if fallback is not None and expected_shape is None:
        return fallback
    raise TypeError(f"Could not find a Zarr array in object of type {type(obj)}")


class TiffRegionReader:
    """Region reader backed by tifffile. Crops stay off the full-page copy."""

    def __init__(self, path: str | Path, series: int = 0):
        self.path = Path(path)
        self.series_index = int(series)
        self.tif = None
        self.store = None
        self.array = None
        self.memmap = None
        self.axes = ""
        self.shape: tuple[int, ...] = ()
        self.dtype = None
        self.height = 0
        self.width = 0

    def __enter__(self) -> "TiffRegionReader":
        import tifffile
        import zarr

        self.tif = tifffile.TiffFile(self.path)
        series = self.tif.series[self.series_index]
        self.axes = series.axes or infer_axes(tuple(series.shape))
        self.shape = tuple(int(v) for v in series.shape)
        self.dtype = series.dtype
        self.height, self.width = yx_from_shape(self.shape, self.axes)
        try:
            self.memmap = tifffile.memmap(self.path, series=self.series_index, mode="r")
            self.array = self.memmap
        except Exception:
            self.store = series.aszarr()
            self.array = first_zarr_array(zarr.open(self.store, mode="r"), expected_shape=self.shape)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.store is not None and hasattr(self.store, "close"):
            self.store.close()
        if self.tif is not None:
            self.tif.close()

    def read(self, box: tuple[int, int, int, int]) -> np.ndarray:
        if self.array is None:
            raise RuntimeError("Reader is not open")
        x1, y1, x2, y2 = [int(v) for v in box]
        slices: list[object] = []
        for axis, dim in zip(self.axes, self.shape):
            if axis == "Y":
                slices.append(slice(y1, y2))
            elif axis == "X":
                slices.append(slice(x1, x2))
            elif axis in {"S", "C"} or dim <= 4:
                slices.append(slice(None))
            else:
                slices.append(0)
        arr = np.asarray(self.array[tuple(slices)])
        return to_hwc(arr, self.axes)
