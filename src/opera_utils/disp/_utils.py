from __future__ import annotations

import datetime
import itertools
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from opera_utils._dates import DATETIME_FORMAT, get_dates
from opera_utils.burst_frame_db import get_frame_bbox

PathOrStrT = TypeVar("PathOrStrT", Path, str)


def get_frame_coordinates(frame_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Get the UTM x, y coordinates for a frame.

    Parameters
    ----------
    frame_id : int
        The frame ID to get the coordinates for.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (x, y) arrays of UTM coordinates (in meters).

    """
    _epsg, bbox = get_frame_bbox(frame_id)
    # 30 meter spacing, coords are on the pixel centers.
    x = np.arange(bbox[0], bbox[2], 30)
    # y-coordinates are in decreasing order (north first)
    y = np.arange(bbox[3], bbox[1], -30)
    # Now shift to pixel centers:
    return x + 15, y - 15


def flatten(list_of_lists: Iterable[Iterable[Any]]) -> itertools.chain[Any]:
    """Flatten one level of a nested iterable."""
    return itertools.chain.from_iterable(list_of_lists)


def last_per_ministack(opera_file_list: Sequence[Path | str]) -> list[Path | str]:
    """Get the last file per ministack.

    Parameters
    ----------
    opera_file_list : Sequence[Path | str]
        List of opera files.

    Returns
    -------
    list[Path | str]
        List of last files per ministack.

    """

    def _get_generation_time(fname: Path | str) -> datetime.datetime:
        return get_dates(fname, fmt=DATETIME_FORMAT)[2]

    last_per_ministack = []
    for _d, cur_groupby in itertools.groupby(
        sorted(opera_file_list), key=_get_generation_time
    ):
        # cur_groupby is an iterable of all matching
        # Get the first one, and the last one. ignore the rest
        last_file = list(cur_groupby)[-1]
        last_per_ministack.append(last_file)
    return last_per_ministack


def round_mantissa(z: np.ndarray, keep_bits=10) -> np.ndarray:
    """Zero out mantissa bits of elements of array in place.

    Drops a specified number of bits from the floating point mantissa,
    leaving an array more amenable to compression.

    Parameters
    ----------
    z : numpy.ndarray
        Real or complex array whose mantissas are to be zeroed out
    keep_bits : int, optional
        Number of bits to preserve in mantissa. Defaults to 10.
        Lower numbers will truncate the mantissa more and enable
        more compression.

    Returns
    -------
    np.ndarray
        View of input `z` array with rounded mantissa.

    References
    ----------
    https://numcodecs.readthedocs.io/en/v0.12.1/_modules/numcodecs/bitround.html

    """
    max_bits = {
        "float16": 10,
        "float32": 23,
        "float64": 52,
    }
    # recurse for complex data
    if np.iscomplexobj(z):
        round_mantissa(z.real, keep_bits)
        round_mantissa(z.imag, keep_bits)
        return z

    if z.dtype.kind != "f" or z.dtype.itemsize > 8:
        msg = "Only float arrays (16-64bit) can be bit-rounded"
        raise TypeError(msg)

    bits = max_bits[str(z.dtype)]
    # cast float to int type of same width (preserve endianness)
    a_int_dtype = np.dtype(z.dtype.str.replace("f", "i"))
    all_set = np.array(-1, dtype=a_int_dtype)
    if keep_bits == bits:
        return z
    if keep_bits > bits:
        msg = "keep_bits too large for given dtype"
        raise ValueError(msg)
    b = z.view(a_int_dtype)
    maskbits = bits - keep_bits
    mask = (all_set >> maskbits) << maskbits
    half_quantum1 = (1 << (maskbits - 1)) - 1
    b += ((b >> maskbits) & 1) + half_quantum1
    b &= mask
    return b.view(z.dtype)


def _get_border(data_arrays: NDArray[np.floating]) -> NDArray[np.floating]:
    top_row = data_arrays[:, 0, :]
    bottom_row = data_arrays[:, -1, :]
    left_col = data_arrays[:, :, 0]
    right_col = data_arrays[:, :, -1]
    all_pixels = np.hstack([top_row, bottom_row, left_col, right_col])
    return np.nanmedian(all_pixels, axis=1)[:, np.newaxis, np.newaxis]


def _clamp_chunk_dict(
    requested_chunks: Mapping[str, int] | None, data_shape: tuple[int, int, int]
) -> dict[str, int]:
    """Ensure requested_chunks are smaller than the downloaded size."""
    chunks = {**(requested_chunks or {})}
    chunks["time"] = min(chunks.get("time", data_shape[0]), data_shape[0])
    chunks["y"] = min(chunks.get("y", data_shape[1]), data_shape[1])
    chunks["x"] = min(chunks.get("x", data_shape[2]), data_shape[2])
    return chunks


def _get_netcdf_encoding(
    ds: xr.Dataset,
    chunks: tuple[int, int, int],
    compression_level: int = 6,
    data_vars: Sequence[str] = [],
) -> dict:
    encoding = {}
    comp = {"zlib": True, "complevel": compression_level, "chunksizes": chunks}
    if not data_vars:
        data_vars = list(ds.data_vars)
    encoding = {var: comp for var in data_vars if ds[var].ndim >= 2}
    for var in data_vars:
        if ds[var].ndim < 2:
            continue
        encoding[var] = comp
        if ds[var].ndim == 2:
            encoding[var]["chunksizes"] = chunks[-2:]
    return encoding
