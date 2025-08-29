from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd
import xarray as xr

from opera_utils.disp._utils import _clamp_chunk_dict

logger = logging.getLogger("opera_utils")

__all__ = [
    "create_rebased_displacement",
]


class NaNPolicy(str, Enum):
    """Policy for handling NaN values in rebase_timeseries."""

    propagate = "propagate"
    omit = "omit"

    def __str__(self) -> str:
        return self.value


def create_rebased_displacement(
    ds: xr.DataArray,
    # reference_datetimes: Sequence[datetime | pd.DatetimeIndex],
    process_chunk_size: tuple[int, int] = (1024, 1024),
    add_reference_time: bool = False,
    nan_policy: str | NaNPolicy = NaNPolicy.propagate,
) -> xr.DataArray:
    """Rebase and stack displacement products with different reference dates.

    This function combines displacement products that may have different reference
    dates by accumulating displacements when the reference date changes.
    When a new reference date is encountered, the displacement values from the
    previous stack's final epoch are added to all epochs in the new stack.

    Parameters
    ----------
    da_displacement : xr.DataArray
        Displacement dataarray to rebase.
    reference_datetimes : Sequence[datetime | pd.DatetimeIndex]
        Reference datetime for each epoch.
        Must be same length as `da_displacement.time`.
    process_chunk_size : tuple[int, int], optional
        Chunk size for processing. Defaults to (512, 512).
    add_reference_time : bool, optional
        Whether to add a zero array for the reference time.
        Defaults to False.
    nan_policy : choices = ["propagate", "omit"]
        Whether to propagate or omit (zero out) NaNs in the data.
        By default "propagate", which means any ministack, or any "reference crossover"
        product, with nan at a pixel causes all subsequent data to be nan.
        If "omit", then any nan causes the pixel to be zeroed out, which is
        equivalent to assuming that 0 displacement occurred during that time.

    Returns
    -------
    xr.DataArray
        Stacked displacement dataarray with rebased displacements.

    """
    logger.info("Starting displacement stack rebasing")

    process_chunks = {
        "time": -1,
        "y": process_chunk_size[0],
        "x": process_chunk_size[1],
    }
    da_displacement: xr.DataArray = ds["displacement"]
    process_chunks = _clamp_chunk_dict(process_chunks, da_displacement.shape)
    reference_datetimes = ds.reference_time.to_numpy()
    secondary_datetimes = ds.time.to_numpy()

    # Make the map_blocks-compatible function to accumulate the displacement
    def process_block(arr: xr.DataArray) -> xr.DataArray:
        out = rebase_timeseries(
            arr.to_numpy(),
            reference_datetimes,
            secondary_datetimes,
            nan_policy=nan_policy,
        )
        return xr.DataArray(out, coords=arr.coords, dims=arr.dims)

    # Process the dataset in blocks
    rebased_da = da_displacement.chunk(process_chunks).map_blocks(process_block)

    if add_reference_time:
        # Add initial reference epoch of zeros, and rechunk
        rebased_da = xr.concat(
            [xr.full_like(rebased_da[0], 0), rebased_da],
            dim="time",
        )
        # Ensure correct dimension order
        rebased_da = rebased_da.transpose("time", "y", "x")

    return rebased_da


def rebase_timeseries_old(
    raw_data: np.ndarray,
    reference_dates: Sequence[datetime],
    nan_policy: str | NaNPolicy = NaNPolicy.propagate,
) -> np.ndarray:
    """Adjust for moving reference dates to create a continuous time series.

    DISP-S1 products have a reference date which changes over time.
    For example, shortening to YYYY-MM-DD notation, the products may be

        (2020-01-01, 2020-01-13)
        (2020-01-01, 2020-01-25)
        ...
        (2020-01-01, 2020-06-17)
        (2020-06-17, 2020-06-29)
        ...


    This function sums up the "crossover" values (the displacement image where the
    reference date moves forward) so that the output is referenced to the first input
    time.

    Parameters
    ----------
    raw_data : np.ndarray
        3D array of displacement values with moving reference dates
        shape = (time, rows, cols)
    reference_dates : Sequence[datetime]
        Reference dates for each time step
    nan_policy : choices = ["propagate", "omit"]
        Whether to propagate or omit (zero out) NaNs in the data.
        By default "propagate", which means any ministack, or any "reference crossover"
        product, with nan at a pixel causes all subsequent data to be nan.
        If "omit", then any nan causes the pixel to be zeroed out, which is
        equivalent to assuming that 0 displacement occurred during that time.

    Returns
    -------
    np.ndarray
        Continuous displacement time series with consistent reference date

    """
    if len(set(reference_dates)) == 1:
        return raw_data.copy()

    shape2d = raw_data.shape[1:]
    cumulative_offset = np.zeros(shape2d, dtype=np.float32)
    previous_displacement = np.zeros(shape2d, dtype=np.float32)

    # Set initial reference date
    current_reference_date = reference_dates[0]

    output = np.zeros_like(raw_data)
    # Process each time step
    for cur_ref_date, current_displacement, out_layer in zip(
        reference_dates, raw_data, output
    ):
        # Check for shift in temporal reference date
        if cur_ref_date != current_reference_date:
            # When reference date changes, accumulate the previous displacement
            if nan_policy == NaNPolicy.omit:
                np.nan_to_num(previous_displacement, copy=False)
            cumulative_offset += previous_displacement
            current_reference_date = cur_ref_date

        # Store current displacement for next iteration
        previous_displacement = current_displacement.copy()

        # Add cumulative offset to get consistent reference
        out_layer[:] = current_displacement + cumulative_offset

    return output


def rebase_timeseries(
    raw_data: np.ndarray,
    reference_dates: list[datetime] | pd.DatetimeIndex,
    secondary_dates: list[datetime] | pd.DatetimeIndex,
    nan_policy: str | NaNPolicy = NaNPolicy.propagate,
) -> np.ndarray:
    """Rebase to the first reference by composing true crossovers, using fuzzy
    matching between reference and secondary datetimes (±crossover_tolerance).

    raw_data: (T, Y, X)
    reference_dates[t]: reference of epoch t (len T)
    secondary_dates[t]: secondary (time) of epoch t (len T)
    """
    T = raw_data.shape[0]
    if T == 0:
        return raw_data.copy()

    # Normalize to numpy int64 ns for fast matching
    ref_ns = reference_dates.astype("datetime64[D]")
    sec_ns = secondary_dates.astype("datetime64[D]")
    if len(set(ref_ns)) == 1:
        return raw_data.copy()

    # Build a nearest-neighbor matcher: for each ref r, find index k with sec≈r
    # within tolerance; otherwise return -1.
    sec_sorted_idx = np.argsort(sec_ns)
    sec_sorted = sec_ns[sec_sorted_idx]

    def match_secondary_index(r_ns: np.int64) -> int:
        pos = np.searchsorted(sec_sorted, r_ns)
        candidates = []
        if pos < sec_sorted.size:
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
        if not candidates:
            return -1
        # choose nearest
        cand = min(candidates, key=lambda i: abs(sec_sorted[i] - r_ns))
        if sec_sorted[cand] == r_ns:
            return int(sec_sorted_idx[cand])
        return -1

    # Unique references in temporal order, keep first ref as r0
    refs_unique, _inv = np.unique(ref_ns, return_inverse=True)
    r0_ns = refs_unique[0]

    # Map each crossover (p,r) to an index k in raw_data via fuzzy match on r
    # (We’ll look up (p,r) existence by checking whether there exists a t with
    # ref_ns[t]==p and sec_ns[t]≈r within tol.)
    # For fast lookup, collect all indices per reference p:
    idx_by_ref: dict[np.int64, np.ndarray] = {}
    for p in refs_unique:
        idx_by_ref[p] = np.nonzero(ref_ns == p)[0]

    def edge_index(p_ns: np.int64, r_ns: np.int64) -> int:
        """Return t where ref==p and sec≈r (within tol), else -1."""
        k_r = match_secondary_index(r_ns)
        if k_r == -1:
            return -1
        # Need t with ref==p and same secondary index k_r
        # Compare by secondary value (ns), not position, because order differs.
        target_sec = sec_ns[k_r]
        t_candidates = idx_by_ref[p_ns]
        # find any t with sec_ns[t] within tol of target_sec
        if t_candidates.size == 0:
            return -1
        # Check nearest among those t_candidates
        diffs = np.abs(sec_ns[t_candidates] - target_sec)
        t_local = t_candidates[np.argmin(diffs)]
        if sec_ns[t_local] == target_sec:
            return int(t_local)
        return -1

    # Build per-reference offset images O(r) by dynamic programming.
    offsets: dict[np.int64, np.ndarray] = {
        r0_ns: np.zeros(raw_data.shape[1:], dtype=raw_data.dtype)
    }

    def _accum(dst: np.ndarray, src: np.ndarray) -> np.ndarray:
        if nan_policy == NaNPolicy.omit:
            src = np.nan_to_num(src, copy=False)
        return dst + src

    for r_ns in refs_unique[1:]:
        # Prefer direct (r0, r)
        k = edge_index(r0_ns, r_ns)
        if k != -1:
            offsets[r_ns] = _accum(offsets[r0_ns], raw_data[k])
            continue

        # Otherwise chain via a prior p<r with known O(p) and available (p,r)
        prior = refs_unique[refs_unique < r_ns]
        # try latest first (works well for sliding windows)
        found = False
        for p_ns in prior[::-1]:
            if p_ns not in offsets:
                continue
            k = edge_index(p_ns, r_ns)
            if k != -1:
                offsets[r_ns] = _accum(offsets[p_ns], raw_data[k])
                found = True
                break
        if not found:
            # As a last resort, if r has no inbound edge (rare), carry forward O of the nearest prior ref
            # so the series stays continuous (optional: raise instead).
            if prior.size > 0 and prior[-1] in offsets:
                offsets[r_ns] = offsets[prior[-1]].copy()
            else:
                msg = (
                    "Cannot construct offset for reference "
                    f"{pd.to_datetime(r_ns)}: no crossover within tolerance."
                )
                raise ValueError(
                    msg
                )

    # Apply per-epoch
    out = np.empty_like(raw_data)
    for t in range(T):
        out[t] = raw_data[t] + offsets[ref_ns[t]]
    return out
