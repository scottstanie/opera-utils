from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger("opera_utils")

__all__ = ["rebase", "rebase_np"]


class NaNPolicy(str, Enum):
    """Policy for handling NaN values in rebase_timeseries."""

    propagate = "propagate"
    omit = "omit"

    def __str__(self) -> str:
        return self.value


def rebase_np(
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


def rebase(
    da_displacement: xr.DataArray,
    reference_time: Sequence[datetime | pd.DatetimeIndex] | xr.DataArray,
    nan_policy: str | NaNPolicy = NaNPolicy.propagate,
) -> xr.DataArray:
    """Rebase displacement products with different reference dates.

    This function combines displacement products that may have different reference
    dates by accumulating displacements when the reference date changes.
    When a new reference date is encountered, the displacement values from the
    previous stack's final epoch are added to all epochs in the new stack.

    Parameters
    ----------
    da_displacement : xr.DataArray
        Displacement dataarray to rebase.
    reference_time : Sequence[datetime | pd.DatetimeIndex]
        Reference datetime for each epoch.
        Must be same length as `da_time`.
    nan_policy : str | NaNPolicy
        Policy for handling NaN values in rebase_timeseries.
        choices = ["propagate", "omit"]

    Returns
    -------
    xr.DataArray
        Stacked displacement dataarray with rebased displacements.

    """
    da_time = da_displacement.time  # .compute()
    nan_policy = NaNPolicy(nan_policy)

    # Find the last epoch of every ministack, where the reference date changes.
    # These frames contain the offsets that must be accumulated.
    ref_dates = np.unique(da_time)
    # Short-circuit — nothing to rebase
    if len(ref_dates) == 1:
        return da_displacement
    last_idxs = np.where(da_time.isin(ref_dates))[0]

    # Take just those “crossover” frames,
    crossover = da_displacement.isel(time=last_idxs)
    # decide what to do with NaNs in them.
    if nan_policy is NaNPolicy.omit:
        crossover = crossover.fillna(0)

    # Cumulative sum of the crossover frames gives the running offset to apply
    ref_offsets = crossover.cumsum(dim="time")  # same length as ref_dates

    for idx, ref in enumerate(ref_dates[1:]):
        # Boolean mask that selects *all* epochs belonging to this ministack
        # mask = (reference_time == ref).compute()
        mask = reference_time == ref

        # Add the cumulative offset on pixels where `mask` is True
        da_displacement = xr.where(
            mask, da_displacement + ref_offsets.isel(time=idx), da_displacement
        )

    return da_displacement.transpose("time", "y", "x")
