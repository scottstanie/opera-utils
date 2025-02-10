from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TypeVar

import numpy as np

from ._dates import get_dates
from ._utils import flatten
from .constants import DISP_FILE_REGEX

T = TypeVar("T")
__all__ = [
    "OperaDispFile",
    "parse_disp_datetimes",
]


@dataclass
class OperaDispFile:
    """Class for information from one DISP-S1 production filename."""

    sensor: str
    acquisition_mode: str
    frame_id: int
    polarization: str
    reference_datetime: datetime
    secondary_datetime: datetime
    version: str
    generation_datetime: datetime

    @classmethod
    def from_filename(cls, name: Path | str) -> "OperaDispFile":
        """Create a OperaDispFile from a filename."""
        if not (match := DISP_FILE_REGEX.match(Path(name).name)):
            raise ValueError(f"Invalid filename format: {name}")

        data = match.groupdict()
        data["reference_datetime"] = datetime.fromisoformat(data["reference_datetime"])
        data["secondary_datetime"] = datetime.fromisoformat(data["secondary_datetime"])
        data["generation_datetime"] = datetime.fromisoformat(
            data["generation_datetime"]
        )
        data["frame_id"] = int(data["frame_id"])

        return cls(**data)  # type: ignore


def open_stack(filenames: Sequence[Path | str]):
    """Open a stack of files as a single xarray dataset.

    Parameters
    ----------
    filenames : list[Filename]
        List of filenames to open.

    Returns
    -------
    xr.Dataset
        The dataset containing all files.
    """
    import pandas as pd
    import xarray as xr

    def _prep(ds):
        fname = ds.encoding["source"]
        if len(ds.band) == 1:
            ds = ds.sel(band=ds.band[0])

        ref_dts, sec_dts, gen_dts = parse_disp_datetimes([fname])
        # TODO: how should we store reference/generation times?
        return ds.expand_dims(time=[pd.to_datetime(sec_dts[0])])

    ds = xr.open_mfdataset(
        filenames,
        preprocess=_prep,
        engine="rasterio",
    )
    return ds


def parse_disp_datetimes(
    opera_disp_file_list: Sequence[Path | str],
) -> tuple[tuple[datetime], tuple[datetime], tuple[datetime]]:
    """Parse the datetimes from a list of OPERA DISP-S1 filenames."""
    dts = [get_dates(f, fmt="%Y%m%dT%H%M%SZ") for f in opera_disp_file_list]

    reference_times: tuple[datetime]
    secondary_times: tuple[datetime]
    generation_times: tuple[datetime]
    reference_times, secondary_times, generation_times = zip(*dts)
    return reference_times, secondary_times, generation_times


def _get_first_file_per_ministack(
    opera_file_list: Sequence[Path | str],
) -> list[Path | str]:
    def _get_generation_time(fname):
        return parse_disp_datetimes([fname])[2][0]

    first_per_ministack = []
    for d, cur_groupby in itertools.groupby(
        sorted(opera_file_list), key=_get_generation_time
    ):
        # cur_groupby is an iterable of all matching
        # Get the first one
        first_per_ministack.append(next(cur_groupby))
    return first_per_ministack


class DispReader:
    """A reader for a stack of OPERA DISP-S1 files.

    When reading a stack of data along the time dimension, the data are assumed to be stored
    as relative displacements. We convert these to displacement from the first epoch by
    multiplying along the time axis by the pseudo-inverse of the incidence matrix.

    Parameters
    ----------
    filepaths : list[str | Path]
        List of paths to OPERA DISP-S1 files to read.
    page_size : int, optional
        Page size in bytes for HDF5 file system page strategy. Default is 4 MB.
    """

    def __init__(self, filepaths: list[str | Path], page_size: int = 4 * 1024 * 1024):
        self.filepaths = [Path(f) for f in filepaths]
        self.page_size = page_size

        # Parse the reference/secondary times from each file
        self.ref_times, self.sec_times, _ = parse_disp_datetimes(self.filepaths)
        # Get the unique dates in chronological order
        self.unique_dates = sorted(set(self.ref_times + self.sec_times))

        # Create the incidence matrix for the full stack
        # Each row represents one file, with -1 at ref_time and +1 at sec_time
        ifg_pairs = list(zip(self.ref_times, self.sec_times))
        self.incidence_matrix = get_incidence_matrix(
            ifg_pairs, sar_idxs=self.unique_dates, delete_first_date_column=True
        )
        self.incidence_pinv = np.linalg.pinv(self.incidence_matrix)

        # Create a mapping of dates to indices for the time dimension
        self.date_to_idx = {d: i for i, d in enumerate(self.unique_dates)}

        # Open all files with h5netcdf
        import h5netcdf.core

        self.datasets = []
        for f in self.filepaths:
            ds = h5netcdf.core.File(
                f,
                "r",
                decode_vlen_strings=False,
                # Set page size for cloud-optimized HDF5
                libver="latest",
                fs_strategy="page",
                fs_page_size=self.page_size,
            )
            self.datasets.append(ds)

    def __getitem__(self, key):
        """Get a slice of data from the stack.

        Parameters
        ----------
        key : tuple[slice | int]
            Tuple of slices/indices into (time, y, x) dimensions.

        Returns
        -------
        np.ndarray
            Data transformed from relative to absolute displacements.
        """
        # Get the time slice/index
        if isinstance(key, tuple):
            time_key = key[0]
            spatial_key = key[1:]
        else:
            time_key = key
            spatial_key = (slice(None), slice(None))

        # Read the data from each file
        data = []
        for ds in self.datasets:
            data.append(ds["displacement"][spatial_key])
        data = np.stack(data)

        # Transform from relative to absolute displacements
        transformed = np.tensordot(self.incidence_pinv, data, axes=([1], [0]))

        # Return the requested time slice
        if isinstance(time_key, slice):
            return transformed[time_key]
        return transformed[time_key]

    def __del__(self):
        """Close all open datasets."""
        for ds in self.datasets:
            ds.close()


def get_incidence_matrix(
    ifg_pairs: Sequence[tuple[T, T]],
    sar_idxs: Sequence[T] | None = None,
    delete_first_date_column: bool = True,
) -> np.ndarray:
    """Build the indicator matrix from a list of ifg pairs (index 1, index 2).

    Parameters
    ----------
    ifg_pairs : Sequence[tuple[T, T]]
        List of ifg pairs represented as tuples of (day 1, day 2)
        Can be ints, datetimes, etc.
    sar_idxs : Sequence[T], optional
        If provided, used as the total set of indexes which `ifg_pairs`
        were formed from.
        Otherwise, created from the unique entries in `ifg_pairs`.
        Only provide if there are some dates which are not present in `ifg_pairs`.
    delete_first_date_column : bool
        If True, removes the first column of the matrix to make it full column rank.
        Size will be `n_sar_dates - 1` columns.
        Otherwise, the matrix will have `n_sar_dates`, but rank `n_sar_dates - 1`.

    Returns
    -------
    A : np.array 2D
        The incident-like matrix for the system: A*phi = dphi
        Each row corresponds to an ifg, each column to a SAR date.
        The value will be -1 on the early (reference) ifgs, +1 on later (secondary)
        since the ifg phase = (later - earlier)
        Shape: (n_ifgs, n_sar_dates - 1)

    """
    if sar_idxs is None:
        sar_idxs = sorted(set(flatten(ifg_pairs)))

    M = len(ifg_pairs)
    col_iter = sar_idxs[1:] if delete_first_date_column else sar_idxs
    N = len(col_iter)
    A = np.zeros((M, N))

    # Create a dictionary mapping sar dates to matrix columns
    # We take the first SAR acquisition to be time 0, leave out of matrix
    date_to_col = {date: i for i, date in enumerate(col_iter)}
    for i, (early, later) in enumerate(ifg_pairs):
        if early in date_to_col:
            A[i, date_to_col[early]] = -1
        if later in date_to_col:
            A[i, date_to_col[later]] = +1

    return A
