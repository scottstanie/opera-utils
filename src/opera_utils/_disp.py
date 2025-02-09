from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ._dates import get_dates
from .constants import DISP_FILE_REGEX

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


def open_stack(filenames: list[Path | str]):
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
    opera_disp_file_list: list[Path | str],
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
