"""SLC stack parsing utilities for tropospheric corrections.

This module provides functions to extract metadata from SLC stacks
from various SAR sensors (Capella, etc.) for use in tropospheric
correction workflows.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass

__all__ = [
    "extract_stack_info",
    "extract_stack_info_capella",
]

logger = logging.getLogger("opera_utils")


def extract_stack_info(
    slc_files: list[Path],
    sensor: Literal["capella"] = "capella",
) -> tuple[list[datetime], tuple[float, float, float, float]]:
    """Extract datetimes and bounds from an SLC stack.

    Parameters
    ----------
    slc_files : list[Path]
        List of SLC file paths.
    sensor : str
        Sensor type. Currently only "capella" is supported.

    Returns
    -------
    datetimes : list[datetime]
        List of center times (UTC) for each SLC, sorted chronologically.
    bounds : tuple[float, float, float, float]
        Combined bounding box (west, south, east, north) in degrees,
        covering all SLCs in the stack.

    """
    if sensor == "capella":
        return extract_stack_info_capella(slc_files)
    else:
        msg = f"Unsupported sensor: {sensor}"
        raise ValueError(msg)


def extract_stack_info_capella(
    slc_files: list[Path],
) -> tuple[list[datetime], tuple[float, float, float, float]]:
    """Extract datetimes and bounds from a Capella SLC stack.

    Parameters
    ----------
    slc_files : list[Path]
        List of Capella SLC GeoTIFF file paths.

    Returns
    -------
    datetimes : list[datetime]
        List of center times (UTC) for each SLC, sorted chronologically.
    bounds : tuple[float, float, float, float]
        Combined bounding box (west, south, east, north) in degrees,
        covering all SLCs in the stack.

    Raises
    ------
    ImportError
        If capella_reader is not installed.

    """
    try:
        from capella_reader import CapellaSLC
    except ImportError as e:
        msg = (
            "capella_reader is required for Capella SLC parsing. "
            "Install with: pip install capella-reader"
        )
        raise ImportError(msg) from e

    assert len(slc_files) > 0, "At least one SLC file is required"

    datetimes: list[datetime] = []
    west_vals: list[float] = []
    south_vals: list[float] = []
    east_vals: list[float] = []
    north_vals: list[float] = []

    for slc_file in slc_files:
        slc = CapellaSLC.from_file(slc_file)

        # center_time is a string like "2024-06-29T13:49:12.689866839Z"
        # Parse it to datetime
        center_time_str = str(slc.center_time)
        # Handle nanosecond precision by truncating
        if "." in center_time_str:
            base, frac = center_time_str.rsplit(".", 1)
            # Take only first 6 digits of fractional seconds (microseconds)
            frac = frac.rstrip("Z")[:6]
            center_time_str = f"{base}.{frac}"
        center_time_str = center_time_str.rstrip("Z")
        dt = datetime.fromisoformat(center_time_str)
        datetimes.append(dt)

        # bounds is (min_lon, min_lat, max_lon, max_lat) = (west, south, east, north)
        w, s, e, n = slc.bounds  # noqa[misc]
        west_vals.append(w)
        south_vals.append(s)
        east_vals.append(e)
        north_vals.append(n)

        logger.debug(f"Parsed {slc_file}: {dt}, bounds={slc.bounds}")

    # Compute combined bounding box
    combined_bounds = (
        min(west_vals),
        min(south_vals),
        max(east_vals),
        max(north_vals),
    )

    # Sort by datetime
    sorted_indices = sorted(range(len(datetimes)), key=lambda i: datetimes[i])
    datetimes_sorted = [datetimes[i] for i in sorted_indices]

    logger.info(
        f"Extracted {len(datetimes_sorted)} SLCs from "
        f"{datetimes_sorted[0]} to {datetimes_sorted[-1]}"
    )
    logger.info(f"Combined bounds: {combined_bounds}")

    return datetimes_sorted, combined_bounds


def get_incidence_angle_capella(slc_file: Path) -> float:
    """Get the center pixel incidence angle from a Capella SLC.

    Parameters
    ----------
    slc_file : Path
        Path to a Capella SLC GeoTIFF file.

    Returns
    -------
    float
        Incidence angle in degrees at the center pixel.

    """
    try:
        from capella_reader import CapellaSLC
    except ImportError as e:
        msg = (
            "capella_reader is required for Capella SLC parsing. "
            "Install with: pip install capella-reader"
        )
        raise ImportError(msg) from e

    slc = CapellaSLC.from_file(slc_file)
    return slc.collect.image.center_pixel.incidence_angle
