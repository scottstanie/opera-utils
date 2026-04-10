"""SLC stack parsing utilities for tropospheric corrections.

This module provides functions to extract metadata from SLC stacks
from various SAR sensors (Capella, etc.) for use in tropospheric
correction workflows.

A `SLCReader` Protocol defines the interface that every sensor backend
must implement.  Use `register_sensor` to add new backends and
`get_sensor` to look them up by name.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

__all__ = [
    "SLCReader",
    "extract_stack_info",
    "extract_stack_info_capella",
    "get_incidence_angle_capella",
    "get_sensor",
    "register_sensor",
]

logger = logging.getLogger("opera_utils")


# ---------------------------------------------------------------------------
# Protocol + registry
# ---------------------------------------------------------------------------


@runtime_checkable
class SLCReader(Protocol):
    """Interface for reading metadata from a single SLC file."""

    def read_datetime(self, slc_file: Path) -> datetime:
        """Return the center time (UTC) for the SLC acquisition."""
        ...

    def read_bounds(self, slc_file: Path) -> tuple[float, float, float, float]:
        """Return the bounding box as (west, south, east, north) in degrees."""
        ...

    def read_incidence_angle(self, slc_file: Path) -> float:
        """Return the incidence angle in degrees at the center pixel."""
        ...


_sensor_registry: dict[str, SLCReader] = {}


def register_sensor(name: str, reader: SLCReader) -> None:
    """Register an `SLCReader` implementation under *name* (case-insensitive)."""
    _sensor_registry[name.lower()] = reader


def get_sensor(name: str) -> SLCReader:
    """Look up a registered `SLCReader` by *name* (case-insensitive).

    Raises
    ------
    ValueError
        If no reader has been registered for *name*.

    """
    key = name.lower()
    if key not in _sensor_registry:
        available = ", ".join(sorted(_sensor_registry)) or "(none)"
        msg = f"Unknown sensor {name!r}. Registered sensors: {available}"
        raise ValueError(msg)
    return _sensor_registry[key]


# ---------------------------------------------------------------------------
# Capella backend
# ---------------------------------------------------------------------------


class CapellaSLCReader:
    """SLCReader implementation for Capella SAR data.

    The ``capella_reader`` package is imported lazily so that the
    dependency is only required when Capella data is actually used.
    """

    @staticmethod
    def _load():
        from capella_reader import CapellaSLC  # noqa: PLC0415

        return CapellaSLC

    def read_datetime(self, slc_file: Path) -> datetime:
        CapellaSLC = self._load()
        slc = CapellaSLC.from_file(slc_file)
        center_time_str = str(slc.center_time)
        # Handle nanosecond precision by truncating to microseconds
        if "." in center_time_str:
            base, frac = center_time_str.rsplit(".", 1)
            frac = frac.rstrip("Z")[:6]
            center_time_str = f"{base}.{frac}"
        center_time_str = center_time_str.rstrip("Z")
        return datetime.fromisoformat(center_time_str)

    def read_bounds(self, slc_file: Path) -> tuple[float, float, float, float]:
        CapellaSLC = self._load()
        slc = CapellaSLC.from_file(slc_file)
        w, s, e, n = slc.bounds
        return (w, s, e, n)

    def read_incidence_angle(self, slc_file: Path) -> float:
        CapellaSLC = self._load()
        slc = CapellaSLC.from_file(slc_file)
        return slc.collect.image.center_pixel.incidence_angle


# Register the built-in Capella backend
register_sensor("capella", CapellaSLCReader())


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def extract_stack_info(
    slc_files: list[Path],
    sensor: str = "capella",
) -> tuple[list[datetime], tuple[float, float, float, float]]:
    """Extract datetimes and bounds from an SLC stack.

    Parameters
    ----------
    slc_files : list[Path]
        List of SLC file paths.
    sensor : str
        Sensor name (case-insensitive).  Must have been registered via
        `register_sensor`.  Default is ``"capella"``.

    Returns
    -------
    datetimes : list[datetime]
        List of center times (UTC) for each SLC, sorted chronologically.
    bounds : tuple[float, float, float, float]
        Combined bounding box (west, south, east, north) in degrees,
        covering all SLCs in the stack.

    """
    reader = get_sensor(sensor)
    assert len(slc_files) > 0, "At least one SLC file is required"

    datetimes: list[datetime] = []
    west_vals: list[float] = []
    south_vals: list[float] = []
    east_vals: list[float] = []
    north_vals: list[float] = []

    for slc_file in slc_files:
        dt = reader.read_datetime(slc_file)
        datetimes.append(dt)

        w, s, e, n = reader.read_bounds(slc_file)
        west_vals.append(w)
        south_vals.append(s)
        east_vals.append(e)
        north_vals.append(n)

        logger.debug(f"Parsed {slc_file}: {dt}, bounds=({w}, {s}, {e}, {n})")

    combined_bounds = (
        min(west_vals),
        min(south_vals),
        max(east_vals),
        max(north_vals),
    )

    sorted_indices = sorted(range(len(datetimes)), key=lambda i: datetimes[i])
    datetimes_sorted = [datetimes[i] for i in sorted_indices]

    logger.info(
        f"Extracted {len(datetimes_sorted)} SLCs from "
        f"{datetimes_sorted[0]} to {datetimes_sorted[-1]}"
    )
    logger.info(f"Combined bounds: {combined_bounds}")

    return datetimes_sorted, combined_bounds


# ---------------------------------------------------------------------------
# Backward-compatible thin wrappers
# ---------------------------------------------------------------------------


def extract_stack_info_capella(
    slc_files: list[Path],
) -> tuple[list[datetime], tuple[float, float, float, float]]:
    """Extract datetimes and bounds from a Capella SLC stack.

    Thin wrapper around `extract_stack_info` for backward compatibility.

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

    """
    return extract_stack_info(slc_files, sensor="capella")


def get_incidence_angle_capella(slc_file: Path) -> float:
    """Get the center pixel incidence angle from a Capella SLC.

    Thin wrapper using the registered Capella reader.

    Parameters
    ----------
    slc_file : Path
        Path to a Capella SLC GeoTIFF file.

    Returns
    -------
    float
        Incidence angle in degrees at the center pixel.

    """
    reader = get_sensor("capella")
    return reader.read_incidence_angle(slc_file)
