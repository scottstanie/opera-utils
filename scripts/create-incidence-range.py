#!/usr/bin/env python
# /// script
# dependencies = ['rasterio', 'tyro']
# ///
"""Create incidence and approximate slant range distance from line-of-sight rasters.

DISP-S1 static layers include the line-of-sight (LOS) east, north, up (ENU) unit vectors.
From this, we can get the incidence angel as `arccos(up)`, and an approximation of the slant
range distance based on Sentinel-1 orbit altitude.
"""

from pathlib import Path

import numpy as np
import rasterio
import tyro

DEFAULT_RASTERIO_PROFILE = dict(
    dtype=rasterio.float32,
    count=1,
    nodata=0,
    tiled=True,
    compress="deflate",
    predictor=2,
    nbits=16,
)


def compute_incidence_angle(
    los_enu_path: Path | str,
    out_path: Path | str = Path("incidence_angle.tif"),
) -> Path:
    """
    Compute incidence angles from a LOS ENU GeoTIFF and save the output as a new GeoTIFF.

    This function:
      1. Reads band 3 of the input `los_enu` raster (assumed to be cos(incidence)).
      2. Computes the incidence angle in degrees using `incidence_deg = degrees(arccos(cos_values))`.
      3. Writes the result to a GeoTIFF with tiled/deflate compression settings.

    Parameters
    ----------
    los_enu_path : Path or str
        Path to the input 'los_enu.tif'.
    out_path : Path or str, optional
        Path to the output incidence angle GeoTIFF. Default is "incidence_angle.tif".

    Returns
    -------
    Path
        The path to the output incidence angle GeoTIFF.
    """
    nodata = 0
    with rasterio.open(los_enu_path) as src:
        # Read the third band (los_up == cos(incidence) )
        cos_values = src.read(3, masked=True)

        # Compute incidence angle in degrees
        incidence_deg = np.degrees(np.arccos(cos_values))

        profile = src.profile.copy()
        profile.update(**DEFAULT_RASTERIO_PROFILE)

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(incidence_deg.astype(np.float32).filled(nodata), 1)

    return Path(out_path)


def get_slant_range(incidence_raster: Path | str, subsample: int = 1) -> np.ndarray:
    """
    Compute slant-range distance from a raster of incidence angles in degrees.

    The calculation uses the law of sines approach to derive the slant range from:
      - Earth radius (6,371,008.8 meters)
      - Satellite altitude (~693,000 meters)
      - Incidence angle (in degrees)

    Parameters
    ----------
    incidence_raster : Path or str
        Path to the incidence angle GeoTIFF (in degrees).
    subsample : int, optional
        Factor to subsample the raster (e.g., use every N-th pixel). Default is 100.

    Returns
    -------
    np.ndarray
        A 2D array of slant-range distances, matching the shape of the subsampled raster.
    """
    earth_radius = 6_371_008.8  # meters
    sat_altitude = 693_000.0  # meters above Earth's surface
    R = earth_radius + sat_altitude

    with rasterio.open(incidence_raster) as src:
        incidence_deg = src.read(1, masked=True)[::subsample, ::subsample]

    incidence_rad = np.radians(incidence_deg)

    two_times_circ = R / np.sin(incidence_rad)
    look_angle_rad = np.arcsin(earth_radius / two_times_circ)
    range_angle_rad = incidence_rad - look_angle_rad
    slant_range = two_times_circ * np.sin(range_angle_rad)

    return slant_range.filled(0)


def create_inc_range(
    los_enu: Path | str,
    inc_angle_path: Path | str = Path("incidence_angle.tif"),
    slant_range_path: Path | str = Path("slant_range_distance.tif"),
) -> None:
    """
    Create an incidence-angle GeoTIFF from a LOS ENU raster.

    This function uses `compute_incidence_angle` to read the LOS ENU raster,
    compute its incidence angles in degrees, and write the result to the specified output path.

    Parameters
    ----------
    los_enu : Path or str
        Path to the input LOS ENU GeoTIFF.
    inc_angle_path : Path or str
        Path to write the incidence angle GeoTIFF.
        Default is "incidence_angle.tif".
    slant_range_path : Path or str
        Path to write the slant range GeoTIFF.
        Default is "slant_range_distance.tif".
    """
    compute_incidence_angle(los_enu_path=los_enu, out_path=inc_angle_path)
    slant_range = get_slant_range(incidence_raster=inc_angle_path)

    with rasterio.open(inc_angle_path) as src:
        profile = src.profile.copy()
    with rasterio.open(slant_range_path, "w", **profile) as dst:
        dst.write(slant_range.astype(np.float32), 1)


if __name__ == "__main__":
    tyro.cli(create_inc_range)
