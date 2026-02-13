"""High-level workflow for creating tropospheric corrections from SLC stacks."""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import Literal

from opera_utils.tropo._apply import apply_tropo
from opera_utils.tropo._crop import crop_tropo
from opera_utils.tropo._search import search_tropo
from opera_utils.tropo._slc_stack import extract_stack_info

__all__ = ["create_tropo_corrections_for_stack"]

logger = logging.getLogger("opera_utils")


def create_tropo_corrections_for_stack(
    slc_files: list[Path],
    dem_path: Path,
    los_path: Path,
    output_dir: Path = Path("tropo_corrections"),
    sensor: str = "capella",
    los_type: Literal["incidence_angle", "enu"] = "incidence_angle",
    margin_deg: float = 0.3,
    height_max: float = 10000.0,
    subtract_first_date: bool = True,
    skip_time_interpolation: bool = False,
    num_workers: int = 2,
    cropped_tropo_dir: Path | None = None,
) -> list[Path]:
    """Create tropospheric corrections for an SLC stack end-to-end.

    This is a high-level workflow function that:
    1. Parses the SLC stack to extract datetimes and bounds
    2. Searches CMR for TROPO products covering the time range
    3. Crops TROPO products to the AOI
    4. Applies corrections using DEM and LOS geometry

    Parameters
    ----------
    slc_files : list[Path]
        List of SLC file paths.
    dem_path : Path
        DEM GeoTIFF (UTM or WGS84).
    los_path : Path
        LOS geometry raster. Either incidence angle or ENU components
        depending on `los_type`.
    output_dir : Path
        Output directory for correction GeoTIFFs.
        Default is "tropo_corrections".
    sensor : str
        Sensor type. Currently only "capella" is supported.
        Default is "capella".
    los_type : {"incidence_angle", "enu"}
        Type of LOS geometry raster:
        - "incidence_angle": Single-band raster with incidence angle in degrees.
        - "enu": 3-band raster with LOS (east, north, up) unit vector components.
        Default is "incidence_angle".
    margin_deg : float
        Additional margin in degrees around AOI bounds.
        Default is 0.3 degrees.
    height_max : float
        Maximum height in meters to include in TROPO cropping.
        Default is 10000 meters.
    subtract_first_date : bool
        If True, subtract day-1 correction from all subsequent dates.
        Default is True.
    skip_time_interpolation : bool
        If True, use nearest TROPO file instead of time interpolation.
        Default is False.
    num_workers : int
        Number of parallel processes.
        Default is 2.
    cropped_tropo_dir : Path, optional
        Directory to store cropped TROPO files.
        If None, uses a subdirectory of `output_dir`.

    Returns
    -------
    list[Path]
        List of output correction GeoTIFF paths.

    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Step 1: Parse SLC stack to get datetimes and bounds
    logger.info(f"Parsing {len(slc_files)} SLC files from {sensor} sensor...")
    datetimes, bounds = extract_stack_info(slc_files, sensor=sensor)
    logger.info(f"Found {len(datetimes)} SLCs from {datetimes[0]} to {datetimes[-1]}")
    logger.info(f"Combined bounds: {bounds}")

    # Step 2: Search for TROPO products
    # Add buffer to ensure we have bracketing files for interpolation
    start_dt = datetimes[0] - timedelta(hours=12)
    end_dt = datetimes[-1] + timedelta(hours=12)

    logger.info(f"Searching for TROPO products from {start_dt} to {end_dt}...")
    tropo_urls = search_tropo(
        start_datetime=start_dt,
        end_datetime=end_dt,
    )
    logger.info(f"Found {len(tropo_urls)} TROPO products")

    if len(tropo_urls) == 0:
        msg = f"No TROPO products found for date range {start_dt} to {end_dt}"
        raise ValueError(msg)

    # Step 3: Write TROPO URLs to a temporary file and crop
    if cropped_tropo_dir is None:
        cropped_tropo_dir = output_dir / "cropped_tropo"
    cropped_tropo_dir.mkdir(exist_ok=True, parents=True)

    # Write URLs to file for crop_tropo
    urls_file = cropped_tropo_dir / "tropo_urls.txt"
    urls_file.write_text("\n".join(tropo_urls))

    logger.info(f"Cropping TROPO products to AOI with {margin_deg} deg margin...")
    crop_tropo(
        tropo_urls_file=urls_file,
        datetimes=datetimes,
        aoi_bounds=bounds,
        output_dir=cropped_tropo_dir,
        skip_time_interpolation=skip_time_interpolation,
        height_max=height_max,
        margin_deg=margin_deg,
        num_workers=num_workers,
    )

    # Step 4: Collect cropped files and apply corrections
    cropped_files = sorted(cropped_tropo_dir.glob("tropo_cropped_*.nc"))
    logger.info(f"Found {len(cropped_files)} cropped TROPO files")

    assert len(cropped_files) > 0, "No cropped TROPO files found after cropping"

    logger.info("Applying tropospheric corrections...")
    incidence_angle_path = los_path if los_type == "incidence_angle" else None
    los_enu_path = los_path if los_type == "enu" else None

    apply_tropo(
        cropped_tropo_list=cropped_files,
        dem_path=dem_path,
        incidence_angle_path=incidence_angle_path,
        los_enu_path=los_enu_path,
        output_dir=output_dir,
        subtract_first_date=subtract_first_date,
        num_workers=num_workers,
    )

    # Return list of output files
    output_files = sorted(output_dir.glob("tropo_correction_*.tif"))
    logger.info(f"Created {len(output_files)} tropospheric correction files")

    return output_files
