#!/usr/bin/env python
"""Script for extracting displacement time series from OPERA DISP-S1 products.

Supports reading by row/column or lat/lon coordinates and outputs CSV time series.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
import pyproj
import tyro
from rasterio.transform import Affine, AffineTransformer

from opera_utils import _disp
from opera_utils.credentials import get_earthaccess_s3_creds


def read_url_list(file_path: str) -> list[str]:
    """Read a list of URLs from a file.

    Parameters
    ----------
    file_path : str
        Path to file containing URLs, one per line

    Returns
    -------
    list[str]
        list of valid URLs
    """
    with open(file_path, "r") as f:
        urls = [line.strip() for line in f if line.strip()]
    return urls


def initialize_reader(
    urls: list[str], page_size: int = 4 * 1024 * 1024, max_workers: int = 8
) -> _disp.DispReader:
    """Initialize and open the displacement reader.

    Parameters
    ----------
    urls : list[str]
        list of URLs to OPERA DISP-S1 files
    page_size : int, optional
        Page size in bytes for HDF5 file system page strategy. Default is 4MB.
    max_workers : int, optional
        Number of worker processes to use. Default is 8.

    Returns
    -------
    _disp.DispReader
        Initialized reader object
    """
    aws_credentials = get_earthaccess_s3_creds("opera-uat")
    reader = _disp.DispReader(
        urls,
        page_size=page_size,
        max_workers=max_workers,
        use_multiprocessing=max_workers > 1,
        aws_credentials=aws_credentials,
    )
    t0 = time.time()
    reader.open()
    print(f"Reader opened in {time.time() - t0:.2f} seconds")
    return reader


def get_frame_transformers(
    reader: _disp.DispReader,
) -> tuple[AffineTransformer, pyproj.Transformer, int]:
    """Get transformers for coordinate conversions.

    Parameters
    ----------
    reader : _disp.DispReader
        Initialized reader object

    Returns
    -------
    tuple[AffineTransformer, pyproj.Transformer, int]
        tuple of (rowcol_to_utm, utm_to_lonlat, frame_id)

    Raises
    ------
    ValueError
        If more than one frame is found in the reader
    """
    # Get the metadata object for the current frame
    disp_file = reader.disp_files[0]
    frame_id = disp_file.frame_id

    # Check if all files are from the same frame
    if len(set(df.frame_id for df in reader.disp_files)) > 1:
        raise ValueError("More than one frame found in reader")

    frame_metadata = _disp.FrameMetadata.from_frame_id(frame_id)

    # Set up the transformer from row/col to UTM and lat/lon
    transform = Affine.from_gdal(*frame_metadata.geotransform)
    rowcol_to_utm = AffineTransformer(transform)

    # Create transformer between UTM and lat/lon (WGS84)
    utm_to_lonlat = pyproj.Transformer.from_crs(
        frame_metadata.crs.to_epsg(), "EPSG:4326", always_xy=True
    )

    return rowcol_to_utm, utm_to_lonlat, frame_id


def lonlat_to_rowcol(
    lons: list[float],
    lats: list[float],
    utm_to_lonlat: pyproj.Transformer,
    rowcol_to_utm: AffineTransformer,
) -> list[tuple[int, int]]:
    """Convert lon/lat coordinates to row/col indices.

    Parameters
    ----------
    lons : list[float]
        list of longitude values
    lats : list[float]
        list of latitude values
    utm_to_lonlat : pyproj.Transformer
        Transformer for UTM to lon/lat conversion
    rowcol_to_utm : AffineTransformer
        Transformer for row/col to UTM conversion

    Returns
    -------
    list[tuple[int, int]]
        list of (row, col) tuples
    """
    # Create the inverse transformers
    lonlat_to_utm = utm_to_lonlat.inverse()
    utm_to_rowcol = ~rowcol_to_utm

    # Convert each lat/lon pair to row/col
    result = []
    for lon, lat in zip(lons, lats):
        # Convert lat/lon to UTM coordinates
        utm_x, utm_y = lonlat_to_utm.transform(lon, lat)

        # Convert UTM to row/col
        row, col = utm_to_rowcol.rowcol(utm_x, utm_y)

        # Convert to integers and add to result
        result.append((int(round(row)), int(round(col))))

    return result


def get_sample_locations(
    row_col: Optional[tuple[int, int]],
    lats: Optional[list[float]],
    lons: Optional[list[float]],
    rowcol_to_utm: Affine,
    utm_to_lonlat: pyproj.Transformer,
) -> list[dict[str, Any]]:
    """Get sample locations based on input arguments.

    Parameters
    ----------
    row_col : Optional[tuple[int, int]], optional
        Single (row, column) location.
    lats : Optional[list[float]], optional
        list of latitudes.
    lons : Optional[list[float]], optional
        list of longitudes.
    rowcol_to_utm : Affine
        Transformer for row/col to UTM conversion.
    utm_to_lonlat : pyproj.Transformer
        Transformer for UTM to lon/lat conversion.

    Returns
    -------
    list[dict[str, Any]]
        list of locations with metadata.

    Raises
    ------
    ValueError
        If neither row_col nor lats/lons are provided,
        or if lats and lons have different lengths.
    """
    locations = []

    # Determine the base locations from arguments
    if row_col is not None:
        base_locations = [row_col]
    elif lats is not None and lons is not None:
        # Validate lat/lon inputs
        if len(lats) != len(lons):
            raise ValueError(
                f"Number of latitudes ({len(lats)}) must match number of longitudes ({len(lons)})"
            )
        # Convert lat/lon to row/col
        base_locations = lonlat_to_rowcol(lons, lats, utm_to_lonlat, rowcol_to_utm)
    else:
        raise ValueError("Either row_col or both lats and lons must be provided")

    # Process each location
    for i, (row, col) in enumerate(base_locations):
        utm_x, utm_y = rowcol_to_utm.xy(row, col)
        lon, lat = utm_to_lonlat.transform(utm_x, utm_y)
        locations.append(
            {
                "location_id": i + 1,
                "row": int(row),
                "col": int(col),
                "lon": float(lon),
                "lat": float(lat),
                "utm_x": float(utm_x),
                "utm_y": float(utm_y),
            }
        )

    return locations


def read_time_series(
    reader: _disp.DispReader, locations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Read time series data for each location.

    Parameters
    ----------
    reader : _disp.DispReader
        Initialized reader object
    locations : list[dict[str, Any]]
        list of locations to read from

    Returns
    -------
    list[dict[str, Any]]
        Locations with time series data added
    """
    for loc in locations:
        row, col = loc["row"], loc["col"]

        t0 = time.time()
        # Read just 1 pixel at the specified location
        # The shape will be (time_steps, 1, 1)
        data = reader[:, row : row + 1, col : col + 1]
        read_time = time.time() - t0

        # Extract the time series (squeeze removes singleton dimensions)
        time_series = data.squeeze()

        # Add time series to the location data
        loc["time_series"] = time_series

        print(
            f"Location {loc['location_id']}: ({row}, {col}) - "
            f"Read shape {data.shape} in {read_time:.2f} seconds"
        )

    return locations


def create_dataframes(
    locations: list[dict[str, Any]], dates: list[datetime]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create wide and long format DataFrames from time series data.

    Parameters
    ----------
    locations : list[dict[str, Any]]
        list of locations with time series data
    dates : list[datetime]
        list of dates corresponding to time steps

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Wide and long format DataFrames
    """
    # Convert dates to strings
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]

    # Create wide format DataFrame
    df_data = {"date": date_strs}

    for loc in locations:
        location_id = loc["location_id"]
        row = loc["row"]
        col = loc["col"]
        time_series = loc["time_series"]

        # Add this location's time series as a column
        column_name = f"location_{location_id}_r{row}_c{col}"
        df_data[column_name] = time_series

    # Create the wide format DataFrame
    return pd.DataFrame(df_data)


def save_outputs(
    df_wide: pd.DataFrame,
    df_long: pd.DataFrame,
    locations: list[dict[str, Any]],
    dates: list[datetime],
    output_dir: Path,
    plot: bool = False,
) -> dict[str, Path]:
    """Save outputs to files.

    Parameters
    ----------
    df_wide : pd.DataFrame
        Wide format DataFrame
    df_long : pd.DataFrame
        Long format DataFrame
    locations : list[dict[str, Any]]
        list of locations with metadata
    dates : list[datetime]
        list of dates corresponding to time steps
    output_dir : Path
        Directory to save outputs
    plot : bool, optional
        Whether to generate a plot, by default False

    Returns
    -------
    dict[str, Path]
        Dictionary of output filenames
    """
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_files = {}

    # Save wide format DataFrame
    csv_filename = output_dir / f"displacement_timeseries_{timestamp}.csv"
    df_wide.to_csv(csv_filename, index=False)
    output_files["wide_csv"] = csv_filename
    print(f"Time series data saved to {csv_filename}")

    # Save long format DataFrame
    long_csv_filename = output_dir / f"displacement_timeseries_long_{timestamp}.csv"
    df_long.to_csv(long_csv_filename, index=False)
    output_files["long_csv"] = long_csv_filename
    print(f"Long-format time series data saved to {long_csv_filename}")

    # Create metadata file
    metadata = {
        "number_of_locations": len(locations),
        "number_of_time_points": len(dates),
        "first_date": dates[0].isoformat(),
        "last_date": dates[-1].isoformat(),
        "locations": [
            {
                "id": loc["location_id"],
                "row": loc["row"],
                "col": loc["col"],
                "lat": loc["lat"],
                "lon": loc["lon"],
                "utm_x": loc["utm_x"],
                "utm_y": loc["utm_y"],
            }
            for loc in locations
        ],
    }

    # Save metadata to JSON
    metadata_filename = output_dir / f"displacement_metadata_{timestamp}.json"
    with open(metadata_filename, "w") as f:
        json.dump(metadata, f, indent=2)
    output_files["metadata"] = metadata_filename
    print(f"Metadata saved to {metadata_filename}")

    # Optionally create plots
    if plot:
        plt.figure(figsize=(10, 6))
        for loc in locations:
            location_id = loc["location_id"]
            row = loc["row"]
            col = loc["col"]
            time_series = loc["time_series"]

            plt.plot(
                dates,
                time_series,
                marker="o",
                linestyle="-",
                label=f"Loc {location_id}: ({row}, {col})",
            )

        plt.xlabel("Date")
        plt.ylabel("Displacement (m)")
        plt.title("Displacement Time Series")
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot
        plot_filename = output_dir / f"displacement_timeseries_{timestamp}.png"
        plt.savefig(plot_filename, dpi=200)
        output_files["plot"] = plot_filename
        print(f"Plot saved to {plot_filename}")

    return output_files


def main(
    url_file: str,
    row_col: Optional[tuple[int, int]] = None,
    lats: Optional[list[float]] = None,
    lons: Optional[list[float]] = None,
    max_workers: int = 8,
    page_size: int = 4 * 1024 * 1024,
    output_dir: Path = Path("./output"),
    plot: bool = False,
) -> None:
    """Read remove DISP-S1 data.

    Parameters
    ----------
    url_file : str
        File containing S3 URLs to OPERA DISP-S1 files, one per line
    row_col : Optional[tuple[int, int]], optional
        Read one (row, column) within the frame.
    lats : Optional[list[float]], optional
        list of latitudes to read from (used with `lons`).
    lons : Optional[list[float]], optional
        list of longitudes to read from (used with `lats`).
    max_workers : int, optional
        Number of worker processes to use. Default is 8.
    page_size : int, optional
        Page size in bytes for HDF5 file system page strategy. Default is 4MB.
    output_dir : str, optional
        Directory to save output files. Default is "./output".
    plot : bool, optional
        Generate time series plots. Default is False.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    urls = read_url_list(url_file)
    print(f"Found {len(urls)} URLs to process")

    reader = initialize_reader(urls, page_size=page_size, max_workers=max_workers)

    rowcol_to_utm, utm_to_lonlat, frame_id = get_frame_transformers(reader)
    print(f"Procssing OPERA DISP Frame {frame_id:05d}")

    locations = get_sample_locations(
        row_col=row_col,
        lats=lats,
        lons=lons,
        rowcol_to_utm=rowcol_to_utm,
        utm_to_lonlat=utm_to_lonlat,
    )

    # Get unique dates (skip the first reference date where displacement = 0)
    dates = reader.unique_dates[1:]

    locations = read_time_series(reader, locations)

    df_wide, df_long = create_dataframes(locations, dates)
    save_outputs(df_wide, df_long, locations, dates, output_dir, plot)

    print("All processing complete!")
    # Ensure reader is closed
    reader.close()


if __name__ == "__main__":
    tyro.cli(main)
