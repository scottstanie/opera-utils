#!/usr/bin/env python

# /// script
# dependencies = ["opera-utils", "h5py", "numpy", "pyproj", "tqdm", "tyro"]
# ///
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, TypeVar

import h5py
import numpy as np
import tyro
from pyproj import Transformer
from tqdm.auto import tqdm

from opera_utils import Bbox, flatten, get_dates, get_frame_bbox

T = TypeVar("T")


# OPERA_L3_DISP-S1_IW_F11116_VV_20160705T140755Z_20160729T140756Z_v1.0_20241219T231545Z.nc
DISP_FILE_REGEX = re.compile(
    "OPERA_L3_DISP-"
    r"(?P<sensor>(S1|NI))_"
    r"(?P<acquisition_mode>IW)_"  # TODO: What's NISAR's?
    r"F(?P<frame_id>\d{5})_"
    r"(?P<polarization>(VV|HH))_"
    r"(?P<reference_datetime>\d{8}T\d{6}Z)_"
    r"(?P<secondary_datetime>\d{8}T\d{6}Z)_"
    r"v(?P<version>[\d.]+)_"
    r"(?P<generation_datetime>\d{8}T\d{6}Z)",
)


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


class DispReader:
    """A reader for a stack of OPERA DISP-S1 files.

    When reading a stack of data along the time dimension, the data are assumed to be stored
    as relative displacements. We convert these to displacement from the first epoch by
    multiplying along the time axis by the pseudo-inverse of the incidence matrix.

    Parameters
    ----------
    filepaths : list[str | Path]
        List of paths to OPERA DISP-S1 files to read.
    """

    def __init__(
        self,
        filepaths: list[str | Path],
        # TODO: refactor, get full list
        dset_name: Literal[
            "displacement", "short_wavelength_displacement"
        ] = "short_wavelength_displacement",
    ):
        self.filepaths = sorted(
            filepaths,
            key=lambda key: OperaDispFile.from_filename(key).secondary_datetime,
        )

        self.dset_name = dset_name

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
        self._opened = False
        self._h5py_files: list[h5py.File] = []

    def open(self):
        """Open all files with h5py."""
        for f in tqdm(self.filepaths, desc="Opening files"):
            hf = h5py.File(f)
            self._h5py_files.append(hf)
        self._opened = True

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
        if not self._opened:
            self.open()
        # Get the time slice/index
        time, rows, cols = key

        # Read the data from each file
        data = []
        for hf in tqdm(self._h5py_files, desc="Reading data"):
            data.append(hf[self.dset_name][rows, cols])
        data = np.stack(data)

        # Reconstruct single-reference time series from moving reference date
        transformed = self.incidence_pinv @ data

        # Return the requested time slice
        return transformed

    def close(self):
        """Close all open datasets."""
        for ds in self._h5py_files:
            ds.close()
        self._opened = False

    def __del__(self):
        self.close()


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


def get_geospatial_metadata(frame_id: int) -> dict:
    """
    Get the geospatial metadata for a given frame.

    This function retrieves the bounding box and EPSG code for the specified frame
    using `get_frame_bbox`, then constructs a geotransform based on known properties:
    - The image is always in UTM.
    - Pixel spacing is fixed at 30 m by 30 m.
    - The geotransform is defined as (top left x, pixel width, rotation_x, top left y, rotation_y, pixel height),
      where the top left coordinate is (xmin, ymax) because y typically decreases downward in raster data.

    Additionally, the full CRS object is constructed using pyproj.

    Parameters
    ----------
    frame_id : int
        The ID of the frame to get metadata for.

    Returns
    -------
    dict
        A dictionary containing:
          - "crs": pyproj CRS object based on the EPSG code.
          - "bbox": tuple of (xmin, ymin, xmax, ymax).
          - "geotransform": tuple (top left x, pixel width, 0, top left y, 0, pixel height)
            where pixel height is negative for a top-down image orientation.
          - "resolution": (30, 30)
    """
    import pyproj

    # Retrieve the EPSG and bounding box, e.g. (xmin, ymin, xmax, ymax)
    epsg, bbox = get_frame_bbox(frame_id)
    crs = pyproj.CRS.from_epsg(epsg)

    xmin, ymin, xmax, ymax = bbox
    # Construct the geotransform.
    # Top left coordinate: (xmin, ymax)
    # Pixel width: 30, and pixel height is -30 (negative because it goes downward)
    geotransform = (xmin, 30, 0, ymax, 0, -30)

    return {
        "crs": crs,
        "bbox": bbox,
        "geotransform": geotransform,
        "resolution": (30, 30),
    }


def utm_to_rowcol(
    utm_x: float,
    utm_y: float,
    geotransform: tuple[float, float, float, float, float, float],
) -> tuple[int, int]:
    """
    Convert UTM coordinates to pixel row and column indices using the provided geotransform.

    Parameters
    ----------
    utm_x : float
        The UTM x coordinate.
    utm_y : float
        The UTM y coordinate.
    geotransform : tuple
        Geotransform tuple of the form
        (xmin, pixel_width, 0, ymax, 0, pixel_height),
        where pixel_height is negative for top-down images.

    Returns
    -------
    tuple[int, int]
        (row, col) indices corresponding to the UTM coordinate.
    """
    xmin, pixel_width, _, ymax, _, pixel_height = geotransform
    col = int(round((utm_x - xmin) / pixel_width))
    row = int(round((ymax - utm_y) / abs(pixel_height)))
    return row, col


def lonlat_to_utm(lon: float, lat: float, utm_crs) -> tuple[float, float]:
    """
    Convert geographic coordinates (longitude, latitude) to UTM coordinates using a target CRS.

    Parameters
    ----------
    lon : float
        Longitude in degrees.
    lat : float
        Latitude in degrees.
    utm_crs : pyproj.CRS
        Target UTM coordinate system (typically from get_geospatial_metadata).

    Returns
    -------
    tuple[float, float]
        (utm_x, utm_y) coordinates.
    """
    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    utm_x, utm_y = transformer.transform(lon, lat)
    return utm_x, utm_y


def lonlat_to_rowcol(
    lon: float,
    lat: float,
    geotransform: tuple[float, float, float, float, float, float],
    utm_crs,
) -> tuple[int, int]:
    """
    Convert geographic coordinates (lon, lat) to pixel row and column indices.

    This function transforms the (lon, lat) pair to UTM
    and then computes the corresponding pixel indices using the provided geotransform.

    Parameters
    ----------
    lon : float
        Longitude in degrees.
    lat : float
        Latitude in degrees.
    geotransform : tuple
        Geotransform tuple (xmin, pixel_width, 0, ymax, 0, pixel_height).
    utm_crs : pyproj.CRS
        Target UTM coordinate system.

    Returns
    -------
    tuple[int, int]
        (row, col) indices.
    """
    utm_x, utm_y = lonlat_to_utm(lon, lat, utm_crs)
    return utm_to_rowcol(utm_x, utm_y, geotransform)


def bbox_lonlat_to_utm(bbox: Bbox, utm_crs) -> Bbox:
    """
    Convert a bounding box in geographic coordinates (lon, lat) to UTM coordinates.

    The input Bbox has attributes (left, bottom, right, top) in lon/lat.
    The function transforms all four corners and then determines the
    new bounding box in UTM.

    Parameters
    ----------
    bbox : Bbox
        Bounding box in lon/lat.
    utm_crs : pyproj.CRS
        Target UTM coordinate system.

    Returns
    -------
    Bbox
        Bounding box in UTM coordinates.
    """
    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    # Transform each corner: lower-left, upper-left, lower-right, upper-right
    ll = transformer.transform(bbox.left, bbox.bottom)
    lt = transformer.transform(bbox.left, bbox.top)
    rl = transformer.transform(bbox.right, bbox.bottom)
    rt = transformer.transform(bbox.right, bbox.top)
    xs = [ll[0], lt[0], rl[0], rt[0]]
    ys = [ll[1], lt[1], rl[1], rt[1]]
    return Bbox(left=min(xs), bottom=min(ys), right=max(xs), top=max(ys))


def bbox_utm_to_rowcol(
    bbox: Bbox, geotransform: tuple[float, float, float, float, float, float]
) -> tuple[int, int, int, int]:
    """
    Convert a bounding box in UTM coordinates to pixel row/column indices.

    The provided geotransform should be of the form
        (xmin, pixel_width, 0, ymax, 0, pixel_height)
    with pixel_height negative for top-down orientation.

    Parameters
    ----------
    bbox : Bbox
        Bounding box in UTM coordinates (left, bottom, right, top).
    geotransform : tuple
        Geotransform tuple as described above.

    Returns
    -------
    tuple[int, int, int, int]
        Pixel indices as (row_min, row_max, col_min, col_max).
    """
    # Use the top-left corner for the minimum indices...
    row_min, col_min = utm_to_rowcol(bbox.left, bbox.top, geotransform)
    # ...and the bottom-right for the maximum indices.
    row_max, col_max = utm_to_rowcol(bbox.right, bbox.bottom, geotransform)
    return row_min, row_max, col_min, col_max


def bbox_lonlat_to_rowcol(
    bbox: Bbox, geotransform: tuple[float, float, float, float, float, float], utm_crs
) -> tuple[int, int, int, int]:
    """
    Convert a bounding box in geographic coordinates (lon, lat) to pixel row and column indices.

    This function transforms the Bbox from lon/lat to UTM using the provided target CRS,
    then computes the pixel indices using the geotransform.

    Parameters
    ----------
    bbox : Bbox
        Bounding box in lon/lat (left, bottom, right, top).
    geotransform : tuple
        Geotransform tuple (xmin, pixel_width, 0, ymax, 0, pixel_height).
    utm_crs : pyproj.CRS
        Target UTM coordinate system.

    Returns
    -------
    tuple[int, int, int, int]
        (row_min, row_max, col_min, col_max) pixel indices.
    """
    utm_bbox = bbox_lonlat_to_utm(bbox, utm_crs)
    return bbox_utm_to_rowcol(utm_bbox, geotransform)


# -------------------------
# Command-line Interface
# -------------------------


def extract(
    files: list[Path],
    output: Path = Path("output.npy"),
    layer: Literal[
        "displacement", "short_wavelength_displacement"
    ] = "short_wavelength_displacement",
    format: Literal["npy", "csv"] = "npy",
    rowcol: tuple[int, int] | None = None,
    latlon: tuple[float, float] | None = None,
) -> None:
    """
    Extract data from input files and save to output file.

    Parameters
    ----------
    files : list of Path
        One or more input file paths.
    output : Path, optional
        Output file path (.npy or .csv). Default is Path("output.npy").
    layer : str, choices = { "displacement", "short_wavelength_displacement"}
        HDF5 layer to read pixels from.
        Default is "short_wavelength_displacement".
    format : {"npy", "csv"}, optional
        Output format. Default is "npy".
    rowcol : tuple of int, optional
        Pixel coordinates as (row, col) if using pixel indexing.
    latlon : tuple of float, optional
        Geographic coordinates as (lat, lon) if using geographic coordinates.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If neither or both rowcol and latlon are specified.

    Notes
    -----
    This function extracts data from the input files based on either pixel
    coordinates (rowcol) or geographic coordinates (latlon), and saves the
    result to the specified output file in the chosen format.
    """
    # Validate coordinate input:
    if rowcol is not None and latlon is not None:
        raise ValueError("Specify either rowcol or latlon, not both.")
    if rowcol is None and latlon is None:
        raise ValueError("Must specify either rowcol or latlon.")

    # Determine row and column indices
    if latlon is not None:
        # Use metadata from the first file to convert lat/lon to row/col.
        first_file = files[0]
        op_file = OperaDispFile.from_filename(first_file)
        metadata = get_geospatial_metadata(op_file.frame_id)
        geotransform = metadata["geotransform"]
        utm_crs = metadata["crs"]
        row, col = lonlat_to_rowcol(latlon[1], latlon[0], geotransform, utm_crs)
    else:
        row, col = rowcol

    disp_reader = DispReader(filepaths=files, dset_name=layer)
    try:
        time_series = disp_reader[:, row, col]
    finally:
        disp_reader.close()

    # Save the extracted time series.
    if format == "npy":
        np.save(output, time_series)
        print(f"Saved time series to {output} as a NumPy binary file.")
    elif format == "csv":
        np.savetxt(output, time_series, delimiter=",")
        print(f"Saved time series to {output} as CSV.")


if __name__ == "__main__":
    tyro.cli(extract)
