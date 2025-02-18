from __future__ import annotations

import asyncio
import itertools
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, TypeVar

import h5netcdf
import h5py
import numpy as np
from pyproj import Transformer
from tqdm.auto import tqdm

from ._dates import get_dates
from ._types import Bbox
from ._utils import flatten
from .burst_frame_db import get_frame_bbox
from .constants import DISP_FILE_REGEX
from .credentials import AWSCredentials

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


def get_remote_h5(
    url: str,
    aws_credentials: AWSCredentials | None = None,
    page_size: int = 4 * 1024 * 1024,
) -> h5netcdf.File:
    from .credentials import get_frozen_credentials

    secret_id, secret_key, session_token = get_frozen_credentials(
        aws_credentials=aws_credentials
    )
    # ROS3 driver uses weirdly different names
    driver_kwds = dict(
        aws_region=b"us-west-2",
        secret_id=secret_id.encode(),
        secret_key=secret_key.encode(),
        session_token=session_token.encode(),
    )

    cloud_opts = dict(
        # Set page size for cloud-optimized HDF5
        libver="latest",
        fs_page_size=page_size,
        rdcc_nbytes=1024 * 1024 * 100,  # 100 MB per file
    )
    # return h5netcdf.File(
    return h5py.File(
        url,
        "r",
        driver="ros3",
        **driver_kwds,
        **cloud_opts,
    )


from typing import Literal


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

    def __init__(
        self,
        filepaths: list[str | Path],
        page_size: int = 4 * 1024 * 1024,
        # TODO: refactor, get full list
        dset_name: Literal[
            "displacement", "short_wavelength_displacement"
        ] = "displacement",
        aws_credentials=None,
    ):
        # self.filepaths = [Path(f) for f in filepaths]
        self.filepaths = sorted(
            filepaths,
            key=lambda key: OperaDispFile.from_filename(key).secondary_datetime,
        )
        self.page_size = page_size
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
        self.date_to_idx = {d: i for i, d in enumerate(self.unique_dates)}
        self.aws_credentials = aws_credentials
        self._opened = False
        self.datasets = []

    async def _load_file(self, filepath: str | Path) -> None:
        """Load a single HDF5 file asynchronously."""
        # Run get_remote_h5 in a thread pool since it's a blocking operation
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            ds = await loop.run_in_executor(
                pool,
                lambda: get_remote_h5(
                    str(filepath),
                    page_size=self.page_size,
                    aws_credentials=self.aws_credentials,
                ),
            )
            self.datasets.append(ds)

    async def open(self, aws_credentials=None):
        """Asynchronously open all files using a semaphore to limit concurrent operations."""
        if self._opened:
            return

        creds = aws_credentials or self.aws_credentials
        self.aws_credentials = creds

        # Create a semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def load_with_semaphore(filepath):
            async with semaphore:
                await self._load_file(filepath)

        # Create tasks for all files
        tasks = [load_with_semaphore(f) for f in self.filepaths]

        # Use tqdm to show progress
        for f in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Loading files"
        ):
            await f

        self._opened = True

    def __getitem__(self, key):
        """Synchronous version of data access. For better performance, use aget instead."""
        return asyncio.run(self.aget(key))

    async def aget(self, key):
        """Asynchronously get a slice of data from the stack."""
        if not self._opened:
            print("Not opened yet, running...")
            await self.open()

        # Get the time slice/index
        if isinstance(key, tuple):
            time_key = key[0]
            spatial_key = key[1:]
        else:
            time_key = key
            spatial_key = (slice(None), slice(None))

        # Read the data from each file asynchronously
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            # Create tasks for reading data from each dataset
            tasks = [
                loop.run_in_executor(
                    pool, lambda ds=ds: ds[self.dset_name][spatial_key]
                )
                for ds in self.datasets
            ]
            # Wait for all reads to complete
            data = await asyncio.gather(*tasks)

        data = np.stack(data)

        # Transform from relative to absolute displacements
        transformed = np.tensordot(self.incidence_pinv, data, axes=([1], [0]))

        # Return the requested time slice
        if isinstance(time_key, slice):
            return transformed[time_key]
        return transformed[time_key]

    async def aclose(self):
        """Asynchronously close all open datasets."""
        if not self._opened:
            return

        # Create a thread pool for closing operations
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            # Close all datasets concurrently
            await asyncio.gather(
                *[loop.run_in_executor(pool, ds.close) for ds in self.datasets]
            )
        self.datasets = []
        self._opened = False

    def close(self):
        """Synchronous wrapper for aclose."""
        asyncio.run(self.aclose())

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
