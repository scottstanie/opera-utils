from __future__ import annotations

import logging
import multiprocessing as mp
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, NamedTuple, TypeVar

import h5netcdf
import h5py
import numpy as np
import pyproj
from tqdm.auto import tqdm

from ._dates import get_dates
from ._utils import flatten
from .burst_frame_db import get_frame_bbox
from .constants import DISP_FILE_REGEX
from .credentials import AWSCredentials

T = TypeVar("T")
__all__ = [
    "OperaDispFile",
    "parse_disp_datetimes",
]

logger = logging.getLogger("opera_utils")


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


def get_remote_h5(
    url: str,
    aws_credentials: AWSCredentials | None = None,
    page_size: int = 4 * 1024 * 1024,
    rdcc_nbytes=1024 * 1024 * 100,
) -> h5netcdf.File:
    from .credentials import get_frozen_credentials

    secret_id, secret_key, session_token = get_frozen_credentials(
        aws_credentials=aws_credentials
    )
    # ROS3 driver uses weirdly different names
    ros3_kwargs = dict(
        aws_region=b"us-west-2",
        secret_id=secret_id.encode(),
        secret_key=secret_key.encode(),
        session_token=session_token.encode(),
    )

    # Set page size for cloud-optimized HDF5
    cloud_kwargs = dict(fs_page_size=page_size, rdcc_nbytes=rdcc_nbytes)
    # return h5netcdf.File(
    return h5py.File(
        url,
        "r",
        driver="ros3",
        **ros3_kwargs,
        **cloud_kwargs,
    )


class DispReader:
    """A reader for a stack of OPERA DISP-S1 files.

    When reading a stack of data along the time dimension, the data are assumed to be stored
    as relative displacements. We convert these to displacement from the first epoch by
    multiplying along the time axis by the pseudo-inverse of the incidence matrix.

    Parameters
    ----------
    filepaths : list[str | Path]
        list of paths to OPERA DISP-S1 files to read.
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
        max_concurrent: int = 50,
    ):
        # self.filepaths = [Path(f) for f in filepaths]
        self.filepaths = sorted(
            filepaths,
            key=lambda key: OperaDispFile.from_filename(key).secondary_datetime,
        )
        self._is_s3 = "s3://" in filepaths[0]
        # TODO: provide the formatting like
        # 'HDF5:"/vsicurl/https://datapool...OPERA_...240Z.nc"://displacement'
        self._is_http = "http:" in filepaths[0]

        self.page_size = page_size
        self.dset_name = dset_name
        self.max_concurrent = max_concurrent

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

        self.aws_credentials = aws_credentials
        self._opened = False
        self.datasets = []

    def open(self, aws_credentials=None):
        # Open all files with h5netcdf
        if self._is_s3:
            creds = aws_credentials or self.aws_credentials
            for f in tqdm(self.filepaths):
                ds = get_remote_h5(
                    str(f), page_size=self.page_size, aws_credentials=creds
                )
                self.datasets.append(ds)
        elif self._is_http:
            from osgeo import gdal

            for f in tqdm(self.filepaths):
                ds = gdal.Open(f)
                self.datasets.append(ds)

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
            logging.debug("Not opened yet, running...")
            self.open()

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
            data.append(ds[self.dset_name][spatial_key])
        data = np.stack(data)

        cur_shape = data.shape
        return (self.incidence_pinv @ data.reshape(cur_shape[0], -1)).reshape(cur_shape)

    def close(self):
        """Close all open datasets."""
        for ds in self.datasets:
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
        list of ifg pairs represented as tuples of (day 1, day 2)
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


class GeoTransform(NamedTuple):
    """Named tuple for a GDAL Geotransform tuple.

    References
    ----------
    https://gdal.org/en/stable/tutorials/geotransforms_tut.html
    """

    top_left_x: float
    pixel_width: float
    rotation_x: float
    top_left_y: float
    pixel_height: float
    rotation_y: float


@dataclass
class FrameMetadata:
    crs: pyproj.CRS
    bbox: tuple[float, float, float, float]
    geotransform: GeoTransform

    @property
    def resolution(self) -> tuple[float, float]:
        """Resolution of the geospatial data."""
        return (self.geotransform.pixel_width, self.geotransform.pixel_height)

    @classmethod
    def from_frame_id(cls, frame_id: int) -> "FrameMetadata":
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
        FrameMetadata
            Dataclass containing Frame's geospatial metadata
        """
        # Retrieve the EPSG and bounding box, e.g. (xmin, ymin, xmax, ymax)
        epsg, bbox = get_frame_bbox(frame_id)
        crs = pyproj.CRS.from_epsg(epsg)

        xmin, ymin, xmax, ymax = bbox
        return cls(
            crs=crs, bbox=bbox, geotransform=GeoTransform(xmin, 30, 0, ymax, 0, -30)
        )


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


def worker_main(
    worker_id, urls_for_worker, dset_name, aws_credentials, request_queue, result_queue
):
    """Opens all the URLs assigned to this worker exactly once.
    Then repeatedly waits for read requests on `request_queue`.
    For each request, reads the slice from the appropriate open file
    and puts the result on `result_queue`.
    """

    # Open each file once
    file_handles = {}
    for url in urls_for_worker:
        file_handles[url] = get_remote_h5(url, aws_credentials=aws_credentials)

    logger.debug(f"[Worker {worker_id}] opened {len(file_handles)} files")

    while True:
        msg = request_queue.get()
        if msg is None:
            # Sentinel -> time to shut down
            break

        idx, url, slice_obj = msg

        # Read the data
        # (We assume 'dset_name' exists in the opened file)
        fh = file_handles[url]
        if fh is None:
            # The file failed to open, send back an error
            result_queue.put((idx, f"Error: Could not open {url}"))
        else:
            try:
                arr = fh[dset_name][slice_obj]
                # Return the array
                result_queue.put((idx, arr))
            except Exception as e:
                result_queue.put((idx, f"Error reading {url}: {str(e)}"))

    # Close all open handles
    for h in file_handles.values():
        if h is not None:
            h.close()

    logger.debug(f"[Worker {worker_id}] shutting down")


class DispReaderPool:
    """A reader for a stack of OPERA DISP-S1 files with multiprocessing support.

    Parameters
    ----------
    filepaths : list[str | Path]
        list of paths to OPERA DISP-S1 files to read.
    page_size : int, optional
        Page size in bytes for HDF5 file system page strategy. Default is 4 MB.
    dset_name : Literal["displacement", "short_wavelength_displacement"], optional
        Name of the dataset to read. Default is "displacement".
    aws_credentials : dict[str, Any], optional
        AWS credentials for accessing S3.
    max_concurrent : int, optional
        Maximum number of worker processes to use. Default is 4.
    """

    def __init__(
        self,
        filepaths: list[str | Path],
        page_size: int = 4 * 1024 * 1024,
        dset_name: Literal[
            "displacement", "short_wavelength_displacement"
        ] = "displacement",
        aws_credentials=None,
        max_concurrent: int = 4,
    ):
        # Sort filepaths by secondary datetime
        self.filepaths = sorted(
            filepaths,
            key=lambda key: OperaDispFile.from_filename(key).secondary_datetime,
        )
        self._is_s3 = "s3://" in str(filepaths[0])
        self._is_http = "http:" in str(filepaths[0])

        self.page_size = page_size
        self.dset_name = dset_name
        self.max_concurrent = max_concurrent
        self.aws_credentials = aws_credentials

        # Parse the reference/secondary times from each file
        self.ref_times, self.sec_times, _ = parse_disp_datetimes(self.filepaths)
        # Get the unique dates in chronological order
        self.unique_dates = sorted(set(self.ref_times + self.sec_times))

        # Create the incidence matrix for the full stack
        ifg_pairs = list(zip(self.ref_times, self.sec_times))
        self.incidence_matrix = get_incidence_matrix(
            ifg_pairs, sar_idxs=self.unique_dates, delete_first_date_column=True
        )
        self.incidence_pinv = np.linalg.pinv(self.incidence_matrix)

        # 1) Partition the filepaths across workers
        # E.g. round-robin or hash or chunk them
        # Simple round-robin approach:
        self.worker_file_lists = [[] for _ in range(max_concurrent)]
        for i, url in enumerate(self.filepaths):
            w_id = i % max_concurrent
            self.worker_file_lists[w_id].append(url)

        # 2) Make a lookup: which worker ID is responsible for each URL?
        self.url_to_worker = {}
        for w_id, url_list in enumerate(self.worker_file_lists):
            for url in url_list:
                self.url_to_worker[url] = w_id

        # Initialize multiprocessing resources
        # self.task_queue = mp.Queue()
        self.task_queues: list[mp.Queue] = []
        self.result_queue = mp.Queue()
        self.workers = []
        self._opened = False

    def open(self, aws_credentials=None):
        """Start worker processes for parallel reading.

        Parameters
        ----------
        aws_credentials : dict[str, Any], optional
            AWS credentials for accessing S3. If not provided, uses the credentials
            specified during initialization.
        """
        if self._opened:
            return

        # Make sure credentials are set
        self.aws_credentials = aws_credentials or self.aws_credentials

        for w_id in range(self.max_concurrent):
            q = mp.Queue()
            self.task_queues.append(q)

            # The subset of URLs for this worker
            urls_for_worker = self.worker_file_lists[w_id]

            p = mp.Process(
                target=worker_main,
                args=(
                    w_id,
                    urls_for_worker,
                    self.dset_name,
                    self.aws_credentials,
                    q,
                    self.result_queue,
                ),
            )
            p.start()
            self.workers.append(p)

        self._opened = True
        logger.debug(
            f"Started {len(self.workers)} worker processes for parallel reading"
        )

    def __getitem__(self, key):
        """Get a slice of data from the stack using worker processes.

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

        # Extract time and spatial keys
        if isinstance(key, tuple):
            time_key = key[0]
            spatial_key = key[1:]
        else:
            time_key = key
            spatial_key = (slice(None), slice(None))

        # Submit tasks to worker processes
        for i, url in enumerate(self.filepaths):
            # self.task_queue.put((i, str(url), self.dset_name, spatial_key))
            w_id = self.url_to_worker[url]
            self.task_queues[w_id].put((i, url, spatial_key))

        # Collect results with progress tracking
        results = [None] * len(self.filepaths)
        for _ in tqdm(range(len(self.filepaths)), desc="Reading data"):
            i, data = self.result_queue.get()
            if isinstance(data, str):  # Error message
                raise RuntimeError(f"Error processing file {i}: {data}")
            results[i] = data

        # Stack the results
        data = np.stack(results)
        cur_shape = data.shape

        # Transform from relative to absolute displacements
        return (self.incidence_pinv @ data.reshape(cur_shape[0], -1)).reshape(cur_shape)

    def close(self):
        """Close all worker processes and clean up resources."""
        if not self._opened:
            return

        # Send sentinel values to stop workers
        for q in self.task_queues:
            q.put(None)

        # Wait for workers to finish
        for p in self.workers:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()

        # Clean up
        self.workers = []
        self._opened = False
        logger.debug("All worker processes stopped")

    def __del__(self):
        self.close()
