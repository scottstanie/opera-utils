from __future__ import annotations

import logging
import multiprocessing as mp
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Any, NamedTuple, Optional, TypeVar, Union

import h5py
import numpy as np
import pyproj
from tqdm.auto import tqdm

T = TypeVar("T")
PathOrStr = Union[str, Path]

__all__ = [
    "OperaDispFile",
    "parse_disp_datetimes",
    "DispReader",
    "GeoTransform",
    "FrameMetadata",
    "utm_to_rowcol",
]

logger = logging.getLogger("opera_utils")

# Names in the DISP-S1 product:
# [p.name for p in DISPLACEMENT_PRODUCTS]
# ['displacement',
#  'short_wavelength_displacement',
#  'recommended_mask',
#  'connected_component_labels',
#  'temporal_coherence',
#  'estimated_phase_quality',
#  'persistent_scatterer_mask',
#  'shp_counts',
#  'water_mask',
#  'phase_similarity',
#  'timeseries_inversion_residuals']


# For use in newer pythons, if we want to type the dataset arg:
class DispLayers(str, Enum):
    displacement = "displacement"
    short_wavelength_displacement = "short_wavelength_displacement"


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
    def from_filename(cls, name: PathOrStr) -> "OperaDispFile":
        """Create a OperaDispFile from a filename.

        Parameters
        ----------
        name : str or Path
            Filename to parse for OPERA DISP-S1 information.

        Returns
        -------
        OperaDispFile
            Parsed file information.

        Raises
        ------
        ValueError
            If the filename format is invalid.
        """
        from .constants import DISP_FILE_REGEX

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
    opera_disp_file_list: Sequence[PathOrStr],
) -> tuple[tuple[datetime, ...], tuple[datetime, ...], tuple[datetime, ...]]:
    """Parse the datetimes from a list of OPERA DISP-S1 filenames.

    Parameters
    ----------
    opera_disp_file_list : Sequence[str or Path]
        List of OPERA DISP-S1 filenames to parse.

    Returns
    -------
    tuple[tuple[datetime, ...], tuple[datetime, ...], tuple[datetime, ...]]
        Tuple of (reference_times, secondary_times, generation_times).
    """
    from ._dates import get_dates

    dts = [get_dates(f, fmt="%Y%m%dT%H%M%SZ") for f in opera_disp_file_list]

    reference_times = tuple(dt[0] for dt in dts)
    secondary_times = tuple(dt[1] for dt in dts)
    generation_times = tuple(dt[2] for dt in dts)

    return reference_times, secondary_times, generation_times


def get_remote_h5(
    url: str,
    aws_credentials=None,
    page_size: int = 4 * 1024 * 1024,
    rdcc_nbytes: int = 1024 * 1024 * 100,
) -> h5py.File:
    """Open a remote HDF5 file using the ROS3 driver.

    Parameters
    ----------
    url : str
        S3 URL to the HDF5 file.
    aws_credentials : AWSCredentials, optional
        AWS credentials for accessing S3.
    page_size : int, optional
        File system page size in bytes. Default is 4 MB.
    rdcc_nbytes : int, optional
        Raw data chunk cache size in bytes. Default is 100 MB.

    Returns
    -------
    h5py.File
        Opened HDF5 file.
    """
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

    return h5py.File(
        url,
        "r",
        driver="ros3",
        **ros3_kwargs,
        **cloud_kwargs,
    )


def get_incidence_matrix(
    ifg_pairs: Sequence[tuple[T, T]],
    sar_idxs: Optional[Sequence[T]] = None,
    delete_first_date_column: bool = True,
) -> np.ndarray:
    """Build the indicator matrix from a list of interferogram pairs.

    Parameters
    ----------
    ifg_pairs : Sequence[tuple[T, T]]
        List of interferogram pairs represented as tuples of (day 1, day 2).
        Can be ints, datetimes, etc.
    sar_idxs : Sequence[T], optional
        If provided, used as the total set of indexes which `ifg_pairs`
        were formed from.
        Otherwise, created from the unique entries in `ifg_pairs`.
        Only provide if there are some dates which are not present in `ifg_pairs`.
    delete_first_date_column : bool, optional
        If True, removes the first column of the matrix to make it full column rank.
        Size will be `n_sar_dates - 1` columns.
        Otherwise, the matrix will have `n_sar_dates`, but rank `n_sar_dates - 1`.
        Default is True.

    Returns
    -------
    np.ndarray
        The incident-like matrix for the system: A*phi = dphi
        Each row corresponds to an interferogram, each column to a SAR date.
        The value will be -1 on the early (reference) ifgs, +1 on later (secondary)
        since the ifg phase = (later - earlier)
        Shape: (n_ifgs, n_sar_dates - 1) if delete_first_date_column=True
               (n_ifgs, n_sar_dates) otherwise
    """

    def flatten(list_of_lists: Iterable[Iterable[Any]]) -> chain[Any]:
        """Flatten one level of a nested iterable."""
        return chain.from_iterable(list_of_lists)

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

    Attributes
    ----------
    top_left_x : float
        X coordinate of the top-left corner.
    pixel_width : float
        Width of a pixel in map units.
    rotation_x : float
        X rotation (usually 0).
    top_left_y : float
        Y coordinate of the top-left corner.
    rotation_y : float
        Y rotation (usually 0).
    pixel_height : float
        Height of a pixel in map units (typically negative).

    References
    ----------
    https://gdal.org/en/stable/tutorials/geotransforms_tut.html
    """

    top_left_x: float
    pixel_width: float
    rotation_x: float
    top_left_y: float
    rotation_y: float
    pixel_height: float


@dataclass
class FrameMetadata:
    """Geospatial metadata for a frame.

    Attributes
    ----------
    crs : pyproj.CRS
        Coordinate reference system.
    bbox : tuple[float, float, float, float]
        Bounding box as (xmin, ymin, xmax, ymax).
    geotransform : GeoTransform
        GDAL-style geotransform.
    """

    crs: pyproj.CRS
    bbox: tuple[float, float, float, float]
    geotransform: GeoTransform

    @property
    def resolution(self) -> tuple[float, float]:
        """Resolution of the geospatial data in (x, y) directions.

        Returns
        -------
        tuple[float, float]
            Resolution as (x_resolution, y_resolution) in CRS units.
        """
        return (abs(self.geotransform.pixel_width), abs(self.geotransform.pixel_height))

    @classmethod
    def from_frame_id(cls, frame_id: int) -> "FrameMetadata":
        """Get the geospatial metadata for a given frame.

        This function retrieves the bounding box and EPSG code for the specified frame
        using `get_frame_bbox`, then constructs a geotransform based on known properties:
        - The image is always in UTM.
        - Pixel spacing is fixed at 30 m by 30 m.
        - The geotransform is defined as (top left x, pixel width, rotation_x,
          top left y, rotation_y, pixel height), where the top left coordinate is
          (xmin, ymax) because y typically decreases downward in raster data.

        Parameters
        ----------
        frame_id : int
            The ID of the frame to get metadata for.

        Returns
        -------
        FrameMetadata
            Dataclass containing frame's geospatial metadata.
        """
        # Retrieve the EPSG and bounding box, e.g. (xmin, ymin, xmax, ymax)
        from .burst_frame_db import get_frame_bbox

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
    """Convert UTM coordinates to pixel row and column indices.

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


@dataclass
class DispReader:
    """A reader for a stack of OPERA DISP-S1 files with optional multiprocessing.

    When reading a stack of data along the time dimension, the data are assumed to be stored
    as relative displacements. We convert these to displacement from the first epoch by
    multiplying along the time axis by the pseudo-inverse of the incidence matrix.

    Parameters
    ----------
    filepaths : list[str | Path]
        List of paths to OPERA DISP-S1 files to read.
    page_size : int, optional
        Page size in bytes for HDF5 file system page strategy. Default is 4 MB.
    dset_name : {"displacement", "short_wavelength_displacement"}, optional
        Name of the dataset to read. Default is "displacement".
    aws_credentials : AWSCredentials, optional
        AWS credentials for accessing S3 files.
    use_multiprocessing : bool, optional
        Whether to use multiprocessing for parallel reading. Default is False.
    max_workers : int, optional
        Maximum number of worker processes to use when multiprocessing is enabled.
        Default is 4.
    """

    filepaths: Sequence[PathOrStr]
    page_size: int = 4 * 1024 * 1024
    dset_name: DispLayers = DispLayers.displacement
    aws_credentials: Optional[Any] = None
    use_multiprocessing: bool = False
    max_workers: int = 4
    zero_nans: bool = True

    def __post_init__(self):
        # Sort filepaths by secondary datetime
        self.filepaths = sorted(
            self.filepaths,
            key=lambda key: OperaDispFile.from_filename(key).secondary_datetime,
        )
        self.disp_files = [OperaDispFile.from_filename(fp) for fp in self.filepaths]
        self._is_s3 = "s3://" in str(self.filepaths[0])
        self._is_http = "https" in str(self.filepaths[0])

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

        # Multiprocessing setup if enabled
        if self.use_multiprocessing:
            # Partition the filepaths across workers
            self.worker_file_lists = [[] for _ in range(self.max_workers)]
            for i, url in enumerate(self.filepaths):
                w_id = i % self.max_workers
                self.worker_file_lists[w_id].append(str(url))

            # Make a lookup: which worker ID is responsible for each URL?
            self.url_to_worker = {}
            for w_id, url_list in enumerate(self.worker_file_lists):
                for url in url_list:
                    self.url_to_worker[url] = w_id

            # Initialize multiprocessing resources
            self.task_queues = [mp.Queue() for _ in range(self.max_workers)]
            self.result_queue = mp.Queue()
            self.workers = []
        else:
            # For non-multiprocessing mode
            self.datasets = []

        self._opened = False

    def open(self, aws_credentials: Optional[Any] = None) -> None:
        """Open the datasets for reading.

        Parameters
        ----------
        aws_credentials : AWSCredentials, optional
            AWS credentials for accessing S3. If not provided, uses the credentials
            specified during initialization.
        """
        if self._opened:
            return

        creds = aws_credentials or self.aws_credentials

        if self.use_multiprocessing:
            # Start worker processes for parallel reading
            for w_id in range(self.max_workers):
                # The subset of URLs for this worker
                urls_for_worker = self.worker_file_lists[w_id]

                p = mp.Process(
                    target=self._worker_main,
                    args=(
                        w_id,
                        urls_for_worker,
                        self.dset_name.value,
                        creds,
                        self.task_queues[w_id],
                        self.result_queue,
                    ),
                )
                p.start()
                self.workers.append(p)

            logger.debug(
                f"Started {len(self.workers)} worker processes for parallel reading"
            )
        else:
            # Open all files directly
            if self._is_s3:
                for f in tqdm(self.filepaths, desc="Opening files"):
                    ds = get_remote_h5(
                        str(f), page_size=self.page_size, aws_credentials=creds
                    )
                    self.datasets.append(ds)
            elif self._is_http:
                from osgeo import gdal

                for f in tqdm(self.filepaths, desc="Opening files"):
                    gdal_str = f'HDF5:"/vsicurl/{f}"://{self.dset_name.strip("/")}'
                    ds = gdal.Open(gdal_str)
                    self.datasets.append(ds)
            else:
                # Local files
                for f in tqdm(self.filepaths, desc="Opening files"):
                    ds = h5py.File(str(f), "r")
                    self.datasets.append(ds)

        self._opened = True

    @staticmethod
    def _worker_main(
        worker_id: int,
        urls: list[str],
        dset_name: str,
        aws_credentials: Any,
        request_queue: mp.Queue,
        result_queue: mp.Queue,
    ) -> None:
        """Worker process function for multiprocessing mode.

        Opens all the URLs assigned to this worker exactly once.
        Then repeatedly waits for read requests on `request_queue`.
        For each request, reads the slice from the appropriate open file
        and puts the result on `result_queue`.

        Parameters
        ----------
        worker_id : int
            ID of the worker process.
        urls : list[str]
            List of URLs to read.
        dset_name : str
            Name of the dataset to read.
        aws_credentials : AWSCredentials
            AWS credentials for accessing S3.
        request_queue : mp.Queue
            Queue for read requests.
        result_queue : mp.Queue
            Queue for results.
        """
        # Open each file once
        file_handles = {}
        for url in urls:
            try:
                file_handles[url] = get_remote_h5(url, aws_credentials=aws_credentials)
            except Exception as e:
                logger.error(f"[Worker {worker_id}] Failed to open {url}: {str(e)}")
                file_handles[url] = None

        logger.debug(f"[Worker {worker_id}] opened {len(file_handles)} files")

        while True:
            msg = request_queue.get()
            if msg is None:
                # Sentinel -> time to shut down
                break

            idx, url, slice_obj = msg

            # Read the data
            fh = file_handles.get(url)
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

    def __getitem__(self, key) -> np.ndarray:
        """Get a slice of data from the stack.

        Parameters
        ----------
        key : tuple[slice | int]
            Tuple of slices/indices into (time, y, x) dimensions.

        Returns
        -------
        np.ndarray
            Data transformed from relative to absolute displacements.

        Raises
        ------
        RuntimeError
            If there's an error reading the data.
        """
        if not self._opened:
            logger.debug("Reader not opened yet, opening now...")
            self.open()

        # Extract time and spatial keys
        if isinstance(key, tuple):
            # time_key = key[0]
            spatial_key = key[1:]
        else:
            # time_key = key
            spatial_key = (slice(None), slice(None))

        if self.use_multiprocessing:
            # Submit tasks to worker processes
            for i, url in enumerate(self.filepaths):
                w_id = self.url_to_worker[str(url)]
                self.task_queues[w_id].put((i, str(url), spatial_key))

            # Collect results with progress tracking
            results = [None] * len(self.filepaths)
            for _ in tqdm(range(len(self.filepaths)), desc="Reading data"):
                i, data = self.result_queue.get()
                if isinstance(data, str):  # Error message
                    raise RuntimeError(f"Error processing file {i}: {data}")
                results[i] = data

            # Stack the results
            data = np.stack(results)
        else:
            # Read the data from each file directly
            data = []
            for ds in tqdm(self.datasets, desc="Reading data"):
                if self._is_s3:
                    data.append(ds[self.dset_name.value][spatial_key])
                else:
                    # TODO: Need to store the image size to cut off edges same as slice
                    xoff, yoff, xsize, ysize = _slice_to_offsets(
                        *spatial_key,
                    )
                    data.append(ds.ReadAsArray(xoff, yoff, xsize, ysize))
            data = np.stack(data)

        # Transform from relative to absolute displacements
        if self.zero_nans:
            np.nan_to_num(data, copy=False)
        cur_shape = data.shape
        return (self.incidence_pinv @ data.reshape(cur_shape[0], -1)).reshape(cur_shape)

    def close(self) -> None:
        """Close all open datasets or worker processes."""
        if not self._opened:
            return

        if self.use_multiprocessing:
            # Send sentinel values to stop workers
            for q in self.task_queues:
                q.put(None)

            # Wait for workers to finish with timeout
            for p in self.workers:
                p.join(timeout=1.0)
                if p.is_alive():
                    p.terminate()

            # Clean up
            self.workers = []
            logger.debug("All worker processes stopped")
        else:
            # Close all open datasets
            for ds in self.datasets:
                ds.close()
            self.datasets = []

        self._opened = False

    def __del__(self):
        """Ensure resources are cleaned up when the object is garbage collected."""
        self.close()


def _slice_to_offsets(rows: slice, cols: slice) -> tuple[int, int, int, int]:
    xoff, yoff = int(cols.start), int(rows.start)
    # row_stop = min(rows.stop, nrows)
    # col_stop = min(cols.stop, ncols)
    row_stop = rows.stop
    col_stop = cols.stop
    xsize, ysize = int(col_stop - cols.start), int(row_stop - rows.start)
    return xoff, yoff, xsize, ysize
