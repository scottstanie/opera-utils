from __future__ import annotations

import datetime
from typing import Any, Callable

import h5py
import numpy as np
from pyproj import CRS, Transformer
from typing_extensions import TypeAlias

from ._types import Filename

__all__ = [
    "get_zero_doppler_time",
    "get_radar_wavelength",
    "get_orbit_arrays",
    "get_xy_coords",
    "get_lonlat_grid",
    "get_cslc_orbit",
]

FilenameOrHandle: TypeAlias = Filename | h5py.File


def get_zero_doppler_time(
    filename: FilenameOrHandle, type_: str = "start"
) -> datetime.datetime:
    """Get the full acquisition time from the CSLC product.

    Uses `/identification/zero_doppler_{type_}_time` from the CSLC product.

    Parameters
    ----------
    filename : FilenameOrHandle
        Path to the CSLC product.
    type_ : str, optional
        Either "start" or "stop", by default "start".

    Returns
    -------
    str
        Full acquisition time.
    """
    DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

    def get_dt(in_str):
        return datetime.datetime.strptime(in_str.decode("utf-8"), DATETIME_FORMAT)

    dset = f"/identification/zero_doppler_{type_}_time"
    value = _get_dset_and_attrs(filename, dset, parse_func=get_dt)[0]
    return value


def _get_dset_and_attrs(
    filename: FilenameOrHandle,
    dset_name: str,
    parse_func: Callable = lambda x: x,
) -> tuple[Any, dict[str, Any]]:
    """Get one dataset's value and attributes from the CSLC product.

    Parameters
    ----------
    filename : FilenameOrHandle
        Path to the CSLC product.
    dset_name : str
        Name of the dataset.
    parse_func : Callable, optional
        Function to parse the dataset value, by default lambda x: x
        For example, could be parse_func=lambda x: x.decode("utf-8") to decode,
        or getting a datetime object from a string.

    Returns
    -------
    dset : Any
        The value of the scalar
    attrs : dict
        Attributes.
    """

    def _process(hf):
        dset = hf[dset_name]
        attrs = dict(dset.attrs)
        value = parse_func(dset[()])
        return value, attrs

    if isinstance(filename, h5py.File):
        # Don't close it if they pass the opened `File`
        return _process(filename)
    else:
        with h5py.File(filename, "r") as hf:
            return _process(hf)


def get_radar_wavelength(filename: FilenameOrHandle):
    """Get the radar wavelength from the CSLC product.

    Parameters
    ----------
    filename : FilenameOrHandle
        Path to the CSLC product.

    Returns
    -------
    wavelength : float
        Radar wavelength in meters.
    """
    dset = "/metadata/processing_information/input_burst_metadata/wavelength"
    value = _get_dset_and_attrs(filename, dset)[0]
    return value


def get_orbit_arrays(
    h5file: FilenameOrHandle,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, datetime.datetime]:
    """Parse orbit info from OPERA S1 CSLC HDF5 file into python types.

    Parameters
    ----------
    h5file : FilenameOrHandle
        Path to OPERA S1 CSLC HDF5 file.

    Returns
    -------
    times : np.ndarray
        Array of times in seconds since reference epoch.
    positions : np.ndarray
        Array of positions in meters.
    velocities : np.ndarray
        Array of velocities in meters per second.
    reference_epoch : datetime.datetime
        Reference epoch of orbit.

    """
    input_is_file = isinstance(h5file, h5py.File)
    hf = h5file if input_is_file else h5py.File(h5file)

    orbit_group = hf["/metadata/orbit"]
    times = orbit_group["time"][:]
    positions = np.stack([orbit_group[f"position_{p}"] for p in ["x", "y", "z"]]).T
    velocities = np.stack([orbit_group[f"velocity_{p}"] for p in ["x", "y", "z"]]).T
    reference_epoch = datetime.datetime.fromisoformat(
        orbit_group["reference_epoch"][()].decode()
    )

    if not input_is_file:
        hf.close()

    return times, positions, velocities, reference_epoch


def get_cslc_orbit(h5file: FilenameOrHandle):
    """Parse orbit info from OPERA S1 CSLC HDF5 file into an isce3.core.Orbit.

    `isce3` must be installed to use this function.

    Parameters
    ----------
    h5file : FilenameOrHandle
        Path to OPERA S1 CSLC HDF5 file.

    Returns
    -------
    orbit : isce3.core.Orbit
        Orbit object.

    """
    from isce3.core import DateTime, Orbit, StateVector

    times, positions, velocities, reference_epoch = get_orbit_arrays(h5file)
    orbit_svs = []

    for t, x, v in zip(times, positions, velocities):
        orbit_svs.append(
            StateVector(
                DateTime(reference_epoch + datetime.timedelta(seconds=t)),
                x,
                v,
            )
        )

    return Orbit(orbit_svs)


def get_xy_coords(h5file: FilenameOrHandle, subsample: int = 100) -> tuple:
    """Get x and y grid from OPERA S1 CSLC HDF5 file.

    Parameters
    ----------
    h5file : FilenameOrHandle
        Path to OPERA S1 CSLC HDF5 file.
    subsample : int, optional
        Subsampling factor, by default 100

    Returns
    -------
    x : np.ndarray
        Array of x coordinates in meters.
    y : np.ndarray
        Array of y coordinates in meters.
    projection : int
        EPSG code of projection.

    """
    input_is_file = isinstance(h5file, h5py.File)
    hf = h5file if input_is_file else h5py.File(h5file)

    x = hf["/data/x_coordinates"][:]
    y = hf["/data/y_coordinates"][:]
    projection = hf["/data/projection"][()]

    if not input_is_file:
        hf.close()
    return x[::subsample], y[::subsample], projection


def get_lonlat_grid(
    h5file: FilenameOrHandle, subsample: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Get 2D latitude and longitude grid from OPERA S1 CSLC HDF5 file.

    Parameters
    ----------
    h5file : FilenameOrHandle
        Path to OPERA S1 CSLC HDF5 file.
    subsample : int, optional
        Subsampling factor, by default 100

    Returns
    -------
    lat : np.ndarray
        2D Array of latitude coordinates in degrees.
    lon : np.ndarray
        2D Array of longitude coordinates in degrees.
    projection : int
        EPSG code of projection.

    """
    x, y, projection = get_xy_coords(h5file, subsample)
    X, Y = np.meshgrid(x, y)
    xx = X.flatten()
    yy = Y.flatten()
    crs = CRS.from_epsg(projection)
    transformer = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
    lon, lat = transformer.transform(xx=xx, yy=yy, radians=True)
    lon = lon.reshape(X.shape)
    lat = lat.reshape(Y.shape)
    return lon, lat
