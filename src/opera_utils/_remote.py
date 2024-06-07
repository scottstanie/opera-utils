import netrc

import aiohttp
import fsspec
import h5py


def get_fsspec_filesystem(protocol="http", host="urs.earthdata.nasa.gov"):
    """Create an fsspec filesystem object authenticated using netrc for HTTP access.

    Parameters
    ----------
    protocol : str, optional
        The protocol to use for the filesystem, by default "http".
    use_netrc : bool, optional
        Whether to use .netrc for authentication, by default True.
    host : str, optional
        The host for which to authenticate using netrc, by default "urs.earthdata.nasa.gov".

    Returns
    -------
    fsspec.AbstractFileSystem
        An authenticated fsspec filesystem object for HTTP access.

    Raises
    ------
    ValueError
        If the .netrc file does not contain authentication information for the specified host.
    """
    result = netrc.netrc().authenticators(host)
    if result is None:
        raise ValueError(".netrc file has no 'urs.earthdata.nasa.gov' entry")

    (username, _, password) = result
    fs = fsspec.filesystem(
        protocol, client_kwargs={"auth": aiohttp.BasicAuth(username, password)}
    )
    return fs


def get_h5py_handle(url: str) -> h5py.File:
    """Open an HDF5 file from a given URL and return an h5py.File object.

    Parameters
    ----------
    url : str
        The URL of the HDF5 file to be accessed.

    Returns
    -------
    h5py.File
        A handle to the HDF5 file.

    Raises
    ------
    ValueError
        If the .netrc file does not contain authentication information for the specified host.
    """
    fs = get_fsspec_filesystem()
    byte_stream = fs.open(path=url, mode="rb", cache_type="first")
    return h5py.File(byte_stream)
