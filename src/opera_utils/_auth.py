import netrc
import os

import aiohttp
import boto3.s3
import cachetools.func
import fsspec
import h5py
import requests
import s3fs

from .credentials import ENDPOINTS, AWSCredentials


# For instance, 50 minutes so that credentials are refreshed at least 10 minutes
# before they're set to expire.
@cachetools.func.ttl_cache(ttl=60 * 50)
def get_earthaccess_s3_creds(dataset: str = "opera") -> AWSCredentials:
    """Get S3 credentials for the specified dataset.

    Parameters
    ----------
    dataset : str, optional
        The name of the dataset to get credentials for.
        Options are "opera" or "sentinel1". Default is "opera".

    Returns
    -------
    AWSCredentials
        Object containing S3 credentials

    Raises
    ------
    ValueError
        If an unknown dataset is specified.

    Notes
    -----
    Uses the `earthaccess` library to login, which requires one of the following
    auth strategies:
        - "all": (default) try all methods until one works
        - "interactive": enter username and password.
        - "netrc": retrieve username and password from ~/.netrc.
        - "environment": retrieve username and password from
            `$EARTHDATA_USERNAME` and `$EARTHDATA_PASSWORD`.

    """
    import earthaccess

    auth = earthaccess.login()
    if dataset not in set(ENDPOINTS.keys()):
        raise ValueError(f"Unknown dataset: {dataset}")
    return AWSCredentials(**auth.get_s3_credentials(endpoint=ENDPOINTS[dataset]))


@cachetools.func.ttl_cache(ttl=60 * 50)
def get_authorized_s3_client(
    dataset: str = "opera",
    aws_credentials: AWSCredentials | None = None,
):
    """Get an authorized S3 client for the specified dataset.

    Parameters
    ----------
    dataset : str, optional
        The name of the dataset to get credentials for. Default is "opera".
    aws_credentials : AWSCredentials, optional
        Pre-configured s3 credentials.
        If not provided, fetches using earthaccess

    Returns
    -------
    boto3.S3Client
        An authorized S3 client.

    """
    if aws_credentials is None:
        aws_credentials = get_earthaccess_s3_creds(dataset=dataset)

    return boto3.client(
        "s3",
        aws_access_key_id=aws_credentials.access_key_id,
        aws_secret_access_key=aws_credentials.secret_access_key,
        aws_session_token=aws_credentials.session_token,
        region_name="us-west-2",
    )


def get_aws_session(
    dataset: str = "opera",
    aws_credentials: AWSCredentials | None = None,
) -> boto3.Session:
    """Create a Rasterio AWS Session with AWSCredentials."""
    if aws_credentials is None:
        aws_credentials = get_earthaccess_s3_creds(dataset=dataset)
    return boto3.Session(
        aws_access_key_id=aws_credentials.access_key_id,
        aws_secret_access_key=aws_credentials.secret_access_key,
        aws_session_token=aws_credentials.session_token,
        region_name="us-west-2",
    )


def get_frozen_credentials(
    aws_credentials: AWSCredentials | None = None, dataset: str = "opera"
) -> tuple[str, str, str]:
    if aws_credentials is None:
        session = get_aws_session(dataset=dataset)
    else:
        session = aws_credentials.to_session()
    current_creds = session.get_credentials()

    frozen_creds = current_creds.get_frozen_credentials()
    return frozen_creds.access_key, frozen_creds.secret_key, frozen_creds.token


def set_s3_creds(
    access_key_id: str, secret_access_key: str, session_token: str
) -> None:
    """Set S3 credentials as environment variables.

    Parameters
    ----------
    access_key_id : str
        The AWS access key ID.
    secret_access_key : str
        The AWS secret access key.
    session_token : str
        The AWS session token.

    """
    d = {
        "AWS_ACCESS_KEY_ID": access_key_id,
        "AWS_SECRET_ACCESS_KEY": secret_access_key,
        "AWS_SESSION_TOKEN": session_token,
    }
    for env_name, val in d.items():
        os.environ[env_name] = val


def print_export(dataset: str = "opera") -> None:
    """Print export commands for S3 credentials.

    Parameters
    ----------
    dataset : str, optional
        The name of the dataset to get credentials for. Default is "opera".

    """
    creds = get_earthaccess_s3_creds(dataset)
    for env_name, val in creds.to_env().items():
        print(f"export {env_name}='{val}'")


# Set TTL to number of seconds to cache.
@cachetools.func.ttl_cache(ttl=60 * 50)
def get_temporary_aws_credentials() -> dict:
    """Get temporary AWS S3 access credentials.

    Requests new credentials if credentials are expired, or gets from the cache.

    Assumes Earthdata Login credentials are available via a .netrc file,
     or via EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables.

    Returns
    -------
    dict:
        JSON reqponse from s3credentials URL.

    """
    resp = requests.get("https://cumulus.asf.alaska.edu/s3credentials")
    if resp.status_code == 401 and resp.url.startswith(
        "https://urs.earthdata.nasa.gov/oauth/authorize?"
    ):
        auth = (os.environ["EARTHDATA_USERNAME"], os.environ["EARTHDATA_PASSWORD"])
        resp = requests.get(resp.url, auth=auth)
    resp.raise_for_status()
    return resp.json()


@cachetools.func.ttl_cache(ttl=60 * 50)
def get_temporary_s3_fs() -> s3fs.S3FileSystem:
    creds = get_temporary_aws_credentials()
    s3_fs = s3fs.S3FileSystem(
        key=creds["accessKeyId"],
        secret=creds["secretAccessKey"],
        token=creds["sessionToken"],
    )
    return s3_fs


def get_https_fs(host="urs.earthdata.nasa.gov", protocol="http"):
    """Create an fsspec filesystem object authenticated using netrc for HTTP access.

    Parameters
    ----------
    protocol : str, optional
        The protocol to use for the filesystem, by default "http".
    use_netrc : bool, optional
        Whether to use .netrc for authentication, by default True.
    host : str, optional
        The host for which to authenticate using netrc.
        Default is "urs.earthdata.nasa.gov".

    Returns
    -------
    fsspec.AbstractFileSystem
        An authenticated fsspec filesystem object for HTTP access.

    Raises
    ------
    ValueError
        If the .netrc file does not contain authentication information for the
        specified host.

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
        If the .netrc file does not contain authentication information for the
        specified host.

    """
    if url.startswith("http"):
        fs = get_https_fs()
    elif url.startswith("s3://"):
        fs = get_temporary_s3_fs()
    else:
        raise ValueError(f"Unrecognized scheme for {url}")

    byte_stream = fs.open(path=url, mode="rb", cache_type="first")
    return h5py.File(byte_stream)


if __name__ == "__main__":
    import logging
    import sys

    logging.basicConfig()
    print_export(sys.argv[1])
