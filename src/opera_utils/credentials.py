import os
from datetime import datetime

from pydantic import BaseModel, Field


class AWSCredentials(BaseModel):
    """Class for AWS credentials (accessKeyId, secretAccessKey, sessionToken)."""

    access_key_id: str = Field(alias="accessKeyId")  # use aliases to match JSON case
    secret_access_key: str = Field(alias="secretAccessKey")
    session_token: str = Field(alias="sessionToken")
    expiration: datetime | None = None


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
    if dataset == "opera":
        return _generate_earthaccess_s3_creds_opera()
    elif dataset == "sentinel1":
        return _generate_earthaccess_s3_creds_sentinel_level1()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def _generate_earthaccess_s3_creds_opera() -> AWSCredentials:
    """Get S3 credentials for OPERA dataset.

    Returns
    -------
    AWSCredentials

    """
    import earthaccess

    auth = earthaccess.login()
    results = earthaccess.search_datasets(daac="ASF", keyword="OPERA_L2_CSLC-S1_V1")
    bucket = results[0].s3_bucket()
    creds = auth.get_s3_credentials(endpoint=bucket["S3CredentialsAPIEndpoint"])
    return AWSCredentials(**creds)


def _generate_earthaccess_s3_creds_sentinel_level1() -> AWSCredentials:
    """Get S3 credentials for Sentinel-1 Level 1 dataset.

    Returns
    -------
    AWSCredentials

    """
    import earthaccess

    earthaccess.login()
    return earthaccess.get_s3_credentials(daac="ASF")


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
    import boto3

    if aws_credentials is None:
        aws_credentials = get_earthaccess_s3_creds(dataset=dataset)

    return boto3.client(
        "s3",
        aws_access_key_id=aws_credentials.access_key_id,
        aws_secret_access_key=aws_credentials.secret_access_key,
        aws_session_token=aws_credentials.session_token,
        region_name="us-west-2",
    )


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
    access_key_id, secret_access_key, session_token = get_earthaccess_s3_creds(dataset)
    d = {
        "AWS_ACCESS_KEY_ID": access_key_id,
        "AWS_SECRET_ACCESS_KEY": secret_access_key,
        "AWS_SESSION_TOKEN": session_token,
    }
    for env_name, val in d.items():
        print(f"export {env_name}='{val}'")
