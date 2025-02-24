import os
from datetime import datetime

import boto3.s3
from pydantic import BaseModel, Field

ENDPOINTS = {
    # ESA's SAFE granules via ASF:
    "sentinel1": "https://sentinel1.asf.alaska.edu/s3credentials",
    # OPERA products via ASF
    "opera": "https://cumulus.asf.alaska.edu/s3credentials",
    # Test OPERA products (the UAT venue, uat.urs.earthdata.nasa.gov)
    "opera-uat": "https://cumulus-test.asf.alaska.edu/s3credentials",
}


class AWSCredentials(BaseModel):
    """Class for AWS credentials (accessKeyId, secretAccessKey, sessionToken)."""

    access_key_id: str = Field(alias="accessKeyId")
    secret_access_key: str = Field(alias="secretAccessKey")
    session_token: str = Field(alias="sessionToken")
    expiration: datetime | None = None

    def to_session(self) -> boto3.Session:
        return boto3.Session(
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            aws_session_token=self.session_token,
            region_name="us-west-2",
        )

    def to_env(self) -> dict[str, str]:
        """Return the environment variable format of values: `AWS_`.

        Settable using os.environ.
        """
        return {
            "AWS_ACCESS_KEY_ID": self.access_key_id,
            "AWS_SECRET_ACCESS_KEY": self.secret_access_key,
            "AWS_SESSION_TOKEN": self.session_token,
        }


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


if __name__ == "__main__":
    import logging
    import sys

    logging.basicConfig()
    print_export(sys.argv[1])
