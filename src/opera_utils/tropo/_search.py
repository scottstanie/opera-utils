"""Search for OPERA TROPO products from CMR.

Examples
--------
>>> from opera_utils.tropo import search_tropo
>>> from datetime import datetime
>>> urls = search_tropo(
...     start_datetime=datetime(2024, 6, 1),
...     end_datetime=datetime(2024, 6, 30),
... )

"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from enum import Enum

import requests

__all__ = ["TropoProduct", "search_tropo"]

logger = logging.getLogger("opera_utils")

TROPO_FILE_REGEX = re.compile(
    r"OPERA_L4_TROPO-ZENITH_"
    r"(?P<start_datetime>\d{8}T\d{6})Z_"
    r"(?P<end_datetime>\d{8}T\d{6})Z_"
    r"(?P<resolution>HRES|LRES)_"
    r"v(?P<version>[\d.]+)"
    r"\.nc$"
)


class UrlType(str, Enum):
    """Choices for URL protocol type."""

    S3 = "s3"
    HTTPS = "https"

    def __str__(self) -> str:
        return str(self.value)


class TropoProduct:
    """Class representing a single TROPO product."""

    def __init__(
        self,
        url: str,
        start_datetime: datetime,
        end_datetime: datetime,
        resolution: str,
        version: str,
    ) -> None:
        self.url = url
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.resolution = resolution
        self.version = version

    @classmethod
    def from_url(cls, url: str) -> TropoProduct:
        """Parse a TROPO product URL to extract metadata."""
        # Extract filename from URL
        filename = url.rsplit("/", maxsplit=1)[-1]
        match = TROPO_FILE_REGEX.match(filename)
        assert match is not None, f"Invalid TROPO filename: {filename}"

        def _to_datetime(dt: str) -> datetime:
            return datetime.strptime(dt, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)

        data = match.groupdict()
        return cls(
            url=url,
            start_datetime=_to_datetime(data["start_datetime"]),
            end_datetime=_to_datetime(data["end_datetime"]),
            resolution=data["resolution"],
            version=data["version"],
        )

    def __repr__(self) -> str:
        return f"TropoProduct({self.start_datetime.isoformat()}, {self.resolution})"


def search_tropo(
    start_datetime: datetime,
    end_datetime: datetime,
    resolution: str = "HRES",
    url_type: UrlType = UrlType.HTTPS,
    use_uat: bool = False,
) -> list[str]:
    """Search CMR for OPERA TROPO products in a date range.

    TROPO products are global, so no spatial filtering is needed.
    Products are released every 6 hours.

    Parameters
    ----------
    start_datetime : datetime
        Start of the temporal range (UTC).
    end_datetime : datetime
        End of the temporal range (UTC).
    resolution : str
        Resolution of TROPO products: "HRES" (high resolution) or "LRES".
        Default is "HRES".
    url_type : UrlType
        Protocol for URLs: "s3" for S3 URLs, "https" for HTTPS URLs.
        Default is HTTPS.
    use_uat : bool
        Whether to use the UAT environment instead of production Earthdata.
        Default is False.

    Returns
    -------
    list[str]
        List of TROPO product URLs covering the requested time range,
        sorted by start datetime.

    """
    # Expand the search window to ensure we get bracketing files
    # TROPO products are every 6 hours, so add a 6-hour buffer on each side
    buffer = timedelta(hours=6)
    search_start = start_datetime - buffer
    search_end = end_datetime + buffer

    edl_host = "uat.earthdata" if use_uat else "earthdata"
    search_url = f"https://cmr.{edl_host}.nasa.gov/search/granules.umm_json"

    params: dict[str, int | str | list[str]] = {
        "short_name": "OPERA_L4_TROPO-ZENITH_V1",
        "page_size": 500,
    }

    # Add temporal range
    start_str = search_start.isoformat() if search_start else ""
    end_str = search_end.isoformat() if search_end else ""
    params["temporal"] = f"{start_str},{end_str}"

    headers: dict[str, str] = {}
    urls: list[str] = []

    while True:
        response = requests.get(search_url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()

        for item in data["items"]:
            url = _get_tropo_url(item["umm"], protocol=url_type)
            # Filter by resolution if specified
            if resolution:
                product = TropoProduct.from_url(url)
                if product.resolution != resolution:
                    continue
            urls.append(url)

        if "CMR-Search-After" not in response.headers:
            break

        headers["CMR-Search-After"] = response.headers["CMR-Search-After"]

    # Sort by start datetime (parsed from URL)
    urls = sorted(urls, key=lambda u: TropoProduct.from_url(u).start_datetime)
    return urls


def _get_tropo_url(umm_data: dict, protocol: UrlType = UrlType.HTTPS) -> str:
    """Extract a download URL from the product's UMM metadata.

    Parameters
    ----------
    umm_data : dict
        The product's UMM metadata dictionary.
    protocol : UrlType
        The protocol to use for downloading, either "s3" or "https".

    Returns
    -------
    str
        The download URL for the .nc file.

    """
    for url_info in umm_data.get("RelatedUrls", []):
        url = url_info.get("URL", "")
        url_type = url_info.get("Type", "")
        if (
            url_type.startswith("GET DATA")
            and url.endswith(".nc")
            and url.startswith(str(protocol))
        ):
            return url

    # Fallback: find any .nc URL
    for url_info in umm_data.get("RelatedUrls", []):
        url = url_info.get("URL", "")
        if url.endswith(".nc"):
            return url

    msg = (
        "No TROPO download URL found for granule"
        f" {umm_data.get('GranuleUR', 'unknown')}"
    )
    raise ValueError(msg)
