import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import requests
import tyro
from pydantic import BaseModel, StringConstraints
from rich.console import Console
from rich.table import Table

logger = logging.getLogger("opera_utils")


class Granule(BaseModel):
    """
    Model representing a single granule from CMR.

    Attributes
    ----------
    frame_id : str | None
        The frame number of the granule.
    url : str
        URL (https or s3) containing granule download location.
    ascending_descending : str
        The direction (ascending or descending).
    start : datetime
        The beginning date/time of the granule.
    end : datetime
        The ending date/time of the granule.
    """

    frame_id: int
    url: str
    ascending_descending: Annotated[str, StringConstraints(to_lower=True)]
    start: datetime
    end: datetime

    @classmethod
    def from_umm(
        cls,
        umm_data: dict[str, Any],
        url_type: Literal["s3", "https"] = "https",
    ) -> "Granule":
        """
        Construct a Granule instance from a raw dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            The raw granule data from the CMR API.

        Returns
        -------
        Granule
            The parsed Granule instance.

        Raises
        ------
        ValueError
            If required temporal extent data is missing.
        """
        url = get_download_url(umm_data, protocol=url_type)
        additional_attributes = umm_data.get("AdditionalAttributes", [])
        frame = _get_attr(additional_attributes, "FRAME_NUMBER")
        direction = _get_attr(additional_attributes, "ASCENDING_DESCENDING")
        temporal_extent = umm_data.get("TemporalExtent", {}).get("RangeDateTime", {})
        begin_str = temporal_extent.get("BeginningDateTime")
        end_str = temporal_extent.get("EndingDateTime")
        if begin_str is None or end_str is None:
            raise ValueError("Missing temporal extent data")
        return cls(
            frame_id=frame,
            url=url,
            ascending_descending=direction,
            start=begin_str,
            end=end_str,
        )


def get_download_url(
    umm_data: dict[str, Any],
    protocol: Literal["s3", "https"] = "https",
) -> str:
    """Extract a download URL from the product's UMM metadata.

    Parameters
    ----------
    product : dict[str, Any]
        The product's umm metadata dictionary
    protocol : Literal["s3", "https"]
        The protocol to use for downloading, either "s3" or "https"

    Returns
    -------
    str
        The download URL

    Raises
    ------
    ValueError
        If no URL with the specified protocol is found or if the protocol is invalid
    """
    if protocol not in ["https", "s3"]:
        raise ValueError(f"Unknown protocol {protocol}; must be https or s3")

    for url in umm_data["RelatedUrls"]:
        if url["Type"].startswith("GET DATA") and url["URL"].startswith(protocol):
            return url["URL"]

    raise ValueError(f"No download URL found for granule {product['umm']['GranuleUR']}")


def _get_attr(attrs: list[dict[str, Any]], name: str) -> str | None:
    """Get the first attribute value for a given name."""
    for attr in attrs:
        if attr.get("Name") == name:
            values = attr.get("Values", [])
            if values:
                return values[0]
    return None


class FrameGranuleData(BaseModel):
    """Aggregated metadata for a specific frame.

    Attributes
    ----------
    frame_id : str
        The frame number.
    ascending_descending : Optional[str]
        The direction (ASCENDING or DESCENDING).
    granule_count : int
        The number of granules in this frame.
    start : str
        The earliest date (YYYY-MM-DD) of the granules in this frame.
    end : str
        The latest date (YYYY-MM-DD) of the granules in this frame.
    """

    frame_id: int
    ascending_descending: Annotated[str, StringConstraints(to_lower=True)]
    granule_count: int
    start: datetime
    end: datetime


def fetch_granules() -> list[Granule]:
    """
    Fetch DISP-S1 granule metadata from the CMR API and parse it into Granule objects.

    Returns
    -------
    list[Granule]
        A list of parsed Granule instances.
    """
    url = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
    params = {"short_name": "OPERA_L3_DISP-S1_V1", "page_size": "2000"}
    headers: dict[str, str] = {}
    granules_raw: list[dict[str, Any]] = []

    while True:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        items = data.get("items", [])
        granules_raw.extend(items)
        logger.debug(len(granules_raw))
        search_after = response.headers.get("CMR-Search-After")
        if not search_after:
            break
        headers["CMR-Search-After"] = search_after

    granules: list[Granule] = []
    for item in granules_raw:
        try:
            granule = Granule.from_umm(item["umm"])
            granules.append(granule)
        except Exception as e:
            logging.warning(f"Skipping granule due to error: {e}")
    return granules


def aggregate_frames(granules: list[Granule]) -> list[FrameGranuleData]:
    """
    Group granules by frame number and aggregate metadata for each frame.

    Parameters
    ----------
    granules : list[Granule]
        list of Granule objects.

    Returns
    -------
    list[FrameGranuleData]
        A list of aggregated frame metadata.
    """
    grouped: dict[str, list[Granule]] = defaultdict(list)
    for granule in granules:
        frame = granule.frame_id
        grouped[frame].append(granule)

    aggregated_data: list[FrameGranuleData] = []
    for frame_id, group in grouped.items():
        direction = group[0].ascending_descending
        granule_count = len(group)
        start_date = min(g.start for g in group).date().isoformat()
        end_date = max(g.end for g in group).date().isoformat()
        aggregated_data.append(
            FrameGranuleData(
                frame_id=frame_id,
                ascending_descending=direction,
                granule_count=granule_count,
                start=start_date,
                end=end_date,
            )
        )
    return aggregated_data


def main(
    save_to: Path | None = None, print_output: bool = True
) -> tuple[list[Granule], list[FrameGranuleData]]:
    """Fetch and aggregate DISP-S1 granule metadata from CMR.

    Parameters
    ----------
    print_output : bool, optional
        If True, prints the aggregated results in a formatted table using rich, by default False.

    Returns
    -------
    list[FrameGranuleData]
        A list of aggregated frame metadata.
    """
    granules = fetch_granules()
    aggregated = aggregate_frames(granules)

    if print_output:
        console = Console()
        table = Table(title="Aggregated DISP-S1 Granule Metadata")
        table.add_column("Direction")
        table.add_column("Frame")
        table.add_column("Granule Count", justify="right")
        table.add_column("Start")
        table.add_column("End")
        for data in aggregated:
            table.add_row(
                data.ascending_descending,
                str(data.frame_id),
                str(data.granule_count),
                str(data.start),
                str(data.end),
            )
        console.print(table)

    return granules, aggregated


if __name__ == "__main__":
    tyro.cli(main)
