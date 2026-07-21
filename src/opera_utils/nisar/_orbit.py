"""Search for and download NISAR orbit ephemeris files.

NISAR distributes four flavors of orbit ephemeris as ancillary XML files, which
differ in latency and accuracy:

============ ================================ ==================
Product type Description                      Typical latency
============ ================================ ==================
POE          Precise Orbit Ephemeris          days to weeks
MOE          Medium-precision Orbit Ephemeris hours
NOE          Near-real-time Orbit Ephemeris   minutes to hours
FOE          Forecast Orbit Ephemeris         predicted forward
============ ================================ ==================

Each file covers a fixed window of validity encoded in its name, and a granule
can only be processed with an orbit whose window fully contains the granule's
acquisition (plus some padding for interpolation at the edges).

Note that the orbit files ASF Vertex lists alongside an L0/L1 granule are *not*
filtered by validity window, so they routinely do not cover the granule they are
attached to. Matching on the validity window, as this module does, is the
reliable approach.

Examples
--------
$ python -m opera_utils.nisar._orbit NISAR_L0_PR_RRSD_025_131_A_148S_20260717T063521_20260717T063549_P05023_F_J_001

"""  # noqa: E501

from __future__ import annotations

import logging
import netrc
import re
import shutil
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import requests
from tqdm.auto import tqdm
from typing_extensions import Self

from opera_utils._cmr import fetch_cmr_pages, get_download_url
from opera_utils._types import PathOrStr
from opera_utils.constants import NISAR_ORBIT_FILE_REGEX, UrlType

__all__ = [
    "NisarOrbit",
    "OrbitType",
    "download_orbits",
    "get_orbit_for_granule",
    "search_orbits",
]

logger = logging.getLogger("opera_utils")

# CMR collection holding all four orbit ephemeris types
NISAR_ORBIT_SHORT_NAME = "NISAR_OE"

_CMR_GRANULE_URL = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"

# Orbit interpolation near a file's edge degrades, so require the validity
# window to extend past the acquisition by this much on each side.
DEFAULT_PAD = timedelta(minutes=5)

_DATETIME_RE = re.compile(r"\d{8}T\d{6}")


class OrbitType(str, Enum):
    """Type of NISAR orbit ephemeris, ordered from most to least accurate."""

    POE = "POE"
    MOE = "MOE"
    NOE = "NOE"
    FOE = "FOE"

    def __str__(self) -> str:
        return str(self.value)

    @property
    def precedence(self) -> int:
        """Rank of this type when choosing a "best" orbit (0 is best)."""
        return _ORBIT_TYPE_PRECEDENCE[self]


# Preference order when several types cover the same acquisition.
_ORBIT_TYPE_PRECEDENCE = {
    OrbitType.POE: 0,
    OrbitType.MOE: 1,
    OrbitType.NOE: 2,
    OrbitType.FOE: 3,
}


def _to_datetime(dt: str) -> datetime:
    """Parse NISAR datetime string format (YYYYMMDDThhmmss)."""
    return datetime.strptime(dt, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


def _as_utc(dt: datetime) -> datetime:
    """Attach UTC to a naive datetime, or convert an aware one to UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True)
class NisarOrbit:
    """One NISAR orbit ephemeris file, parsed from its name."""

    name: str
    """Granule name, without the ``.xml`` suffix."""
    orbit_type: OrbitType
    creation_datetime: datetime
    """When the file was generated."""
    start_datetime: datetime
    """Start of the validity window."""
    end_datetime: datetime
    """End of the validity window."""
    url: str | None = None
    """Download URL. Constructed from `name` if not supplied by CMR."""
    size_in_bytes: int | None = None

    @classmethod
    def from_filename(cls, name: PathOrStr) -> Self:
        """Parse an orbit ephemeris filename.

        Parameters
        ----------
        name : PathOrStr
            Orbit filename or granule name. A ``.xml`` suffix is optional, and
            a full URL or path is accepted.

        Returns
        -------
        NisarOrbit
            The parsed orbit file information.

        Raises
        ------
        ValueError
            If the name is not a valid NISAR orbit ephemeris filename.

        Examples
        --------
        >>> orbit = NisarOrbit.from_filename(
        ...     "NISAR_ANC_J_PR_POE_20260702T171245_20260619T205942_20260621T025942.xml"
        ... )
        >>> orbit.orbit_type
        <OrbitType.POE: 'POE'>
        >>> orbit.start_datetime.isoformat()
        '2026-06-19T20:59:42+00:00'

        """
        stem = Path(str(name)).name.removesuffix(".xml")
        if not (match := re.match(NISAR_ORBIT_FILE_REGEX, stem)):
            msg = f"Invalid NISAR orbit filename format: {name}"
            raise ValueError(msg)

        g = match.groupdict()
        return cls(
            name=stem,
            orbit_type=OrbitType(g["orbit_type"]),
            creation_datetime=_to_datetime(g["creation_datetime"]),
            start_datetime=_to_datetime(g["start_datetime"]),
            end_datetime=_to_datetime(g["end_datetime"]),
        )

    @classmethod
    def from_umm(
        cls, umm_data: dict[str, Any], url_type: UrlType = UrlType.HTTPS
    ) -> Self:
        """Construct a `NisarOrbit` from a raw CMR UMM dictionary.

        Parameters
        ----------
        umm_data : dict[str, Any]
            The raw granule UMM data from the CMR API.
        url_type : UrlType
            Protocol to use for the download URL, either "s3" or "https".

        Returns
        -------
        NisarOrbit
            The parsed orbit, with `url` taken from the CMR metadata.

        """
        url = get_download_url(umm_data, protocol=url_type, filename_suffix=".xml")
        orbit = cls.from_filename(url)
        archive_info = umm_data.get("DataGranule", {}).get(
            "ArchiveAndDistributionInformation", []
        )
        size_in_bytes = archive_info[0].get("SizeInBytes") if archive_info else None
        # Frozen dataclass, so rebuild rather than mutate
        return cls(
            name=orbit.name,
            orbit_type=orbit.orbit_type,
            creation_datetime=orbit.creation_datetime,
            start_datetime=orbit.start_datetime,
            end_datetime=orbit.end_datetime,
            url=url,
            size_in_bytes=size_in_bytes,
        )

    @property
    def download_url(self) -> str:
        """HTTPS URL for the orbit XML, from CMR or built from the name."""
        if self.url is not None:
            return self.url
        return (
            "https://nisar.asf.earthdatacloud.nasa.gov/NISAR/"
            f"{self.orbit_type}/{self.name}/{self.name}.xml"
        )

    @property
    def filename(self) -> str:
        """Name of the orbit file on disk, including the ``.xml`` suffix."""
        return f"{self.name}.xml"

    def covers(
        self,
        start_datetime: datetime,
        end_datetime: datetime,
        pad: timedelta = DEFAULT_PAD,
    ) -> bool:
        """Check whether this orbit's validity window contains a time range.

        Parameters
        ----------
        start_datetime : datetime
            Start of the range to cover. Naive datetimes are assumed to be UTC.
        end_datetime : datetime
            End of the range to cover.
        pad : timedelta
            Extra margin required on each side of the range.
            Default is 5 minutes.

        Returns
        -------
        bool
            True if ``[start - pad, end + pad]`` falls inside the validity window.

        """
        return (
            self.start_datetime <= _as_utc(start_datetime) - pad
            and self.end_datetime >= _as_utc(end_datetime) + pad
        )


def parse_granule_datetimes(granule: PathOrStr) -> tuple[datetime, datetime]:
    """Extract the acquisition start and end times from a NISAR granule name.

    Works across product levels by pulling the timestamps out of the name
    directly, rather than requiring a full filename parse.

    Parameters
    ----------
    granule : PathOrStr
        NISAR granule name, with or without a file suffix. A full path or URL
        is accepted.

    Returns
    -------
    tuple[datetime, datetime]
        The acquisition start and end times, in UTC.

    Raises
    ------
    ValueError
        If the name does not contain at least two ``YYYYMMDDThhmmss`` timestamps.

    Notes
    -----
    GUNW filenames carry four timestamps (reference and secondary acquisitions);
    only the first pair, the reference acquisition, is returned.

    Examples
    --------
    >>> start, end = parse_granule_datetimes(
    ...     "NISAR_L0_PR_RRSD_025_131_A_148S_20260717T063521_20260717T063549"
    ...     "_P05023_F_J_001"
    ... )
    >>> start.isoformat()
    '2026-07-17T06:35:21+00:00'

    """
    stem = Path(str(granule)).name
    matches = _DATETIME_RE.findall(stem)
    if len(matches) < 2:
        msg = (
            f"Could not find an acquisition time range in {granule!r}: "
            f"expected at least two YYYYMMDDThhmmss timestamps, found {len(matches)}"
        )
        raise ValueError(msg)

    start_datetime = _to_datetime(matches[0])
    end_datetime = _to_datetime(matches[1])
    if end_datetime < start_datetime:
        msg = f"Parsed end time before start time in {granule!r}"
        raise ValueError(msg)
    return start_datetime, end_datetime


def search_orbits(
    start_datetime: datetime,
    end_datetime: datetime,
    orbit_types: Sequence[OrbitType] | None = None,
    pad: timedelta = DEFAULT_PAD,
    url_type: UrlType = UrlType.HTTPS,
) -> list[NisarOrbit]:
    """Query CMR for orbit files covering a time range.

    Parameters
    ----------
    start_datetime : datetime
        Start of the range that must be covered. Naive datetimes are UTC.
    end_datetime : datetime
        End of the range that must be covered.
    orbit_types : Sequence[OrbitType] | None
        Restrict the search to these types. If None, searches all four.
    pad : timedelta
        Extra margin the orbit must cover on each side of the range.
        Default is 5 minutes.
    url_type : UrlType
        Protocol to use for the download URLs, either "s3" or "https".

    Returns
    -------
    list[NisarOrbit]
        Orbits whose validity window fully covers the padded range, sorted best
        first: by type precedence (POE, MOE, NOE, FOE), then most recently
        created.

    Examples
    --------
    >>> from datetime import datetime
    >>> orbits = search_orbits(  # doctest: +SKIP
    ...     datetime(2026, 6, 20, 12), datetime(2026, 6, 20, 12, 1)
    ... )

    """
    start_datetime = _as_utc(start_datetime)
    end_datetime = _as_utc(end_datetime)

    params: dict[str, int | str | list[str]] = {
        "short_name": NISAR_ORBIT_SHORT_NAME,
        "provider": "ASF",
        "page_size": 500,
        # CMR returns any granule *intersecting* this range; containment,
        # including `pad`, is enforced below.
        "temporal": f"{start_datetime.isoformat()},{end_datetime.isoformat()}",
    }
    if orbit_types is not None:
        params["attribute[]"] = [
            f"string,PRODUCT_TYPE,{OrbitType(t)}" for t in orbit_types
        ]

    items = fetch_cmr_pages(_CMR_GRANULE_URL, params)
    logger.debug(f"CMR returned {len(items)} orbit granules before filtering")

    orbits = [NisarOrbit.from_umm(item["umm"], url_type=url_type) for item in items]
    covering = [o for o in orbits if o.covers(start_datetime, end_datetime, pad=pad)]

    return sorted(
        covering,
        key=lambda o: (o.orbit_type.precedence, -o.creation_datetime.timestamp()),
    )


def get_orbit_for_granule(
    granule: PathOrStr,
    orbit_types: Sequence[OrbitType] | None = None,
    pad: timedelta = DEFAULT_PAD,
    url_type: UrlType = UrlType.HTTPS,
) -> NisarOrbit:
    """Find the best orbit file covering a NISAR granule's acquisition.

    Parameters
    ----------
    granule : PathOrStr
        NISAR granule name, with or without a file suffix.
    orbit_types : Sequence[OrbitType] | None
        Restrict the search to these types. If None, searches all four and
        returns the most accurate available.
    pad : timedelta
        Extra margin the orbit must cover on each side of the acquisition.
        Default is 5 minutes.
    url_type : UrlType
        Protocol to use for the download URL, either "s3" or "https".

    Returns
    -------
    NisarOrbit
        The best available covering orbit.

    Raises
    ------
    ValueError
        If no orbit file covers the granule's acquisition.

    """
    start_datetime, end_datetime = parse_granule_datetimes(granule)
    orbits = search_orbits(
        start_datetime,
        end_datetime,
        orbit_types=orbit_types,
        pad=pad,
        url_type=url_type,
    )
    if not orbits:
        types = (
            "any type" if orbit_types is None else str([str(t) for t in orbit_types])
        )
        msg = (
            f"No orbit file of {types} covers "
            f"{start_datetime.isoformat()} to {end_datetime.isoformat()} "
            f"(with {pad} padding) for granule {Path(str(granule)).name!r}"
        )
        raise ValueError(msg)
    return orbits[0]


def _make_earthdata_session() -> requests.Session:
    """Build a `requests.Session` authenticated from ``~/.netrc``."""
    auth = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
    if auth is None:
        msg = "No .netrc entry found for urs.earthdata.nasa.gov"
        raise ValueError(msg)
    username, _, password = auth
    session = requests.Session()
    session.auth = (username, password)
    return session


def download_orbits(
    granules: Iterable[PathOrStr],
    output_dir: PathOrStr = Path("orbits"),
    orbit_types: Sequence[OrbitType] | None = None,
    pad: timedelta = DEFAULT_PAD,
    max_jobs: int = 4,
) -> list[Path]:
    """Download the best orbit file for each of several NISAR granules.

    Granules sharing an orbit file download it only once.

    Parameters
    ----------
    granules : Iterable[PathOrStr]
        NISAR granule names, with or without file suffixes.
    output_dir : PathOrStr
        Directory to write the orbit XML files to. Created if needed.
        Default is ``./orbits``.
    orbit_types : Sequence[OrbitType] | None
        Restrict the search to these types. If None, searches all four and
        picks the most accurate available for each granule.
    pad : timedelta
        Extra margin the orbit must cover on each side of each acquisition.
        Default is 5 minutes.
    max_jobs : int
        Number of parallel downloads. Default is 4.

    Returns
    -------
    list[Path]
        Paths to the downloaded orbit files, sorted by name.

    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    orbits: dict[str, NisarOrbit] = {}
    for granule in granules:
        orbit = get_orbit_for_granule(granule, orbit_types=orbit_types, pad=pad)
        logger.info(f"{Path(str(granule)).name} -> {orbit.filename}")
        orbits[orbit.name] = orbit

    session = _make_earthdata_session()

    def _download_one(orbit: NisarOrbit) -> Path:
        out_path = output_dir / orbit.filename
        if out_path.exists():
            logger.info(f"Skipped (exists): {out_path.name}")
            return out_path
        with session.get(orbit.download_url, stream=True) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)
        return out_path

    out_paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=max_jobs) as pool:
        futures = [pool.submit(_download_one, o) for o in orbits.values()]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Downloading orbits"
        ):
            out_paths.append(future.result())

    return sorted(out_paths)


def _read_granule_list(entries: Sequence[str]) -> list[str]:
    """Expand CLI arguments, reading any that name an existing text file."""
    granules: list[str] = []
    for entry in entries:
        path = Path(entry)
        if path.suffix != ".h5" and path.is_file():
            lines = path.read_text(encoding="utf-8").splitlines()
            granules.extend(line.strip() for line in lines if line.strip())
        else:
            granules.append(entry)
    return granules


def run_orbit_download(
    granules: list[str],
    output_dir: Path = Path("orbits"),
    orbit_types: list[OrbitType] | None = None,
    pad_minutes: float = 5.0,
    max_jobs: int = 4,
) -> list[Path]:
    """Download NISAR orbit ephemeris files matching a list of granules.

    Parameters
    ----------
    granules : list[str]
        NISAR granule names, with or without file suffixes. Any argument naming
        an existing text file is read as a newline-separated list of granules.
    output_dir : Path
        Directory to write the orbit XML files to. Default is ``./orbits``.
    orbit_types : list[OrbitType] | None
        Restrict the search to these types. If None, picks the most accurate
        available for each granule.
    pad_minutes : float
        Extra margin, in minutes, the orbit must cover on each side of each
        acquisition. Default is 5.
    max_jobs : int
        Number of parallel downloads. Default is 4.

    Returns
    -------
    list[Path]
        Paths to the downloaded orbit files.

    """
    return download_orbits(
        _read_granule_list(granules),
        output_dir=output_dir,
        orbit_types=orbit_types,
        pad=timedelta(minutes=pad_minutes),
        max_jobs=max_jobs,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "granules",
        nargs="+",
        help="NISAR granule names, or a text file listing one per line",
    )
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("orbits"))
    parser.add_argument(
        "-t",
        "--orbit-types",
        nargs="+",
        type=OrbitType,
        choices=list(OrbitType),
        default=None,
        help="Restrict to these orbit types (default: best available)",
    )
    parser.add_argument("--pad-minutes", type=float, default=5.0)
    parser.add_argument("--max-jobs", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for path in run_orbit_download(**vars(args)):
        print(path)
