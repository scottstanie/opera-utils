from __future__ import annotations

import pooch

__all__ = [
    "fetch_frame_geometries_simple",
    "fetch_burst_id_geometries_simple",
    "fetch_burst_to_frame_mapping_file",
    "fetch_frame_to_burst_mapping_file",
]

# See: https://github.com/opera-adt/burst_db/tree/main/src/burst_db/data
# BASE_URL = "https://github.com/opera-adt/burst_db/raw/v{version}/src/burst_db/data"
# BASE_URL = "https://github.com/opera-adt/burst_db/raw/v0.3.0/src/burst_db/data"
BASE_URL = "https://github.com/opera-adt/burst_db/releases/download/v{version}/"

# $ ls *json.zip | xargs -n1 shasum -a 256
# a86b3365317d3857bebedc50d709c09b18e3481c2cf257a48ef06f7e31d4c1b6  burst-id-geometries-simple-0.4.0.geojson.zip
# dc5c3bde33a8ad54bfbeddc84a4f13e88980c8756287752434d95f2d0550ebd2  frame-geometries-simple-0.4.0.geojson.zip
# 3cd2737c3f03897755c8449cc18ef6fa6720789fc7bae3c347ffacecf1667d1f  opera-s1-disp-0.4.0-burst-to-frame.json.zip
# a0c28df480bb0d88c03dd650de41264c5f30fbb778d967904ee7cb8a36000a37  opera-s1-disp-0.4.0-frame-to-burst.json.zip

BURST_DB_VERSION = "0.4.0"

POOCH = pooch.create(
    # Folder where the data will be stored. For a sensible default, use the
    # default cache folder for your OS.
    path=pooch.os_cache("opera_utils"),
    # Base URL of the remote data store. Will call .format on this string
    # to insert the version (see below).
    base_url=BASE_URL,
    # Pooches are versioned so that you can use multiple versions of a
    # package simultaneously. Use PEP440 compliant version number. The
    # version will be appended to the path.
    version=BURST_DB_VERSION,
    # If a version as a "+XX.XXXXX" suffix, we'll assume that this is a dev
    # version and replace the version with this string.
    version_dev="main",
    # An environment variable that overwrites the path.
    env="OPERA_UTILS_DATA_DIR",
    # The cache file registry. A dictionary with all files managed by this
    # pooch. Keys are the file names (relative to *base_url*) and values
    # are their respective SHA256 hashes. Files will be downloaded
    # automatically when needed.
    registry={
        f"burst-id-geometries-simple-{BURST_DB_VERSION}.geojson.zip": "a86b3365317d3857bebedc50d709c09b18e3481c2cf257a48ef06f7e31d4c1b6",
        f"frame-geometries-simple-{BURST_DB_VERSION}.geojson.zip": "dc5c3bde33a8ad54bfbeddc84a4f13e88980c8756287752434d95f2d0550ebd2",
        f"opera-s1-disp-{BURST_DB_VERSION}-burst-to-frame.json.zip": "3cd2737c3f03897755c8449cc18ef6fa6720789fc7bae3c347ffacecf1667d1f",
        f"opera-s1-disp-{BURST_DB_VERSION}-frame-to-burst.json.zip": "a0c28df480bb0d88c03dd650de41264c5f30fbb778d967904ee7cb8a36000a37",
    },
)


def fetch_frame_geometries_simple() -> str:
    """Get the simplified frame geometries for the burst database."""
    return POOCH.fetch(f"frame-geometries-simple-{BURST_DB_VERSION}.geojson.zip")


def fetch_burst_id_geometries_simple() -> str:
    """Get the simplified burst ID geometries for the burst database."""
    return POOCH.fetch(f"burst-id-geometries-simple-{BURST_DB_VERSION}.geojson.zip")


def fetch_burst_to_frame_mapping_file() -> str:
    """Get the burst-to-frame mapping for the burst database."""
    return POOCH.fetch(f"opera-s1-disp-{BURST_DB_VERSION}-burst-to-frame.json.zip")


def fetch_frame_to_burst_mapping_file() -> str:
    """Get the frame-to-burst mapping for the burst database."""
    return POOCH.fetch(f"opera-s1-disp-{BURST_DB_VERSION}-frame-to-burst.json.zip")


def fetch_full_db_file() -> str:
    """Get full geopackage containing the frame/burst ID database."""
    return POOCH.fetch(f"opera-s1-disp-{BURST_DB_VERSION}.gpkg.zip")
