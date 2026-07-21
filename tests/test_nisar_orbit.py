"""Tests for opera_utils.nisar._orbit."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from opera_utils.constants import UrlType
from opera_utils.nisar._orbit import (
    NisarOrbit,
    OrbitType,
    get_orbit_for_granule,
    parse_granule_datetimes,
    search_orbits,
)

# A real POE granule: created 2026-07-02, valid 2026-06-19T20:59:42 - 2026-06-21T02:59:42
POE_NAME = "NISAR_ANC_J_PR_POE_20260702T171245_20260619T205942_20260621T025942"
# The four orbits ASF Vertex attaches to the L0 granule below. None of them
# actually cover its acquisition, which is what motivated this module.
POE_LATEST = "NISAR_ANC_J_PR_POE_20260715T185257_20260703T205942_20260705T025942"
MOE_LATEST = "NISAR_ANC_J_PR_MOE_20260721T132436_20260719T205942_20260721T025942"
NOE_LATEST = "NISAR_ANC_J_PR_NOE_20260721T171818_20260720T095942_20260721T153442"
FOE_LATEST = "NISAR_ANC_J_PR_FOE_20260720T222320_20260720T221515_20260727T221515"

L0_GRANULE = (
    "NISAR_L0_PR_RRSD_025_131_A_148S_20260717T063521_20260717T063549_P05023_F_J_001"
)
GSLC_GRANULE = (
    "NISAR_L2_PR_GSLC_004_076_A_022_2005_QPDH_A_20251103T110514"
    "_20251103T110549_X05007_N_F_J_001.h5"
)
GUNW_GRANULE = (
    "NISAR_L2_PR_GUNW_004_151_A_011_005_4000_SH_20251108T155041_20251108T155058"
    "_20251120T155041_20251120T155058_X05010_N_P_J_001.h5"
)


def _make_umm_item(name: str, protocol: str = "https") -> dict:
    """Build a minimal CMR UMM item dict for an orbit granule."""
    if protocol == "https":
        orbit_type = name.split("_")[4]
        url = (
            "https://nisar.asf.earthdatacloud.nasa.gov/NISAR/"
            f"{orbit_type}/{name}/{name}.xml"
        )
    else:
        url = f"s3://sds-n-cumulus-prod-nisar-products/{name}/{name}.xml"
    return {
        "umm": {
            "GranuleUR": name,
            "RelatedUrls": [{"URL": url, "Type": "GET DATA"}],
            "DataGranule": {
                "ArchiveAndDistributionInformation": [{"SizeInBytes": 30_599_938}]
            },
        }
    }


class TestNisarOrbitParsing:
    def test_from_filename(self):
        orbit = NisarOrbit.from_filename(POE_NAME)
        assert orbit.name == POE_NAME
        assert orbit.orbit_type == OrbitType.POE
        assert orbit.creation_datetime == datetime(
            2026, 7, 2, 17, 12, 45, tzinfo=timezone.utc
        )
        assert orbit.start_datetime == datetime(
            2026, 6, 19, 20, 59, 42, tzinfo=timezone.utc
        )
        assert orbit.end_datetime == datetime(
            2026, 6, 21, 2, 59, 42, tzinfo=timezone.utc
        )

    @pytest.mark.parametrize(
        "name",
        [
            POE_NAME,
            f"{POE_NAME}.xml",
            f"/some/local/dir/{POE_NAME}.xml",
            f"https://nisar.asf.earthdatacloud.nasa.gov/NISAR/POE/{POE_NAME}/{POE_NAME}.xml",
        ],
    )
    def test_suffix_and_path_are_optional(self, name):
        assert NisarOrbit.from_filename(name).name == POE_NAME

    @pytest.mark.parametrize("name", [MOE_LATEST, NOE_LATEST, FOE_LATEST])
    def test_all_orbit_types(self, name):
        assert NisarOrbit.from_filename(name).orbit_type == OrbitType(
            name.split("_")[4]
        )

    @pytest.mark.parametrize(
        "bad_name",
        [
            "NISAR_ANC_J_PR_XYZ_20260702T171245_20260619T205942_20260621T025942",
            "NISAR_ANC_J_PR_POE_20260702T171245_20260619T205942",
            GSLC_GRANULE,
            "not-a-nisar-file.xml",
        ],
    )
    def test_invalid_names_raise(self, bad_name):
        with pytest.raises(ValueError, match="Invalid NISAR orbit filename"):
            NisarOrbit.from_filename(bad_name)

    def test_download_url_constructed_from_name(self):
        orbit = NisarOrbit.from_filename(POE_NAME)
        assert orbit.url is None
        assert (
            orbit.download_url
            == "https://nisar.asf.earthdatacloud.nasa.gov/NISAR/POE/"
            f"{POE_NAME}/{POE_NAME}.xml"
        )
        assert orbit.filename == f"{POE_NAME}.xml"

    def test_from_umm_uses_cmr_url(self):
        orbit = NisarOrbit.from_umm(_make_umm_item(POE_NAME)["umm"])
        assert orbit.name == POE_NAME
        assert orbit.size_in_bytes == 30_599_938
        assert orbit.download_url.endswith(f"{POE_NAME}.xml")

    def test_from_umm_s3(self):
        orbit = NisarOrbit.from_umm(
            _make_umm_item(POE_NAME, protocol="s3")["umm"], url_type=UrlType.S3
        )
        assert orbit.download_url.startswith("s3://")


class TestCovers:
    orbit = NisarOrbit.from_filename(POE_NAME)  # valid 06-19T20:59:42 - 06-21T02:59:42

    def test_acquisition_well_inside_window(self):
        start = datetime(2026, 6, 20, 12, 0, 0, tzinfo=timezone.utc)
        assert self.orbit.covers(start, start + timedelta(seconds=30))

    def test_acquisition_outside_window(self):
        start = datetime(2026, 6, 25, 12, 0, 0, tzinfo=timezone.utc)
        assert not self.orbit.covers(start, start + timedelta(seconds=30))

    def test_pad_excludes_acquisition_near_the_edge(self):
        # Two minutes after validity starts: inside the window, but there is not
        # enough margin before it for the default 5-minute pad.
        start = datetime(2026, 6, 19, 21, 1, 42, tzinfo=timezone.utc)
        end = start + timedelta(seconds=30)
        assert self.orbit.covers(start, end, pad=timedelta(0))
        assert not self.orbit.covers(start, end)

    def test_naive_datetimes_treated_as_utc(self):
        naive = datetime(2026, 6, 20, 12, 0, 0)
        aware = naive.replace(tzinfo=timezone.utc)
        assert self.orbit.covers(naive, naive) == self.orbit.covers(aware, aware)


class TestParseGranuleDatetimes:
    def test_l0_granule(self):
        start, end = parse_granule_datetimes(L0_GRANULE)
        assert start == datetime(2026, 7, 17, 6, 35, 21, tzinfo=timezone.utc)
        assert end == datetime(2026, 7, 17, 6, 35, 49, tzinfo=timezone.utc)

    def test_gslc_granule_with_suffix(self):
        start, end = parse_granule_datetimes(GSLC_GRANULE)
        assert start == datetime(2025, 11, 3, 11, 5, 14, tzinfo=timezone.utc)
        assert end == datetime(2025, 11, 3, 11, 5, 49, tzinfo=timezone.utc)

    def test_gunw_returns_reference_acquisition(self):
        start, end = parse_granule_datetimes(GUNW_GRANULE)
        assert start == datetime(2025, 11, 8, 15, 50, 41, tzinfo=timezone.utc)
        assert end == datetime(2025, 11, 8, 15, 50, 58, tzinfo=timezone.utc)

    def test_full_path_accepted(self):
        assert parse_granule_datetimes(f"/data/gslcs/{GSLC_GRANULE}") == (
            parse_granule_datetimes(GSLC_GRANULE)
        )

    def test_too_few_timestamps_raises(self):
        with pytest.raises(ValueError, match="at least two"):
            parse_granule_datetimes("NISAR_L2_PR_GSLC_004_076_A_20251103T110514.h5")


class TestSearchOrbits:
    def test_filters_out_non_covering_orbits(self):
        """CMR returns intersecting granules; only covering ones survive."""
        items = [_make_umm_item(n) for n in (POE_NAME, POE_LATEST, MOE_LATEST)]
        start = datetime(2026, 6, 20, 12, 0, 0, tzinfo=timezone.utc)
        with patch(
            "opera_utils.nisar._orbit.fetch_cmr_pages", return_value=items
        ) as mock_fetch:
            results = search_orbits(start, start + timedelta(seconds=30))

        assert [o.name for o in results] == [POE_NAME]
        params = mock_fetch.call_args[0][1]
        assert params["short_name"] == "NISAR_OE"
        assert "attribute[]" not in params

    def test_orbit_type_becomes_a_cmr_attribute_filter(self):
        start = datetime(2026, 6, 20, 12, 0, 0, tzinfo=timezone.utc)
        with patch(
            "opera_utils.nisar._orbit.fetch_cmr_pages", return_value=[]
        ) as mock_fetch:
            search_orbits(start, start, orbit_types=[OrbitType.POE, OrbitType.MOE])

        params = mock_fetch.call_args[0][1]
        assert params["attribute[]"] == [
            "string,PRODUCT_TYPE,POE",
            "string,PRODUCT_TYPE,MOE",
        ]

    def test_sorted_by_type_then_newest_creation(self):
        # Same validity window, three types plus an older duplicate POE
        older_poe = "NISAR_ANC_J_PR_POE_20260701T000000_20260619T205942_20260621T025942"
        moe = "NISAR_ANC_J_PR_MOE_20260622T000000_20260619T205942_20260621T025942"
        foe = "NISAR_ANC_J_PR_FOE_20260618T000000_20260619T205942_20260621T025942"
        items = [_make_umm_item(n) for n in (foe, older_poe, moe, POE_NAME)]
        start = datetime(2026, 6, 20, 12, 0, 0, tzinfo=timezone.utc)

        with patch("opera_utils.nisar._orbit.fetch_cmr_pages", return_value=items):
            results = search_orbits(start, start + timedelta(seconds=30))

        assert [o.name for o in results] == [POE_NAME, older_poe, moe, foe]


class TestGetOrbitForGranule:
    def test_picks_the_best_covering_orbit(self):
        covering_poe = (
            "NISAR_ANC_J_PR_POE_20260801T000000_20260716T205942_20260718T025942"
        )
        items = [_make_umm_item(n) for n in (FOE_LATEST, covering_poe)]
        with patch("opera_utils.nisar._orbit.fetch_cmr_pages", return_value=items):
            orbit = get_orbit_for_granule(L0_GRANULE)

        assert orbit.name == covering_poe
        assert orbit.orbit_type == OrbitType.POE

    def test_vertex_linked_orbits_do_not_cover_the_granule(self):
        """Regression guard: the orbits ASF lists for this L0 all miss it."""
        items = [
            _make_umm_item(n) for n in (POE_LATEST, MOE_LATEST, NOE_LATEST, FOE_LATEST)
        ]
        with (
            patch("opera_utils.nisar._orbit.fetch_cmr_pages", return_value=items),
            pytest.raises(ValueError, match="No orbit file of any type covers"),
        ):
            get_orbit_for_granule(L0_GRANULE)

    def test_error_message_names_restricted_types(self):
        with (
            patch("opera_utils.nisar._orbit.fetch_cmr_pages", return_value=[]),
            pytest.raises(ValueError, match=r"No orbit file of \['POE'\] covers"),
        ):
            get_orbit_for_granule(L0_GRANULE, orbit_types=[OrbitType.POE])
