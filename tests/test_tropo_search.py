from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from opera_utils.tropo._search import TROPO_FILE_REGEX, TropoProduct, search_tropo


class TestTropoFileRegex:
    def test_valid_filename(self):
        filename = (
            "OPERA_L4_TROPO-ZENITH_20260102T000000Z_20260106T000635Z_HRES_v1.0.nc"
        )
        match = TROPO_FILE_REGEX.match(filename)
        assert match is not None
        assert match.group("start_datetime") == "20260102T000000"
        assert match.group("end_datetime") == "20260106T000635"
        assert match.group("resolution") == "HRES"
        assert match.group("version") == "1.0"

    def test_invalid_filename(self):
        filename = "OPERA_L3_DISP-S1_IW_F11116_VV_20160705T140755Z.nc"
        match = TROPO_FILE_REGEX.match(filename)
        assert match is None


class TestTropoProduct:
    def test_from_url(self):
        url = "https://cumulus.asf.earthdatacloud.nasa.gov/OPERA/OPERA_L4_TROPO-ZENITH_V1/OPERA_L4_TROPO-ZENITH_20260102T000000Z_20260106T000635Z_HRES_v1.0/OPERA_L4_TROPO-ZENITH_20260102T000000Z_20260106T000635Z_HRES_v1.0.nc"
        product = TropoProduct.from_url(url)
        assert product.start_datetime == datetime(
            2026, 1, 2, 0, 0, 0, tzinfo=timezone.utc
        )
        assert product.end_datetime == datetime(
            2026, 1, 6, 0, 6, 35, tzinfo=timezone.utc
        )
        assert product.resolution == "HRES"
        assert product.version == "1.0"

    def test_from_url_invalid(self):
        url = "https://example.com/invalid_file.nc"
        with pytest.raises(AssertionError):
            TropoProduct.from_url(url)


@pytest.fixture
def cmr_tropo_response_json():
    return {
        "items": [
            {
                "meta": {"concept-id": "C1234-OPERA"},
                "umm": {
                    "GranuleUR": (
                        "OPERA_L4_TROPO-ZENITH_20260102T000000Z_20260102T060000Z_HRES_v1.0"
                    ),
                    "RelatedUrls": [
                        {
                            "URL": (
                                "https://cumulus.asf.earthdatacloud.nasa.gov/OPERA/OPERA_L4_TROPO-ZENITH_V1/OPERA_L4_TROPO-ZENITH_20260102T000000Z_20260102T060000Z_HRES_v1.0.nc"
                            ),
                            "Type": "GET DATA",
                        },
                    ],
                },
            },
            {
                "meta": {"concept-id": "C1235-OPERA"},
                "umm": {
                    "GranuleUR": (
                        "OPERA_L4_TROPO-ZENITH_20260102T060000Z_20260102T120000Z_HRES_v1.0"
                    ),
                    "RelatedUrls": [
                        {
                            "URL": (
                                "https://cumulus.asf.earthdatacloud.nasa.gov/OPERA/OPERA_L4_TROPO-ZENITH_V1/OPERA_L4_TROPO-ZENITH_20260102T060000Z_20260102T120000Z_HRES_v1.0.nc"
                            ),
                            "Type": "GET DATA",
                        },
                    ],
                },
            },
        ]
    }


class MockResponse:
    def __init__(self, json_data, status_code=200, headers=None):
        self.json_data = json_data
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        return self.json_data

    def raise_for_status(self):
        if self.status_code != 200:
            msg = f"Status code {self.status_code}"
            raise ValueError(msg)


class TestSearchTropo:
    def test_search_with_mock_response(self, cmr_tropo_response_json):
        with patch("requests.get") as mock_get:
            mock_get.return_value = MockResponse(cmr_tropo_response_json)

            urls = search_tropo(
                start_datetime=datetime(2026, 1, 2, 0, 0, 0),
                end_datetime=datetime(2026, 1, 2, 12, 0, 0),
            )

            args, kwargs = mock_get.call_args
            assert args[0] == "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
            assert kwargs["params"]["short_name"] == "OPERA_L4_TROPO-ZENITH_V1"

            assert len(urls) == 2
            assert all(url.endswith(".nc") for url in urls)

    def test_search_with_uat(self, cmr_tropo_response_json):
        with patch("requests.get") as mock_get:
            mock_get.return_value = MockResponse(cmr_tropo_response_json)

            _urls = search_tropo(
                start_datetime=datetime(2026, 1, 2, 0, 0, 0),
                end_datetime=datetime(2026, 1, 2, 12, 0, 0),
                use_uat=True,
            )

            args, _ = mock_get.call_args
            assert "uat.earthdata" in args[0]

    def test_search_results_sorted(self, cmr_tropo_response_json):
        with patch("requests.get") as mock_get:
            mock_get.return_value = MockResponse(cmr_tropo_response_json)

            urls = search_tropo(
                start_datetime=datetime(2026, 1, 2, 0, 0, 0),
                end_datetime=datetime(2026, 1, 2, 12, 0, 0),
            )

            products = [TropoProduct.from_url(u) for u in urls]
            for i in range(len(products) - 1):
                assert products[i].start_datetime <= products[i + 1].start_datetime
