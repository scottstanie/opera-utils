"""Tests for `opera_utils.disp._stack` and the download robustness it relies on."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import requests

from opera_utils.disp import DispProduct
from opera_utils.disp import _download as download_mod
from opera_utils.disp import _stack as stack_mod
from opera_utils.disp._enums import ReferenceMethod

geopandas = pytest.importorskip("geopandas")
shapely_geometry = pytest.importorskip("shapely.geometry")

BBOX = (-118.68, 34.40, -118.61, 34.46)


@pytest.fixture
def frame_gdf(monkeypatch):
    """Three frames: ascending and descending ones that contain BBOX, one partial."""
    box = shapely_geometry.box
    gdf = geopandas.GeoDataFrame(
        {
            "orbit_pass": ["ASCENDING", "ASCENDING", "DESCENDING", "DESCENDING"],
            "geometry": [
                box(-119.5, 33.5, -118.0, 35.0),  # contains
                box(-118.65, 34.45, -117.0, 36.0),  # partial
                box(-119.0, 34.0, -118.0, 35.0),  # contains
                box(-100.0, 40.0, -99.0, 41.0),  # elsewhere
            ],
        },
        index=[16941, 16940, 18904, 99999],
    )
    from opera_utils import burst_frame_db

    monkeypatch.setattr(burst_frame_db, "get_frame_geodataframe", lambda: gdf)
    return gdf


def _product(frame_id: int, ref: str, sec: str) -> DispProduct:
    name = (
        f"OPERA_L3_DISP-S1_IW_F{frame_id:05d}_VV_{ref}T015011Z_{sec}T015012Z"
        "_v1.0_20250617T175359Z.nc"
    )
    return DispProduct.from_filename(name)


@pytest.mark.usefixtures("frame_gdf")
class TestSelectFrames:
    def test_keeps_frames_that_contain_bbox(self):
        frames = stack_mod.select_frames(BBOX)
        assert frames == [(16941, "ASCENDING", True), (18904, "DESCENDING", True)]

    def test_orbit_pass_filter(self):
        frames = stack_mod.select_frames(BBOX, orbit_pass="DESCENDING")
        assert frames == [(18904, "DESCENDING", True)]

    def test_explicit_frame_may_be_partial(self):
        frames = stack_mod.select_frames(BBOX, frame_ids=[16940])
        assert frames == [(16940, "ASCENDING", False)]

    def test_no_frame_raises(self):
        with pytest.raises(ValueError, match="No DISP-S1 frame"):
            stack_mod.select_frames((0.0, 0.0, 1.0, 1.0))


@pytest.fixture
def fake_pipeline(monkeypatch, tmp_path, frame_gdf):
    """Replace CMR search, download and reformat; record the calls."""
    catalog = {
        16941: [_product(16941, "20260101", "20260113")],
        18904: [_product(18904, "20260105", "20260117")],
    }
    calls: dict[str, list] = {"download": [], "reformat": []}

    def fake_search(frame_id, **kwargs):
        return list(catalog[frame_id])

    def fake_download(products, bbox, output_dir, num_workers=8):
        calls["download"].append((products[0].frame_id, bbox))
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for p in products:
            path = output_dir / Path(p.filename).name
            path.touch()
            paths.append(path)
        return paths

    def fake_reformat(input_files, output_name, **kwargs):
        calls["reformat"].append((Path(output_name).name, kwargs))
        Path(output_name).mkdir(parents=True)

    monkeypatch.setattr(stack_mod, "search", fake_search)
    monkeypatch.setattr(stack_mod, "download_products", fake_download)
    monkeypatch.setattr(stack_mod, "reformat_stack", fake_reformat)
    return catalog, calls, tmp_path / "out"


class TestMakeStack:
    def test_list_only_downloads_nothing(self, fake_pipeline, capsys):
        _, calls, out = fake_pipeline
        stacks = stack_mod.make_stack(bbox=BBOX, output_dir=out, list_only=True)
        assert [s.frame_id for s in stacks] == [16941, 18904]
        assert stacks[0].num_products == 1
        assert stacks[0].first_date == "2026-01-01"
        assert stacks[0].last_date == "2026-01-13"
        assert calls == {"download": [], "reformat": []}
        assert "16941" in capsys.readouterr().out
        assert not (out / stack_mod.MANIFEST_NAME).exists()

    def test_writes_stacks_and_manifest_without_mask(self, fake_pipeline):
        _, calls, out = fake_pipeline
        stack_mod.make_stack(bbox=BBOX, output_dir=out)

        assert [c[0] for c in calls["download"]] == [16941, 18904]
        assert calls["download"][0][1] == BBOX
        names = [c[0] for c in calls["reformat"]]
        assert names == ["F16941.zarr", "F18904.zarr"]
        kwargs = calls["reformat"][0][1]
        assert kwargs["quality_datasets"] is None
        assert kwargs["reference_method"] == ReferenceMethod.HIGH_COHERENCE

        manifest = json.loads((out / stack_mod.MANIFEST_NAME).read_text())
        assert manifest["request"]["bbox"] == list(BBOX)
        assert manifest["request"]["quality_datasets"] == []
        frames = {f["frame_id"]: f for f in manifest["frames"]}
        assert frames[18904]["orbit_pass"] == "DESCENDING"
        assert frames[16941]["files"] == [
            Path(_product(16941, "20260101", "20260113").filename).name
        ]

    def test_rerun_rebuilds_only_changed_frames(self, fake_pipeline):
        catalog, calls, out = fake_pipeline
        stack_mod.make_stack(bbox=BBOX, output_dir=out)
        calls["reformat"].clear()

        stack_mod.make_stack(bbox=BBOX, output_dir=out)
        assert calls["reformat"] == []

        catalog[18904].append(_product(18904, "20260105", "20260129"))
        stack_mod.make_stack(bbox=BBOX, output_dir=out)
        assert [c[0] for c in calls["reformat"]] == ["F18904.zarr"]

        calls["reformat"].clear()
        stack_mod.make_stack(bbox=BBOX, output_dir=out, overwrite=True)
        assert len(calls["reformat"]) == 2

    @pytest.mark.usefixtures("fake_pipeline")
    def test_needs_exactly_one_region(self):
        with pytest.raises(ValueError, match="exactly one"):
            stack_mod.make_stack()
        with pytest.raises(ValueError, match="exactly one"):
            stack_mod.make_stack(bbox=BBOX, wkt="POINT (0 0)")

    def test_wkt_uses_polygon_bounds(self, fake_pipeline):
        _, calls, out = fake_pipeline
        wkt = shapely_geometry.box(*BBOX).wkt
        stack_mod.make_stack(wkt=wkt, output_dir=out)
        assert calls["download"][0][1] == pytest.approx(BBOX)


class TestProcessFile:
    def test_failed_subset_leaves_no_file(self, monkeypatch, tmp_path):
        src = tmp_path / "OPERA_src.nc"
        src.touch()
        out_dir = tmp_path / "subsets"
        out_dir.mkdir()

        def failing_extract(input_obj, outpath, rows, cols):
            Path(outpath).write_bytes(b"half a file")
            msg = "connection dropped"
            raise OSError(msg)

        monkeypatch.setattr(download_mod, "_extract_subset", failing_extract)
        with pytest.raises(OSError, match="connection dropped"):
            download_mod.process_file(str(src), None, None, out_dir)
        assert list(out_dir.iterdir()) == []

        def good_extract(input_obj, outpath, rows, cols):
            Path(outpath).write_bytes(b"complete")

        monkeypatch.setattr(download_mod, "_extract_subset", good_extract)
        out = download_mod.process_file(str(src), None, None, out_dir)
        assert out.read_bytes() == b"complete"
        assert list(out_dir.iterdir()) == [out]


class _Response:
    def __init__(self, status_code: int, content: bytes = b""):
        self.status_code = status_code
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            msg = f"{self.status_code} error"
            raise requests.HTTPError(msg)


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def get(self, _url):
        self.calls += 1
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class TestGetWithRetries:
    @pytest.fixture(autouse=True)
    def _no_sleep(self, monkeypatch):
        monkeypatch.setattr(download_mod.time, "sleep", lambda _: None)

    def test_retries_server_and_connection_errors(self):
        session = _Session(
            [_Response(503), requests.ConnectionError("dns"), _Response(200, b"ok")]
        )
        content = download_mod._get_with_retries(session, "https://x", max_attempts=3)
        assert content == b"ok"
        assert session.calls == 3

    def test_gives_up_after_max_attempts(self):
        session = _Session([_Response(503), _Response(503)])
        with pytest.raises(requests.HTTPError, match="503"):
            download_mod._get_with_retries(session, "https://x", max_attempts=2)

    def test_does_not_retry_client_errors(self):
        session = _Session([_Response(401), _Response(200, b"ok")])
        with pytest.raises(requests.HTTPError, match="401"):
            download_mod._get_with_retries(session, "https://x", max_attempts=3)
        assert session.calls == 1


def test_cli_help(capsys, monkeypatch):
    from opera_utils.cli import cli_app

    monkeypatch.setattr(sys, "argv", ["opera-utils", "disp-s1-stack", "--help"])
    with pytest.raises(SystemExit):
        cli_app()
    assert "Find, download, crop, and reformat DISP-S1" in capsys.readouterr().out
