"""Tests for opera_utils.nisar._rslc_download module."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from opera_utils.nisar._product import NISAR_RSLC_GEOLOCATION, NISAR_RSLC_SWATHS
from opera_utils.nisar._rslc_download import (
    _extract_rslc_subset_from_h5,
    process_file,
    run_rslc_download,
)

FILE_1 = "NISAR_L1_PR_RSLC_005_172_A_008_2005_DHDH_A_20251122T024618_20251122T024652_X05007_N_F_J_001.h5"

NROWS = 50
NCOLS = 80
ZDT = np.linspace(9978.0, 10013.0, NROWS)
SR = np.linspace(880000.0, 1050000.0, NCOLS)
GEOLOC_NAZ = 10
GEOLOC_NRG = 15
GEOLOC_NHEIGHT = 3


@pytest.fixture
def rslc_h5(tmp_path):
    """Create a minimal RSLC HDF5 file with known data."""
    h5path = tmp_path / FILE_1
    with h5py.File(h5path, "w") as f:
        # Swaths structure
        f.create_dataset(f"{NISAR_RSLC_SWATHS}/zeroDopplerTime", data=ZDT)
        f.create_dataset(f"{NISAR_RSLC_SWATHS}/zeroDopplerTimeSpacing", data=0.001)

        freq_path = f"{NISAR_RSLC_SWATHS}/frequencyA"
        f.create_dataset(f"{freq_path}/slantRange", data=SR)
        f.create_dataset(f"{freq_path}/slantRangeSpacing", data=5.0)
        f.create_dataset(f"{freq_path}/numberOfSubSwaths", data=np.uint8(1))

        rng = np.random.default_rng(42)
        data = (
            rng.standard_normal((NROWS, NCOLS))
            + 1j * rng.standard_normal((NROWS, NCOLS))
        ).astype(np.complex64)

        hh = f.create_dataset(f"{freq_path}/HH", data=data)
        hh.attrs["description"] = "HH polarization"
        f.create_dataset(f"{freq_path}/VV", data=data * 0.5)
        f.create_dataset(f"{freq_path}/listOfPolarizations", data=[b"HH", b"VV"])

        # validSamplesSubSwath1
        valid_samples = np.column_stack(
            [
                np.full(NROWS, 5, dtype=np.uint32),
                np.full(NROWS, NCOLS - 5, dtype=np.uint32),
            ]
        )
        f.create_dataset(f"{freq_path}/validSamplesSubSwath1", data=valid_samples)

        # Geolocation grid
        geoloc_zdt = np.linspace(ZDT[0], ZDT[-1], GEOLOC_NAZ)
        geoloc_sr = np.linspace(SR[0], SR[-1], GEOLOC_NRG)
        heights = np.array([-500.0, 0.0, 500.0])

        # Simple linear lon/lat grids
        lons = np.linspace(40.0, 43.0, GEOLOC_NRG)
        lats = np.linspace(12.0, 14.0, GEOLOC_NAZ)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        coord_x = np.broadcast_to(
            lon_grid[np.newaxis, :, :],
            (GEOLOC_NHEIGHT, GEOLOC_NAZ, GEOLOC_NRG),
        ).copy()
        coord_y = np.broadcast_to(
            lat_grid[np.newaxis, :, :],
            (GEOLOC_NHEIGHT, GEOLOC_NAZ, GEOLOC_NRG),
        ).copy()

        f.create_dataset(f"{NISAR_RSLC_GEOLOCATION}/zeroDopplerTime", data=geoloc_zdt)
        f.create_dataset(f"{NISAR_RSLC_GEOLOCATION}/slantRange", data=geoloc_sr)
        f.create_dataset(f"{NISAR_RSLC_GEOLOCATION}/heightAboveEllipsoid", data=heights)
        f.create_dataset(f"{NISAR_RSLC_GEOLOCATION}/coordinateX", data=coord_x)
        f.create_dataset(f"{NISAR_RSLC_GEOLOCATION}/coordinateY", data=coord_y)
        f.create_dataset(f"{NISAR_RSLC_GEOLOCATION}/epsg", data=4326)

    return h5path


class TestRunRslcDownloadValidation:
    def test_bbox_and_rows_mutually_exclusive(self):
        with pytest.raises(ValueError, match="Cannot specify both bbox/wkt and rows"):
            run_rslc_download(bbox=(0, 0, 1, 1), rows=(0, 100))

    def test_bbox_and_cols_mutually_exclusive(self):
        with pytest.raises(ValueError, match="Cannot specify both bbox/wkt and rows"):
            run_rslc_download(bbox=(0, 0, 1, 1), cols=(0, 100))

    def test_wkt_and_rows_mutually_exclusive(self):
        with pytest.raises(ValueError, match="Cannot specify both bbox/wkt and rows"):
            run_rslc_download(wkt="POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))", rows=(0, 100))

    def test_bbox_and_wkt_mutually_exclusive(self):
        with pytest.raises(ValueError, match="Cannot specify both bbox and wkt"):
            run_rslc_download(
                bbox=(0, 0, 1, 1), wkt="POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))"
            )


class TestExtractRslcSubsetFromH5:
    def test_full_extraction(self, rslc_h5, tmp_path):
        """Extract without subsetting (rows=None, cols=None)."""
        outpath = tmp_path / "output.h5"
        with h5py.File(rslc_h5, "r") as src:
            _extract_rslc_subset_from_h5(src, outpath, rows=None, cols=None)

        with h5py.File(outpath, "r") as dst:
            freq_path = f"{NISAR_RSLC_SWATHS}/frequencyA"
            # Both polarizations should be extracted
            assert "HH" in dst[freq_path]
            assert "VV" in dst[freq_path]
            assert dst[f"{freq_path}/HH"].shape == (NROWS, NCOLS)
            # Coordinate axes should be full length
            assert dst[f"{NISAR_RSLC_SWATHS}/zeroDopplerTime"].shape == (NROWS,)
            assert dst[f"{freq_path}/slantRange"].shape == (NCOLS,)
            # Scalar metadata should be copied
            assert f"{freq_path}/slantRangeSpacing" in dst

    def test_spatial_subset(self, rslc_h5, tmp_path):
        """Extract a spatial subset."""
        outpath = tmp_path / "subset.h5"
        rows = slice(10, 30)
        cols = slice(20, 50)
        with h5py.File(rslc_h5, "r") as src:
            _extract_rslc_subset_from_h5(src, outpath, rows=rows, cols=cols)

        with h5py.File(outpath, "r") as dst:
            freq_path = f"{NISAR_RSLC_SWATHS}/frequencyA"
            assert dst[f"{freq_path}/HH"].shape == (20, 30)
            assert dst[f"{NISAR_RSLC_SWATHS}/zeroDopplerTime"].shape == (20,)
            assert dst[f"{freq_path}/slantRange"].shape == (30,)

    def test_polarization_filter(self, rslc_h5, tmp_path):
        """Extract only selected polarizations."""
        outpath = tmp_path / "hh_only.h5"
        with h5py.File(rslc_h5, "r") as src:
            _extract_rslc_subset_from_h5(
                src, outpath, rows=None, cols=None, polarizations=["HH"]
            )

        with h5py.File(outpath, "r") as dst:
            freq_path = f"{NISAR_RSLC_SWATHS}/frequencyA"
            assert "HH" in dst[freq_path]
            from opera_utils.nisar._product import NISAR_POLARIZATIONS

            pols_in_output = [k for k in dst[freq_path] if k in NISAR_POLARIZATIONS]
            assert pols_in_output == ["HH"]

    def test_valid_samples_adjusted(self, rslc_h5, tmp_path):
        """validSamplesSubSwath column indices should be adjusted by col_start."""
        outpath = tmp_path / "valid_samples.h5"
        col_start = 20
        with h5py.File(rslc_h5, "r") as src:
            _extract_rslc_subset_from_h5(
                src, outpath, rows=None, cols=slice(col_start, 50)
            )

        with h5py.File(outpath, "r") as dst:
            freq_path = f"{NISAR_RSLC_SWATHS}/frequencyA"
            valid = dst[f"{freq_path}/validSamplesSubSwath1"][:]
            # Original values were (5, NCOLS-5). After adjustment:
            # start: max(0, 5 - 20) = 0, stop: max(0, 75 - 20) = 55
            assert valid[0, 0] == 0
            assert valid[0, 1] == NCOLS - 5 - col_start

    def test_data_values_match_source(self, rslc_h5, tmp_path):
        """Subsetted data should exactly match the source slice."""
        outpath = tmp_path / "values.h5"
        rows = slice(5, 25)
        cols = slice(10, 60)
        with h5py.File(rslc_h5, "r") as src:
            expected = src[f"{NISAR_RSLC_SWATHS}/frequencyA/HH"][rows, cols]
            _extract_rslc_subset_from_h5(src, outpath, rows=rows, cols=cols)

        with h5py.File(outpath, "r") as dst:
            actual = dst[f"{NISAR_RSLC_SWATHS}/frequencyA/HH"][:]
            np.testing.assert_array_equal(actual, expected)

    def test_output_is_compressed(self, rslc_h5, tmp_path):
        """Output datasets should use gzip compression."""
        outpath = tmp_path / "compressed.h5"
        with h5py.File(rslc_h5, "r") as src:
            _extract_rslc_subset_from_h5(src, outpath, rows=None, cols=None)

        with h5py.File(outpath, "r") as dst:
            hh_dset = dst[f"{NISAR_RSLC_SWATHS}/frequencyA/HH"]
            assert hh_dset.compression == "gzip"

    def test_geolocation_grid_subsetted(self, rslc_h5, tmp_path):
        """Geolocation grid should be subsetted to match swath extent."""
        outpath = tmp_path / "geoloc.h5"
        rows = slice(10, 30)
        cols = slice(20, 50)
        with h5py.File(rslc_h5, "r") as src:
            _extract_rslc_subset_from_h5(src, outpath, rows=rows, cols=cols)

        with h5py.File(outpath, "r") as dst:
            # Geolocation grid should exist and be subsetted
            assert f"{NISAR_RSLC_GEOLOCATION}/zeroDopplerTime" in dst
            assert f"{NISAR_RSLC_GEOLOCATION}/slantRange" in dst
            assert f"{NISAR_RSLC_GEOLOCATION}/coordinateX" in dst
            assert f"{NISAR_RSLC_GEOLOCATION}/coordinateY" in dst
            assert f"{NISAR_RSLC_GEOLOCATION}/heightAboveEllipsoid" in dst
            # Heights should not be subsetted
            assert dst[f"{NISAR_RSLC_GEOLOCATION}/heightAboveEllipsoid"].shape == (
                GEOLOC_NHEIGHT,
            )
            # Geolocation time/range should be smaller than the full grid
            geoloc_zdt = dst[f"{NISAR_RSLC_GEOLOCATION}/zeroDopplerTime"]
            geoloc_sr = dst[f"{NISAR_RSLC_GEOLOCATION}/slantRange"]
            assert geoloc_zdt.shape[0] <= GEOLOC_NAZ
            assert geoloc_sr.shape[0] <= GEOLOC_NRG

    def test_subset_metadata_stored(self, rslc_h5, tmp_path):
        """Output file should record subset metadata."""
        outpath = tmp_path / "meta.h5"
        with h5py.File(rslc_h5, "r") as src:
            _extract_rslc_subset_from_h5(
                src, outpath, rows=slice(5, 15), cols=slice(10, 40)
            )

        with h5py.File(outpath, "r") as dst:
            assert "subset_rows" in dst.attrs
            assert "subset_cols" in dst.attrs
            assert "source_file" in dst.attrs


class TestRslcProcessFile:
    def test_skips_existing_output(self, rslc_h5, tmp_path):
        """process_file returns early if output already exists."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        existing = output_dir / FILE_1
        existing.write_bytes(b"placeholder")

        result = process_file(
            url=str(rslc_h5),
            rows=None,
            cols=None,
            output_dir=output_dir,
        )
        assert result == existing
        assert existing.read_bytes() == b"placeholder"

    def test_local_file_extraction(self, rslc_h5, tmp_path):
        """process_file extracts a subset from a local file."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = process_file(
            url=str(rslc_h5),
            rows=slice(0, 20),
            cols=slice(0, 40),
            output_dir=output_dir,
            polarizations=["HH"],
        )
        assert result.exists()
        with h5py.File(result) as f:
            assert f[f"{NISAR_RSLC_SWATHS}/frequencyA/HH"].shape == (20, 40)
