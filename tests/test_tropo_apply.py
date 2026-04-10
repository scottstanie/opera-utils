import numpy as np
import pytest

from opera_utils.tropo._apply import _read_los_up


class TestReadLosUp:
    def test_requires_one_input(self):
        with pytest.raises(ValueError, match="Must provide either"):
            _read_los_up()

    def test_rejects_both_inputs(self, tmp_path):
        # Create dummy files
        inc_file = tmp_path / "incidence.tif"
        enu_file = tmp_path / "los_enu.tif"
        inc_file.touch()
        enu_file.touch()

        with pytest.raises(ValueError, match="Provide only one"):
            _read_los_up(
                incidence_angle_path=inc_file,
                los_enu_path=enu_file,
            )

    def test_incidence_angle_path(self, tmp_path):
        import rasterio
        from rasterio.transform import from_bounds

        inc_file = tmp_path / "incidence.tif"
        data = np.full((100, 100), 45.0, dtype=np.float32)
        transform = from_bounds(0, 0, 100, 100, 100, 100)

        with rasterio.open(
            inc_file,
            "w",
            driver="GTiff",
            height=100,
            width=100,
            count=1,
            dtype=data.dtype,
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data, 1)

        result = _read_los_up(incidence_angle_path=inc_file)

        expected_cos = np.cos(np.radians(45.0))
        np.testing.assert_array_almost_equal(result.values, expected_cos)

    def test_los_enu_path(self, tmp_path):
        import rasterio
        from rasterio.transform import from_bounds

        enu_file = tmp_path / "los_enu.tif"
        # 3-band raster: E, N, U
        data_e = np.full((100, 100), 0.5, dtype=np.float32)
        data_n = np.full((100, 100), 0.3, dtype=np.float32)
        data_u = np.full((100, 100), 0.8124, dtype=np.float32)
        transform = from_bounds(0, 0, 100, 100, 100, 100)

        with rasterio.open(
            enu_file,
            "w",
            driver="GTiff",
            height=100,
            width=100,
            count=3,
            dtype=data_e.dtype,
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data_e, 1)
            dst.write(data_n, 2)
            dst.write(data_u, 3)

        result = _read_los_up(los_enu_path=enu_file)

        # Should return the 'up' component (band 3)
        np.testing.assert_array_almost_equal(result.values, 0.8124, decimal=3)
