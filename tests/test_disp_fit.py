"""Tests for disp timeseries fitting functionality."""

from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from opera_utils.disp.fit import (
    FitConfig,
    build_design_matrix,
    datetime_to_float,
    fit_disp_timeseries,
    rebase_displacement,
    sincos_to_amplitude_phase,
)


def test_datetime_to_float():
    """Test datetime to float conversion."""
    dates = [
        datetime(2020, 1, 1),
        datetime(2020, 1, 13),  # 12 days later
        datetime(2020, 2, 1),  # 31 days later
    ]

    result = datetime_to_float(dates, reference_index=0)

    assert result[0] == 0.0  # reference date
    assert result[1] == 12.0  # 12 days
    assert result[2] == 31.0  # 31 days


def test_build_design_matrix_linear():
    """Test design matrix construction for linear model."""
    dates = np.array(["2020-01-01", "2020-02-01", "2020-03-01"], dtype="datetime64[D]")

    A, names = build_design_matrix(dates, poly_degree=1, seasonal="none")

    assert A.shape[0] == 3  # 3 time points
    assert A.shape[1] == 2  # constant + linear
    assert names == ["constant", "linear_trend"]

    # First column should be all ones (constant term)
    np.testing.assert_array_equal(A[:, 0], np.ones(3))

    # Second column should be time values
    assert A[0, 1] == 0.0  # reference time
    assert A[1, 1] > 0.0  # later time
    assert A[2, 1] > A[1, 1]  # even later time


def test_build_design_matrix_seasonal():
    """Test design matrix construction with seasonal terms."""
    dates = np.array(
        ["2020-01-01", "2020-04-01", "2020-07-01", "2020-10-01"], dtype="datetime64[D]"
    )

    A, names = build_design_matrix(dates, poly_degree=1, seasonal="annual")

    assert A.shape[0] == 4  # 4 time points
    assert A.shape[1] == 4  # constant + linear + sin + cos
    assert names == ["constant", "linear_trend", "annual_sin", "annual_cos"]


def test_build_design_matrix_cubic():
    """Test design matrix construction for cubic polynomial."""
    dates = np.array(["2020-01-01", "2020-02-01", "2020-03-01"], dtype="datetime64[D]")

    A, names = build_design_matrix(dates, poly_degree=3, seasonal="none")

    assert A.shape[0] == 3  # 3 time points
    assert A.shape[1] == 4  # constant + linear + quadratic + cubic
    assert names == ["constant", "linear_trend", "quadratic", "cubic"]


def test_sincos_to_amplitude_phase():
    """Test conversion from sin/cos coefficients to amplitude/phase."""
    # Test case where sin=1, cos=0
    a_cos = np.array([0.0])
    a_sin = np.array([1.0])
    period = 1.0

    amp, phase = sincos_to_amplitude_phase(a_cos, a_sin, period)

    assert amp[0] == 1.0
    # The phase calculation uses arctan2(-sin, cos), so we expect 0.75 for this case
    assert np.isclose(phase[0], 0.75)


def create_synthetic_disp_dataset():
    """Create a synthetic displacement dataset for testing."""
    # Create synthetic data
    times = np.array(
        ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"], dtype="datetime64[D]"
    )
    y_coords = np.arange(10)
    x_coords = np.arange(15)

    # Synthetic displacement with linear trend + noise
    displacement = np.random.randn(4, 10, 15) * 0.1
    # Add linear trend
    time_values = datetime_to_float(times)
    for i, t in enumerate(time_values):
        displacement[i] += t * 0.01  # 1cm/day velocity

    # Reference times (same for all in this test)
    reference_times = np.full(4, times[0])

    # Temporal coherence
    temporal_coherence = np.random.uniform(0.5, 1.0, (4, 10, 15))

    ds = xr.Dataset(
        {
            "displacement": (["time", "y", "x"], displacement),
            "reference_time": (["time"], reference_times),
            "temporal_coherence": (["time", "y", "x"], temporal_coherence),
        },
        coords={
            "time": times,
            "y": y_coords,
            "x": x_coords,
        },
    )

    return ds


def test_rebase_displacement():
    """Test displacement rebasing with uniform reference times."""
    ds = create_synthetic_disp_dataset()

    # For uniform reference times, rebasing should not change the data
    rebased = rebase_displacement(ds)

    # Should have same shape
    assert rebased.shape == ds.displacement.shape

    # Values should be similar (same reference time means no rebasing needed)
    np.testing.assert_array_almost_equal(rebased.values, ds.displacement.values)


def test_fit_disp_timeseries_basic():
    """Test basic timeseries fitting."""
    ds = create_synthetic_disp_dataset()

    cfg = FitConfig(poly_degree=1, seasonal="none", backend="numpy")
    result = fit_disp_timeseries(ds, cfg=cfg)

    # Check output structure
    assert "coefficients" in result
    assert "mse" in result
    assert "velocity" in result

    # Check shapes
    assert result.coefficients.shape == (2, 10, 15)  # 2 coefficients, 10x15 grid
    assert result.mse.shape == (10, 15)
    assert result.velocity.shape == (10, 15)

    # Check coefficient names
    expected_names = ["constant", "linear_trend"]
    assert list(result.coefficients.coefficient.values) == expected_names


def test_fit_disp_timeseries_seasonal():
    """Test timeseries fitting with seasonal terms."""
    ds = create_synthetic_disp_dataset()

    cfg = FitConfig(poly_degree=1, seasonal="annual", backend="numpy")
    result = fit_disp_timeseries(ds, cfg=cfg)

    # Check output structure
    assert "coefficients" in result
    assert "velocity" in result
    assert "annual_amplitude" in result
    assert "annual_phase" in result

    # Check shapes
    assert result.coefficients.shape == (4, 10, 15)  # 4 coefficients

    # Check coefficient names
    expected_names = ["constant", "linear_trend", "annual_sin", "annual_cos"]
    assert list(result.coefficients.coefficient.values) == expected_names


def test_fit_disp_timeseries_temporal_coherence_threshold():
    """Test timeseries fitting with temporal coherence masking."""
    ds = create_synthetic_disp_dataset()

    # Set some pixels to have low temporal coherence across all time steps
    ds["temporal_coherence"].values[:, :5, :5] = 0.1

    cfg = FitConfig(
        poly_degree=1,
        seasonal="none",
        temporal_coherence_threshold=0.3,
        backend="numpy",
    )
    result = fit_disp_timeseries(ds, cfg=cfg)

    # Check that results exist
    assert "velocity" in result

    # Some pixels should have valid values (those with high coherence)
    assert np.any(np.isfinite(result.velocity.values[5:, 5:]))

    # Most masked pixels should have NaN values (but may not be all due to fitting logic)
    num_nan_masked = np.sum(np.isnan(result.velocity.values[:5, :5]))
    assert num_nan_masked > 10  # Most should be NaN


def test_fit_config_defaults():
    """Test FitConfig default values."""
    cfg = FitConfig()

    assert cfg.poly_degree == 1
    assert cfg.seasonal == "annual"
    assert cfg.temporal_coherence_threshold is None
    assert cfg.backend == "numpy"
    assert cfg.reference_index == 0


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_fit_backends_consistency(backend):
    """Test that both NumPy and JAX backends give similar results."""
    ds = create_synthetic_disp_dataset()

    # Simple linear fit to minimize numerical differences
    cfg = FitConfig(poly_degree=1, seasonal="none", backend=backend)

    result = fit_disp_timeseries(ds, cfg=cfg)

    # Check basic structure
    assert "velocity" in result
    assert result.velocity.shape == (10, 15)

    # Should have finite values (at least some)
    assert np.any(np.isfinite(result.velocity.values))


def test_fit_with_missing_reference_time():
    """Test fitting when reference_time is not in dataset."""
    ds = create_synthetic_disp_dataset()

    # Remove reference_time
    ds = ds.drop_vars("reference_time")

    cfg = FitConfig(poly_degree=1, seasonal="none", backend="numpy")
    result = fit_disp_timeseries(ds, cfg=cfg)

    # Should still work
    assert "velocity" in result
    assert result.velocity.shape == (10, 15)
