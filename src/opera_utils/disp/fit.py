"""Timeseries fitting for OPERA DISP products.

Fit velocity + seasonal + polynomial models to displacement timeseries.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from math import factorial
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

from ._product import DispProductStack
from ._rebase import rebase_timeseries

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

logger = logging.getLogger(__name__)


def rebase_displacement(
    ds: xr.Dataset,
    displacement_var: str = "displacement",
    reference_time_var: str = "reference_time",
) -> xr.DataArray:
    """Return a rebased displacement (time,y,x) DataArray.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with displacement data.
    displacement_var : str
        Name of displacement variable.
    reference_time_var : str
        Name of reference time variable.

    Returns
    -------
    xr.DataArray
        Rebased displacement array.

    """
    da = ds[displacement_var]
    refs = ds[reference_time_var].values

    # Work in process-friendly chunks (time chunk = -1 so we see all crossovers)
    chunks = {"time": -1}
    if da.chunks is not None:
        if len(da.chunks) > 1:
            chunks["y"] = da.chunks[1][0]
        if len(da.chunks) > 2:
            chunks["x"] = da.chunks[2][0]
    else:
        chunks.update({"y": 1024, "x": 1024})
    da_c = da.chunk(chunks)

    def _block(arr: xr.DataArray) -> xr.DataArray:
        data = rebase_timeseries(arr.data, refs)
        return xr.DataArray(data, coords=arr.coords, dims=arr.dims)

    return da_c.map_blocks(_block)


def datetime_to_float(
    dates: Sequence[date | datetime], reference_index: int = 0
) -> np.ndarray:
    """Convert a sequence of datetime objects to a float representation.

    Output units are in days since the first item in `dates`.

    Parameters
    ----------
    dates : Sequence[DateOrDatetime]
        List of datetime objects to convert to floats
    reference_index : int
        Index of reference date (typically 0).

    Returns
    -------
    date_arr : np.array 1D
        The float representation of the datetime objects

    """
    sec_per_day = 60 * 60 * 24
    date_arr = np.asarray(dates).astype("datetime64[s]")
    # Reference the 0 to the first date
    date_arr = date_arr - date_arr[reference_index]
    return date_arr.astype(float) / sec_per_day


def build_design_matrix(
    dates: np.ndarray,
    poly_degree: int = 1,
    seasonal: Literal["none", "annual", "annual+semi"] = "annual",
    reference_index: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """Build design matrix for timeseries fitting.

    Parameters
    ----------
    dates : np.ndarray
        Array of datetime64 values.
    poly_degree : int
        Polynomial degree (1=velocity, up to 3).
    seasonal : {"none", "annual", "annual+semi"}
        Seasonal terms to include.
    reference_index : int
        Index of reference date (typically 0).

    Returns
    -------
    tuple[np.ndarray, list[str]]
        Design matrix and coefficient names.

    """
    t = datetime_to_float(dates, reference_index)

    cols = []
    names: list[str] = []

    # Poly terms: constant, linear (velocity), quadratic, cubic
    # factorial normalization for conditioning
    for i, name in enumerate(
        ["constant", "linear_trend", "quadratic", "cubic"][: poly_degree + 1]
    ):
        if i == 0:
            cols.append(np.ones_like(t))
        else:
            cols.append((t**i) / factorial(i))
        names.append(name)

    # Seasonal terms: sin/cos pairs
    periods = []
    if seasonal in ("annual", "annual+semi"):
        periods.append(1.0)
    if seasonal == "annual+semi":
        periods.append(0.5)

    for P in periods:
        w = 2 * np.pi / P
        if np.isclose(P, 1.0):
            names += ["annual_sin", "annual_cos"]
        elif np.isclose(P, 0.5):
            names += ["semiannual_sin", "semiannual_cos"]
        else:
            names += [f"period_{P:.3f}y_sin", f"period_{P:.3f}y_cos"]
        cols += [np.sin(w * t), np.cos(w * t)]

    A = np.column_stack(cols).astype(np.float64, copy=False)
    return A, names


def sincos_to_amplitude_phase(
    a_cos: np.ndarray, a_sin: np.ndarray, period: float
) -> tuple[np.ndarray, np.ndarray]:
    """Convert sin/cos coefficients to amplitude and phase.

    Parameters
    ----------
    a_cos : np.ndarray
        Cosine coefficients.
    a_sin : np.ndarray
        Sine coefficients.
    period : float
        Period in years.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Amplitude and phase arrays.

    """
    amp = np.sqrt(a_cos**2 + a_sin**2)
    phase = (period / (2 * np.pi)) * np.arctan2(-a_sin, a_cos)
    return amp, np.mod(phase, period)


# Fitting (NumPy default, optional JAX)
def _fit_block_numpy(
    A: np.ndarray,  # noqa: N803
    Y: np.ndarray,  # noqa: N803
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve (A,b) per pixel with missing data ignored.

    Parameters
    ----------
    A : np.ndarray
        Design matrix, shape (T, P).
    Y : np.ndarray
        Observations, shape (T, N).
    valid : np.ndarray
        Valid data mask, shape (T, N).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Coefficients (P, N) and MSE (N,).

    """
    _T, P = A.shape
    N = Y.shape[1]
    coeffs = np.full((P, N), np.nan, dtype=np.float64)
    mse = np.full((N,), np.nan, dtype=np.float64)

    # precompute normal matrix parts for each pixel via masking
    for j in range(N):
        m = valid[:, j]
        nobs = int(m.sum())
        if nobs < P:
            continue
        Aj = A[m]
        bj = Y[m, j]
        # stable via lstsq
        x, resid, rank, _ = np.linalg.lstsq(Aj, bj, rcond=None)
        coeffs[:, j] = x
        dof = max(1, nobs - rank)
        if resid.size:
            mse[j] = resid.item() / dof
        else:
            # compute residuals explicitly if lstsq didn't return aggregate
            mse[j] = np.sum((Aj @ x - bj) ** 2) / dof
    return coeffs, mse


def _maybe_jax_fit(
    design_matrix: np.ndarray,
    observations: np.ndarray,
    valid: np.ndarray,
    use_jax: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit using JAX if available, otherwise fall back to NumPy."""
    if not use_jax or not JAX_AVAILABLE:
        logger.debug("Falling back to NumPy fit (JAX unavailable).")
        return _fit_block_numpy(design_matrix, observations, valid)

    design_jax = jnp.asarray(design_matrix)  # (T, P)
    obs_jax = jnp.asarray(observations)  # (T, N)
    valid_jax = jnp.asarray(valid)  # (T, N)
    _T, P = design_matrix.shape

    @jax.vmap
    def _solve_pixel(y_col: jnp.ndarray, v_col: jnp.ndarray):
        # v_col is bool-like (T,). Build row weights w in {0,1} without bool indexing
        w = v_col.astype(design_jax.dtype)  # (T,)
        nobs = jnp.sum(v_col.astype(jnp.int32))  # scalar int

        # Weighted rows: zero-out invalid rows by multiplication (JAX-friendly).
        A_w = design_jax * w[:, None]  # (T, P)
        y_w = y_col * w  # (T,)

        def _bad():
            return jnp.full((P,), jnp.nan), jnp.array(jnp.nan)

        def _ok():
            x, _residuals, rank, _ = jnp.linalg.lstsq(A_w, y_w, rcond=None)
            # Compute MSE on valid rows only (weighted residuals == masked residuals)
            r = (design_jax @ x - y_col) * w
            dof = jnp.maximum(1, nobs - rank)
            mse = jnp.sum(r * r) / dof.astype(r.dtype)
            return x, mse

        return jax.lax.cond(nobs < P, _bad, _ok)

    coef_result, mse_result = _solve_pixel(obs_jax.T, valid_jax.T)  # (N,P), (N,)
    return np.asarray(coef_result.T), np.asarray(mse_result)


# Public API
@dataclass
class FitConfig:
    """Configuration for timeseries fitting."""

    poly_degree: int = 1  # 1 = velocity
    seasonal: Literal["none", "annual", "annual+semi"] = "annual"
    temporal_coherence_threshold: float | None = None
    backend: Literal["jax", "numpy"] = "jax"
    reference_index: int = 0  # 0 = first epoch


DEFAULT_CONFIG = FitConfig()


def fit_disp_timeseries(
    ds: xr.Dataset,
    *,
    cfg: FitConfig = DEFAULT_CONFIG,
    displacement_var: str = "displacement",
    reference_time_var: str = "reference_time",
    temporal_coherence_var: str = "temporal_coherence",
) -> xr.Dataset:
    """Fit velocity + seasonal (+ up to cubic) over a DISP stack.

    Parameters
    ----------
    ds : xr.Dataset
        Input DISP dataset.
    cfg : FitConfig
        Fitting configuration.
    displacement_var : str
        Name of displacement variable.
    reference_time_var : str
        Name of reference time variable.
    temporal_coherence_var : str
        Name of temporal coherence variable.

    Returns
    -------
    xr.Dataset
        Fitted results with coefficients and derived parameters.

    """
    # 1) Rebase moving references (lazy)
    if reference_time_var in ds:
        disp = rebase_displacement(ds, displacement_var, reference_time_var)
    else:
        disp = ds[displacement_var]

    # 2) Apply temporal coherence threshold if requested
    if cfg.temporal_coherence_threshold is not None and temporal_coherence_var in ds:
        tc = ds[temporal_coherence_var]
        disp = disp.where(tc >= cfg.temporal_coherence_threshold)

    # 3) Design matrix
    A, names = build_design_matrix(
        disp.time.values,
        poly_degree=cfg.poly_degree,
        seasonal=cfg.seasonal,
        reference_index=cfg.reference_index,
    )

    # 4) Chunked solve
    chunks = {"time": -1}
    if disp.chunks is not None:
        if len(disp.chunks) > 1:
            chunks["y"] = disp.chunks[1][0]
        if len(disp.chunks) > 2:
            chunks["x"] = disp.chunks[2][0]
    else:
        chunks.update({"y": 1024, "x": 1024})
    disp_c = disp.chunk(chunks)

    def _solve_block(arr: xr.DataArray) -> xr.Dataset:
        T, H, W = arr.shape
        data = arr.data  # dask array -> materialized inside block
        # to numpy
        Y = np.asarray(data).reshape(T, H * W)
        valid = np.isfinite(Y)
        coeffs, mse = _maybe_jax_fit(A, Y, valid, use_jax=(cfg.backend == "jax"))
        P = coeffs.shape[0]
        coeffs = coeffs.reshape(P, H, W)
        mse = mse.reshape(H, W)

        out = xr.Dataset(
            {
                "coefficients": (["coefficient", "y", "x"], coeffs),
                "mse": (["y", "x"], mse),
            },
            coords={
                "coefficient": names,
                "y": arr.y if "y" in arr.coords else np.arange(H),
                "x": arr.x if "x" in arr.coords else np.arange(W),
            },
        )
        return out

    template_chunks = {"coefficient": -1}
    if disp_c.chunks is not None:
        y_dim_idx = list(disp_c.dims).index("y")
        x_dim_idx = list(disp_c.dims).index("x")
        template_chunks["y"] = disp_c.chunks[y_dim_idx][0]
        template_chunks["x"] = disp_c.chunks[x_dim_idx][0]
    else:
        template_chunks.update({"y": 1024, "x": 1024})

    template = xr.Dataset(
        {
            "coefficients": (
                ["coefficient", "y", "x"],
                np.full(
                    (len(names), disp.sizes["y"], disp.sizes["x"]),
                    np.nan,
                    dtype=np.float64,
                ),
            ),
            "mse": (
                ["y", "x"],
                np.full((disp.sizes["y"], disp.sizes["x"]), np.nan, dtype=np.float64),
            ),
        },
        coords={"coefficient": names, "y": disp.y, "x": disp.x},
    ).chunk(template_chunks)

    ds_fit = xr.map_blocks(_solve_block, disp_c, template=template)

    # 5) Derived seasonal diagnostics
    def _maybe(name: str) -> xr.DataArray | None:
        return (
            ds_fit["coefficients"].sel(coefficient=name, drop=True)
            if name in names
            else None
        )

    ds_fit = ds_fit.assign_attrs(
        model=(
            f"poly_degree={cfg.poly_degree}, seasonal={cfg.seasonal},"
            f" ref_idx={cfg.reference_index}"
        )
    )

    # Convenience layers
    # velocity (linear term) = 1st-order coefficient scaled by factorial(1)=1
    vel = _maybe("linear_trend")
    if vel is not None:
        ds_fit["velocity"] = vel

    if "annual_sin" in names and "annual_cos" in names:
        A_cos = ds_fit["coefficients"].sel(coefficient="annual_cos")
        A_sin = ds_fit["coefficients"].sel(coefficient="annual_sin")
        amp, ph = sincos_to_amplitude_phase(A_cos, A_sin, period=1.0)
        ds_fit["annual_amplitude"] = amp
        ds_fit["annual_phase"] = ph

    if "semiannual_sin" in names and "semiannual_cos" in names:
        A_cos = ds_fit["coefficients"].sel(coefficient="semiannual_cos")
        A_sin = ds_fit["coefficients"].sel(coefficient="semiannual_sin")
        amp, ph = sincos_to_amplitude_phase(A_cos, A_sin, period=0.5)
        ds_fit["semiannual_amplitude"] = amp
        ds_fit["semiannual_phase"] = ph

    return ds_fit


def fit_cli(
    opera_netcdfs: Sequence[Path],
    output_dir: Path,
    *,
    poly_degree: int = 1,
    seasonal: Literal["none", "annual", "annual+semi"] = "annual",
    temporal_coherence_threshold: float | None = None,
    x_chunks: int = 1024,
    y_chunks: int = 1024,
    backend: Literal["jax", "numpy"] = "jax",
) -> None:
    """Fit velocity + seasonal + up to 3rd order over a DISP stack.

    Parameters
    ----------
    opera_netcdfs : Sequence[Path]
        Directory or single NetCDF of OPERA DISP products.
    output_dir : Path
        Output Zarr or NetCDF path.
    poly_degree : int
        Polynomial degree (1=velocity, up to 3), by default 1.
    seasonal : {"none", "annual", "annual+semi"}
        Seasonal terms to include, by default "annual".
    temporal_coherence_threshold : float, optional
        If set, mask pixels with temporal_coherence < threshold.
    x_chunks : int
        Chunk size for x dimension, by default 1024.
    y_chunks : int
        Chunk size for y dimension, by default 1024.
    backend : {"jax", "numpy"}
        Computation backend, by default "jax".

    """
    cfg = FitConfig(
        poly_degree=poly_degree,
        seasonal=seasonal,
        temporal_coherence_threshold=temporal_coherence_threshold,
        backend=backend,
    )

    dps = DispProductStack.from_file_list(opera_netcdfs)
    ds = xr.open_mfdataset(dps.filenames, chunks={"x": x_chunks, "y": y_chunks})
    ds_fit = fit_disp_timeseries(ds, cfg=cfg)

    output_dir = output_dir.with_suffix(output_dir.suffix or ".zarr")
    if output_dir.suffix == ".zarr":
        ds_fit.to_zarr(output_dir, mode="w")
    else:
        ds_fit.to_netcdf(output_dir)

    logger.info(f"Wrote {output_dir}")
