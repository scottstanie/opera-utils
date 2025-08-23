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

import dask.array as da
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


def rebase_displacement(ds: xr.Dataset) -> xr.DataArray:
    """Return a rebased displacement (time,y,x) DataArray.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with displacement data.

    Returns
    -------
    xr.DataArray
        Rebased displacement array.

    """
    displacement_var = "displacement"
    reference_time_var = "reference_time"
    da_disp = ds[displacement_var]

    # If no reference time provided, leave as-is
    if reference_time_var not in ds:
        return da_disp

    refs = ds[reference_time_var].values

    def _block(arr: xr.DataArray) -> xr.DataArray:
        # arr is one (time,y,x) block
        out = rebase_timeseries(np.asarray(arr), refs)
        return xr.DataArray(out, coords=arr.coords, dims=arr.dims)

    # Preserve existing chunking; ensure time chunk = -1 for correct crossover behavior
    chunks = {"time": -1}
    if da_disp.chunks:
        if len(da_disp.chunks) > 1:
            chunks["y"] = da_disp.chunks[1][0]
        if len(da_disp.chunks) > 2:
            chunks["x"] = da_disp.chunks[2][0]
    else:
        chunks |= {"y": 1024, "x": 1024}

    return da_disp.chunk(chunks).map_blocks(_block).transpose("time", "y", "x")


def datetime_to_float(
    dates: Sequence[date | datetime], reference_index: int = 0
) -> np.ndarray:
    """Convert a sequence of datetime objects to a float representation.

    Output units are in years since the first item in `dates`.

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
    sec_per_year = 60 * 60 * 24 * 365.25
    date_arr = np.asarray(dates).astype("datetime64[s]")
    # Reference the 0 to the first date
    date_arr = date_arr - date_arr[reference_index]
    return date_arr.astype(float) / sec_per_year


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


def sin_cos_to_amplitude_phase(
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


def _fit_block_numpy(A, B, valid):
    """Solves least squares problem subject to missing data in the right hand side.

    Parameters
    ----------
    A : ndarray
        m x n system matrix.
    B : ndarray
        m x k matrix representing the k right hand side data vectors of size m.
    valid : ndarray
        m x k boolean matrix of missing data (`False` indicate missing values)

    Returns
    -------
    X : ndarray
        n x k matrix that minimizes norm(valid*(AX - B))
    residuals : np.array 1D
        Sums of (k,) squared residuals: squared Euclidean 2-norm for `b - A @ x`

    Reference
    ---------
    http://alexhwilliams.info/itsneuronalblog/2018/02/26/censored-lstsq/

    """
    import jax.numpy as np  # noqa: PLC0415

    # if B is a vector, simply drop out corresponding rows in A
    if B.ndim == 1 or B.shape[1] == 1:
        return np.linalg.lstsq(A[valid], B[valid])[0]

    # else solve via tensor representation
    rhs = np.dot(A.T, valid * B).T[:, :, None]  # k x n x 1 tensor
    T = np.matmul(
        A.T[None, :, :], valid.T[:, :, None] * A[None, :, :]
    )  # k x n x n tensor
    x = np.squeeze(np.linalg.solve(T, rhs)).T  # transpose to get n x k
    residuals = np.linalg.norm(A @ x - (B * valid.astype(int)), axis=0)
    return x, residuals


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
    backend: Literal["jax", "numpy"] = "numpy"
    reference_index: int = 0  # 0 = first epoch


DEFAULT_CONFIG = FitConfig()


import jax.numpy as jnp
from jax import Array, jit, vmap


@jit
def weighted_lstsq_single(
    A: Array,
    b: Array,
    weights: Array,
) -> Array:
    r"""Perform weighted least squares for one data vector.

    Minimizes the weighted 2-norm of the residual vector:

    \[
        || b - A x ||^2_W
    \]

    where \(W\) is a diagonal matrix of weights.

    Parameters
    ----------
    A : Array
        Incidence matrix of shape (n_ifgs, n_sar_dates - 1)
    b : Array, 1D
        The phase differences between the ifg pairs
    weights : Array, 1D, optional
        The weights for each element of `b`.

    Returns
    -------
    x : np.array 1D
        The estimated phase for each SAR acquisition
    residuals : np.array 1D
        Sums of squared residuals: Squared Euclidean 2-norm for `b - A @ x`
        For a 1D `b`, this is a scalar.

    """
    # scale both A and b by sqrt so we are minimizing
    sqrt_weights = jnp.sqrt(weights)
    # Multiply each data point by sqrt(weight)
    b_scaled = b * sqrt_weights
    # Multiply each row by sqrt(weight)
    A_scaled = A * sqrt_weights[:, None]

    # Run the weighted least squares
    x, residuals, _rank, _sing_vals = jnp.linalg.lstsq(A_scaled, b_scaled)
    # TODO: do we need special handling?
    # if rank < A.shape[1]:
    #     # logger.warning("Rank deficient solution")

    return x, residuals.ravel()


@jit
def invert_stack(
    A: Array,
    dphi: Array,
    weights: Array | None = None,
) -> tuple[Array, Array]:
    """Solve the SBAS problem for a stack of unwrapped phase differences.

    Parameters
    ----------
    A : Array
        Incidence matrix of shape (n_ifgs, n_sar_dates - 1)
    dphi : Array
        The phase differences between the ifg pairs, shape=(n_ifgs, n_rows, n_cols)
    weights : Array, optional
        The weights for each element of `dphi`.
        Same shape as `dphi`.
        If not provided, all weights are set to 1 (ordinary least squares).

    Returns
    -------
    phi : np.array 3D
        The estimated phase for each SAR acquisition
        Shape is (n_sar_dates - 1, n_rows, n_cols)
    residuals : np.array 2D
        Sums of squared residuals: Squared Euclidean 2-norm for `dphi - A @ x`
        Shape is (n_rows, n_cols)

    Notes
    -----
    To mask out data points of a pixel, the weight can be set to 0.
    When `A` remains full rank, setting the weight to zero is the same as removing
    the entry from the data vector and the corresponding row from `A`.

    """
    n_ifgs, n_rows, n_cols = dphi.shape

    if weights is None:
        # Can use ordinary least squares with no weights
        # Reshape to be size (M, K) instead of 3D
        b = dphi.reshape(n_ifgs, -1)
        phase_cols, residuals_cols, _, _ = jnp.linalg.lstsq(A, b)
        # Reshape the phase and residuals to be 3D
        phase = phase_cols.reshape(-1, n_rows, n_cols)
        residuals = residuals_cols.reshape(n_rows, n_cols)
    else:
        # vectorize the solve function to work on 2D and 3D arrays
        # We are not vectorizing over the A matrix, only the dphi vector
        # Solve 2d shapes: (nrows, n_ifgs) -> (nrows, n_sar_dates)
        invert_2d = vmap(weighted_lstsq_single, in_axes=(None, 1, 1), out_axes=(1, 1))
        # Solve 3d shapes: (nrows, ncols, n_ifgs) -> (nrows, ncols, n_sar_dates)
        invert_3d = vmap(invert_2d, in_axes=(None, 2, 2), out_axes=(2, 2))
        phase, residuals = invert_3d(A, dphi, weights)
        # Reshape the residuals to be 2D
        residuals = residuals[0]

    return phase, residuals


def fit_disp_timeseries(
    ds: xr.Dataset,
    *,
    cfg: FitConfig = DEFAULT_CONFIG,
    chunks: tuple[int, int] = (512, 512),
) -> xr.Dataset:
    """Fit velocity + seasonal (+ up to cubic) over a DISP stack.

    Parameters
    ----------
    ds : xr.Dataset
        Input DISP dataset.
    cfg : FitConfig
        Fitting configuration.
    chunks : tuple[int, int]
        Size of spatial blocks to process at a time

    Returns
    -------
    xr.Dataset
        Fitted results with coefficients and derived parameters.

    """
    # 1) Rebase moving references (lazy)
    disp = rebase_displacement(ds)

    # 2) Apply temporal coherence threshold if requested
    if cfg.temporal_coherence_threshold is not None:
        tc = ds["temporal_coherence"]
        disp = disp.where(tc >= cfg.temporal_coherence_threshold)

    # 3) Design matrix
    A, names = build_design_matrix(
        disp.time.values,
        poly_degree=cfg.poly_degree,
        seasonal=cfg.seasonal,
        reference_index=cfg.reference_index,
    )

    # 4) Chunked solve
    chunk_dict = {"time": -1, "y": chunks[0], "x": chunks[1]}
    disp_c = disp.chunk(chunk_dict)

    def _solve_block(arr: xr.DataArray) -> xr.Dataset:
        _T, H, W = arr.shape
        data = arr.data  # dask array -> materialized inside block
        # weights = ... # TODO: how do you get map_blocks to take 2 dataarrays
        coeffs, mse = invert_stack(A, data, weights=None)

        # # to numpy
        # Y = np.asarray(data).reshape(T, H * W)
        # valid = np.isfinite(Y)
        # coeffs, mse = _maybe_jax_fit(A, Y, valid, use_jax=(cfg.backend == "jax"))
        # P = coeffs.shape[0]
        # coeffs = coeffs.reshape(P, H, W)
        # mse = mse.reshape(H, W)

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

    P = len(names)
    H = disp.sizes["y"]
    W = disp.sizes["x"]

    template_chunks = {"coefficient": -1}
    if disp.chunks is not None:
        y_dim_idx = list(disp.dims).index("y")
        x_dim_idx = list(disp.dims).index("x")
        template_chunks["y"] = disp.chunks[y_dim_idx][0]
        template_chunks["x"] = disp.chunks[x_dim_idx][0]
    else:
        template_chunks.update({"y": 1024, "x": 1024})

    coeffs_da = da.empty(
        (P, H, W),
        dtype="float32",
        chunks=(P, template_chunks["y"], template_chunks["x"]),
    )
    mse_da = da.empty(
        (H, W),
        dtype="float32",
        chunks=(template_chunks["y"], template_chunks["x"]),
    )

    template = xr.Dataset(
        {
            "coefficients": (["coefficient", "y", "x"], coeffs_da),
            "mse": (["y", "x"], mse_da),
        },
        coords={"coefficient": names, "y": disp.y, "x": disp.x},
    )

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
        amp, ph = sin_cos_to_amplitude_phase(A_cos, A_sin, period=1.0)
        ds_fit["annual_amplitude"] = amp
        ds_fit["annual_phase"] = ph

    if "semiannual_sin" in names and "semiannual_cos" in names:
        A_cos = ds_fit["coefficients"].sel(coefficient="semiannual_cos")
        A_sin = ds_fit["coefficients"].sel(coefficient="semiannual_sin")
        amp, ph = sin_cos_to_amplitude_phase(A_cos, A_sin, period=0.5)
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
    ds = ds.chunk({"time": -1, "y": y_chunks, "x": x_chunks})
    print(ds)
    ds_fit = fit_disp_timeseries(ds, cfg=cfg)

    output_dir = output_dir.with_suffix(output_dir.suffix or ".zarr")
    if output_dir.suffix == ".zarr":
        ds_fit.to_zarr(output_dir, mode="w")
    else:
        ds_fit.to_netcdf(output_dir)

    logger.info(f"Wrote {output_dir}")
