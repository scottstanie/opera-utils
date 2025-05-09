#!/usr/bin/env python
"""Concatenate displacement tiles into a CF-compliant virtual dataset.

Usage
-----
python create_virtual_stack.py --output disp_stack_vds.nc disp-s1/*.nc

The output file contains a single 3-D variable
    <dataset_name>(time, y, x)
whose slices reference the original NetCDF files through HDF5 Virtual Datasets,
plus the usual CF coordinate variables (time, y, x) and grid-mapping metadata.

Note that in-place modification of the virtual file will change the original
underlying data.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import h5netcdf
import h5py
import numpy as np
import tyro
from pyproj import CRS

import opera_utils
from opera_utils.disp import _netcdf


def create_virtual_stack(
    input_files: Sequence[Path],
    /,
    output: Path,
    dataset_names: Sequence[str] | None = None,
) -> None:
    """Concatenate single-date NetCDF displacement files into a stack.

    Uses the virtual HDF5 stack along the time dimension.
    See https://docs.h5py.org/en/stable/vds.html for more info.

    Parameters
    ----------
    input_files : list[pathlib.Path]
        Source NetCDF/CF files.
    output : pathlib.Path
        Path of the VDS file to create. Suffix may be “.nc” or “.h5”.
    dataset_names : Sequence[str], optional
        Names of the 2-D variable to concatenate and include in `output`.
        If None provided, creates for all 2D datasets in `input_files`.
    """
    if not input_files:
        raise ValueError("No input files provided.")

    if dataset_names is None:
        dataset_names = _get_2d_datasets(input_files[0])

    # Inspect the first file to determine shapes, dtype, & metadata
    with h5py.File(input_files[0], "r") as hf0:
        ny, nx = hf0["displacement"].shape

        # Coordinate values and attributes we will copy later
        x_data = hf0["x"][:]
        y_data = hf0["y"][:]

        spatial_ref_attrs = dict(hf0["spatial_ref"].attrs)
        geo_transform = [float(n) for n in spatial_ref_attrs["GeoTransform"].split(" ")]
        crs = CRS.from_wkt(spatial_ref_attrs["crs_wkt"])
        name_to_dtype = {name: hf0[name].dtype for name in dataset_names}

    n_time = len(input_files)
    file_datetimes = [
        opera_utils.get_dates(f, fmt="%Y%m%dT%H%M%S")[1] for f in input_files
    ]

    # Create the VDS container
    with h5netcdf.File(output, "w", libver="latest") as f:
        _netcdf._create_grid_mapping(group=f, crs=crs, gt=geo_transform)
        _netcdf._create_dimension_variables(
            group=f, datetimes=file_datetimes, y=y_data, x=x_data
        )

    with h5py.File(output, "r+", libver="latest") as hf:
        time_ds = hf["time"]
        y_ds = hf["y"]
        x_ds = hf["x"]
        for dataset_name in dataset_names:
            dtype = name_to_dtype[dataset_name]
            layout = h5py.VirtualLayout(shape=(n_time, ny, nx), dtype=dtype)
            for k, src in enumerate(map(str, input_files)):
                layout[k, :, :] = h5py.VirtualSource(src, dataset_name, shape=(ny, nx))
            dset = hf.create_virtual_dataset(dataset_name, layout, fillvalue=np.nan)

            # Attach dimension scales → proper CF & GDAL recognition
            dset.dims[0].attach_scale(time_ds)
            dset.dims[1].attach_scale(y_ds)
            dset.dims[2].attach_scale(x_ds)
            dset.attrs["grid_mapping"] = "spatial_ref"


def _get_2d_datasets(filename: Path | str) -> list[str]:
    names = []

    def _add_name(name):
        if isinstance(hf[name], h5py.Dataset) and hf[name].ndim == 2:
            names.append(name)

    with h5py.File(filename) as hf:
        hf.visit(_add_name)
    return names


if __name__ == "__main__":
    import tyro

    tyro.cli(create_virtual_stack)
