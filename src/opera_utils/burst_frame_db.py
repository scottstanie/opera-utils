from __future__ import annotations

import json
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path

from . import datasets
from ._types import Bbox, PathOrStr
from .bursts import _normalize


def read_zipped_json(filename: PathOrStr):
    """Read a zipped JSON file and returns its contents as a dictionary.

    Parameters
    ----------
    filename : PathOrStr
        The path to the zipped JSON file.

    Returns
    -------
    dict
        The contents of the zipped JSON file as a dictionary.
    """
    if Path(filename).suffix == ".zip":
        with zipfile.ZipFile(filename) as zf:
            bytes = zf.read(str(Path(filename).name).replace(".zip", ""))
            return json.loads(bytes.decode())
    else:
        with open(filename) as f:
            return json.load(f)


def get_frame_to_burst_mapping(
    frame_id: int, json_file: PathOrStr | None = None
) -> dict:
    """Get the frame data for one frame ID.

    Parameters
    ----------
    frame_id : int
        The ID of the frame to get the bounding box for.
    json_file : PathOrStr, optional
        The path to the JSON file containing the frame-to-burst mapping.
        If `None`, uses the zip file contained in `data/`
    Returns
    -------
    dict
        The frame data for the given frame ID.
    """
    if json_file is None:
        json_file = datasets.fetch_frame_to_burst_mapping_file()
    js = read_zipped_json(json_file)
    return js["data"][str(frame_id)]


def get_frame_geojson(
    as_geodataframe: bool = False,
    columns: Sequence[str] | None = None,
    frame_ids: Sequence[str] | None = None,
) -> dict:
    """Get the GeoJSON for the frame geometries."""
    where = _form_where_in_query(frame_ids, "frame_id") if frame_ids else None
    return _get_geojson(
        datasets.fetch_frame_geometries_simple(),
        as_geodataframe=as_geodataframe,
        columns=columns,
        where=where,
        index_name="frame_id",
    )


def get_burst_id_geojson(
    as_geodataframe: bool = False,
    columns: Sequence[str] | None = None,
    burst_ids: Sequence[str] | None = None,
) -> dict:
    """Get the GeoJSON for the burst_id geometries."""
    where = (
        _form_where_in_query(map(_normalize, burst_ids), "burst_id_jpl")
        if burst_ids
        else None
    )
    return _get_geojson(
        datasets.fetch_burst_id_geometries_simple(),
        as_geodataframe=as_geodataframe,
        columns=columns,
        where=where,
        index_name="burst_id_jpl",
    )


def _form_where_in_query(values: Iterable[str], column_name):
    # Example:
    # "burst_id_jpl in ('t005_009471_iw2','t007_013706_iw2','t008_015794_iw1')"
    where_in_str = ",".join(f"'{b}'" for b in values)
    return f"{column_name} IN ({where_in_str})"


def _get_geojson(
    f,
    as_geodataframe: bool = False,
    columns: Sequence[str] | None = None,
    where: str | None = None,
    index_name: str | None = None,
) -> dict:
    # https://gdal.org/user/ogr_sql_dialect.html#where
    # https://pyogrio.readthedocs.io/en/latest/introduction.html#filter-records-by-attribute-value
    if as_geodataframe:
        from pyogrio import read_dataframe

        # import geopandas as gpd
        # return gpd.read_file(f)
        gdf = read_dataframe(f, columns=columns, where=where, fid_as_index=True)
        if index_name:
            if index_name in gdf.columns:
                return gdf.drop_duplicates(subset=index_name).set_index(index_name)
            else:
                gdf.index.name = index_name
                return gdf

    return read_zipped_json(f)


def get_frame_bbox(
    frame_id: int, json_file: PathOrStr | None = None
) -> tuple[int, Bbox]:
    """Get the bounding box of a frame from a JSON file.

    Parameters
    ----------
    frame_id : int
        The ID of the frame to get the bounding box for.
    json_file : PathOrStr, optional
        The path to the JSON file containing the frame-to-burst mapping.
        If `None`, fetches the remote zip file from `datasets`

    Returns
    -------
    epsg : int
        EPSG code for the bounds coordinates
    tuple[float, float, float, float]
        bounding box coordinates (xmin, ymin, xmax, ymax)
    """
    frame_dict = get_frame_to_burst_mapping(frame_id=frame_id, json_file=json_file)
    epsg = int(frame_dict["epsg"])
    bounds = (
        float(frame_dict["xmin"]),
        float(frame_dict["ymin"]),
        float(frame_dict["xmax"]),
        float(frame_dict["ymax"]),
    )
    return epsg, Bbox(*bounds)


def get_burst_ids_for_frame(
    frame_id: int, json_file: PathOrStr | None = None
) -> list[str]:
    """Get the burst IDs for one frame ID.

    Parameters
    ----------
    frame_id : int
        The ID of the frame to get the bounding box for.
    json_file : PathOrStr, optional
        The path to the JSON file containing the frame-to-burst mapping.
        If `None`, fetches the remote zip file from `datasets`

    Returns
    -------
    list[str]
        The burst IDs for the given frame ID.
    """
    frame_data = get_frame_to_burst_mapping(frame_id, json_file)
    return frame_data["burst_ids"]


def get_burst_to_frame_mapping(
    burst_id: str, json_file: PathOrStr | None = None
) -> dict:
    """Get the burst data for one burst ID.

    Parameters
    ----------
    burst_id : str
        The ID of the burst to get the frame IDs for.
    json_file : PathOrStr, optional
        The path to the JSON file containing the burst-to-frame mapping.
        If `None`, uses the zip file fetched from `datasets`

    Returns
    -------
    dict
        The burst data for the given burst ID.
    """
    if json_file is None:
        json_file = datasets.fetch_burst_to_frame_mapping_file()
    js = read_zipped_json(json_file)
    return js["data"][_normalize(burst_id)]


def get_frame_ids_for_burst(
    burst_id: str, json_file: PathOrStr | None = None
) -> list[int]:
    """Get the frame IDs for one burst ID.

    Parameters
    ----------
    burst_id : str
        The ID of the burst to get the frame IDs for.
    json_file : PathOrStr, optional
        The path to the JSON file containing the burst-to-frame mapping.
        If `None`, fetches the remote zip file from `datasets`

    Returns
    -------
    list[int]
        The frame IDs for the given burst ID.
        Most burst IDs have 1, but burst IDs in the overlap are in
        2 frames.
    """
    burst_data = get_burst_to_frame_mapping(burst_id, json_file)
    return burst_data["frame_ids"]
