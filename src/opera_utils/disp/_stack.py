"""Make analysis-ready DISP-S1 stacks for a bounding box in one call."""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from opera_utils import burst_frame_db
from opera_utils._types import Bbox
from opera_utils._version import version as opera_utils_version

from ._download import download_products
from ._enums import QualityDataset, ReferenceMethod
from ._reformat import reformat_stack
from ._search import UrlType, search

logger = logging.getLogger("opera_utils")

MANIFEST_NAME = "stack_manifest.json"
OrbitPassName = Literal["ASCENDING", "DESCENDING"]


@dataclass
class FrameStack:
    """Summary of one DISP-S1 frame in a stack request."""

    frame_id: int
    orbit_pass: str
    contains_bbox: bool
    num_products: int
    first_date: str | None
    last_date: str | None
    subset_dir: str | None = None
    stack_path: str | None = None
    files: list[str] = field(default_factory=list)


def select_frames(
    bbox: tuple[float, float, float, float],
    frame_ids: Sequence[int] | None = None,
    orbit_pass: OrbitPassName | None = None,
) -> list[tuple[int, str, bool]]:
    """Pick the DISP-S1 frames to use for a bounding box.

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        (west, south, east, north) in degrees longitude/latitude.
    frame_ids : Sequence[int] | None
        Use only these frames. If None, use every frame whose footprint
        contains `bbox`. When no frame contains it, use every intersecting frame.
    orbit_pass : str | None
        Keep only "ASCENDING" or "DESCENDING" frames. None keeps both.

    Returns
    -------
    list[tuple[int, str, bool]]
        (frame ID, orbit pass, footprint contains `bbox`) for each frame,
        sorted by frame ID.

    Raises
    ------
    ValueError
        If no frame is left after the selection.

    """
    from shapely.geometry import box  # noqa: PLC0415

    gdf = burst_frame_db.get_frame_geodataframe()
    region = box(*bbox)
    frames = gdf[gdf.geometry.intersects(region)]
    if frame_ids is not None:
        frames = frames[frames.index.isin([int(f) for f in frame_ids])]
    if orbit_pass is not None:
        frames = frames[frames.orbit_pass == orbit_pass]
    contains = frames.geometry.contains(region)
    if frame_ids is None and contains.any():
        frames = frames[contains]
        contains = contains[contains]
    if frames.empty:
        msg = (
            f"No DISP-S1 frame intersects {bbox} with frame_ids={frame_ids},"
            f" orbit_pass={orbit_pass}"
        )
        raise ValueError(msg)
    if not contains.all():
        logger.warning(
            "Frames %s cover only part of %s",
            sorted(int(f) for f in frames.index[~contains]),
            bbox,
        )
    return sorted(
        (int(fid), str(row.orbit_pass), bool(contains[fid]))
        for fid, row in frames.iterrows()
    )


def make_stack(
    bbox: tuple[float, float, float, float] | None = None,
    wkt: str | None = None,
    output_dir: Path = Path("./disp_s1_stack"),
    start_datetime: datetime | None = None,
    end_datetime: datetime | None = None,
    frame_ids: list[int] | None = None,
    orbit_pass: OrbitPassName | None = None,
    list_only: bool = False,
    product_version: str | None = "1.0",
    url_type: UrlType = UrlType.HTTPS,
    num_workers: int = 8,
    quality_datasets: Sequence[QualityDataset] | None = None,
    quality_thresholds: Sequence[float] | None = None,
    reference_method: ReferenceMethod = ReferenceMethod.HIGH_COHERENCE,
    apply_solid_earth_corrections: bool = True,
    apply_ionospheric_corrections: bool = False,
    overwrite: bool = False,
) -> list[FrameStack]:
    """Find, download, crop, and reformat DISP-S1 data for a bounding box.

    For every frame that covers the region, this downloads the products in
    the date range, crops them to the region, and writes one Zarr stack per
    frame (``F<frame_id>.zarr``). The displacement is rebased to the first
    date of each frame. A ``stack_manifest.json`` in `output_dir` records the
    request, the frames and the files.

    Run it again with a later `end_datetime` to extend the stacks: complete
    subsets are kept, and a stack is rebuilt only when its file list changed.

    Parameters
    ----------
    bbox : tuple[float, float, float, float] | None
        (west, south, east, north) in degrees longitude/latitude.
    wkt : str | None
        Alternative to `bbox`: a WKT polygon. Its bounds are used.
    output_dir : Path
        Directory for the subsets, the stacks and the manifest.
        Default is "./disp_s1_stack".
    start_datetime : datetime | None
        Start of the date range. If None, start at the first product (2016).
    end_datetime : datetime | None
        End of the date range. If None, end at the latest product.
    frame_ids : list[int] | None
        Use only these frames. If None, use every frame that contains the
        region (both orbit passes, and overlapping tracks). When no frame
        contains it, use every frame that intersects it.
    orbit_pass : str | None
        Keep only "ASCENDING" or "DESCENDING" frames. None keeps both.
    list_only : bool
        Print the selected frames with product counts and dates, then stop
        without downloading. Default is False.
    product_version : str | None
        DISP-S1 product version. Default is "1.0".
    url_type : UrlType
        HTTPS (default) or S3. S3 works only in the us-west-2 AWS region.
    num_workers : int
        Number of parallel downloads. Default is 8.
    quality_datasets : Sequence[QualityDataset] | None
        Quality layers to mask the displacement with while it is rebased.
        Default is None: no mask. The quality layers stay in each stack,
        so a viewer or analysis can apply a mask later.
    quality_thresholds : Sequence[float] | None
        Thresholds for `quality_datasets`, one per dataset.
    reference_method : ReferenceMethod
        Spatial reference for each date. Default is HIGH_COHERENCE.
    apply_solid_earth_corrections : bool
        Subtract the solid earth tide correction. Default is True.
    apply_ionospheric_corrections : bool
        Subtract the ionospheric delay correction. Default is False.
    overwrite : bool
        Rebuild every stack, even when its file list did not change.

    Returns
    -------
    list[FrameStack]
        One entry per selected frame with products in the date range.

    Raises
    ------
    ValueError
        If neither or both of `bbox` and `wkt` are given, or no frame is found.

    """
    if (bbox is None) == (wkt is None):
        msg = "Provide exactly one of `bbox` or `wkt`"
        raise ValueError(msg)
    if wkt is not None:
        from shapely import from_wkt  # noqa: PLC0415

        bbox = tuple(from_wkt(wkt).bounds)  # type: ignore[assignment]
    assert bbox is not None
    region = Bbox(*bbox)

    frames = select_frames(region, frame_ids=frame_ids, orbit_pass=orbit_pass)
    output_dir = Path(output_dir)
    manifest_path = output_dir / MANIFEST_NAME
    previous = _read_previous_files(manifest_path)

    stacks: list[FrameStack] = []
    for frame_id, frame_pass, contains in frames:
        products = search(
            frame_id=frame_id,
            product_version=product_version,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            url_type=url_type,
        )
        stack = FrameStack(
            frame_id=frame_id,
            orbit_pass=frame_pass,
            contains_bbox=contains,
            num_products=len(products),
            first_date=_date_str(products[0].reference_datetime) if products else None,
            last_date=_date_str(products[-1].secondary_datetime) if products else None,
        )
        logger.info(
            "Frame %s (%s): %s products, %s to %s",
            frame_id,
            frame_pass,
            stack.num_products,
            stack.first_date,
            stack.last_date,
        )
        if not products:
            continue
        stacks.append(stack)
        if list_only:
            continue

        subset_dir = output_dir / f"F{frame_id:05d}"
        files = download_products(
            products, bbox=region, output_dir=subset_dir, num_workers=num_workers
        )
        stack.subset_dir = str(subset_dir)
        stack.files = sorted(Path(f).name for f in files)

        stack_path = output_dir / f"F{frame_id:05d}.zarr"
        stack.stack_path = str(stack_path)
        if (
            stack_path.exists()
            and not overwrite
            and previous.get(frame_id) == stack.files
        ):
            logger.info("Keeping %s: its file list did not change", stack_path)
            continue
        if stack_path.exists():
            shutil.rmtree(stack_path)
        reformat_stack(
            input_files=[Path(f) for f in files],
            output_name=str(stack_path),
            quality_datasets=quality_datasets,
            quality_thresholds=quality_thresholds,
            reference_method=reference_method,
            apply_solid_earth_corrections=apply_solid_earth_corrections,
            apply_ionospheric_corrections=apply_ionospheric_corrections,
        )

    if list_only:
        _print_frames(stacks)
        return stacks

    _write_manifest(
        manifest_path,
        stacks,
        request={
            "bbox": list(region),
            "start_datetime": _iso(start_datetime),
            "end_datetime": _iso(end_datetime),
            "frame_ids": frame_ids,
            "orbit_pass": orbit_pass,
            "product_version": product_version,
            "quality_datasets": [str(q.value) for q in quality_datasets or []],
            "quality_thresholds": list(quality_thresholds or []),
            "reference_method": reference_method.value,
            "apply_solid_earth_corrections": apply_solid_earth_corrections,
            "apply_ionospheric_corrections": apply_ionospheric_corrections,
        },
    )
    return stacks


def _date_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def _read_previous_files(manifest_path: Path) -> dict[int, list[str]]:
    if not manifest_path.exists():
        return {}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {f["frame_id"]: f["files"] for f in data["frames"]}


def _write_manifest(path: Path, stacks: list[FrameStack], request: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "created": datetime.now(tz=timezone.utc).isoformat(),
        "opera_utils_version": opera_utils_version,
        "request": request,
        "frames": [asdict(s) for s in stacks],
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", path)


def _print_frames(stacks: list[FrameStack]) -> None:
    print("frame_id  orbit_pass   contains_bbox  products  first_date  last_date")
    for s in stacks:
        print(
            f"{s.frame_id:<9} {s.orbit_pass:<12} {s.contains_bbox!s:<14}"
            f" {s.num_products:<9} {s.first_date}  {s.last_date}"
        )
