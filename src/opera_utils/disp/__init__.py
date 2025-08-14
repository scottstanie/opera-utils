"""OPERA DISP-S1 utilities for reading and analyzing displacement data."""

from __future__ import annotations

from ._download import download_static_products
from ._product import (
    DispProduct,
    DispProductStack,
    DispStaticProduct,
    ProductType,
    StaticAsset,
    UrlType,
)
from ._rebase import create_rebased_displacement, rebase_timeseries
from ._reformat import reformat_stack
from ._remote import open_file, open_h5
from ._search import search

__all__ = [
    "DispProduct",
    "DispProductStack",
    "DispStaticProduct",
    "ProductType",
    "StaticAsset",
    "UrlType",
    "create_rebased_displacement",
    "download_static_products",
    "open_file",
    "open_h5",
    "rebase_timeseries",
    "reformat_stack",
    "search",
]
