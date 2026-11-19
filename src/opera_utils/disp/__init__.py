"""OPERA DISP-S1 utilities for reading and analyzing displacement data."""

from __future__ import annotations

from ._product import DispProduct, DispProductStack
from ._rebase import create_rebased_displacement, rebase_timeseries
from ._reformat import reformat_stack
from ._remote import open_file, open_h5
from ._search import search
from ._stack import make_stack, select_frames

__all__ = [
    "DispProduct",
    "DispProductStack",
    "create_rebased_displacement",
    "make_stack",
    "open_file",
    "open_h5",
    "rebase_timeseries",
    "reformat_stack",
    "search",
    "select_frames",
]
