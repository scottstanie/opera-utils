from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from itertools import groupby
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from ._dates import get_dates
from ._helpers import flatten, powerset, sorted_deduped_values
from ._types import Filename
from .bursts import S1BurstId, group_by_burst

logger = logging.getLogger(__name__)

__all__ = [
    "BurstSubsetOption",
    "get_burst_id_to_dates",
    "get_missing_data_options",
    "get_burst_id_date_incidence",
]


@dataclass(frozen=True)
class BurstSubsetOption:
    """Dataclass for a possible subset of SLC data."""

    total_num_bursts: int
    """Total number of bursts used in this subset."""
    burst_ids: tuple[S1BurstId, ...]
    """Burst IDs used in this subset."""
    dates: tuple[date, ...]
    """Dates used in this subset."""
    # subset_selected: list[bool]
    # """True if the corresponding file/ (burst ID, date) was selected."""

    @property
    def num_dates(self) -> int:
        return len(self.dates)

    @property
    def num_burst_ids(self) -> int:
        return len(self.burst_ids)


def get_burst_id_to_dates(
    slc_files: Optional[Iterable[Filename]] = None,
    burst_id_date_tuples: Optional[Iterable[tuple[str | S1BurstId, date]]] = None,
) -> dict[S1BurstId, list[date]]:
    """Get a mapping of burst ID to list of dates.

    Assumes that the `slc_files` have only one date in the name, or
    that the first date in the `burst_id_date_tuples` is the relevant
    one (as is the case for OPERA CSLCs).


    Parameters
    ----------
    slc_files : Optional[Iterable[Filename]]
        List of OPERA CSLC filenames.
    burst_id_date_tuples : Optional[Iterable[tuple[str | S1BurstId, date]]]
        Alternative input: list of all existing (burst_id, date) tuples.

    Returns
    -------
    dict[S1BurstId, list[date]]
        Mapping of burst ID to list of dates.
    """
    if slc_files is not None:
        return _burst_id_mapping_from_files(slc_files)
    elif burst_id_date_tuples is not None:
        return _burst_id_mapping_from_tuples(burst_id_date_tuples)
    else:
        raise ValueError("Must provide either slc_files or burst_id_date_tuples")


def get_missing_data_options(
    slc_files: Optional[Iterable[Filename]] = None,
    burst_id_date_tuples: Optional[Iterable[tuple[S1BurstId | str, date]]] = None,
) -> list[BurstSubsetOption]:
    """Get a list of possible data subsets for a set of burst SLCs.

    The default optimization criteria for choosing among these subsets is

        maximize        total number of bursts used
        subject to      dates used for each burst ID are all equal

    The constraint that the same dates are used for each burst ID is to
    avoid spatial discontinuities the estimated displacement/velocity,
    which can occur if different dates are used for different burst IDs.

    Parameters
    ----------
    slc_files : Optional[Iterable[Filename]]
        list of OPERA CSLC filenames.
    burst_id_date_tuples : Optional[Iterable[tuple[str, date]]]
        Alternative input: list of all existing (burst_id, date) tuples.

    Returns
    -------
    list[BurstSubsetOption]
        List of possible subsets of the given SLC data.
        The options will be sorted by the total number of bursts used, so
        that the first option is the one that uses the most data.
    """
    burst_id_to_dates = get_burst_id_to_dates(
        slc_files=slc_files, burst_id_date_tuples=burst_id_date_tuples
    )

    all_burst_ids = list(burst_id_to_dates.keys())
    all_dates = sorted_deduped_values(burst_id_to_dates)

    B = get_burst_id_date_incidence(burst_id_to_dates)
    # In this matrix,
    # - Each row corresponds to one of the possible burst IDs
    # - Each column corresponds to one of the possible dates
    return generate_burst_subset_options(B, all_burst_ids, all_dates)


def get_burst_id_date_incidence(
    burst_id_to_dates: Mapping[S1BurstId, list[date]]
) -> np.ndarray:
    """Create a matrix of burst ID vs. date incidence.

    Parameters
    ----------
    burst_id_to_dates : Mapping[S1BurstId, list[date]]
        Mapping of burst ID to list of dates.

    Returns
    -------
    np.ndarray[bool]
        Matrix of burst ID vs. date incidence.
        Rows correspond to burst IDs, columns correspond to dates.
        A value of True indicates that the burst ID was acquired on that date.
    """
    all_dates = sorted_deduped_values(burst_id_to_dates)

    # Construct the incidence matrix of dates vs. burst IDs
    burst_id_to_date_incidence = {}
    for burst_id, date_list in burst_id_to_dates.items():
        cur_incidences = np.zeros(len(all_dates), dtype=bool)
        idxs = np.searchsorted(all_dates, date_list)
        cur_incidences[idxs] = True
        burst_id_to_date_incidence[burst_id] = cur_incidences

    return np.array(list(burst_id_to_date_incidence.values()))


def _burst_id_mapping_from_tuples(
    burst_id_date_tuples: Iterable[tuple[str | S1BurstId, date]]
) -> dict[S1BurstId, list[date]]:
    """Create a {burst_id -> [date,...]} (burst_id, date) tuples."""
    # Don't exhaust the iterator for multiple groupings
    burst_id_date_tuples = list(burst_id_date_tuples)

    # Group the possible SLC files by their date and by their Burst ID
    return {
        burst_id: [date for _, date in g]
        for burst_id, g in groupby(burst_id_date_tuples, key=lambda x: x[0])
    }


def _burst_id_mapping_from_files(
    slc_files: Iterable[Filename],
) -> dict[S1BurstId, list[date]]:
    """Create a {S1BurstId -> [date,...]} mapping from filenames."""
    # Don't exhaust the iterator for multiple groupings
    slc_file_list = list(map(str, slc_files))

    # Group the possible SLC files by their date and by their Burst ID
    burst_id_to_files = group_by_burst(slc_file_list)

    date_tuples = [get_dates(f) for f in slc_file_list]
    assert all(len(tup) == 1 for tup in date_tuples)

    return {
        burst_id: [get_dates(f)[0] for f in file_list]
        for (burst_id, file_list) in burst_id_to_files.items()
    }


def generate_burst_subset_options(
    B: np.ndarray, burst_ids: Sequence[S1BurstId], dates: Sequence[date]
) -> list[BurstSubsetOption]:
    """Generate possible valid subsets of the given SLC data.

    Parameters
    ----------
    B : NDArray[np.bool]
        Matrix of burst ID vs. date incidence.
        Rows correspond to burst IDs, columns correspond to dates.
        A value of True indicates that the burst ID was acquired on that date.
    burst_ids : Sequence[S1BurstId]
        List of all burst IDs.
    dates : Sequence[date]
        List of all dates.

    Returns
    -------
    list[BurstSubsetOption]
        List of possible subsets of the given SLC data.
        The options will be sorted by the total number of bursts used, so
        that the first option is the one that uses the most data.
    """
    options = []

    # Get the idxs where there are any missing dates for each burst
    # We're going to try all possible combinations of these *groups*,
    # not all possible combinations of the individual missing dates
    missing_date_idxs = set()
    for row in B:
        missing_date_idxs.add(tuple(np.where(~row)[0]))

    # Generate all unique combinations of idxs to exclude
    date_idxs_to_exclude_combinations = [
        set(flatten(combo)) for combo in powerset(missing_date_idxs)
    ]

    all_column_idxs = set(range(B.shape[1]))
    all_row_idxs = set(range(B.shape[0]))

    # Track the row/col combinations that we've already
    tested_combinations = set()
    # Now iterate over these combinations
    for idxs_to_exclude in date_idxs_to_exclude_combinations:
        valid_col_idxs = all_column_idxs - idxs_to_exclude

        # Create sub-matrix with the remaining columns
        col_selector = sorted(valid_col_idxs)
        B_sub = B[:, col_selector]

        # We've decided which columns to exclude
        # Now we have to decide if we're throwing away rows
        # We'll get rid of any row that's not fully populated
        rows_to_exclude = set()
        for i, row in enumerate(B_sub):
            if not row.all():
                rows_to_exclude.add(i)

        # Get which indexes we're keeping
        valid_row_idxs = all_row_idxs - rows_to_exclude

        # Check if we've already tested this combination
        combo = (tuple(valid_row_idxs), tuple(valid_col_idxs))
        if combo in tested_combinations:
            continue
        tested_combinations.add(combo)

        # Remove the rows that we're excluding
        row_selector = sorted(valid_row_idxs)
        B_sub2 = B_sub[row_selector, :]

        # Check if all rows have the same pattern in the remaining columns
        if not (B_sub2.size > 0):
            logger.debug("No remaining entries in B_sub2")
            continue
        if not np.all(B_sub2 == B_sub2[[0]]):
            logger.debug("Not all rows have the same pattern in the remaining columns")
            continue
        # Create a BurstSubsetOption if we have at least one burst and one date
        assert np.all(B_sub2)

        selected_burst_ids = tuple(burst_ids[i] for i in valid_row_idxs)
        selected_dates = tuple(dates[i] for i in valid_col_idxs)
        total_num_bursts = B_sub2.sum()
        # breakpoint()
        options.append(
            BurstSubsetOption(
                total_num_bursts=total_num_bursts,
                burst_ids=selected_burst_ids,
                dates=selected_dates,
            )
        )

    return sorted(
        options, key=lambda x: (x.total_num_bursts, x.num_burst_ids), reverse=True
    )
