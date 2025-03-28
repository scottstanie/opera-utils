import itertools
from collections.abc import Sequence
from pathlib import Path

from opera_utils._dates import get_dates


def _last_per_ministack(
    opera_file_list: Sequence[Path | str],
) -> tuple[list[Path | str], list[Path | str]]:
    def _get_generation_time(fname):
        return get_dates(fname)[2]

    last_per_ministack = []
    for d, cur_groupby in itertools.groupby(
        sorted(opera_file_list), key=_get_generation_time
    ):
        # cur_groupby is an iterable of all matching
        # Get the first one, and the last one. ignore the rest
        last_file = list(cur_groupby)[-1]
        last_per_ministack.append(last_file)
    return last_per_ministack
