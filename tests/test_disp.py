from datetime import datetime, timezone
from pathlib import Path

import pytest

from opera_utils._disp import (
    OperaDispFile,
    _get_first_file_per_ministack,
    parse_disp_datetimes,
)


@pytest.fixture
def sample_files():
    return [
        "OPERA_L3_DISP-S1_IW_F11116_VV_20160705T140755Z_20160729T140756Z_v0.9_20241219T231545Z.nc",
        "OPERA_L3_DISP-S1_IW_F11116_VV_20160705T140755Z_20160810T140757Z_v0.9_20241219T231545Z.nc",
        "OPERA_L3_DISP-S1_IW_F11116_VV_20160705T140755Z_20160822T140758Z_v0.9_20241219T231545Z.nc",
        "OPERA_L3_DISP-S1_IW_F11116_VV_20160705T140755Z_20160903T140758Z_v0.9_20241219T231545Z.nc",
    ]


def test_opera_disp_file_from_filename():
    filename = "OPERA_L3_DISP-S1_IW_F11116_VV_20160705T140755Z_20160729T140756Z_v0.9_20241219T231545Z.nc"
    for f in [filename, Path(filename)]:
        op = OperaDispFile.from_filename(filename)

        assert op.frame_id == 11116
        assert op.reference_datetime == datetime(
            2016, 7, 5, 14, 7, 55, tzinfo=timezone.utc
        )
        assert op.secondary_datetime == datetime(
            2016, 7, 29, 14, 7, 56, tzinfo=timezone.utc
        )
        assert op.version == "0.9"
        assert op.generation_datetime == datetime(
            2024, 12, 19, 23, 15, 45, tzinfo=timezone.utc
        )


def test_opera_disp_file_from_filename_invalid():
    filename = "INVALID_FILENAME.nc"
    with pytest.raises(ValueError):
        OperaDispFile.from_filename(filename)


def test_parse_disp_datetimes(sample_files):
    reference_times, secondary_times, generation_times = parse_disp_datetimes(
        sample_files
    )

    assert len(reference_times) == 4
    assert len(set(reference_times)) == 1
    assert len(secondary_times) == 4
    assert len(generation_times) == 4
    assert len(set(generation_times)) == 1

    assert reference_times[0] == datetime(2016, 7, 5, 14, 7, 55)
    assert secondary_times[0] == datetime(2016, 7, 29, 14, 7, 56)
    assert generation_times[0] == datetime(2024, 12, 19, 23, 15, 45)


def test_get_first_file_per_ministack(sample_files):
    result = _get_first_file_per_ministack(sample_files)

    assert len(result) == 1
    assert result[0] == sample_files[0]
