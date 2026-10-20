import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mod():
    pytest.importorskip("ee")
    spec = importlib.util.spec_from_file_location("a3i", ROOT / "scripts" / "a3_stage1_interactive.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_profile_eecu_sums_first_column(mod):
    # format of ee.profilePrinting output (header, then EECU-s, PeakMem, Count, Description)
    text = (" EECU·s PeakMem Count  Description\n"
            " 75.183    512k   444  Algorithm Image.and computing pixels\n"
            " 20.123    4.0M   108  Reprojection precalculation between EPSG:32639 and PROJCS[...]\n"
            "  0.193     19k   674  Algorithm Image.updateMask\n"
            "      -    3.3k     1  Algorithm GeometryConstructors.Polygon\n")
    assert mod.profile_eecu(text) == pytest.approx(95.499)


def test_profile_eecu_empty(mod):
    assert mod.profile_eecu(" EECU·s PeakMem Count  Description\n") == 0.0
