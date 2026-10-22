import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def guard():
    pytest.importorskip("ee")
    argv, sys.argv = sys.argv, ["ee_guard.py"]
    try:
        spec = importlib.util.spec_from_file_location("ee_guard", ROOT / "tools" / "ee_guard.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod


def test_cancels_when_over_budget(guard):
    assert guard.should_cancel(used=3700, rate=0, limit=3600, poll_s=15)


def test_cancels_before_overshoot_at_high_burn_rate(guard):
    # 20 EECU-s per second (as in the pilot) would pass 3,600 before the next poll
    assert guard.should_cancel(used=3400, rate=20, limit=3600, poll_s=15)


def test_keeps_slow_task(guard):
    assert not guard.should_cancel(used=1000, rate=1, limit=3600, poll_s=15)
