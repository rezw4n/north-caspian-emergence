"""scripts/g1_ee.py run_counts checkpointing and the month-to-date sum,
with a fake request function: no Earth Engine call is made."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("g1_ee", ROOT / "scripts" / "g1_ee.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)

GROUPS = [f"E2_b{b:03d}_p{p}" for b in (70, 71, 72) for p in (0, 1)][:5]
DIMS = (10, 8)


def _target():
    return {"t_00_00": {"region": None, "transform": [60.0, 0.0, 0.0, 0.0, -60.0, 480.0], "dims": DIMS,
                        "groups": list(GROUPS)}}


def _value(g, kind):
    return (GROUPS.index(g) + 1) * 10 + "wvo".index(kind)


class Fake:
    """Returns per-group constant arrays; optionally raises on the n-th call."""

    def __init__(self, fail_on=None):
        self.calls, self.fail_on = [], fail_on

    def __call__(self, job, t):
        key, c0, r0, w, h, chunk = job
        self.calls.append(tuple(chunk))
        if self.fail_on is not None and len(self.calls) == self.fail_on:
            raise RuntimeError("simulated crash")
        arr = {f"{k}_{g}": np.full((h, w), _value(g, k), np.uint16) for g in chunk for k in "wvo"}
        return arr, 10.0, 0.0


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "LEDGER", tmp_path / "ledger.jsonl")
    return tmp_path / "ledger.jsonl"


def _check_tile(out):
    with np.load(out / "tile_t_00_00.npz") as z:
        for g in GROUPS:
            for k in "wvo":
                assert (z[f"{k}_{g}"] == _value(g, k)).all()


def test_crash_loses_only_the_request_in_flight(tmp_path, ledger):
    out = tmp_path / "s2"
    out.mkdir()
    f1 = Fake(fail_on=3)                                 # 3 requests of 2, 2 and 1 groups; the third crashes
    res = E.run_counts("great_salt_lake", None, _target(), 1e9, "test", groups_per_req=2, workers=1, out=out,
                       request_fn=f1, first_guess=10.0)
    assert res == {} and not (out / "tile_t_00_00.npz").exists()
    assert len(list((out / "partial" / "t_00_00").glob("*.npz"))) == 2
    f2 = Fake()
    res = E.run_counts("great_salt_lake", None, _target(), 1e9, "test", groups_per_req=2, workers=1, out=out,
                       request_fn=f2, first_guess=10.0)
    assert f2.calls == [(GROUPS[4],)]                     # only the lost request is repeated
    assert "t_00_00" in res
    _check_tile(out)
    assert not (out / "partial" / "t_00_00").exists()
    assert len(ledger.read_text().splitlines()) == 3      # 2 + 1 finished requests logged


def test_budget_stop_keeps_finished_requests(tmp_path, ledger):
    out = tmp_path / "s2"
    out.mkdir()
    f1 = Fake()
    E.run_counts("great_salt_lake", None, _target(), 25.0, "test", groups_per_req=2, workers=1, out=out,
                 request_fn=f1, first_guess=10.0)
    assert len(f1.calls) == 2                             # the third would pass 25 EECU-s
    f2 = Fake()
    E.run_counts("great_salt_lake", None, _target(), 1e9, "test", groups_per_req=2, workers=1, out=out,
                 request_fn=f2, first_guess=10.0)
    assert len(f2.calls) == 1
    _check_tile(out)


def test_guard_stops_the_run(tmp_path, ledger):
    out = tmp_path / "s2"
    out.mkdir()
    f1 = Fake()
    E.run_counts("great_salt_lake", None, _target(), 1e9, "test", groups_per_req=1, workers=1, out=out,
                 request_fn=f1, guard=lambda spent: spent < 20, guard_every=1, first_guess=10.0)
    assert len(f1.calls) == 2


def test_all_checkpointed_target_is_written_without_requests(tmp_path, ledger):
    out = tmp_path / "s2"
    out.mkdir()
    E.run_counts("great_salt_lake", None, _target(), 1e9, "test", groups_per_req=5, workers=1, out=out,
                 request_fn=Fake(fail_on=None), first_guess=10.0)
    (out / "tile_t_00_00.npz").unlink()                   # simulate a crash after the request, before the tile
    arr = {f"{k}_{g}": np.full((8, 10), _value(g, k), np.uint16) for g in GROUPS for k in "wvo"}
    E.save_partial(out, "t_00_00", 0, 0, GROUPS, arr)
    f = Fake()
    res = E.run_counts("great_salt_lake", None, _target(), 1e9, "test", groups_per_req=5, workers=1, out=out,
                       request_fn=f, first_guess=10.0)
    assert f.calls == [] and res == {"t_00_00": 0.0}
    _check_tile(out)


def test_interactive_logs_sum(tmp_path):
    (tmp_path / "a_interactive_log.jsonl").write_text(json.dumps({"eecu_s": 1.5}) + "\n" + json.dumps({"eecu_s": 2}) + "\n")
    (tmp_path / "ee_guard_log.jsonl").write_text(json.dumps({"eecu_s": 999}) + "\n")
    (tmp_path / "b.jsonl").write_text(json.dumps({"other": 1}) + "\n\n")
    s = E.interactive_logs_s(tmp_path)
    assert s == {"a_interactive_log.jsonl": 3.5, "b.jsonl": 0.0}
