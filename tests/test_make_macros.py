"""tools/make_macros.py: number formatting for manuscript/numbers.tex."""

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from make_macros import fmt  # noqa: E402

SEPARATED_INT = re.compile(r"^(\\textminus\{\})?\d{1,3}(\\,\d{3})*$")


def test_sector_areas_take_the_km2_rule():
    # study_area.sector_km2.<name> does not end in _km2; it used to print '11298.00'
    assert fmt("study_area.sector_km2.interfluve", 11298.0, unit="km2") == "11\\,298"
    assert fmt("study_area.sector_km2.interfluve", 11298.0) == "11\\,298"  # by key alone
    assert fmt("study_area.domain_km2", 224136.6, unit="km2") == "224\\,137"


def test_unit_km2_and_days_format_at_0dp_when_no_key_rule_matches():
    assert fmt("hindcast.gebco2026.bias", 1234.4, unit="km2") == "1\\,234"
    assert fmt("x.window", 20.0, unit="days") == "20"
    assert fmt("x.other", 1234.4, unit="m") == "1\\,234.40"  # default 2 dp, separators as on the int path


def test_every_km2_metric_in_metrics_json_is_0dp_with_separators():
    metrics = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))
    km2 = {k: r for k, r in metrics.items() if r.get("unit") == "km2"}
    assert km2, "no km2 metric found"
    for k, r in km2.items():
        s = fmt(k, r["value"], r.get("decimals"), r.get("unit"))
        assert SEPARATED_INT.match(s), (k, s)


def test_negative_zero_dp_uses_textminus_not_hyphen():
    assert fmt("hindcast.area_error_km2", -1234.4) == "\\textminus{}1\\,234"
    assert fmt("x.max_gap_days", -3.6) == "\\textminus{}4"
    assert fmt("x_km2", -1234) == "\\textminus{}1\\,234"  # int path agrees with float path


@pytest.mark.parametrize("key,v,dec,expected", [
    ("levels.crosscheck.offset_median_m", -0.0004, None, "0.000"),
    ("x.y_m", -0.004, None, "0.00"),
    ("lit.x", -0.0004, 1, "0.0"),
    ("lit.x", -0.04, 1, "0.0"),
    ("lit.x", -0.0, 1, "0.0"),
    ("hindcast.area_error_km2", -0.3, None, "0"),
])
def test_values_rounding_to_zero_are_unsigned(key, v, dec, expected):
    assert fmt(key, v, dec) == expected


def test_existing_formats_unchanged():
    assert fmt("levels.crosscheck.offset_median_m", 0.114) == "0.114"
    assert fmt("levels.crosscheck.drift_m_per_decade", -0.016) == "\\textminus{}0.016"
    assert fmt("lit.ginzburg2025.level_end2024_bhs_m", -29.34, 2) == "\\textminus{}29.34"
    assert fmt("inventory.n_scenes", 75603) == "75\\,603"
    assert fmt("x.date", "2026-10-20") == "20 October 2026"
    assert fmt("x.q_m2_per_day", 1.234e-5) == "\\ensuremath{1.23\\times10^{-5}}"
    assert fmt("x.q_m2_per_day", 0.0) == "0"


def test_integers_are_formatted_exactly_beyond_float_precision():
    # 10**17 + 1 is not representable as a float; going through float printed ...000
    assert fmt("x.n", 10**17 + 1) == "100\\,000\\,000\\,000\\,000\\,001"
    assert fmt("x.n", -(10**17 + 1)) == "\\textminus{}100\\,000\\,000\\,000\\,000\\,001"
    assert fmt("lit.x", 10**17 + 1, 1) == "100\\,000\\,000\\,000\\,000\\,001.0"
    assert fmt("lit.x", 1234, 2) == "1\\,234.00"
    assert fmt("x.n", 0) == "0"


def test_no_ascii_minus_in_any_generated_number():
    metrics = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))
    for k, r in metrics.items():
        if isinstance(r["value"], (int, float)) and not isinstance(r["value"], bool):
            s = fmt(k, r["value"], r.get("decimals"), r.get("unit"))
            s = re.sub(r"\^\{-?\d+\}", "", s)  # a math-mode exponent's '-' is a minus sign
            assert not re.search(r"(^|[^\\a-z])-\d", s), (k, s)


def test_rounded_forms_have_raw_values_for_every_finite_number():
    # \metricsf and \metricdp round the raw value in LaTeX; every finite numeric metric needs one.
    tex = (ROOT / "manuscript" / "numbers.tex").read_text(encoding="utf-8")
    assert r"\newcommand{\metricsf}[2]" in tex and r"\newcommand{\metricdp}[2]" in tex
    raw = dict(re.findall(r"\\@namedef\{metricraw@([^}]*)\}\{([^}]*)\}", tex))
    metrics = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))
    for k, rec in metrics.items():
        v = rec["value"]
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf"):
            assert float(raw[k]) == pytest.approx(v), k
