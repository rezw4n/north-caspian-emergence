import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from caspian import gauges as G


# ----------------------------------------------------------------------------- synthetic workbooks

def _col(c: int) -> str:
    s = ""
    c += 1
    while c:
        c, r = divmod(c - 1, 26)
        s = chr(65 + r) + s
    return s


def _xlsx(cells: dict) -> bytes:
    """A minimal .xlsx with one sheet; str values go to sharedStrings."""
    strings = sorted({v for v in cells.values() if isinstance(v, str)})
    idx = {s: i for i, s in enumerate(strings)}
    rows = {}
    for (r, c), v in cells.items():
        rows.setdefault(r, []).append((c, v))
    xml_rows = []
    for r in sorted(rows):
        cs = []
        for c, v in sorted(rows[r]):
            ref = f"{_col(c)}{r + 1}"
            cs.append(f'<c r="{ref}" t="s"><v>{idx[v]}</v></c>' if isinstance(v, str)
                      else f'<c r="{ref}"><v>{v}</v></c>')
        xml_rows.append(f'<row r="{r + 1}">{"".join(cs)}</row>')
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    sheet = f'<?xml version="1.0"?><worksheet {ns}><sheetData>{"".join(xml_rows)}</sheetData></worksheet>'
    sst = (f'<?xml version="1.0"?><sst {ns}>' + "".join(f"<si><t>{s}</t></si>" for s in strings) + "</sst>")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("xl/worksheets/sheet1.xml", sheet)
        z.writestr("xl/sharedStrings.xml", sst)
    return buf.getvalue()


def _sheet_cells(name="Testpost", datum="____-28,0____ metres; Altitude system ____Baltic system____",
                 years=((2001, [10.0] * 12), (2002, [20.0] * 11 + ["-"]))):
    cells = {(1, 0): "MONTHLY SEA LEVEL DATA", (2, 0): f"Post ___{name}___",
             (5, 0): "Observation data are adjusted to the unified datum for the whole observation period:",
             (6, 0): f"Post datum equals to {datum}", (7, 0): "AVERAGE SEA LEVEL",
             (9, 0): "Years ", (9, 1): "Months"}
    for m in range(1, 13):
        cells[(10, m)] = float(m)
    for i, (y, vals) in enumerate(years):
        cells[(11 + i, 0)] = float(y) if i % 2 == 0 else str(y)
        for m, v in enumerate(vals, start=1):
            cells[(11 + i, m)] = v
    return cells


# ----------------------------------------------------------------------------- parsing

@pytest.mark.parametrize("text,expected", [
    ("Post datum equals to ____-28,0_____ metres; Altitude system __baltic system__", (-28.0, "baltic", False)),
    ("Post datum equals to ______28,0____ metres; Altitude system _Baltic system______", (-28.0, "baltic", True)),
    ("Post datum equals to ______-26,075____ metres  from the Persian Gulf", (-26.075, "persian gulf", False)),
    ("Post datum equals to _________ metres  from the Persian Gulf", (None, "persian gulf", False)),
])
def test_parse_datum(text, expected):
    assert G.parse_datum(text) == expected


def test_xlsx_reader_and_sheet_parser():
    cells = G.read_xlsx_cells(_xlsx(_sheet_cells()))
    assert cells[(2, 0)] == "Post ___Testpost___"
    assert cells[(11, 3)] == 10.0
    s = G.parse_post_sheet(cells)
    assert s.name == "Testpost" and s.datum_m == -28.0 and s.altitude_system == "baltic"
    m = s.monthly
    assert len(m) == 24                          # string year "2002" is read too
    assert m.loc[(m.year == 2002) & (m.month == 12), "value_cm"].isna().all()
    a = G.annual_means(m.assign(post="x"), value="value_cm")
    assert a.set_index("year").loc[2001, "value_cm"] == 10.0
    assert np.isnan(a.set_index("year").loc[2002, "value_cm"])   # 11 months only


def test_catalogue_zip_levels_in_bhs(tmp_path):
    p = tmp_path / "cat.zip"
    persian = _sheet_cells(name="Iran", datum="__-26,075__ metres  from the Persian Gulf")
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("AVERAGE/TEST_av.xlsx", _xlsx(_sheet_cells()))
        z.writestr("AVERAGE/IRAN_av.xlsx", _xlsx(persian))
    monthly, meta = G.read_catalogue_zip(p)
    t = monthly[monthly.post == "TEST"]
    assert np.allclose(t.loc[t.year == 2001, "level_m"], -27.9)      # -28.0 + 10 cm
    assert monthly.loc[monthly.post == "IRAN", "level_m"].isna().all()  # not Baltic: no BHS level
    assert meta["IRAN"]["altitude_system"] == "persian gulf"


# ----------------------------------------------------------------------------- DAHITI monthly means

def test_monthly_means_of_daily_interpolant():
    t = pd.date_range("2001-01-01", "2001-12-31", freq="10D")
    days = (t - pd.Timestamp("2001-01-01")) / pd.Timedelta(days=1)
    h = -27.0 + 0.001 * days.to_numpy()          # linear: the interpolant is exact
    m = G.monthly_means_daily(t, h, max_gap_days=60)
    feb = m[(m.year == 2001) & (m.month == 2)].iloc[0]
    # mean of the line over 1-28 Feb at 12:00: day index 31.5 .. 58.5
    assert feb.level_m == pytest.approx(-27.0 + 0.001 * 45.0, abs=1e-9)
    assert feb.n_valid == feb.n_days == 28
    dec = m[(m.year == 2001) & (m.month == 12)].iloc[0]
    assert np.isnan(dec.level_m)                 # last record 2001-12-27: month not covered


def test_month_in_long_gap_is_dropped():
    t = pd.to_datetime(["2001-01-01", "2001-01-11", "2001-05-01", "2001-05-11"])
    m = G.monthly_means_daily(t, [0.0, 0.0, 1.0, 1.0], max_gap_days=60)
    assert m[(m.month >= 2) & (m.month <= 4)].level_m.isna().all()
    m2 = G.monthly_means_daily(t, [0.0, 0.0, 1.0, 1.0], max_gap_days=None)
    assert m2[(m2.month >= 2) & (m2.month <= 4)].level_m.notna().all()


# ----------------------------------------------------------------------------- offset and bootstrap

def _panel(delta=-0.4, n_years=30, seed=1, sd=0.03, missing=True):
    rng = np.random.default_rng(seed)
    years = np.arange(1993, 1993 + n_years)
    e = rng.normal(0, sd, years.size)            # common year noise
    effects = {"A": 0.05, "B": -0.05, "C": 0.10, "D": -0.10}
    rows = []
    for p, a in effects.items():
        for i, y in enumerate(years):
            if missing and p == "D" and i % 3 == 0:
                continue
            rows.append({"post": p, "year": int(y), "d": delta + a + e[i]})
    return pd.DataFrame(rows), years, e


def test_offset_recovers_constant_and_post_effects():
    diffs, years, e = _panel(missing=False)
    # balanced: mean over posts of per-post means = delta + mean(effects) + mean(e)
    assert G.offset(diffs, list("ABCD")) == pytest.approx(-0.4 + 0.0 + e.mean(), abs=1e-12)
    po = G.post_offsets(diffs, list("ABCD"))
    assert po.set_index("post").loc["C", "m"] == pytest.approx(-0.3 + e.mean(), abs=1e-12)
    assert np.allclose(po["trend_m_per_decade"], po["trend_m_per_decade"].iloc[0])


def test_post_with_too_few_years_is_dropped():
    diffs, years, _ = _panel(missing=False)
    short = diffs[~((diffs.post == "D") & (diffs.year > 1997))]
    po = G.post_offsets(short, list("ABCD"), min_years=10)
    assert np.isnan(po.set_index("post").loc["D", "m"])
    assert G.offset(short, list("ABCD")) == pytest.approx(po["m"].dropna().mean())


def test_moving_blocks():
    rng = np.random.default_rng(0)
    idx = G.moving_block_indices(10, 3, rng)
    assert idx.size == 10 and idx.min() >= 0 and idx.max() <= 9
    # consecutive within each block of 3
    assert all(idx[k + 1] - idx[k] == 1 for k in (0, 1, 3, 4, 6, 7))
    with pytest.raises(ValueError):
        G.moving_block_indices(5, 6, rng)


def test_bootstrap_sigma_matches_iid_theory():
    diffs, years, e = _panel(n_years=40, sd=0.05, missing=False, seed=3)
    reps = G.bootstrap_offset(diffs, list("ABCD"), years, block=1, n_boot=4000, seed=0)
    # common iid year noise: SE of the mean = sd/sqrt(n) (post effects cancel)
    se = np.std(e, ddof=0) / np.sqrt(len(years))
    assert np.std(reps) == pytest.approx(se, rel=0.1)
    assert np.mean(reps) == pytest.approx(G.offset(diffs, list("ABCD")), abs=3 * se / np.sqrt(4000) + 1e-3)


def test_bootstrap_is_seeded_and_handles_missing():
    diffs, years, _ = _panel()
    a = G.bootstrap_offset(diffs, list("ABCD"), years, block=3, n_boot=200, seed=7)
    b = G.bootstrap_offset(diffs, list("ABCD"), years, block=3, n_boot=200, seed=7)
    assert np.array_equal(a, b) and np.isfinite(a).all()
    med = G.bootstrap_offset(diffs, list("ABCD"), years, block=3, n_boot=50, seed=7, stat="median")
    assert np.isfinite(med).all()
    two = G.bootstrap_offset(diffs, list("ABCD"), years, block=3, n_boot=500, seed=7, resample_posts=True)
    assert np.std(two) > np.std(a)               # post effects add spread when posts are resampled
