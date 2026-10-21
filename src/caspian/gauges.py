"""Tide-gauge (post) levels of the Caspian Sea and the DAHITI -> BHS-77 offset.

Source
------
The CASPCOM General Catalogue of the Caspian Sea level, monthly data
(http://www.caspcom.com, page "Уровень", archive ``Sea-Level-EN.zip``; docs/DATUMS.md,
refs/notes/caspcom_catalogue.md). One legacy Excel workbook per post (``.xls``; one
``.xlsx``), each with a header block and a table of monthly mean levels:

    r2  Post ___Makhachkala___
    r5  Observation data are adjusted to the unified datum for the whole observation period:
    r6  Post datum equals to ___-28,0___ metres; Altitude system _Baltic System_
    r9  Years | Months ... | year
    r10       | 1 ... 12   |
    r11 1900  | 238 ... 241 | 242.08

Monthly values are centimetres above the post datum, so the level in the Baltic
Height System is ``datum + value / 100`` m. Missing months are text ("-", "н.д.") or
empty.

Offset (PREREG section 5; D-093)
--------------------------------
DAHITI is averaged the way the posts are: the linear interpolant of the DAHITI
records is evaluated daily, a calendar month is kept only if every day lies within a
gap of at most ``max_gap_days`` between records, and a year only if all 12 months are
kept. The annual difference of post s in year y is d[s, y] = H_post[s, y] - H_DAHITI[y];
the post offset m[s] is its mean over the overlap years, and the offset is the mean of
m[s] over the chosen posts. Its sigma comes from a moving-block bootstrap over years.
"""

from __future__ import annotations

import io
import re
import warnings
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from caspian import levels as L

_XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


# ----------------------------------------------------------------------------- workbook reading

def read_xlsx_cells(data: bytes) -> dict[tuple[int, int], object]:
    """Cell values of the first worksheet of an .xlsx file, keyed (row, col), 0-based.
    Numbers are floats, shared and inline strings are str. Minimal reader (no styles,
    no formulas beyond their cached values)."""
    z = zipfile.ZipFile(io.BytesIO(data))
    names = z.namelist()
    shared: list[str] = []
    if "xl/sharedStrings.xml" in names:
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(f"{_XLSX_NS}si"):
            shared.append("".join(t.text or "" for t in si.iter(f"{_XLSX_NS}t")))
    sheets = sorted((n for n in names if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n)),
                    key=lambda n: int(re.search(r"(\d+)\.xml$", n).group(1)))
    if not sheets:
        raise ValueError("no worksheet in .xlsx")
    root = ET.fromstring(z.read(sheets[0]))
    cells: dict[tuple[int, int], object] = {}
    for c in root.iter(f"{_XLSX_NS}c"):
        m = re.fullmatch(r"([A-Z]+)(\d+)", c.get("r", ""))
        if not m:
            continue
        col = 0
        for ch in m.group(1):
            col = col * 26 + ord(ch) - 64
        key = (int(m.group(2)) - 1, col - 1)
        t = c.get("t")
        if t == "inlineStr":
            cells[key] = "".join(x.text or "" for x in c.iter(f"{_XLSX_NS}t"))
            continue
        v = c.find(f"{_XLSX_NS}v")
        if v is None or v.text is None:
            continue
        if t == "s":
            cells[key] = shared[int(v.text)]
        elif t in (None, "n"):
            cells[key] = float(v.text)
        else:
            cells[key] = v.text
    return cells


def read_xls_cells(data: bytes) -> dict[tuple[int, int], object]:
    """Cell values of the first sheet of a legacy .xls (BIFF) workbook, as for
    ``read_xlsx_cells``. Empty cells are omitted."""
    import xlrd

    sh = xlrd.open_workbook(file_contents=data).sheet_by_index(0)
    cells: dict[tuple[int, int], object] = {}
    for r in range(sh.nrows):
        for c in range(sh.ncols):
            v = sh.cell_value(r, c)
            if v != "":
                cells[(r, c)] = float(v) if isinstance(v, (int, float)) else str(v)
    return cells


def read_cells(name: str, data: bytes) -> dict[tuple[int, int], object]:
    return read_xlsx_cells(data) if name.lower().endswith(".xlsx") else read_xls_cells(data)


# ----------------------------------------------------------------------------- one post sheet

@dataclass
class PostSheet:
    name: str                      # post name as written in the header
    datum_m: float | None          # post zero in metres in the stated altitude system
    datum_text: str                # the header line, verbatim
    altitude_system: str           # "baltic", "persian gulf" or "" (not stated)
    datum_sign_inferred: bool      # True if a Baltic datum was written without its minus sign
    monthly: pd.DataFrame = field(default_factory=pd.DataFrame)  # year, month, value_cm


def _row_text(cells, r: int, ncols: int = 14) -> str:
    return " ".join(str(cells[(r, c)]) for c in range(ncols) if (r, c) in cells).strip()


def parse_datum(text: str) -> tuple[float | None, str, bool]:
    """(datum in metres, altitude system, sign inferred) from the header line
    "Post datum equals to ____-28,0____ metres; Altitude system ___Baltic system___".

    The Baltic-system posts of the catalogue put their zero at -28.0 m. Five
    Azerbaijani sheets write "28,0" with no minus sign; a post zero 28 m *above* the
    Baltic zero would put their levels about 56 m above the neighbouring posts, so the
    missing sign is read as a typing slip and the datum taken as -28.0 m (flagged).
    """
    low = text.lower()
    system = "baltic" if "baltic" in low else ("persian gulf" if "persian gulf" in low else "")
    m = re.search(r"equals\s+to[\s_]*([-−–]?)\s*_*\s*(\d+(?:[.,]\d+)?)", text, re.I)
    if not m:
        return None, system, False
    val = float(m.group(2).replace(",", "."))
    negative = bool(m.group(1))
    inferred = False
    if system == "baltic" and not negative:
        inferred = True
        negative = True
    return (-val if negative else val), system, inferred


def parse_post_sheet(cells: dict[tuple[int, int], object]) -> PostSheet:
    """Header and monthly table of one catalogue sheet."""
    nrow = 1 + max(r for r, _ in cells) if cells else 0
    name, datum_line = "", ""
    header_end = None
    for r in range(min(nrow, 15)):
        t = _row_text(cells, r)
        if t.lower().startswith("post ") and not name:
            name = re.sub(r"^post\s*", "", t, flags=re.I).strip(" _")
        if "datum equals" in t.lower():
            datum_line = t
        if t.lower().startswith(("years", "годы")):
            header_end = r
    if header_end is None:
        raise ValueError("no 'Years' header row in sheet")
    datum, system, inferred = parse_datum(datum_line)
    rows = []
    for r in range(header_end + 1, nrow):
        y = cells.get((r, 0))
        if isinstance(y, str) and re.fullmatch(r"\s*\d{4}\s*", y):
            y = float(y)
        if not isinstance(y, float) or not 1800 <= y <= 2100:
            continue
        for mth in range(1, 13):
            v = cells.get((r, mth))
            rows.append((int(y), mth, float(v) if isinstance(v, float) else np.nan))
    monthly = pd.DataFrame(rows, columns=["year", "month", "value_cm"])
    return PostSheet(name=name, datum_m=datum, datum_text=datum_line, altitude_system=system,
                     datum_sign_inferred=inferred, monthly=monthly)


def read_catalogue_zip(path: str | Path) -> tuple[pd.DataFrame, dict[str, dict]]:
    """All post sheets of a catalogue archive.

    Returns (monthly, meta): ``monthly`` has columns post (file stem without "_av"),
    year, month, value_cm, level_m (= datum + value/100, NaN unless the post states
    the Baltic system), and ``meta`` maps post -> header facts."""
    frames, meta = [], {}
    with zipfile.ZipFile(path) as z:
        for info in sorted(z.infolist(), key=lambda i: i.filename):
            fn = info.filename
            if info.is_dir() or not fn.lower().endswith((".xls", ".xlsx")):
                continue
            post = re.sub(r"_av$", "", Path(fn).stem)
            sheet = parse_post_sheet(read_cells(fn, z.read(info)))
            m = sheet.monthly.copy()
            m.insert(0, "post", post)
            baltic = sheet.altitude_system == "baltic" and sheet.datum_m is not None
            m["level_m"] = (sheet.datum_m + m["value_cm"] / 100.0) if baltic else np.nan
            frames.append(m)
            meta[post] = {"file": fn, "name": sheet.name, "datum_m": sheet.datum_m,
                          "altitude_system": sheet.altitude_system, "datum_text": sheet.datum_text,
                          "datum_sign_inferred": sheet.datum_sign_inferred}
    return pd.concat(frames, ignore_index=True), meta


def annual_means(monthly: pd.DataFrame, value: str = "level_m", by: tuple[str, ...] = ("post",)) -> pd.DataFrame:
    """Mean of the 12 monthly means per (group, year); NaN unless all 12 are present."""
    g = monthly.groupby(list(by) + ["year"])[value]
    out = g.agg(["mean", "count"]).reset_index()
    out.loc[out["count"] < 12, "mean"] = np.nan
    return out.rename(columns={"mean": value, "count": "n_months"})


# ----------------------------------------------------------------------------- DAHITI, averaged as the posts

def monthly_means_daily(times, levels_m, max_gap_days: float | None = 60.0) -> pd.DataFrame:
    """Calendar-month means of the daily (12:00) linear interpolant of a level series.

    A day is valid if it lies between two records at most ``max_gap_days`` apart
    (``None``: any gap); a month is kept only if all its days are valid. Columns:
    year, month, level_m, n_days, n_valid."""
    t = pd.to_datetime(pd.Series(times))
    h = np.asarray(levels_m, float)
    start = t.min().normalize().replace(day=1)
    end = (t.max().normalize() + pd.offsets.MonthEnd(0))
    days = pd.date_range(start, end, freq="D") + pd.Timedelta(hours=12)
    gap = np.inf if max_gap_days is None else float(max_gap_days)
    it = L.interpolate(t, h, np.zeros_like(h), days, q=0.0, max_gap_days=gap)
    df = pd.DataFrame({"year": days.year, "month": days.month, "level_m": it["level_m"].to_numpy(),
                       "valid": ~it["in_gap"].to_numpy()})
    g = df.groupby(["year", "month"])
    out = g.agg(level_m=("level_m", "mean"), n_days=("valid", "size"), n_valid=("valid", "sum")).reset_index()
    out.loc[out["n_valid"] < out["n_days"], "level_m"] = np.nan
    return out


# ----------------------------------------------------------------------------- offset and bootstrap

def post_offsets(diffs: pd.DataFrame, posts, stat: str = "mean", min_years: int = 10) -> pd.DataFrame:
    """Per-post offset m[s] from annual differences (columns post, year, d).

    Returns post, n_years, m (mean or median of d), sd, first, last, and the OLS
    trend of d in m per decade; posts with fewer than ``min_years`` years get m = NaN."""
    rows = []
    for p in posts:
        d = diffs[(diffs["post"] == p) & np.isfinite(diffs["d"])]
        n = len(d)
        if n:
            m = float(d["d"].mean() if stat == "mean" else d["d"].median())
            sd = float(d["d"].std(ddof=1)) if n > 1 else np.nan
            tr = float(np.polyfit(d["year"], d["d"], 1)[0] * 10) if n > 2 else np.nan
            first, last = int(d["year"].min()), int(d["year"].max())
        else:
            m = sd = tr = np.nan
            first = last = None
        rows.append({"post": p, "n_years": n, "m": m if n >= min_years else np.nan, "sd": sd,
                     "trend_m_per_decade": tr, "first": first, "last": last})
    return pd.DataFrame(rows)


def combine(m: np.ndarray, stat: str = "mean") -> float:
    m = np.asarray(m, float)
    m = m[np.isfinite(m)]
    if not m.size:
        return float("nan")
    return float(m.mean() if stat == "mean" else np.median(m))


def offset(diffs: pd.DataFrame, posts, stat: str = "mean", min_years: int = 10) -> float:
    """The offset: ``stat`` over posts of the per-post ``stat`` of annual differences."""
    return combine(post_offsets(diffs, posts, stat, min_years)["m"].to_numpy(), stat)


def moving_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Indices 0..n-1 of one moving-block bootstrap sample of length n (non-circular
    blocks of ``block`` consecutive positions, truncated to n)."""
    if block < 1 or block > n:
        raise ValueError("block length must be in 1..n")
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=k)
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]


def bootstrap_offset(diffs: pd.DataFrame, posts, years, block: int = 3, n_boot: int = 10000,
                     seed: int = 0, stat: str = "mean", min_years: int = 10,
                     resample_posts: bool = False) -> np.ndarray:
    """Bootstrap replicates of ``offset``: years resampled in moving blocks of
    ``block`` consecutive calendar years (all posts share the drawn years, so the
    common interannual signal stays paired); optionally posts resampled with
    replacement as well. A post with fewer than ``min_years`` drawn years with data
    drops out of that replicate."""
    years = np.asarray(sorted(years))
    posts = list(posts)
    rng = np.random.default_rng(seed)
    tab = (diffs[diffs["post"].isin(posts)].pivot_table(index="year", columns="post", values="d")
           .reindex(index=years, columns=posts))
    a = tab.to_numpy(float)                       # years x posts, NaN where missing
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = moving_block_indices(len(years), block, rng)
        cols = rng.integers(0, len(posts), size=len(posts)) if resample_posts else np.arange(len(posts))
        x = a[np.ix_(idx, cols)]
        n = np.isfinite(x).sum(axis=0)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN columns
            if stat == "mean":
                m = np.where(n >= min_years, np.nansum(x, axis=0) / np.maximum(n, 1), np.nan)
            else:
                m = np.where(n >= min_years, np.nanmedian(x, axis=0), np.nan)
        out[b] = combine(m, stat)
    return out
