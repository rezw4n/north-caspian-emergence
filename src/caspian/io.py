"""Readers for input data.

DAHITI water-level series
-------------------------
DAHITI (Schwatke et al., 2015, doi:10.5194/hess-19-4345-2015) distributes a
target's series in several layouts. Two CSV layouts are documented:

* legacy web download:  ``date,water level,error`` (``yyyy-mm-dd``)
* API v2:               ``datetime;wse;wse_u;mission;phase;timeliness`` where
  each data row carries two more fields (cycle, pass number) than the header.

The ASCII download carries ``#`` metadata lines (``# DAHITI-ID : 10146``,
``# column 2 : water level from altimetry [m]``) followed by whitespace-separated
rows. All variants are read into one schema:

``time`` (datetime64[ns], UTC, naive), ``level_m`` (float), ``sigma_m`` (float),
plus any extra columns (``mission``, ``cycle``, ...).

Heights are in metres. Their vertical reference is recorded in docs/DATUMS.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

_TIME_NAMES = ("datetime", "date", "time", "epoch")
_LEVEL_NAMES = ("wse", "water level", "water_level", "waterlevel", "height", "level", "h")
_SIGMA_NAMES = ("wse_u", "error", "uncertainty", "sigma", "std", "err", "u")
_V2_EXTRA = ("cycle", "pass_nr")
_TIME_TOKEN = re.compile(r"^\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?$")


@dataclass
class LevelSeries:
    data: pd.DataFrame
    meta: dict[str, str] = field(default_factory=dict)
    source: str = ""
    layout: str = ""

    @property
    def n(self) -> int:
        return len(self.data)


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().strip('"').lower())


def _pick(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    normed = {_norm(c): c for c in columns}
    for cand in candidates:
        if cand in normed:
            return normed[cand]
    return None


def _parse_meta(comment_lines: list[str]) -> dict[str, str]:
    meta: dict[str, str] = {}
    for line in comment_lines:
        body = line.lstrip("#").strip()
        if ":" in body:
            key, value = body.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta


def _looks_numeric(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _split(line: str, delim: str | None) -> list[str]:
    return [t.strip() for t in (line.split(delim) if delim else line.split())]


def read_dahiti(path: str | Path) -> LevelSeries:
    """Read a DAHITI water-level file (CSV legacy, CSV API v2 or ASCII)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig", errors="strict")
    lines = [ln.rstrip("\r\n") for ln in text.splitlines()]
    comments = [ln for ln in lines if ln.lstrip().startswith("#")]
    body = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    if not body:
        raise ValueError(f"{path}: no data rows")

    first = body[0]
    delim = ";" if ";" in first else ("," if "," in first else None)
    first_tokens = _split(first, delim)
    has_header = not re.match(r"^\d{4}-\d{2}-\d{2}", first_tokens[0])

    if has_header:
        header = first_tokens
        rows = [_split(ln, delim) for ln in body[1:]]
        width = max(len(r) for r in rows)
        if width > len(header):
            is_v2 = _norm(header[0]) == "datetime" and _pick(header, ("wse",)) is not None
            extra = list(_V2_EXTRA) if is_v2 and width - len(header) == 2 else [
                f"extra_{i}" for i in range(width - len(header))
            ]
            header = header + extra
        rows = [r + [""] * (len(header) - len(r)) for r in rows]
        df = pd.DataFrame(rows, columns=header)
        tcol, lcol = _pick(header, _TIME_NAMES), _pick(header, _LEVEL_NAMES)
        scol = _pick(header, _SIGMA_NAMES)
        if tcol is None or lcol is None:
            raise ValueError(f"{path}: cannot identify time/level columns in header {header}")
        layout = "csv-v2" if _norm(header[0]) == "datetime" else f"delimited-header({delim or 'ws'})"
        times = df[tcol]
    else:
        rows = [_split(ln, delim) for ln in body]
        if len(rows[0]) > 1 and _TIME_TOKEN.match(rows[0][1]):
            # API v2 ASCII: date time wse wse_u mission phase timeliness cycle pass
            times = pd.Series([f"{r[0]} {r[1]}" for r in rows])
            rest = [r[2:] for r in rows]
            layout = "ascii-v2"
        else:
            times = pd.Series([r[0] for r in rows])
            rest = [r[1:] for r in rows]
            layout = "ascii-legacy"
        width = max(len(r) for r in rest)
        names = ["level", "sigma", "mission", "phase", "timeliness", "cycle", "pass_nr"]
        names = names[:width] + [f"extra_{i}" for i in range(width - len(names))]
        rest = [r + [""] * (width - len(r)) for r in rest]
        df = pd.DataFrame(rest, columns=names)
        lcol, scol = "level", ("sigma" if width > 1 else None)

    out = pd.DataFrame(
        {
            "time": pd.to_datetime(pd.Series(times).str.strip(), utc=True, errors="raise")
            .dt.tz_convert(None)
            .values,
            "level_m": pd.to_numeric(df[lcol], errors="coerce").values,
            "sigma_m": (pd.to_numeric(df[scol], errors="coerce").values if scol else float("nan")),
        }
    )
    for col in df.columns:
        if col in (lcol, scol) or (has_header and col == tcol):
            continue
        out[_norm(col).replace(" ", "_")] = df[col].values
    n_bad = int(out["level_m"].isna().sum())
    out = out.dropna(subset=["level_m"]).sort_values("time").reset_index(drop=True)
    meta = _parse_meta(comments)
    meta["rows_dropped_nonnumeric_level"] = str(n_bad)
    return LevelSeries(data=out, meta=meta, source=str(path), layout=layout)


def find_dahiti_file(directory: str | Path) -> Path | None:
    """Return the single DAHITI series file in ``directory`` (csv/txt/ascii)."""
    directory = Path(directory)
    if not directory.is_dir():
        return None
    cands = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in {".csv", ".txt", ".asc", ".ascii", ".dat"}
    )
    return cands[0] if cands else None


def read_hydroweb(path: str | Path) -> LevelSeries:
    """Read a Hydroweb (hydroweb.next / LEGOS) lake water-level text file.

    Format (per its header): ``#`` comment lines, one ``key=value;...`` metadata
    line, then ``;``-separated rows: decimal year; yyyy/mm/dd; hh:mm; height above
    the surface of reference (m); standard deviation (m); area (km2); volume (km3);
    mission-trace. 9999.999 marks missing values."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    comments = [ln for ln in lines if ln.startswith("#")]
    meta: dict[str, str] = {}
    rows = []
    for ln in lines:
        if not ln.strip() or ln.startswith("#"):
            continue
        if "=" in ln and ";" in ln and not re.match(r"^\s*\d{4}\.\d+", ln):
            for part in ln.strip().split(";"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    meta[k.strip()] = v.strip()
            continue
        f = [x.strip() for x in ln.split(";")]
        if len(f) >= 5:
            rows.append(f)
    df = pd.DataFrame(rows).iloc[:, :8]
    df.columns = ["decyear", "date", "time", "height", "sd", "area", "volume", "mission"][:df.shape[1]]
    t = pd.to_datetime(df["date"] + " " + df["time"], format="%Y/%m/%d %H:%M")
    out = pd.DataFrame({"time": t.values,
                        "level_m": pd.to_numeric(df["height"], errors="coerce").values,
                        "sigma_m": pd.to_numeric(df["sd"], errors="coerce").values,
                        "mission": df["mission"].values if "mission" in df else ""})
    out.loc[out["level_m"] >= 9999, "level_m"] = float("nan")
    out.loc[out["sigma_m"] >= 9999, "sigma_m"] = float("nan")
    n_bad = int(out["level_m"].isna().sum())
    out = out.dropna(subset=["level_m"]).sort_values("time").reset_index(drop=True)
    for ln in comments:
        body = ln.lstrip("#").strip()
        if body.lower().startswith("surface of reference"):
            meta["surface_of_reference"] = body.split(":", 1)[1].strip()
        if body.startswith("https://doi.org/"):
            meta["doi"] = body
    meta["rows_dropped_missing_level"] = str(n_bad)
    return LevelSeries(data=out, meta=meta, source=str(path), layout="hydroweb-txt")
