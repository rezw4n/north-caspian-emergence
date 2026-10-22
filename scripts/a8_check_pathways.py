"""Check every number of results/levels/projection_pathways.csv against the text
that was actually retrieved (the projection pathways that map
the H3 scenarios to dates, PREREG section 10: "verified before they are used").

For each row: its ``verbatim_quote`` must be found (scripts/check_quotes.py matching:
white space, dashes, quotation marks and line-break hyphens ignored) in the sources of
its ``source`` key (refs/notes/SOURCES.csv, data/raw/lit/<doi stem>.pdf or .txt, files
the note names, the note's abstract block), and |value_m| must appear in the quote as
a number, in metres or (x 100) in centimetres. Rows that fail are listed; the exit
status is 1 if any fails.

Usage: python scripts/a8_check_pathways.py [csv]
"""

from __future__ import annotations

import importlib.util
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PATHWAYS = ROOT / "results" / "levels" / "projection_pathways.csv"
_spec = importlib.util.spec_from_file_location("check_quotes", ROOT / "scripts" / "check_quotes.py")
cq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cq)

_NUM = re.compile(r"(?<![\d.])\d+(?:[.,]\d+)?")


def numbers_in(text: str) -> list[float]:
    """Numbers written in a text (decimal point or comma), without sign."""
    t = unicodedata.normalize("NFKC", text)
    return [float(m.replace(",", ".")) for m in _NUM.findall(t)]


def value_in_quote(value: float, quote: str) -> bool:
    """|value| appears in the quote, in m or in cm (rates in m per year as cm per year)."""
    v = abs(float(value))
    nums = numbers_in(quote)
    return any(abs(n - v) < 1e-9 or abs(n - 100 * v) < 1e-6 for n in nums)


def check_rows(df: pd.DataFrame, root: Path = ROOT) -> pd.DataFrame:
    """Adds quote_found, value_in_quote and sources_used to every row."""
    sources = cq.read_sources(root / "refs" / "notes" / "SOURCES.csv")
    cache: dict = {}
    found, in_q, used = [], [], []
    for r in df.itertuples():
        key = (r.source, r.doi)
        if key not in cache:
            texts, names, errs = cq.gather_texts(r.source, r.doi, sources, root)
            cache[key] = (texts, names, errs)
        texts, names, errs = cache[key]
        q = str(r.verbatim_quote)
        found.append(bool(texts) and cq.found(q, texts))
        in_q.append(value_in_quote(r.value_m, q))
        used.append("; ".join(names) + (f" [errors: {'; '.join(errs)}]" if errs else ""))
    return df.assign(quote_found=found, value_in_quote=in_q, sources_used=used)


def main(argv) -> int:
    path = Path(argv[0]) if argv else PATHWAYS
    df = pd.read_csv(path)
    res = check_rows(df)
    bad = res[~res["quote_found"] | ~res["value_in_quote"]]
    print(f"{len(res)} rows, {int(res['quote_found'].sum())} quotes found, "
          f"{int(res['value_in_quote'].sum())} values in their quote")
    for r in bad.itertuples():
        print(f"FAIL {r.pathway_id} ({r.source}, {r.location}): quote_found={r.quote_found} "
              f"value_in_quote={r.value_in_quote} value={r.value_m}")
    return 1 if len(bad) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
