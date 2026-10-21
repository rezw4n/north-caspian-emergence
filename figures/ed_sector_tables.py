"""Supplementary Tables: the full sector tables of H1 and H2, as LaTeX that prints every
number through \\metric{...} (CLAUDE.md integrity rule 2).

Writes manuscript/sections/ed_tables_draft.tex:
- H2 per sector (test blocks, H2 population, 50 km): n, and bias, NMAD and RMSE of the
  DEM and of each product (matched variant), for ATL06 and ATL08; conformal coverage
  with its test-block bootstrap interval;
- H1 per sector, shape and total variants: evaluable levels whose median displacement
  exceeds the sector noise floor, of the evaluable levels, and the median over those
  levels of the per-level median displacement; the sector noise floor. Bold marks a
  sector the product "mispredicts" (results/hindcast/main/h1_mispredicts.csv; D-072).

Every key is checked against results/metrics.json; the script stops on a missing key,
so the table cannot reference a number that the macros do not define.

Usage: python figures/ed_sector_tables.py
"""

from __future__ import annotations

import sys

import _common as C
import pandas as pd

from caspian.config import RESULTS
from caspian.geography import SECTOR_ORDER

OUT = C.ROOT / "manuscript" / "sections" / "ed_tables_draft.tex"
PK = {"gebco_2026": "gebco2026", "gebco_2022": "gebco2022", "etopo_2022": "etopo2022", "srtm15p_v2.7": "srtm15p",
      "etopo1": "etopo1"}
TEX_NAME = {"west": "Kizlyar Bay--Dagestan--Kalmyk coast", "volga": "Volga delta", "interfluve": "Volga--Ural interfluve",
            "ural": "Ural delta", "northeast": "NE Kazakh coast", "komsomolets": "Komsomolets Bay and Kaidak",
            "open_shelf": "Open shelf", "overall": "All test blocks"}
SHORT = {"west": "Kizlyar--Dagestan", "volga": "Volga delta", "interfluve": "Interfluve", "ural": "Ural delta",
         "northeast": "NE Kazakh coast", "komsomolets": "Komsomolets", "open_shelf": "Open shelf"}
PROD_TEX = {"gebco_2026": "GEBCO\\_2026", "gebco_2022": "GEBCO\\_2022", "etopo_2022": "ETOPO 2022",
            "srtm15p_v2.7": "SRTM15+ V2.7", "etopo1": "ETOPO1"}


class Keys:
    def __init__(self, metrics: dict):
        self.m, self.missing = metrics, []

    def __call__(self, key: str) -> str:
        if key not in self.m:
            self.missing.append(key)
        return f"\\metric{{{key}}}"


def h2_table(k: Keys) -> list[str]:
    out = ["\\begin{table}[!htbp]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
           "\\caption{\\textbf{Accuracy of the waterline DEM and of the products by sector (H2).} Test blocks of "
           f"\\metric{{h2.block_size_km}}~km, H2 population (emerged seabed with an E3 emergent-vegetation share below "
           f"\\metric{{params.veg_cell_share}}). Errors are model minus ICESat-2 (m); products in the matched variant "
           "(product plus its training-block offset). $n$, test-block points (blocks in parentheses). Coverage: share of "
           f"test points inside the \\metric{{params.conformal_level}} conformal interval, with the test-block bootstrap "
           "95\\% interval. Ural has no test block at this size.}",
           "\\label{tab:ed_h2_sectors}",
           "\\begin{tabular}{llrrrrrr}", "\\hline",
           " & & \\multicolumn{3}{c}{ATL06} & \\multicolumn{3}{c}{ATL08} \\\\",
           "Sector & Model & Bias & NMAD & RMSE & Bias & NMAD & RMSE \\\\", "\\hline"]
    models = [("dem", "Waterline DEM")] + [(PK[p], PROD_TEX[p]) for p in C.PRODUCTS]
    for sec in ["overall"] + [s for s in SECTOR_ORDER]:
        if sec == "ural":
            out += [f"{TEX_NAME[sec]} & \\multicolumn{{7}}{{l}}{{no test block}} \\\\", "\\hline"]
            continue
        n6 = (k("h2.atl06.n_test_points"), k("h2.atl06.n_test_blocks")) if sec == "overall" else \
             (k(f"h2.atl06.n_points.{sec}"), k(f"h2.atl06.n_blocks.{sec}"))
        n8 = (k("h2.atl08.n_test_points"), k("h2.atl08.n_test_blocks")) if sec == "overall" else \
             (k(f"h2.atl08.n_points.{sec}"), k(f"h2.atl08.n_blocks.{sec}"))
        out.append(f"\\multicolumn{{2}}{{l}}{{\\textbf{{{TEX_NAME[sec]}}}}} & \\multicolumn{{3}}{{l}}{{$n$ = {n6[0]} ({n6[1]})}}"
                   f" & \\multicolumn{{3}}{{l}}{{$n$ = {n8[0]} ({n8[1]})}} \\\\")
        for key, name in models:
            cells = []
            for s in ("atl06", "atl08"):
                for metric in ("bias", "nmad", "rmse"):
                    cells.append(k(f"h2.{s}.{key}.{metric}_m") if sec == "overall" else k(f"h2.{s}.{key}.{sec}.{metric}_m"))
            out.append(f" & {name} & " + " & ".join(cells) + " \\\\")
        cv = []
        for s in ("atl06", "atl08"):
            cv.append(f"\\multicolumn{{3}}{{l}}{{{k(f'h2.{s}.coverage.{sec}')} "
                      f"[{k(f'h2.{s}.coverage.{sec}_boot_low')}, {k(f'h2.{s}.coverage.{sec}_boot_high')}]}}")
        out.append(" & Coverage & " + " & ".join(cv) + " \\\\")
        out.append("\\hline")
    out += ["\\end{tabular}", "\\end{table}", ""]
    return out


def h1_table(k: Keys, variant: str, mis: pd.Series) -> list[str]:
    label = {"shape": "shape variant (the product shifted by its fitted constant; deciding variant)",
             "total": "total variant (the product on the DAHITI reference by the per-cell geoid correction, no constant)"}
    out = ["\\begin{table}[!htbp]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{2.5pt}",
           f"\\caption{{\\textbf{{Shoreline displacement of the products by sector (H1), {label[variant]}.}} For each "
           "product: the number of evaluable levels whose median displacement exceeds the sector noise floor, of the "
           "evaluable levels, and the median over those evaluable levels of the per-level median displacement (km; "
           "epochs pooled). Noise floor: the median over levels of the 95th percentile of the distances between the two "
           "acquisition halves (km). Bold: the product mispredicts the sector (more than half of its evaluable levels "
           "exceed the noise floor).}",
           f"\\label{{tab:ed_h1_sectors_{variant}}}",
           "\\begin{tabular}{l" + "rr" * len(C.PRODUCTS) + "r}", "\\hline",
           " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{PROD_TEX[p]}}}" for p in C.PRODUCTS) + " & Noise \\\\",
           "Sector & " + " & ".join("Levels & km" for _ in C.PRODUCTS) + " & km \\\\", "\\hline"]
    for sec in SECTOR_ORDER:
        cells = []
        for p in C.PRODUCTS:
            a = f"{k(f'h1.n_exceeding.{variant}.{PK[p]}.{sec}')}/{k(f'h1.n_evaluable.{variant}.{PK[p]}.{sec}')}"
            b = k(f"h1.disp_median.{variant}.{PK[p]}.{sec}_km")
            if bool(mis.get((p, variant, sec), False)):
                a, b = f"\\textbf{{{a}}}", f"\\textbf{{{b}}}"
            cells += [a, b]
        out.append(f"{SHORT[sec]} & " + " & ".join(cells) + f" & {k(f'h1.noise_p95.{sec}_km')} \\\\")
    out += ["\\hline", "\\end{tabular}", "\\end{table}", ""]
    return out


def main() -> int:
    k = Keys(C.metrics())
    mis = pd.read_csv(RESULTS / "hindcast" / "main" / "h1_mispredicts.csv").set_index(["product", "variant", "sector"])
    mis = mis["mispredicts"]
    lines = ["% Generated by figures/ed_sector_tables.py from results/metrics.json keys. Do not edit.",
             "% Supplementary Tables (draft): full sector tables of H2 and H1. Every number is a \\metric{} macro.", ""]
    lines += h2_table(k)
    for v in ("shape", "total"):
        lines += h1_table(k, v, mis)
    if k.missing:
        print("FAIL missing metric keys:", sorted(set(k.missing)))
        return 1
    C.write_crlf(OUT, "\n".join(lines))
    print(f"wrote {OUT.relative_to(C.ROOT).as_posix()}: {sum(l.count(chr(92) + 'metric{') for l in lines)} macros")
    return 0


if __name__ == "__main__":
    sys.exit(main())
