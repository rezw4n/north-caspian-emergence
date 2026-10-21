"""ERL Table 1, the product report card (D-100, D-102): one row per audited product, seven
columns, every number a metrics key. Writes

- manuscript/sections/tables_erl.tex, between the markers "% BEGIN ERL-T1" and
  "% END ERL-T1" (the rest of the file is kept; the file is created if missing), as LaTeX
  that prints every number through the metric macro of manuscript/numbers.tex. The
  preamble needs the array package and xcolor with its table option (cell shading);
- figures/out/erl_tab_report_card.{pdf,tif,png}, a rendered preview of the same cells.

Columns (results/metrics.json):
1. sectors mispredicted, shape variant, of 7 (h1.mispredicts.shape.<p>.n_sectors);
2. range over the seven sectors of the median shoreline displacement, shape variant
   (erl.card.<p>.disp_min_km, disp_max_km, from h1.disp_median.shape.*);
3. median displacement in the lowest level band, below -28.3 m (h1.disp_band.below_m28p3.*);
4. ETOPO 2022 only: the share of its emerged band filled from FABDEM land heights
   (h1.etopo2022.source_share.fabdem);
5. normalised median absolute deviation against ICESat-2 ATL08 on the held-out test
   blocks, matched variant (h2.atl08.<p>.nmad_m);
6. Great Salt Lake arms mispredicted, of 2 (generality.great_salt_lake.<p>.shape.
   n_sectors_mispredict);
7. emerged area of the domain 1 m below the 2026 level, product as used, with its 90%
   interval (h3.emergence.<p>.uncorrected.1.*).

Shading (a reading aid with stated cut-offs, not a test; D-102): columns 1, 2 (by the
largest sector), 3, 5 and 6 are graded good, fair or poor by the cut-offs below (exported to
results/params.json as params.erl_card_*); poor cells are also set in bold, so the grade
never rests on colour alone. Columns 4 and 7 are not graded. The printed value always
decides.

Usage: python figures/tab_report_card.py
"""

from __future__ import annotations

import sys

import _common as C
import _erl as E

CARD_SECTORS = (1, 4)        # good <= 1 sector, poor >= 4 (a majority of the 7)
CARD_KM = (1.0, 3.0)         # good < 1 km, poor >= 3 km
CARD_NMAD_M = (0.5, 1.0)     # good < 0.5 m, poor >= 1.0 m
CARD_ARMS = (0, 2)           # good 0 arms, poor 2 arms
TABLES = C.ROOT / "manuscript" / "sections" / "tables_erl.tex"
ROWS = ("gebco_2026", "gebco_2022", "etopo_2022", "etopo1", "srtm15p_v2.7")
TEX_NAME = {"gebco_2026": "GEBCO_2026", "gebco_2022": "GEBCO_2022", "etopo_2022": "ETOPO 2022", "etopo1": "ETOPO1",
            "srtm15p_v2.7": "SRTM15+ V2.7"}
SHADE = {"good": "D9EBD3", "fair": "FBF0C8", "poor": "F4D3CE", None: None}
BS = chr(92)


def grade(v: float, cut, integer: bool = False) -> str:
    good, poor = cut
    if integer:
        return "good" if v <= good else ("poor" if v >= poor else "fair")
    return "good" if v < good else ("poor" if v >= poor else "fair")


def cells(m: dict) -> list[dict]:
    """Per product: list of (latex, text, grade) for the seven columns."""
    out = []
    g = "generality.great_salt_lake"
    for p in ROWS:
        k = E.PK[p]
        keys = {"mis": f"h1.mispredicts.shape.{k}.n_sectors", "dmin": f"erl.card.{k}.disp_min_km",
                "dmax": f"erl.card.{k}.disp_max_km", "low": f"h1.disp_band.below_m28p3.{k}_km",
                "nmad": f"h2.atl08.{k}.nmad_m", "gsl": f"{g}.{p}.shape.n_sectors_mispredict",
                "est": f"h3.emergence.{k}.uncorrected.1.estimate_km2", "lo": f"h3.emergence.{k}.uncorrected.1.lo_km2",
                "hi": f"h3.emergence.{k}.uncorrected.1.hi_km2"}
        for key in keys.values():
            if key not in m:
                raise SystemExit(f"metric {key} missing")

        def mac(key):
            return BS + "metric{" + key + "}"

        row = [
            (mac(keys["mis"]), E.num(m, keys["mis"]), grade(E.value(m, keys["mis"]), CARD_SECTORS, integer=True)),
            (mac(keys["dmin"]) + "--" + mac(keys["dmax"]), E.num(m, keys["dmin"]) + "–" + E.num(m, keys["dmax"]),
             grade(E.value(m, keys["dmax"]), CARD_KM)),
            (mac(keys["low"]), E.num(m, keys["low"]), grade(E.value(m, keys["low"]), CARD_KM)),
            ((mac("h1.etopo2022.source_share.fabdem") + BS + "%") if p == "etopo_2022" else "--",
             (E.num(m, "h1.etopo2022.source_share.fabdem") + "%") if p == "etopo_2022" else "–", None),
            (mac(keys["nmad"]), E.num(m, keys["nmad"]), grade(E.value(m, keys["nmad"]), CARD_NMAD_M)),
            (mac(keys["gsl"]), E.num(m, keys["gsl"]), grade(E.value(m, keys["gsl"]), CARD_ARMS, integer=True)),
            (mac(keys["est"]) + BS + "newline [" + mac(keys["lo"]) + ", " + mac(keys["hi"]) + "]",
             E.num(m, keys["est"]) + "\n[" + E.num(m, keys["lo"]) + ", " + E.num(m, keys["hi"]) + "]", None),
        ]
        out.append({"product": p, "cells": row})
    return out


HEAD = ["Sectors mispredicted (of 7)", "Displacement across sectors (km)", "Displacement, lowest level band (km)",
        "Emerged band from FABDEM", "Height error against ICESat-2 (m)", "Great Salt Lake arms mispredicted (of 2)",
        "Emerged area 1 m below 2026 (km2) [90% interval]"]


def latex(m: dict, rows: list[dict]) -> list[str]:
    def esc(s):
        return s.replace("_", BS + "_")

    def mac(key):
        return BS + "metric{" + key + "}"

    head = [h.replace("km2", "km" + BS + "textsuperscript{2}").replace("%", BS + "%") for h in HEAD]
    cap = (BS + "textbf{Product report card.} For each global bathymetry product, as audited against four decades "
           "of observed shorelines of the North Caspian Sea: the number of the seven coastal sectors in which the "
           "product's predicted shoreline is displaced from the observed one by more than the observation noise floor "
           "at most levels (after removing the single vertical constant that best fits each product, the shape "
           "variant); the smallest and largest of the seven sector medians of that displacement; the median "
           "displacement at levels below " + mac("h1.disp_band.edge_1_m") + "~m (the lowest level band, the "
           "seabed exposed last); for ETOPO 2022, the share of its emerged band taken from the land heights of the "
           "FABDEM elevation model, acquired after that seabed had emerged; the normalised median absolute deviation "
           "of the product's heights from ICESat-2 land and canopy heights (ATL08) on the " + mac("h2.atl08.n_test_blocks")
           + " held-out test blocks; the number of the two arms of Great Salt Lake, Utah, in which the product "
           "mispredicts its shorelines in the same audit; and the area of seabed that the product, as users apply it, "
           "places between the 2026 level and 1~m below it, with its split-conformal 90" + BS + "% interval from the "
           "hindcast errors. Shading is a reading aid with fixed cut-offs, not a test: green, good; yellow, fair; red, "
           "poor (also in bold). Sectors: good at most " + mac("params.erl_card_sectors_good") + ", poor from "
           + mac("params.erl_card_sectors_poor") + "; displacements: good below " + mac("params.erl_card_km_good")
           + "~km, poor from " + mac("params.erl_card_km_poor") + "~km (the range by its largest sector); deviation: "
           "good below " + mac("params.erl_card_nmad_good_m") + "~m, poor from " + mac("params.erl_card_nmad_poor_m")
           + "~m; arms: good " + mac("params.erl_card_arms_good") + ", poor " + mac("params.erl_card_arms_poor")
           + ". The FABDEM share and the emerged area are not graded.")
    out = ["% BEGIN ERL-T1 (generated by figures/tab_report_card.py from results/metrics.json keys; do not edit by hand)",
           BS + "begin{table}[!htbp]", BS + "centering", BS + "footnotesize", BS + "setlength{" + BS + "tabcolsep}{3pt}",
           BS + "caption{" + cap + "}", BS + "label{tab:reportcard}",
           BS + "begin{tabular}{" + "".join(">{" + BS + "raggedright" + BS + "arraybackslash}p{" + w + "}" for w in
                                           ("2.2cm", "1.65cm", "1.75cm", "1.6cm", "1.35cm", "1.5cm", "1.6cm", "2.3cm"))
           + "}", BS + "hline",
           "Product & " + " & ".join(head) + " " + BS + BS, BS + "hline"]
    for r in rows:
        cols = []
        for tex, _, gr in r["cells"]:
            body = (BS + "textbf{" + tex + "}") if gr == "poor" else tex
            if SHADE[gr]:
                body = BS + "cellcolor[HTML]{" + SHADE[gr] + "}" + body
            cols.append(body)
        out.append(esc(TEX_NAME[r["product"]]) + " & " + " & ".join(cols) + " " + BS + BS)
    out += [BS + "hline", BS + "end{tabular}", BS + "end{table}", "% END ERL-T1"]
    return out


def preview(m: dict, rows: list[dict]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    E.style()
    W = E.DOUBLE_MM
    colw = [19.0, 17.5, 19.5, 19.0, 14.5, 17.5, 16.5, 24.0]
    scale = (W - 2.0) / sum(colw)
    colw = [c * scale for c in colw]
    hh, rh = 19.0, 9.0
    H = hh + rh * len(rows) + 2.0
    fig = plt.figure(figsize=(W * C.MM, H * C.MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    x = 1.0
    heads = [["Product"], ["Sectors", "mispredicted", "(of 7)"], ["Displacement", "across", "sectors (km)"],
             ["Displacement,", "lowest level", "band (km)"], ["Emerged", "band from", "FABDEM"],
             ["Height error", "against", "ICESat-2 (m)"], ["Great Salt", "Lake arms", "mispredicted", "(of 2)"],
             ["Emerged area", "1 m below 2026", "(km²), with its", "90% interval"]]
    for w, h in zip(colw, heads):
        ax.text(x + 1.0, hh - 1.5, chr(10).join(h), ha="left", va="bottom", fontsize=E.FS, color=E.INK,
                linespacing=1.1)
        x += w
    ax.plot([1.0, W - 1.0], [hh, hh], color=E.INK, lw=0.8)
    for i, r in enumerate(rows):
        y = hh + i * rh
        x = 1.0
        ax.text(x + 1.0, y + rh / 2, E.NAMES[r["product"]], ha="left", va="center", fontsize=E.FS, color=E.INK)
        x += colw[0]
        for w, (_, txt, gr) in zip(colw[1:], r["cells"]):
            if SHADE[gr]:
                ax.add_patch(Rectangle((x + 0.3, y + 0.4), w - 0.6, rh - 0.8, fc="#" + SHADE[gr], ec="none"))
            ax.text(x + 1.0, y + rh / 2, txt, ha="left", va="center", fontsize=E.FS, color=E.INK,
                    fontweight="bold" if gr == "poor" else "normal", linespacing=1.05)
            x += w
    ax.plot([1.0, W - 1.0], [hh + rh * len(rows), hh + rh * len(rows)], color=E.INK, lw=0.8)
    C.save(fig, "erl_tab_report_card")
    plt.close(fig)


def main() -> int:
    m = C.metrics()
    for k in ("params.erl_card_sectors_good", "params.erl_card_km_good", "params.erl_card_nmad_good_m",
              "params.erl_card_arms_good", "h1.disp_band.edge_1_m", "h2.atl08.n_test_blocks"):
        if k not in m:
            raise SystemExit(f"metric {k} missing")
    rows = cells(m)
    block = latex(m, rows)
    if TABLES.exists():
        lines = TABLES.read_bytes().decode("utf-8").replace(chr(13) + chr(10), chr(10)).split(chr(10))
    else:
        lines = ["% ERL tables (D-100, D-102). Generated blocks sit between their BEGIN and END markers; every number",
                 "% is a metric macro from results/metrics.json (manuscript/numbers.tex). Needs, in the",
                 "% preamble, the array package and xcolor with its table option (cell shading).", ""]
    if any(ln.startswith("% BEGIN ERL-T1") for ln in lines):
        i0 = next(i for i, ln in enumerate(lines) if ln.startswith("% BEGIN ERL-T1"))
        i1 = next(i for i, ln in enumerate(lines) if ln.startswith("% END ERL-T1"))
        lines[i0:i1 + 1] = block
    else:
        lines += block + [""]
    C.write_crlf(TABLES, chr(10).join(lines))
    preview(m, rows)
    print(f"wrote {TABLES.relative_to(C.ROOT)} ({len(block)} lines in the ERL-T1 block)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
