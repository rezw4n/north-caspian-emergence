"""ERL Table 1, the product report card (revised version): one row per audited product,
every number a metrics key. Writes

- manuscript/sections/tables_erl.tex, between the markers "% BEGIN ERL-T1" and
  "% END ERL-T1" (the rest of the file is kept; the file is created if missing), as LaTeX
  that prints every number through the metric macro of manuscript/numbers.tex;
- figures/out/erl_tab_report_card.{pdf,tif,png}, a rendered preview of the same cells.

Columns (results/metrics.json):
1. sectors mispredicted, shape variant, of 7 (h1.mispredicts.shape.<p>.n_sectors);
2. validated sectors mispredicted, of 4 (rev.s.item1.shape.<p>.validated4.n_mispredict);
3. range over the seven sectors of the median shoreline displacement, shape variant
   (erl.card.<p>.disp_min_km, disp_max_km);
4. median displacement in the lowest level band, below -28.3 m (h1.disp_band.below_m28p3.*);
5. ETOPO 2022 only: the share of its emerged band filled from FABDEM land heights
   (h1.etopo2022.source_share.fabdem);
6. NMAD and RMSE against ICESat-2 ATL08 on the held-out test blocks, matched variant
   (h2.atl08.<p>.nmad_m; rev.g.h2.rmse.atl08.<p>.matched.rmse_m);
7. seabed in the first metre below the 2026 level, measured from the observed 2026 shoreline
   (rev.n.composite.E.common_location.<p>.1.domain_km2) and on the grid shifted by its
   hindcast constant (rev.n.E.shifted.<p>.1.domain_km2).

The traffic-light shading of the first version is removed: its cut-offs were set by the
authors, not tested, and the columns describe
different populations. The Great Salt Lake column is removed: it was 2 for every grid.

Usage: python figures/tab_report_card.py
"""

from __future__ import annotations

import sys

import _common as C
import _erl as E

TABLES = C.ROOT / "manuscript" / "sections" / "tables_erl.tex"
ROWS = ("gebco_2026", "gebco_2022", "etopo_2022", "etopo1", "srtm15p_v2.7")
TEX_NAME = {"gebco_2026": "GEBCO_2026", "gebco_2022": "GEBCO_2022", "etopo_2022": "ETOPO 2022", "etopo1": "ETOPO1",
            "srtm15p_v2.7": "SRTM15+ V2.7"}
BS = chr(92)


def keys_for(p: str) -> dict:
    k = E.PK[p]
    return {"mis": f"h1.mispredicts.shape.{k}.n_sectors", "val": f"rev.s.item1.shape.{k}.validated4.n_mispredict",
            "dmin": f"erl.card.{k}.disp_min_km", "dmax": f"erl.card.{k}.disp_max_km",
            "low": f"h1.disp_band.below_m28p3.{k}_km", "nmad": f"h2.atl08.{k}.nmad_m",
            "rmse": f"rev.g.h2.rmse.atl08.{k}.matched.rmse_m",
            "obs": f"rev.n.composite.E.common_location.{k}.1.domain_km2", "shift": f"rev.n.E.shifted.{k}.1.domain_km2"}


def cells(m: dict) -> list[dict]:
    """Per product: list of (latex, text) for the columns."""
    out = []
    for p in ROWS:
        keys = keys_for(p)
        for key in keys.values():
            if key not in m:
                raise SystemExit(f"metric {key} missing")

        def mac(key):
            return BS + "metric{" + key + "}"

        row = [
            (mac(keys["mis"]), E.num(m, keys["mis"])),
            (mac(keys["val"]), E.num(m, keys["val"])),
            (mac(keys["dmin"]) + "--" + mac(keys["dmax"]), E.num(m, keys["dmin"]) + "–" + E.num(m, keys["dmax"])),
            (mac(keys["low"]), E.num(m, keys["low"])),
            ((mac("h1.etopo2022.source_share.fabdem") + BS + "%") if p == "etopo_2022" else "--",
             (E.num(m, "h1.etopo2022.source_share.fabdem") + "%") if p == "etopo_2022" else "–"),
            (mac(keys["nmad"]) + " / " + mac(keys["rmse"]), E.num(m, keys["nmad"]) + " / " + E.num(m, keys["rmse"])),
            (mac(keys["obs"]) + " / " + mac(keys["shift"]), E.num(m, keys["obs"]) + " / " + E.num(m, keys["shift"])),
        ]
        out.append({"product": p, "cells": row})
    return out


HEAD = ["Sectors mispredicted (of 7)", "Validated sectors mispredicted (of four)", "Displacement across sectors (km)",
        "Displacement, lowest level band (km)", "Emerged band from FABDEM",
        "NMAD / RMSE against ICESat-2 (m)", "First metre below 2026: from observed shoreline / shifted (km2)"]


def latex(m: dict, rows: list[dict]) -> list[str]:
    def esc(s):
        return s.replace("_", BS + "_")

    def mac(key):
        return BS + "metric{" + key + "}"

    head = [h.replace("km2", "km" + BS + "textsuperscript{2}").replace("%", BS + "%") for h in HEAD]
    cap = (BS + "textbf{Product report card.} For each grid: the sectors (of seven) it mispredicts after its "
           "area-fitted vertical shift, and the validated sectors (of four) among them; the range of its sector "
           "median displacements; its median displacement below " + mac("h1.disp_band.edge_1_m") + "~m; for "
           "ETOPO 2022, the share of its emerged band from FABDEM land heights acquired after emergence; its "
           "normalised median absolute deviation and root-mean-square error against ICESat-2 land heights (ATL08) "
           "on " + mac("h2.atl08.n_test_blocks") + " held-out test blocks, a result that rests largely on the two "
           "northeastern blocks; and the seabed it places in the first metre below the 2026 level, measured from "
           "the observed 2026 shoreline and on the grid shifted by its hindcast constant (Methods). The columns "
           "describe different populations and are not graded.")
    widths = ("2.1cm", "1.4cm", "1.5cm", "1.7cm", "1.6cm", "1.4cm", "1.8cm", "2.6cm")
    out = ["% BEGIN ERL-T1 (generated by figures/tab_report_card.py from results/metrics.json keys; do not edit by hand)",
           BS + "begin{table}[!htbp]", BS + "centering", BS + "footnotesize", BS + "setlength{" + BS + "tabcolsep}{3pt}",
           BS + "caption{" + cap + "}", BS + "label{tab:reportcard}",
           BS + "begin{tabular}{" + "".join(">{" + BS + "raggedright" + BS + "arraybackslash}p{" + w + "}"
                                           for w in widths) + "}", BS + "hline",
           "Product & " + " & ".join(head) + " " + BS + BS, BS + "hline"]
    for r in rows:
        out.append(esc(TEX_NAME[r["product"]]) + " & " + " & ".join(t for t, _ in r["cells"]) + " " + BS + BS)
    out += [BS + "hline", BS + "end{tabular}", BS + "end{table}", "% END ERL-T1"]
    return out


def preview(m: dict, rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    E.style()
    W = E.DOUBLE_MM
    colw = [19.0, 14.0, 15.0, 17.5, 17.5, 14.5, 18.0, 24.0]
    scale = (W - 2.0) / sum(colw)
    colw = [c * scale for c in colw]
    hh, rh = 21.0, 7.0
    H = hh + rh * len(rows) + 2.0
    fig = plt.figure(figsize=(W * C.MM, H * C.MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    x = 1.0
    heads = [["Product"], ["Sectors", "mispredicted", "(of 7)"], ["Validated", "sectors", "(of 4)"],
             ["Displacement", "across", "sectors (km)"], ["Displacement,", "lowest level", "band (km)"],
             ["Emerged", "band from", "FABDEM"], ["NMAD / RMSE", "against", "ICESat-2 (m)"],
             ["First metre", "below 2026 (km²):", "observed shoreline", "/ shifted"]]
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
        for w, (_, txt) in zip(colw[1:], r["cells"]):
            ax.text(x + 1.0, y + rh / 2, txt, ha="left", va="center", fontsize=E.FS, color=E.INK, linespacing=1.05)
            x += w
    ax.plot([1.0, W - 1.0], [hh + rh * len(rows), hh + rh * len(rows)], color=E.INK, lw=0.8)
    C.save(fig, "erl_tab_report_card")
    plt.close(fig)


def main() -> int:
    m = C.metrics()
    for k in ("h1.disp_band.edge_1_m", "h2.atl08.n_test_blocks"):
        if k not in m:
            raise SystemExit(f"metric {k} missing")
    rows = cells(m)
    block = latex(m, rows)
    if TABLES.exists():
        lines = TABLES.read_bytes().decode("utf-8").replace(chr(13) + chr(10), chr(10)).split(chr(10))
    else:
        lines = ["% ERL tables (D-100, D-102). Generated blocks sit between their BEGIN and END markers; every number",
                 "% is a metric macro from results/metrics.json (manuscript/numbers.tex). Needs, in the",
                 "% preamble, the array package.", ""]
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
