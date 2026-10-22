"""ERL figure "Auditing the bathymetry grids" (the Letter's audit figure; 15 cm wide, text
8 pt or more). The ERL version of figures/fig4_audit.py, which
stays the Supplementary figure: the per-block maps (fig4_audit c) are left out here
(they are in the Supplementary), the other panels are kept and enlarged.

(a) Area bias of each product's predicted sea against the observed sea, per (epoch, level)
    unit, inside the domain and within the valid mask (results/hindcast/main/extent.csv,
    sector "domain", area_bias_rel; units with observed sea): left, after each product's
    best-fitting vertical constant (the shape variant, which decides); right,
    on the documented datum chain without a constant (the total variant).
    Both halves share one y-range.
(b) Shoreline displacement by level band, shape variant, split into the four sectors whose
    observations are validated (west, interfluve, northeast, open shelf; A18 item 1) and
    the three that are not (Volga delta, Ural delta, Komsomolets Bay):
    per band, the median over the group's sectors and levels of the per-level
    median displacement, computed as scripts/a7_level_bands.py computes
    displacement_by_level_band.csv from displacement.csv (pooled epochs, observed levels,
    n > 0), which the script checks for all sectors together. Dashed line:
    h1.disp_band.edge_2_m.
(c) Per sector, validated sectors first, each product's per-level ratio D/Q of its median
    displacement to the noise floor (the 95th percentile of the half-against-half
    distances) at every evaluable level (h1_mispredicts_detail.csv, shape variant; dots),
    with the median and quartiles over levels from the sector revision analysis,
    results/revision/s/item2_dq.csv. The script checks that the medians of
    the dots equal item2_dq.csv. The H1 rule is that D/Q exceeds 1 at a majority of the
    levels; filled where the product mispredicts the sector (h1_mispredicts.csv). Ratios
    below the axis (zero displacements) are drawn on it.

Panel letters (a) area bias, (b) bands, (c) sectors keep the Letter's existing references to
(a) and (b) valid. Nothing is recomputed: the values are those of fig4_audit.py, read from the
same tables.
Output: figures/out/erl_fig_audit.{pdf,tif,png}
Usage: python figures/erl_fig_audit.py
"""

from __future__ import annotations

import sys

import _common as C
import _erl as E
import numpy as np
import pandas as pd

from caspian.config import RESULTS
from caspian.geography import SECTOR_ORDER

HC = RESULTS / "hindcast" / "main"
DQ = RESULTS / "revision" / "s" / "item2_dq.csv"
VALIDATED = ("west", "interfluve", "northeast", "open_shelf")      # A18 item 1
NOT_VALIDATED = tuple(s for s in SECTOR_ORDER if s not in VALIDATED)
BANDS = [-29.0, -28.3, -27.7, -27.0, -26.3, -25.5]                 # scripts/a7_level_bands.py
DQ_FLOOR = 0.01                                                     # bottom of the D/Q axis
PRODUCTS = ("gebco_2026", "gebco_2022", "etopo_2022", "etopo1", "srtm15p_v2.7")
EPOCH_STYLE = {"E3": ("-", 1.1), "E2": ((0, (3.5, 1.6)), 1.0), "E1": ((0, (1.0, 1.3)), 1.0)}
SECTOR_LABEL = {"west": "Kizlyar–\nDagestan", "volga": "Volga\ndelta", "interfluve": "Volga–Ural\ninterfluve",
                "ural": "Ural\ndelta", "northeast": "NE Kazakh\ncoast", "komsomolets": "Komsomolets\nBay",
                "open_shelf": "Open\nshelf"}
FLOOR = "#e3e3e3"


def band_table(disp: pd.DataFrame, sectors=None) -> pd.DataFrame:
    """Per level band and product, the median over sectors and levels of the per-level median
    displacement (m), as scripts/a7_level_bands.py; ``sectors`` restricts the sectors."""
    d = disp[(disp["epoch"] == "pooled") & (disp["sector"] != "all") & disp["observed_level"] & (disp["n"] > 0)
             & (disp["variant"] == "shape")].copy()
    if sectors is not None:
        d = d[d["sector"].isin(sectors)]
    d["band"] = pd.cut(d["level_m"], BANDS).astype(str)
    return d.groupby(["band", "product"])["median_m"].median().unstack()


def band_edges(labels) -> list[tuple[float, float]]:
    return [tuple(float(v) for v in b.strip("(]").split(",")) for b in labels]


def main() -> int:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    m = C.metrics()
    ext = pd.read_csv(HC / "extent.csv")
    dom = ext[(ext["sector"] == "domain") & (ext["area_obs_km2"] > 0)].copy()
    dom["bias_pct"] = 100.0 * dom["area_bias_rel"]
    disp = pd.read_csv(HC / "displacement.csv")
    ref = pd.read_csv(HC / "displacement_by_level_band.csv")
    ref = ref[ref["variant"] == "shape"].set_index("band")
    mine = band_table(disp)
    for p in PRODUCTS:
        for b in ref.index:
            if abs(float(mine.loc[b, p]) - float(ref.loc[b, p])) > 0.051:
                raise SystemExit(f"band {b} {p}: {mine.loc[b, p]:.1f} m differs from displacement_by_level_band.csv")
    groups = {"validated": band_table(disp, VALIDATED), "not": band_table(disp, NOT_VALIDATED)}
    det = pd.read_csv(HC / "h1_mispredicts_detail.csv")
    det = det[(det["variant"] == "shape") & (det["evaluable"] == True)].copy()  # noqa: E712
    det["dq"] = det["median"] / det["p95_noise"]
    dq = pd.read_csv(DQ)
    dq = dq[dq["variant"] == "shape"].set_index(["product", "sector"])
    for (p, s), g in det.groupby(["product", "sector"]):
        if abs(float(g["dq"].median()) - float(dq.loc[(p, s), "dq_median"])) > 1e-9 or len(g) != dq.loc[(p, s), "dq_n"]:
            raise SystemExit(f"{p} {s}: per-level D/Q differs from item2_dq.csv")
    mis = pd.read_csv(HC / "h1_mispredicts.csv")
    mis = mis[mis["variant"] == "shape"].set_index(["product", "sector"])["mispredicts"]

    E.style()
    W = E.DOUBLE_MM
    L, R = 15.0, 2.0
    TOP, HA, G1, HB, G2, HD, BOT = 11.0, 40.0, 16.0, 38.0, 19.0, 50.0, 13.0
    H = TOP + HA + G1 + HB + G2 + HD + BOT
    fig = plt.figure(figsize=(W * C.MM, H * C.MM))
    kw = {"fontsize": E.FS, "frameon": False, "borderaxespad": 0, "labelspacing": 0.35}

    # product key, across the top ------------------------------------------------------------------------
    handles = [Line2D([], [], color=E.COLOURS[p], marker=E.MARKERS[p], ms=4.5, mfc=E.COLOURS[p], mec="white", mew=0.5,
                      lw=1.4, label=E.NAMES[p]) for p in PRODUCTS]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(L / W, 1 - 1.0 / H), ncol=len(PRODUCTS),
               handlelength=2.2, columnspacing=1.6, handletextpad=0.5, **kw)

    # (a) area bias, two variants, one y-range ------------------------------------------------------------
    gap_a = 8.0
    aw = (W - L - R - gap_a) / 2
    titles = {"shape": "After the best-fitting vertical shift", "total": "On the datum chain, no shift"}
    for i, variant in enumerate(("shape", "total")):
        ax = E.mm_axes(fig, W, H, L + i * (aw + gap_a), TOP, aw, HA)
        d = dom[dom["variant"] == variant]
        for p in PRODUCTS:
            for ep, (ls, lw) in EPOCH_STYLE.items():
                g = d[(d["product"] == p) & (d["epoch"] == ep)].sort_values("level_m")
                ax.plot(g["level_m"], g["bias_pct"], ls=ls, lw=lw, color=E.COLOURS[p], marker=E.MARKERS[p], ms=3.0,
                        mew=0, markevery=2, zorder=3)
        ax.axhline(0, color=E.MUTED, lw=0.6, zorder=1)
        ax.set_xlabel("Level (m, DAHITI reference)")
        ax.set_ylim(-55, 70)
        ax.set_yticks([-40, -20, 0, 20, 40, 60])
        if i == 0:
            ax.set_ylabel("Area bias (%)")
            E.label(ax, "a", x=-0.2, y=1.02)
        else:
            ax.tick_params(labelleft=False)
        ax.grid(axis="y", color=E.GRID, lw=0.4)
        ax.set_axisbelow(True)
        ax.set_xticks([-28.5, -28.0, -27.5, -27.0, -26.5, -26.0])
        ax.text(0.08 if i == 0 else 0.0, 1.03, titles[variant], transform=ax.transAxes, ha="left", va="bottom",
                fontsize=E.FS, color=E.INK)

    # (b) displacement by level band, validated and not validated sectors ---------------------------------
    leg_w = 36.0
    gap_b = 4.0
    bw = (W - L - R - leg_w - gap_b - 6.0) / 2
    yb = TOP + HA + G1
    split = float(m["h1.disp_band.edge_2_m"]["value"])
    btitle = {"validated": f"Validated sectors ({len(VALIDATED)})", "not": f"Not validated ({len(NOT_VALIDATED)})"}
    for i, key in enumerate(("validated", "not")):
        axb = E.mm_axes(fig, W, H, L + i * (bw + gap_b), yb, bw, HB)
        t = groups[key]
        edges = band_edges(t.index)
        for p in PRODUCTS:
            vals = t[p].to_numpy(float) / 1000.0
            for k, (lo, hi) in enumerate(edges):
                axb.plot([lo, hi], [vals[k]] * 2, color=E.COLOURS[p], lw=1.6, solid_capstyle="butt", zorder=3)
                axb.plot((lo + hi) / 2, vals[k], E.MARKERS[p], ms=4.0, mfc=E.COLOURS[p], mec="white", mew=0.5,
                         zorder=4)
        axb.axvline(split, color=E.MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        for lo, _ in edges:
            axb.axvline(lo, color=E.GRID, lw=0.4, zorder=0)
        axb.set_yscale("log")
        axb.set_ylim(0.15, 40)
        axb.set_yticks([0.3, 1, 3, 10, 30], ["0.3", "1", "3", "10", "30"])
        axb.yaxis.set_minor_locator(plt.NullLocator())
        axb.set_xlim(BANDS[0], BANDS[-1])
        axb.set_xticks([-29, -28, -27, -26])
        axb.set_xlabel("Level band (m, DAHITI reference)")
        axb.text(0.12 if i == 0 else 0.0, 1.03, btitle[key], transform=axb.transAxes, ha="left", va="bottom",
                 fontsize=E.FS, color=E.INK)
        if i == 0:
            axb.set_ylabel("Displacement (km)")
            E.label(axb, "b", x=-0.33, y=1.02)
            axb.text(split + 0.06, 0.17, E.num(m, "h1.disp_band.edge_2_m", "m"), fontsize=E.FS, color=E.MUTED,
                     ha="left", va="bottom")
        else:
            axb.tick_params(labelleft=False)

    # keys for (a) and (c), right of (b) ------------------------------------------------------------------
    kx = (L + 2 * bw + gap_b + 8.0) / W
    ep_h = [Line2D([], [], color=E.INK, ls=EPOCH_STYLE[e][0], lw=EPOCH_STYLE[e][1], label=e) for e in ("E1", "E2", "E3")]
    fig.legend(handles=ep_h, loc="upper left", bbox_to_anchor=(kx, 1 - (yb - 1.0) / H), title="Epoch (a)",
               alignment="left", ncol=3, handlelength=2.0, columnspacing=0.8, handletextpad=0.4, **kw)
    ck = [Line2D([], [], ls="none", marker="o", ms=2.0, mfc=E.MUTED, mec="none", label="One level"),
          Line2D([], [], color=E.INK, lw=1.0, marker="|", ms=0, label="Quartiles over levels"),
          Line2D([], [], ls="none", marker="o", ms=5, mfc=E.INK, mec=E.INK, label="Median; mispredicts"),
          Line2D([], [], ls="none", marker="o", ms=5, mfc="white", mec=E.INK, mew=1.0, label="Median; does not"),
          Patch(fc=FLOOR, ec="none", label="D/Q below 1: within\nthe noise floor")]
    fig.legend(handles=ck, loc="upper left", bbox_to_anchor=(kx, 1 - (yb + 12.0) / H), title="D/Q per sector (c)",
               alignment="left", handlelength=1.4, handletextpad=0.5, **kw)

    # (c) per-level D/Q by sector -------------------------------------------------------------------------
    yd = yb + HB + G2
    axd = E.mm_axes(fig, W, H, L, yd, W - L - R, HD)
    order = list(VALIDATED) + list(NOT_VALIDATED)
    offs = np.linspace(-0.32, 0.32, len(PRODUCTS))
    rng = np.random.default_rng(20261022)
    axd.axhspan(DQ_FLOOR, 1.0, color=FLOOR, lw=0, zorder=0)
    axd.axhline(1.0, color=E.MUTED, lw=0.8, zorder=1)
    for i, s in enumerate(order):
        for p, dx in zip(PRODUCTS, offs):
            g = det[(det["product"] == p) & (det["sector"] == s)]
            v = np.maximum(g["dq"].to_numpy(float), DQ_FLOOR)
            jit = rng.uniform(-0.035, 0.035, v.size)
            axd.plot(i + dx + jit, v, "o", ms=1.6, mfc=E.COLOURS[p], mec="none", alpha=0.55, zorder=2, clip_on=False)
            r = dq.loc[(p, s)]
            axd.plot([i + dx, i + dx], [max(r["dq_p25"], DQ_FLOOR), r["dq_p75"]], color=E.COLOURS[p], lw=1.0,
                     zorder=3, solid_capstyle="butt")
            filled = bool(mis.get((p, s), False))
            axd.plot(i + dx, r["dq_median"], E.MARKERS[p], ms=5.5, mfc=E.COLOURS[p] if filled else "white",
                     mec=E.COLOURS[p], mew=1.0, zorder=4)
    nv = len(VALIDATED)
    axd.axvline(nv - 0.5, color=E.INK, lw=0.6, zorder=1)
    for x, txt in ((nv / 2 - 0.5, "Observations validated"), (nv + len(NOT_VALIDATED) / 2 - 0.5, "Not validated")):
        axd.text(x, 1.01, txt, transform=axd.get_xaxis_transform(), ha="center", va="bottom", fontsize=E.FS,
                 color=E.INK)
    axd.set_yscale("log")
    axd.set_ylim(DQ_FLOOR, 100)
    axd.set_yticks([0.01, 0.1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100"])
    axd.yaxis.set_minor_locator(plt.NullLocator())
    axd.set_xlim(-0.55, len(order) - 0.45)
    axd.set_xticks(range(len(order)), [SECTOR_LABEL[s] for s in order])
    axd.tick_params(axis="x", length=0)
    axd.set_ylabel("D/Q")
    axd.grid(axis="y", color=E.GRID, lw=0.4)
    axd.set_axisbelow(True)
    E.label(axd, "c", x=-0.09, y=1.02)
    C.save(fig, "erl_fig_audit")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())

