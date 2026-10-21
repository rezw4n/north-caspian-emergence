"""Fig. 5: the next metres (H3; D-081, D-090).

All values are read from the frozen tables in results/emergence/main/ (scripts/
a8_emergence.py; caspian.h3); nothing is recomputed.
a, emerged area below each surface's own 2026 shoreline against the decline below the
   2026 level (domain, "emergence" functional; emergence.csv), for each product as
   used (uncorrected) and for the fused surface (the waterline DEM inside the observed
   range, the product corrected by the hindcast error model, linear_hold, below it),
   with the split-conformal 90% intervals (lo_km2, hi_km2). Only the scenario depths
   are computed; no line joins them. The verdict of the pre-registered rule is printed
   (h3_decision.json).
b, the same per sector at 1 m and 2 m: fused estimate (bar) with its interval, and the
   uncorrected estimate (tick). The H3 run stores no per-cell emergence raster
   (only flood-level caches under data/interim/emergence/, which are not committed),
   so the panel is per sector, not a map.
c, exposure in the domain (exposure.csv) for six layers, fused surface (filled, with
   the interval from the equivalent levels of the area-interval bounds) and
   uncorrected (open): platforms and wells newly onshore, pipeline and channel length
   that emerges, ports and settlements that lose the sea within 1 km and 5 km, and the
   ESA WorldCover 2021 herbaceous wetland that emerges. Each surface is referenced to
   its own 2026 shoreline.
d, the calendar dates at which the primary published pathways reach each scenario
   depth (scenario_dates.csv, role "primary"); depths not reached by 2100 are listed.

Output: figures/out/fig5_next_metres.{pdf,tif,png}
Usage: python figures/fig5_next_metres.py
"""

from __future__ import annotations

import sys

import _common as C
import numpy as np
import pandas as pd

from caspian import plotting
from caspian.config import RESULTS
from caspian.geography import SECTOR_ORDER

EM = RESULTS / "emergence" / "main"
PK = {"gebco_2026": "gebco2026", "gebco_2022": "gebco2022", "etopo_2022": "etopo2022", "srtm15p_v2.7": "srtm15p",
      "etopo1": "etopo1"}
UNCORR = "#8c8c8c"
LAYERS = [("osm_platforms", "Platforms and wells\nnewly onshore (n)"),
          ("osm_pipelines", "Pipeline length\nemerged (km)"),
          ("osm_channels", "Channel length\nemerged (km)"),
          ("osm_ports", "Ports losing the sea\nwithin 1 km (n)"),
          ("osm_settlements", "Settlements losing the\nsea within 5 km (n)"),
          ("worldcover2021_herbaceous_wetland", "Herbaceous wetland\nemerged (km²)")]
SOURCE_NAME = {"hoseini2025": "Hoseini et al. 2025", "samant2023": "Samant & Prange 2023"}


def pathway_label(r) -> str:
    stat = {"median": "median", "linear trend": "linear trend"}.get(str(r.statistic), "best fit")
    scen = str(r.scenario).replace(" scenario", "").replace("SSP245", "SSP2-4.5").replace("SSP585", "SSP5-8.5")
    if stat == "linear trend":
        scen, stat = scen.split(" (")[0].replace("-", "\u2013"), "trend"
    return f"{SOURCE_NAME.get(r.source, r.source)}: {scen} ({stat})"


def letter(fig, text: str, y_top_mm: float, H: float) -> None:
    """Panel letter at the left edge of the figure, its top y_top_mm below the figure top."""
    fig.text(1.5 / C.DOUBLE_MM, 1 - y_top_mm / H, text, fontsize=8, fontweight="bold", va="top", ha="left",
             color=plotting.INK)


def main() -> int:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    m = C.metrics()
    e = pd.read_csv(EM / "emergence.csv")
    e = e[e["functional"] == "emergence"]
    dom = e[e["region"] == "domain"]
    x = pd.read_csv(EM / "exposure.csv")
    xd = x[x["region"] == "domain"]
    sd = pd.read_csv(EM / "scenario_dates.csv")
    prim = sd[(sd["role"] == "primary")].copy()
    scen = sorted(dom["scenario_m"].unique())
    near = (0.5, 2.0)

    C.style()
    W = C.DOUBLE_MM
    L, R = 12.0, 2.0
    TOP, HA, G1, HB, G2, HC, G3, HD, BOT = 9.0, 34.0, 15.0, 34.0, 17.0, 27.0, 15.0, 30.0, 9.0
    H = TOP + HA + G1 + HB + G2 + HC + G3 + HD + BOT
    fig = plt.figure(figsize=(W * C.MM, H * C.MM))

    def ax_mm(x0, y_top, w, h):
        return fig.add_axes([x0 / W, (H - y_top - h) / H, w / W, h / H])

    # a: emerged area against decline, one panel per product
    n = len(C.PRODUCTS)
    gap = 4.0
    pw = (W - L - R - (n - 1) * gap) / n
    axs_a = [ax_mm(L + i * (pw + gap), TOP, pw, HA) for i in range(n)]
    ymax = float(np.nanmax(dom[dom["surface"].isin(["fused", "uncorrected"])]["hi_km2"])) / 1000.0 * 1.08
    for ax, p in zip(axs_a, C.PRODUCTS):
        ax.axvspan(*near, color="#f0f0f0", lw=0, zorder=0)
        for surf, dx, style in (("uncorrected", -0.09, "open"), ("fused", 0.09, "filled")):
            g = dom[(dom["product"] == p) & (dom["surface"] == surf)].sort_values("scenario_m")
            xs = g["scenario_m"].to_numpy() + dx
            est, lo, hi = (g[c].to_numpy(float) / 1000.0 for c in ("estimate_km2", "lo_km2", "hi_km2"))
            col = C.PRODUCT_COLOURS[p] if surf == "fused" else UNCORR
            ax.errorbar(xs, est, yerr=[est - lo, hi - est], fmt="none", ecolor=col, elinewidth=0.7, capsize=0, zorder=2)
            ax.plot(xs, est, C.PRODUCT_MARKERS[p], ms=3.4, mfc=col if style == "filled" else "white", mec=col, mew=0.8,
                    zorder=3)
        ax.set_xlim(0, 5.5)
        ax.set_xticks(scen, [f"{s:g}" for s in scen])
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", color=plotting.GRID, lw=0.4)
        ax.set_axisbelow(True)
        ax.set_title(C.PRODUCT_NAMES[p], fontsize=6.5, pad=2)
        k = PK[p]
        ax.text(0.04, 0.96, f"H3: {m[f'h3.verdict.{k}']['value']}\n(at {m[f'h3.relevant_scenarios.{k}']['value']} m)",
                transform=ax.transAxes, ha="left", va="top", fontsize=5.5, color=plotting.INK, linespacing=1.1)
        ax.set_xlabel("Decline below 2026 (m)")
        if ax is not axs_a[0]:
            ax.tick_params(labelleft=False)
    axs_a[0].set_ylabel("Emerged area (10³ km²)")
    fig.legend(handles=[Line2D([], [], ls="none", marker="o", ms=3.4, mfc=plotting.INK, mec=plotting.INK,
                               label="Fused surface (DEM + corrected product), 90% interval"),
                        Line2D([], [], ls="none", marker="o", ms=3.4, mfc="white", mec=UNCORR, mew=0.8,
                               label="Product as used (uncorrected), 90% interval"),
                        Patch(fc="#f0f0f0", ec="none", label="Near-term scenarios (decision rule)")],
               loc="upper left", bbox_to_anchor=(L / W, 1 - 0.6 / H), ncol=3, fontsize=5.5, handlelength=1.2,
               columnspacing=1.2, frameon=False)
    letter(fig, "a", TOP - 1.0, H)

    # b: per sector at 1 m and 2 m
    yb = TOP + HA + G1
    bw = (W - L - R - 8.0) / 2
    axs_b = [ax_mm(L, yb, bw, HB), ax_mm(L + bw + 8.0, yb, bw, HB)]
    offs = np.linspace(-0.32, 0.32, n)
    width = 0.14
    bmax = 0.0
    for ax, s in zip(axs_b, (1.0, 2.0)):
        for i, sec in enumerate(SECTOR_ORDER):
            for p, dx in zip(C.PRODUCTS, offs):
                g = e[(e["region"] == sec) & (e["product"] == p) & (e["scenario_m"] == s)].set_index("surface")
                f, u = g.loc["fused"], g.loc["uncorrected"]
                ax.bar(i + dx, f["estimate_km2"] / 1000.0, width=width, color=C.PRODUCT_COLOURS[p], lw=0, zorder=2)
                ax.plot([i + dx, i + dx], [f["lo_km2"] / 1000.0, f["hi_km2"] / 1000.0], color=plotting.INK, lw=0.5,
                        zorder=3)
                ax.plot([i + dx - width / 2, i + dx + width / 2], [u["estimate_km2"] / 1000.0] * 2, color=plotting.INK,
                        lw=1.0, zorder=4, solid_capstyle="butt")
                bmax = max(bmax, f["hi_km2"] / 1000.0, u["estimate_km2"] / 1000.0)
        ax.set_xticks(range(len(SECTOR_ORDER)), [C.SECTOR_TINY[s_].replace(" ", "\n", 1) for s_ in SECTOR_ORDER])
        ax.tick_params(axis="x", length=0, labelsize=5.5)
        ax.set_xlim(-0.55, len(SECTOR_ORDER) - 0.45)
        ax.grid(axis="y", color=plotting.GRID, lw=0.4)
        ax.set_axisbelow(True)
        ax.text(0.01, 0.97, f"{s:g} m below the 2026 level", transform=ax.transAxes, ha="left", va="top", fontsize=6)
    for ax in axs_b:
        ax.set_ylim(0, bmax * 1.05)
    axs_b[0].set_ylabel("Newly emerged seabed\n(10³ km²)")
    fig.legend(handles=[Patch(fc=C.PRODUCT_COLOURS[p], ec="none", label=C.PRODUCT_NAMES[p]) for p in C.PRODUCTS] +
               [Line2D([], [], color=plotting.INK, lw=0.5, label="fused 90% interval"),
                Line2D([], [], color=plotting.INK, lw=1.0, label="uncorrected estimate")],
               loc="lower left", bbox_to_anchor=(L / W, (H - yb + 0.5) / H), ncol=7, fontsize=5.5, handlelength=1.2,
               columnspacing=1.0, frameon=False)
    letter(fig, "b", yb - 3.5, H)

    # c: exposure by category
    yc = yb + HB + G2
    nl = len(LAYERS)
    cg = 6.5
    cw = (W - L - R - (nl - 1) * cg) / nl
    axs_c = [ax_mm(L + i * (cw + cg), yc, cw, HC) for i in range(nl)]
    xs = np.arange(len(scen))
    coff = np.linspace(-0.3, 0.3, n)
    for ax, (lay, lab) in zip(axs_c, LAYERS):
        g = xd[xd["layer"] == lay]
        for p, dx in zip(C.PRODUCTS, coff):
            for surf in ("uncorrected", "fused"):
                t = g[(g["product"] == p) & (g["surface"] == surf)].set_index("scenario_m").reindex(scen)
                a, lo, hi = (t[c].to_numpy(float) for c in ("amount", "amount_lo", "amount_hi"))
                col = C.PRODUCT_COLOURS[p]
                if surf == "fused":
                    ax.vlines(xs + dx, lo, hi, color=col, lw=0.6, zorder=2)
                    ax.plot(xs + dx, a, C.PRODUCT_MARKERS[p], ms=2.6, mfc=col, mec=col, mew=0.3, zorder=3)
                else:
                    ax.plot(xs + dx, a, C.PRODUCT_MARKERS[p], ms=2.4, mfc="white", mec=col, mew=0.5, zorder=2)
        ax.set_xticks(xs, [f"{s:g}" for s in scen])
        ax.set_xlim(-0.6, len(scen) - 0.4)
        ax.set_ylim(0, None)
        ax.grid(axis="y", color=plotting.GRID, lw=0.4)
        ax.set_axisbelow(True)
        ax.set_title(lab, fontsize=5.5, pad=2, linespacing=1.05)
        ax.set_xlabel("Decline (m)", fontsize=5.5)
        ax.tick_params(labelsize=5.5)
        if lay in ("osm_platforms", "osm_ports", "osm_settlements"):
            ax.yaxis.get_major_locator().set_params(integer=True)
    letter(fig, "c", yc - 4.5, H)
    fig.text(1 - R / W, 1 - (yc - 9.0) / H, "c: filled, fused surface (line: interval from the area-interval bounds); "
             "open, product as used", fontsize=5.5, ha="right", va="top", color=plotting.INK)

    # d: scenario dates of the primary pathways
    yd = yc + HC + G3
    LBL = 62.0
    ax_d = ax_mm(L + LBL, yd, W - L - R - LBL - 30.0, HD)
    ids = list(dict.fromkeys(prim["pathway_id"]))
    first = prim.drop_duplicates("pathway_id").set_index("pathway_id")
    ys = {pid: i for i, pid in enumerate(ids[::-1])}
    for pid in ids:
        g = prim[prim["pathway_id"] == pid].sort_values("delta_m")
        y = ys[pid]
        reached = g[g["reached"] == True]  # noqa: E712
        ax_d.plot([2026, 2100], [y, y], color=plotting.GRID, lw=0.5, zorder=0)
        for r in reached.itertuples():
            ax_d.plot(r.date, y, "o", ms=8.4, mfc="white", mec=plotting.INK, mew=0.6, zorder=3)
            ax_d.text(r.date, y, f"{r.delta_m:g}", ha="center", va="center", fontsize=5.0, zorder=4)
        miss = g[g["reached"] != True]["delta_m"].tolist()  # noqa: E712
        if miss:
            txt = "none reached by 2100" if len(miss) == len(g) else \
                "not by 2100: " + ", ".join(f"{v:g}" for v in miss) + " m"
            ax_d.text(2101.5, y, txt, ha="left", va="center", fontsize=5.2, color=plotting.INK_MUTED, clip_on=False)
    ax_d.set_yticks(list(ys.values()), [pathway_label(first.loc[pid]) for pid in ys])
    ax_d.tick_params(axis="y", length=0, labelsize=5.5)
    ax_d.set_ylim(-0.6, len(ids) - 0.4)
    ax_d.set_xlim(2026, 2100)
    ax_d.set_xticks([2026, 2040, 2060, 2080, 2100])
    ax_d.set_xlabel("Year the pathway reaches the decline (circled, m below the 2026 level)")
    letter(fig, "d", yd - 3.5, H)

    C.save(fig, "fig5_next_metres")
    plt.close(fig)
    print("primary pathways:", ids, "| y max (10^3 km2):", round(ymax, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
