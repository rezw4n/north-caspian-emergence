"""RSE figure: near-term emergence measured from the observed 2026 shoreline (double column).

a, The first-metre increment from the observed 2026 shoreline, re-referenced (PREREG
   Amendment 19 item 1): for every 60 m cell, how many of the five grids place it in
   S_obs and T - 1 m <= F < T, where S_obs is the composite observed sea of 2026, F the
   grid's flood-level map and T the level at which the grid's sea area equals S_obs
   (data/interim/revision2/n/firstmetre_counts_60m.npz, "observed_reref"). Display: per
   240 m block the most frequent count among its cells (ties to the higher count); blocks
   whose most frequent count is 0 are not coloured. Background: the open sea of the
   2025-26 low stand (results/study_area/sea_mask_1km.npz). The script checks that the
   cells placed by at least one grid and by all five give the areas of results/metrics.json
   (rev2.n.agree60.observed_reref.1.*).
b, E(D), the seabed emerging between the 2026 shoreline and a further fall D, for each grid
   on the re-referenced baseline (results/revision2/n/reref_curves_fine.csv.gz, domain,
   0.01 m steps; checked against the 1 m metrics): solid up to the largest decline of the
   hindcast pairs (rev.n.env.max_pair_decline_m), dotted beyond (uncalibrated). Shading,
   per grid, between the estimates with the 2026 level raised and lowered by 0.3 m
   (results/revision2/n/reref_curves.csv at 0.5, 1, 2 and 3 m, joined by straight lines).
c, Decline below the 2026 level from 2026 to 2070 for the seven primary published pathways
   (scripts/a8_emergence.DATE_PATHWAYS, values in results/levels/projection_pathways.csv,
   piecewise-linear knots of caspian.exposure.pathway_curves, as in the ERL timeline
   figure), each with its own colour and dash; grey band, the range pathways. Dotted line,
   the calibrated limit of b.

Output: figures/out/rse_fig_nearterm.{pdf,tif,png}
Usage: python figures/rse_fig_nearterm.py
"""

from __future__ import annotations

import importlib.util
import sys

import _common as C
import _rse as R
import numpy as np
import pandas as pd

from caspian import exposure as ex
from caspian.config import INTERIM, RESULTS

N2 = RESULTS / "revision2" / "n"
COUNTS = INTERIM / "revision2" / "n" / "firstmetre_counts_60m.npz"
FACTOR = 4
PATH_LABEL = {"SAM23-07": "Samant & Prange 2023, SSP2-4.5", "SAM23-10": "Samant & Prange 2023, SSP5-8.5",
              "SAM23-14": "Samant & Prange 2023, observed trend",
              "HOS25-01": "Hoseini et al. 2025, low", "HOS25-04": "Hoseini et al. 2025, medium",
              "HOS25-07": "Hoseini et al. 2025, medium-high", "HOS25-10": "Hoseini et al. 2025, high"}
# Samant & Prange: Paul Tol "muted" indigo, teal, olive, dashed styles; Hoseini et al.: an ordered
# sequence (cmocean "matter") from low to high, solid.
PATH_STYLE = {"SAM23-07": ("#332288", (0, (5, 1.6))), "SAM23-10": ("#44AA99", (0, (5, 1.6))),
              "SAM23-14": ("#999933", (0, (1.2, 1.4))),
              "HOS25-01": (None, "solid"), "HOS25-04": (None, "solid"), "HOS25-07": (None, "solid"),
              "HOS25-10": (None, "solid")}


def a8():
    spec = importlib.util.spec_from_file_location("a8_emergence", C.ROOT / "scripts" / "a8_emergence.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def block_mode(count: np.ndarray, factor: int = FACTOR, kmax: int = 5) -> np.ndarray:
    r, c = (count.shape[0] // factor) * factor, (count.shape[1] // factor) * factor
    best = np.zeros((r // factor, c // factor), np.uint8)
    top = np.full(best.shape, -1, np.int32)
    for k in range(kmax + 1):
        s = (count[:r, :c] == k).reshape(r // factor, factor, c // factor, factor).sum(axis=(1, 3))
        take = s >= top
        best[take], top[take] = k, s[take]
    return best


def main() -> int:
    import cmocean
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    m = C.metrics()
    calib = R.value(m, "rev.n.env.max_pair_decline_m")
    z = np.load(COUNTS)
    cnt = z["observed_reref"]
    a, _, c0, _, e, f0 = z["transform"]
    cell_km2 = abs(a * e) / 1e6
    any_km2, all_km2 = float((cnt > 0).sum()) * cell_km2, float((cnt == 5).sum()) * cell_km2
    for got, key in ((any_km2, "rev2.n.agree60.observed_reref.1.any_km2"),
                     (all_km2, "rev2.n.agree60.observed_reref.1.all_km2")):
        if abs(got - R.value(m, key)) > 0.5:
            raise SystemExit(f"counts give {got:.1f} km2, {key} says {R.value(m, key)}")
    mode = block_mode(cnt)
    r_, c_ = mode.shape[0] * FACTOR, mode.shape[1] * FACTOR
    ext = [c0, c0 + a * c_, f0 + e * r_, f0]
    del cnt
    fine = pd.read_csv(N2 / "reref_curves_fine.csv.gz")
    fine = fine[(fine["surface"] == "observed_reref") & (fine["region"] == "domain")]
    for p in R.PRODUCTS:
        v = float(fine[(fine["product"] == p) & np.isclose(fine["delta_m"], 1.0)]["E_km2"].iloc[0])
        key = f"rev2.n.reref.area.main.{R.PK[p]}.E.1.domain_km2"
        if abs(v - R.value(m, key)) > 0.5:
            raise SystemExit(f"{p}: E(1 m) {v:.1f} differs from {key}")
    cv = pd.read_csv(N2 / "reref_curves.csv")
    cv = cv[(cv["variant"] == "area_match") & (cv["region"] == "domain") & (cv["quantity"] == "E")]
    mod = a8()
    pw = mod.read_pathways(mod.PATHWAYS)
    prim = [pid for pid, role, *_ in mod.DATE_PATHWAYS if role == "primary"]
    rng = [pid for pid, role, *_ in mod.DATE_PATHWAYS if role == "range"]
    allc = ex.pathway_curves(pw)
    if set(prim) != set(PATH_LABEL):
        raise SystemExit(f"primary pathways {sorted(prim)} differ from the labelled set")
    hos = [p for p in ("HOS25-01", "HOS25-04", "HOS25-07", "HOS25-10")]
    for k, pid in enumerate(hos):
        PATH_STYLE[pid] = (cmocean.cm.matter(0.3 + 0.2 * k), "solid")
    t = np.arange(2026.0, 2070.01, 0.25)

    def decline(pid):
        yrs, ch = allc[pid]
        return -(np.interp(t, yrs, ch) - float(np.interp(2026.0, yrs, ch)))

    R.style()
    W = R.DOUBLE_MM
    proj, fwd = C.projection()
    domain, sectors = C.study_geometries()
    mx0, mx1, my0, my1 = -292_000.0, 372_000.0, -172_000.0, 212_000.0
    L, RGT, TOP = 7.0, 2.0, 2.0
    MAP_W = 108.0
    map_h = MAP_W * (my1 - my0) / (mx1 - mx0)
    G1, HC, BOT = 16.0, 46.0, 10.0
    H = TOP + map_h + G1 + HC + BOT
    fig = plt.figure(figsize=(W * C.MM, H * C.MM))

    # a, map ---------------------------------------------------------------------------------------
    ax = R.mm_axes(fig, W, H, L, TOP, MAP_W, map_h)
    ax.set_facecolor(R.LAND)
    sea, sea_ext = C.sea_mask()
    ax.imshow(np.where(sea, 1.0, np.nan), extent=sea_ext, origin="upper", cmap=ListedColormap([R.SEA]),
              interpolation="nearest", zorder=1, rasterized=True)
    ccols = cmocean.cm.tempo(np.linspace(0.3, 1.0, 5))
    cmap = ListedColormap(ccols)
    ax.imshow(np.ma.masked_equal(mode, 0), extent=ext, origin="upper", cmap=cmap,
              norm=BoundaryNorm(np.arange(0.5, 6.0, 1.0), cmap.N), interpolation="nearest", zorder=2, rasterized=True)
    R.laea_graticule(ax, fwd, (mx0, mx1, my0, my1), lons=np.arange(47, 55, 1.0), lats=np.arange(44, 48, 1.0))
    for gg in sectors.values():
        xs, ys = gg.exterior.xy
        ax.plot(xs, ys, color=R.MUTED, lw=0.35, zorder=4)
    xs, ys = domain.exterior.xy
    ax.plot(xs, ys, color=R.INK, lw=0.6, zorder=4)
    R.map_axes_clean(ax)
    ax.set_xlim(mx0, mx1)
    ax.set_ylim(my0, my1)
    R.scale_bar(ax, -60_000.0, my0 + 12_000, C.SCALE_KM * 1000.0, R.num(m, "params.fig_scale_bar_km", "km"))
    R.label(ax, "a", inside=True)
    cax = R.mm_axes(fig, W, H, L + 4.0, TOP + map_h + 7.0, 50.0, 2.0)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=BoundaryNorm(np.arange(0.5, 6.0, 1.0), cmap.N), cmap=cmap), cax=cax,
                      orientation="horizontal")
    cb.set_ticks([1, 2, 3, 4, 5])
    cb.ax.tick_params(labelsize=R.FS_SMALL, length=0, pad=1.5)
    cb.outline.set_linewidth(0.4)
    cb.set_label("Grids placing the cell in the first metre", fontsize=R.FS_SMALL, labelpad=1.5)
    cb.ax.xaxis.set_label_position("top")
    fig.legend(handles=[Patch(fc=R.SEA, ec="none", label="Open sea, 2025–26")], loc="upper left",
               bbox_to_anchor=((L + 60.0) / W, 1 - (TOP + map_h + 6.0) / H), frameon=False, fontsize=R.FS_SMALL,
               borderaxespad=0, handlelength=1.4)

    # b, E(D) ------------------------------------------------------------------------------------------
    bx = L + MAP_W + 14.0
    bw = W - bx - RGT
    axb = R.mm_axes(fig, W, H, bx, TOP + 2.0, bw, map_h - 2.0)
    for p in R.PRODUCTS:
        f = fine[fine["product"] == p].sort_values("delta_m")
        dd, ee = f["delta_m"].to_numpy(float), f["E_km2"].to_numpy(float) / 1000.0
        k = dd <= calib + 1e-9
        axb.plot(dd[k], ee[k], color=R.COLOURS[p], lw=1.1, zorder=3)
        axb.plot(dd[~k], ee[~k], color=R.COLOURS[p], lw=1.1, ls=(0, (1.2, 1.4)), zorder=3)
        s = cv[cv["product"] == p]
        lo = s[s["sensitivity"] == "L26-0.3"].sort_values("delta_m")
        hi = s[s["sensitivity"] == "L26+0.3"].sort_values("delta_m")
        x = lo["delta_m"].to_numpy(float)
        y1, y2 = lo["km2"].to_numpy(float) / 1000.0, hi["km2"].to_numpy(float) / 1000.0
        axb.fill_between(x, np.minimum(y1, y2), np.maximum(y1, y2), color=R.COLOURS[p], alpha=0.12, lw=0, zorder=1)
    axb.axvline(calib, color=R.MUTED, lw=0.6, ls=(0, (2, 2)), zorder=2)
    axb.text(calib - 0.04, 0.97, "calibrated\nlimit", transform=axb.get_xaxis_transform(), ha="right", va="top",
             fontsize=R.FS_SMALL, color=R.MUTED, linespacing=1.0)
    axb.set_xlim(0, 3)
    axb.set_ylim(0, None)
    axb.set_xlabel("Fall below the 2026 level, D (m)")
    axb.set_ylabel("Emerging seabed, E(D) (10³ km²)")
    axb.grid(color=R.GRID, lw=0.4)
    axb.set_axisbelow(True)
    R.fig_label(fig, W, H, "b", bx - 12.0, TOP + 1.0)
    gh = [Line2D([], [], color=R.COLOURS[p], lw=1.1, label=R.NAMES[p]) for p in R.PRODUCTS]
    gh += [Line2D([], [], color=R.INK, lw=1.1, ls=(0, (1.2, 1.4)), label="beyond calibration"),
           Patch(fc=R.MUTED, alpha=0.25, ec="none", label="2026 level ±0.3 m")]
    axb.legend(handles=gh, loc="upper left", fontsize=R.FS_SMALL, handlelength=1.6, labelspacing=0.25,
               borderaxespad=0.3, frameon=True, framealpha=0.9, edgecolor="none")

    # c, pathways --------------------------------------------------------------------------------------
    y3 = TOP + map_h + G1
    axc = R.mm_axes(fig, W, H, 14.0, y3, 105.0, HC)
    rd = np.array([decline(pid) for pid in rng])
    axc.fill_between(t, rd.min(axis=0), rd.max(axis=0), color="#e3e3e3", lw=0, zorder=0)
    for d_ in (1.0, 2.0):
        axc.axhline(d_, color=R.GRID, lw=0.6, zorder=1)
    axc.axhline(calib, color=R.MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
    for pid in ("SAM23-07", "SAM23-10", "SAM23-14", *hos):
        col, ls = PATH_STYLE[pid]
        axc.plot(t, decline(pid), color=col, ls=ls, lw=1.2, zorder=3)
    axc.axhline(0, color=R.INK, lw=0.5, zorder=2)
    axc.set_xlim(2026, 2070)
    axc.set_ylim(np.ceil(max(float(decline(p).max()) for p in prim)) + 0.5, -0.5)   # primary pathways; the range band is clipped
    axc.set_xlabel("Year")
    axc.set_ylabel("Fall below the 2026 level (m)")
    axc.grid(axis="x", color=R.GRID, lw=0.4)
    R.fig_label(fig, W, H, "c", 1.0, y3 - 2.0)
    ph = [Line2D([], [], color=PATH_STYLE[pid][0], ls=PATH_STYLE[pid][1], lw=1.2, label=PATH_LABEL[pid])
          for pid in ("SAM23-07", "SAM23-10", "SAM23-14", *hos)]
    ph += [Patch(fc="#e3e3e3", ec="none", label="Range pathways"),
           Line2D([], [], color=R.MUTED, lw=0.6, ls=(0, (2, 2)), label="Calibrated limit of b")]
    fig.legend(handles=ph, loc="upper left", bbox_to_anchor=((14.0 + 105.0 + 6.0) / W, 1 - y3 / H),
               fontsize=R.FS_SMALL, frameon=False, borderaxespad=0, handlelength=2.4, labelspacing=0.35)
    R.save(fig, "rse_fig_nearterm", W, H)
    plt.close(fig)
    print("any", round(any_km2), "all", round(all_km2), "| calibrated limit", calib)
    return 0


if __name__ == "__main__":
    sys.exit(main())
