"""RSE figure: where the five grids put the shoreline, and one cross-shore profile (double column).

The example level is one late-fall (E3) level bin chosen by a fixed rule in
scripts/f5_rse_display.py (the bin whose per-level displacements are jointly closest to
each grid's median over levels in the Kizlyar-Dagestan, interfluve and NE Kazakh coasts;
results/display/rse_shorelines.json).

a, The North Caspian at the example level: observed state of every cell on a 240 m display
   grid (share of sea among the known 60 m cells), the observed shoreline cells (black; the
   observed waterline cells of the audit), the sector outlines, the zoom boxes of c-e and the
   transect of b.
b, Cross-shore profile on the Volga-Ural interfluve coast along one ICESat-2 ATL08 beam
   (chosen by a fixed rule, scripts/f5_rse_display.py): the heights of the five grids as
   the audit reads them (bilinear on the 60 m grid, datum correction applied, minus each
   grid's area-fitted offset c), the waterline DEM (inside the emerged band), the ICESat-2
   ATL08 heights of the beam (orthometric height plus the training-block datum constant),
   the example level (dashed); under it, ticks where the observed shoreline (black) and each
   grid's predicted shoreline (coloured) cross the transect at that level.
c-e, Zooms (80 km x 70 km) on the Kizlyar-Dagestan coast, the Volga-Ural interfluve coast and
   the NE Kazakh coast, one panel per grid: observed state (fill), observed shoreline
   cells (black), and each predicted shoreline cell (60 m) coloured by its distance to the
   nearest observed shoreline cell; grey, predicted shoreline cells where the observed
   state is unknown (left out of D). D: the grid's median displacement in the sector at
   this level (both directions pooled; results/hindcast/main/displacement.csv); Q: the
   sector's noise floor at this level (results/hindcast/main/noise_floor.csv).

Data: data/interim/figures/rse/shorelines_main.npz and profile_main.npz, results/display/
rse_shorelines.json (scripts/f5_rse_display.py, which checks every per-sector median
displacement against displacement.csv).

Output: figures/out/rse_fig_shorelines.{pdf,tif,png}
Usage: python figures/rse_fig_shorelines.py
"""

from __future__ import annotations

import json
import sys

import _common as C
import _rse as R
import numpy as np
import pandas as pd

from caspian.config import INTERIM, RESULTS

CACHE = INTERIM / "figures" / "rse"
ZOOMS = (("west", "c", (-262_000.0, -170_000.0, -182_000.0, -100_000.0)),
         ("interfluve", "d", (2_000.0, 103_000.0, 82_000.0, 173_000.0)),
         ("northeast", "e", (178_000.0, 22_000.0, 258_000.0, 92_000.0)))
SEA_C, LAND_C, UNK_C, OUT_C = "#d4e5f2", "#e8e5de", "#ffffff", "#f4f4f4"
D_EDGES = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 1e9]       # km
D_LABELS = ["<0.5", "0.5–1", "1–2", "2–5", "5–10", ">10"]
TRANSECT = "#D55E00"


def coords(ext, shape):
    ny, nx = shape
    xs = ext[0] + (np.arange(nx) + 0.5) * (ext[1] - ext[0]) / nx
    ys = ext[3] - (np.arange(ny) + 0.5) * (ext[3] - ext[2]) / ny
    return xs, ys


def classes(field):
    f = np.asarray(field, float)
    out = np.zeros(f.shape, np.int8)
    out[np.isfinite(f) & (f >= 0) & (f < 0.5)] = 1
    out[np.isfinite(f) & (f >= 0.5)] = 2
    return out


def crop(arr, ext, box):
    xs, ys = coords(ext, arr.shape)
    ci = np.flatnonzero((xs >= box[0]) & (xs < box[2]))
    ri = np.flatnonzero((ys >= box[1]) & (ys < box[3]))
    sub = arr[ri[0]:ri[-1] + 1, ci[0]:ci[-1] + 1]
    a = (ext[1] - ext[0]) / arr.shape[1]
    return sub, [xs[ci[0]] - a / 2, xs[ci[-1]] + a / 2, ys[ri[-1]] - a / 2, ys[ri[0]] + a / 2]


def shoreline(ax, field, ext, colour, lw, zorder=10, ls="solid"):
    f = np.asarray(field, float).copy()
    f[f < 0] = np.nan
    xs, ys = coords(ext, f.shape)
    cs = ax.contour(xs, ys, f, [0.5], colors=[colour], linewidths=lw, zorder=zorder)
    cs.set_linestyle(ls)
    return cs


def crossings(s, state):
    """Positions along the transect where the state changes between sea (1) and land (0);
    unknown cells (-1) are skipped."""
    k = np.flatnonzero(state >= 0)
    st = state[k]
    ch = np.flatnonzero(np.diff(st) != 0)
    return [(s[k[i]] + s[k[i + 1]]) / 2 for i in ch]


def main() -> int:
    import cmocean
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    m = C.metrics()
    rec = json.loads((RESULTS / "display" / "rse_shorelines.json").read_text(encoding="utf-8"))
    if not all(c.get("equal", False) for c in rec["checks"]):
        raise SystemExit("rse_shorelines.json holds a failed check")
    z = np.load(CACHE / "shorelines_main.npz")
    pr = np.load(CACHE / "profile_main.npz")
    units = {(r["epoch"], r["bin"]): r for r in rec["units"]}
    picks = rec["rule_units"]
    if len({(v["epoch"], v["bin"]) for v in picks.values()}) != 1:
        raise SystemExit("the rule picked different levels per sector; the figure draws one level")
    ep, b = picks["west"]["epoch"], picks["west"]["bin"]
    u = f"{ep}_b{b:03d}"
    unit = units[(ep, b)]
    level = float(unit["level_m"])
    disp = pd.read_csv(RESULTS / "hindcast" / "main" / "displacement.csv")
    disp = disp[(disp["variant"] == "shape") & (disp["epoch"] == ep) & (disp["bin"] == b)]
    nf = pd.read_csv(RESULTS / "hindcast" / "main" / "noise_floor.csv")
    nf = nf[(nf["kind"] == "pooled_level") & (nf["bin"] == b)].set_index("sector")
    _, fwd = C.projection()
    domain, sectors = C.study_geometries()
    halo = [patheffects.withStroke(linewidth=1.8, foreground="white")]
    state_cmap = ListedColormap([UNK_C, LAND_C, SEA_C])
    dcols = cmocean.cm.matter(np.linspace(0.18, 0.95, len(D_LABELS)))
    dcmap = ListedColormap(dcols)
    dnorm = BoundaryNorm(D_EDGES, dcmap.N)

    R.style()
    W = R.DOUBLE_MM
    L, RGT, TOP = 7.0, 1.5, 2.0
    mx0, mx1, my0, my1 = -292_000.0, 372_000.0, -172_000.0, 212_000.0
    MAP_W = 112.0
    map_h = MAP_W * (my1 - my0) / (mx1 - mx0)
    G1, HEAD, GAPZ = 9.0, 5.5, 1.6
    zw = (W - L - RGT - 4 * GAPZ) / 5
    zh = zw * 70.0 / 80.0
    KEY, BOT = 12.0, 1.0
    H = TOP + map_h + G1 + 3 * (HEAD + zh) + 2 * 1.0 + KEY + BOT
    fig = plt.figure(figsize=(W * C.MM, H * C.MM))

    # a, overview ----------------------------------------------------------------------------------
    ax = R.mm_axes(fig, W, H, L, TOP, MAP_W, map_h)
    ax.set_facecolor(OUT_C)
    ax.imshow(classes(z[f"obs_{u}"]), extent=z["extent"], origin="upper", cmap=state_cmap, vmin=0, vmax=2,
              interpolation="nearest", zorder=1, rasterized=True)
    R.laea_graticule(ax, fwd, (mx0, mx1, my0, my1), lons=np.arange(47, 55, 1.0), lats=np.arange(44, 48, 1.0))
    for g in sectors.values():
        xs, ys = g.exterior.xy
        ax.plot(xs, ys, color=R.MUTED, lw=0.35, zorder=4)
    xs, ys = domain.exterior.xy
    ax.plot(xs, ys, color=R.INK, lw=0.6, zorder=4)
    ocx, ocy = z[f"obscells_{u}_gebco_2026_x"], z[f"obscells_{u}_gebco_2026_y"]
    ax.scatter(ocx, ocy, s=0.15, marker="s", c=R.OBS, linewidths=0, zorder=8, rasterized=True)
    for sid, letter, (x0, y0, x1, y1) in ZOOMS:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none", ec=R.INK, lw=0.9, zorder=12))
        ax.text(x0 + 2500, y1 - 2500, letter, ha="left", va="top", fontsize=R.FS, fontweight="bold", color=R.INK,
                zorder=13, path_effects=halo)
    ax.plot(pr["x"], pr["y"], color=TRANSECT, lw=1.6, zorder=14, solid_capstyle="butt")
    ax.annotate("b", (pr["x"][-1], pr["y"][-1]), xytext=(3, 0), textcoords="offset points", ha="left", va="center",
                fontsize=R.FS, fontweight="bold", color=TRANSECT, path_effects=halo, zorder=14)
    R.map_axes_clean(ax)
    ax.set_xlim(mx0, mx1)
    ax.set_ylim(my0, my1)
    R.scale_bar(ax, -60_000.0, my0 + 12_000, C.SCALE_KM * 1000.0, R.num(m, "params.fig_scale_bar_km", "km"))
    R.label(ax, "a", inside=True)
    lv_txt = R.fmt_num(level, 2)
    ax.text(0.985, 0.975, f"Level {lv_txt} m (DAHITI)\n{unit['n_acquisitions']} Landsat acquisitions, "
            f"{unit['first_year']}–{unit['last_year']}", transform=ax.transAxes, ha="right", va="top",
            fontsize=R.FS_SMALL, color=R.INK, zorder=20, linespacing=1.15,
            bbox={"boxstyle": "square,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.9})

    # b, profile, with a rug of shoreline crossings under it -------------------------------------------
    PX = L + MAP_W + 13.0
    PW = W - PX - RGT - 1.0
    PH = map_h - 36.0
    RUG = 15.0
    axp = R.mm_axes(fig, W, H, PX, TOP + 3.0, PW, PH)
    axr = R.mm_axes(fig, W, H, PX, TOP + 3.0 + PH + 1.2, PW, RUG, sharex=axp)
    s_km = (pr["s"] - pr["s"][0]) / 1000.0
    is2_km = (pr["is2_s"] - pr["s"][0]) / 1000.0
    for p in R.PRODUCTS:
        axp.plot(s_km, pr[f"z_{p}"], color=R.COLOURS[p], lw=0.9, ls=R.DASHES[p], zorder=3)
    dem = np.where(pr["band"], pr["dem"], np.nan)
    axp.plot(is2_km, pr["is2_h"], "o", ms=1.3, mfc="#8c8c8c", mec="none", zorder=2)
    axp.plot(s_km, dem, color=R.INK, lw=1.1, zorder=4)
    axp.axhline(level, color=R.INK, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ylo, yhi = -29.5, -25.5
    obs_x = crossings(s_km, pr[f"obs_{u}"])
    rows = [("Observed", R.OBS, obs_x)] + [(R.NAMES[p], R.COLOURS[p], crossings(s_km, pr[f"pred_{u}_{p}"]))
                                           for p in R.PRODUCTS]
    for k_, (name, col, xs_) in enumerate(rows):
        yy = len(rows) - 1 - k_
        axr.plot(xs_, [yy] * len(xs_), ls="none", marker="|", ms=4.5, mew=1.3, color=col, zorder=3)
    for xv in obs_x:
        axp.axvline(xv, color=R.INK, lw=0.4, ls=(0, (1, 1.5)), zorder=1)
        axr.axvline(xv, color=R.INK, lw=0.4, ls=(0, (1, 1.5)), zorder=1)
    axr.set_ylim(-0.7, len(rows) - 0.3)
    axr.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=R.FS_SMALL)
    axr.tick_params(axis="y", length=0, pad=1.0)
    for t, (_, col, _) in zip(axr.get_yticklabels(), rows[::-1]):
        t.set_color(col)
    axp.set_ylim(ylo, yhi)
    axp.set_xlim(0, s_km[-1])
    axp.tick_params(labelbottom=False)
    axr.set_xlabel("Distance along the transect, sea to land (km)")
    axp.set_ylabel("Height (m, DAHITI reference)")
    axp.text(0.01, (level - ylo) / (yhi - ylo) + 0.012, f"level {lv_txt} m", transform=axp.transAxes, ha="left",
             va="bottom", fontsize=R.FS_SMALL, color=R.INK)
    axp.grid(color=R.GRID, lw=0.4)
    axp.set_axisbelow(True)
    R.label(axp, "b", x=-0.22, y=1.0)
    gh = [Line2D([], [], color=R.INK, lw=1.1, label="Waterline DEM"),
          Line2D([], [], ls="none", marker="o", ms=2.0, mfc="#8c8c8c", mec="none", label="ICESat-2 ATL08")]
    gh += [Line2D([], [], color=R.COLOURS[p], lw=1.0, ls=R.DASHES[p], label=R.NAMES[p]) for p in R.PRODUCTS]
    fig.legend(handles=gh, loc="upper left", bbox_to_anchor=((PX - 9.0) / W, 1 - (TOP + 3.0 + PH + RUG + 8.5) / H),
               ncol=3, fontsize=R.FS_SMALL, frameon=False, borderaxespad=0, handlelength=2.2, columnspacing=0.8,
               labelspacing=0.25, handletextpad=0.4)

    # c-e, zooms --------------------------------------------------------------------------------------
    y = TOP + map_h + G1
    for r_, (sid, letter, box) in enumerate(ZOOMS):
        q = float(nf.loc[sid, "p95_m"]) / 1000.0
        head = (f"{R.NAME[sid]} (checked): noise floor Q = {q:.2f} km at this level")
        fig.text((L + 4.0) / W, 1 - (y + HEAD - 1.2) / H, head, ha="left", va="bottom", fontsize=R.FS, color=R.INK)
        R.fig_label(fig, W, H, letter, 1.0, y + 0.2)
        yt = y + HEAD
        ext = z[f"ext_{sid}"]
        obs_sub, oext = crop(z[f"obs_{u}_{sid}"], ext, box)
        for j, p in enumerate(R.PRODUCTS):
            axz = R.mm_axes(fig, W, H, L + j * (zw + GAPZ), yt, zw, zh)
            axz.imshow(classes(obs_sub), extent=oext, origin="upper", cmap=state_cmap, vmin=0, vmax=2,
                       interpolation="nearest", zorder=1, rasterized=True)
            R.laea_graticule(axz, fwd, (box[0], box[2], box[1], box[3]), lons=np.arange(46.0, 55.0, 0.5),
                             lats=np.arange(44.0, 48.0, 0.5), inside=True, label_lon=(j == 0), label_lat=(j == 0))
            ob = (ocx >= box[0]) & (ocx < box[2]) & (ocy >= box[1]) & (ocy < box[3])
            axz.scatter(ocx[ob], ocy[ob], s=0.45, marker="s", c=R.OBS, linewidths=0, zorder=8, rasterized=True)
            key = f"predcells_{u}_{p}"
            cx, cy, cd, kn = z[f"{key}_x"], z[f"{key}_y"], z[f"{key}_d"], z[f"{key}_known"]
            inb = (cx >= box[0]) & (cx < box[2]) & (cy >= box[1]) & (cy < box[3])
            unk = inb & ~kn
            axz.scatter(cx[unk], cy[unk], s=0.35, marker="s", c="#a8a8a8", linewidths=0, zorder=9, rasterized=True)
            ok = inb & kn
            order = np.argsort(cd[ok])
            axz.scatter(cx[ok][order], cy[ok][order], s=0.45, marker="s", c=cd[ok][order] / 1000.0, cmap=dcmap,
                        norm=dnorm, linewidths=0, zorder=10, rasterized=True)
            R.map_axes_clean(axz)
            axz.set_xlim(box[0], box[2])
            axz.set_ylim(box[1], box[3])
            dval = disp[(disp["product"] == p) & (disp["sector"] == sid)]["median_m"]
            dk = float(dval.iloc[0]) / 1000.0
            axz.text(0.03, 0.97, f"{R.NAMES[p]}\nD = {dk:.1f} km (D/Q {dk / q:.1f})", transform=axz.transAxes,
                     ha="left", va="top", fontsize=R.FS_SMALL, color=R.INK, zorder=30, linespacing=1.1,
                     bbox={"boxstyle": "square,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.88})
            if j == 4:
                R.scale_bar(axz, box[2] - 4_000 - 20_000.0, box[1] + 5_000, 20_000.0, "20 km", box=True)
            if sid == "interfluve":
                axz.plot(pr["x"], pr["y"], color=TRANSECT, lw=0.8, zorder=11)
        y = yt + zh + 1.0

    # keys ----------------------------------------------------------------------------------------------
    yk = y + 3.5
    cax = R.mm_axes(fig, W, H, L + 4.0, yk, 60.0, 2.0)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=BoundaryNorm(np.arange(len(D_LABELS) + 1), dcmap.N), cmap=dcmap),
                      cax=cax, orientation="horizontal")
    cb.set_ticks(np.arange(len(D_LABELS)) + 0.5, labels=D_LABELS)
    cb.ax.tick_params(labelsize=R.FS_SMALL, length=0, pad=1.5)
    cb.outline.set_linewidth(0.4)
    cb.set_label("Distance of a predicted shoreline cell to the observed shoreline (km)", fontsize=R.FS_SMALL,
                 labelpad=1.5)
    cb.ax.xaxis.set_label_position("top")
    sk = [Line2D([], [], color=R.OBS, lw=0.8, label="Observed shoreline"),
          Patch(fc="#a8a8a8", ec="none", label="Predicted, observed state unknown"),
          Patch(fc=SEA_C, ec=R.MUTED, lw=0.3, label="Observed sea"),
          Patch(fc=LAND_C, ec=R.MUTED, lw=0.3, label="Observed land"),
          Patch(fc=UNK_C, ec=R.MUTED, lw=0.3, label="Not determined"),
          Line2D([], [], color=TRANSECT, lw=1.2, label="Transect of b")]
    fig.legend(handles=sk, loc="upper left", bbox_to_anchor=((L + 72.0) / W, 1 - (yk - 3.2) / H), ncol=3,
               fontsize=R.FS_SMALL, frameon=False, borderaxespad=0, handlelength=1.4, columnspacing=1.0,
               labelspacing=0.3, handletextpad=0.4)
    R.save(fig, "rse_fig_shorelines", W, H)
    plt.close(fig)
    print("unit", u, "level", level, "| observed crossings (km):", [round(v, 2) for v in obs_x])
    return 0


if __name__ == "__main__":
    sys.exit(main())
