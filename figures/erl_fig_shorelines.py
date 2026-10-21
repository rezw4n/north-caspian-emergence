"""ERL figure "Where the maps put the shoreline" (D-100, D-102; 15 cm wide).

(a) The North Caspian at the hindcast unit E3, level bin nearest -28.0 m (DAHITI
    reference): the observed state of every cell (sea-connected water, land, or not
    determined) on a 240 m display grid (share of water among the known 60 m cells), the
    observed shoreline (solid black) and the shorelines that GEBCO_2026, ETOPO 2022 and
    SRTM15+ predict for the same level (dashed, dash-dot and dotted): each product's flood
    fill from the open sea at the level plus its D-070 constant (the shape variant).
(b-d) Zooms at 60 m on the Volga delta front, the NE Kazakh coast and Komsomolets Bay, same
    unit; (e-g) the same zooms at the E3 unit nearest -27.0 m, above the -27.7 m edge where
    ETOPO 2022's emerged band switches from GEBCO lake depths to FABDEM (D-087). Arrows: per
    product, from the observed waterline cell of the zoom's sector whose distance to the
    product's nearest predicted waterline cell is closest to that sector's median
    displacement at the unit, to that cell; labels give the arrow's length.

Data: data/interim/figures/erl/hero_main.npz and results/display/erl_hero.json, written by
scripts/f1_erl_display.py caspian with the hindcast's own functions (every per-sector
median displacement of the drawn units is recomputed there and equals
results/hindcast/main/displacement.csv). Numbers: results/metrics.json (erl.hero.*).

Output: figures/out/erl_fig_shorelines.{pdf,tif,png}
Usage: python figures/erl_fig_shorelines.py
"""

from __future__ import annotations

import json
import sys

import _common as C
import _erl as E
import numpy as np

UNITS = ("E3_b019", "E3_b029")
PRODUCTS = ("gebco_2026", "etopo_2022", "srtm15p_v2.7")
INSETS = (("volga", "Volga delta front"), ("northeast", "NE Kazakh coast"), ("komsomolets", "Komsomolets Bay"))
SEA, LAND, UNKNOWN = "#cfe3f1", "#e9e6df", "#ffffff"
OUTSIDE = "#f4f4f4"
LW = 1.1


def coords(ext, shape):
    ny, nx = shape
    xs = ext[0] + (np.arange(nx) + 0.5) * (ext[1] - ext[0]) / nx
    ys = ext[3] - (np.arange(ny) + 0.5) * (ext[3] - ext[2]) / ny
    return xs, ys


def classes(field):
    """0 not determined, 1 land, 2 sea, from a water share (NaN unknown) or a state (-1/0/1)."""
    f = np.asarray(field, float)
    out = np.zeros(f.shape, np.int8)
    out[np.isfinite(f) & (f >= 0) & (f < 0.5)] = 1
    out[np.isfinite(f) & (f >= 0.5)] = 2
    return out


def shoreline(ax, field, ext, colour, dash, lw=LW, zorder=10):
    f = np.asarray(field, float).copy()
    f[f < 0] = np.nan
    xs, ys = coords(ext, f.shape)
    cs = ax.contour(xs, ys, f, [0.5], colors=[colour], linewidths=lw, zorder=zorder)
    cs.set_linestyle(dash if dash != "solid" else "solid")
    return cs


def crop(arr, ext, box):
    xs, ys = coords(ext, arr.shape)
    ci = np.flatnonzero((xs >= box[0]) & (xs < box[2]))
    ri = np.flatnonzero((ys >= box[1]) & (ys < box[3]))
    sub = arr[ri[0]:ri[-1] + 1, ci[0]:ci[-1] + 1]
    a = (ext[1] - ext[0]) / arr.shape[1]
    return sub, [xs[ci[0]] - a / 2, xs[ci[-1]] + a / 2, ys[ri[-1]] - a / 2, ys[ri[0]] + a / 2]


def main() -> int:
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    m = C.metrics()
    h = json.loads((C.ROOT / "results" / "display" / "erl_hero.json").read_text(encoding="utf-8"))
    z = np.load(C.ROOT / "data" / "interim" / "figures" / "erl" / "hero_main.npz")
    _, fwd = C.projection()
    domain, _ = C.study_geometries()
    ext_main = z["extent"]
    cmap = ListedColormap([UNKNOWN, LAND, SEA])
    halo = [patheffects.withStroke(linewidth=2.2, foreground="white")]

    def ukey(u):
        return f"erl.hero.{u.lower()}"

    E.style()
    W = E.DOUBLE_MM
    L, R, TOP = 9.0, 1.5, 2.0
    mx0, mx1, my0, my1 = -292_000.0, 372_000.0, -172_000.0, 212_000.0
    map_w = W - L - R
    map_h = map_w * (my1 - my0) / (mx1 - mx0)
    G_LEG, H_LEG = 6.5, 14.0
    H_HEAD = 6.0
    gap = 3.0
    iw = (map_w - 2 * gap) / 3
    ih = iw * 70.0 / 80.0
    BOT = 2.0
    H = TOP + map_h + G_LEG + H_LEG + 2 * (H_HEAD + ih) + 1.5 + BOT
    fig = plt.figure(figsize=(W * C.MM, H * C.MM))

    # (a) main map --------------------------------------------------------------------------------
    u0 = UNITS[0]
    ax = E.mm_axes(fig, W, H, L, TOP, map_w, map_h)
    ax.set_facecolor(OUTSIDE)
    obs = z[f"obs_{u0}"]
    ax.imshow(classes(obs), extent=ext_main, origin="upper", cmap=cmap, vmin=0, vmax=2, interpolation="nearest",
              zorder=1, rasterized=True)
    E.laea_graticule(ax, fwd, (mx0, mx1, my0, my1), lons=np.arange(47, 55, E.GRATICULE_MAP_DEG),
                     lats=np.arange(44, 48, E.GRATICULE_MAP_DEG))
    xs, ys = domain.exterior.xy
    ax.plot(xs, ys, color=E.MUTED, lw=0.6, zorder=4)
    for p in PRODUCTS:
        shoreline(ax, z[f"pred_{u0}_{p}"], ext_main, E.COLOURS[p], E.DASHES[p], lw=0.9, zorder=8)
    shoreline(ax, obs, ext_main, E.OBS, "solid", lw=0.9, zorder=9)
    E.map_axes_clean(ax)
    ax.set_xlim(mx0, mx1)
    ax.set_ylim(my0, my1)
    for j, (k, name) in enumerate(INSETS):
        x0, y0, x1, y1 = h["insets"][k]["box_m"]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none", ec=E.INK, lw=1.0, zorder=12))
        if j == 0:      # above the box; the eastern two boxes are stacked, so their labels go to the right
            ax.text(x0 + (x1 - x0) / 2, y1 + 2500, f"({'bcd'[j]}, {'efg'[j]})", ha="center", va="bottom",
                    fontsize=E.FS, fontweight="bold", color=E.INK, zorder=13, path_effects=halo)
        else:
            ax.text(x1 + 3000, y0 + (y1 - y0) / 2, f"({'bcd'[j]}, {'efg'[j]})", ha="left", va="center",
                    fontsize=E.FS, fontweight="bold", color=E.INK, zorder=13, path_effects=halo)
    E.scale_bar(ax, mx0 + 20_000, my0 + 18_000, E.SCALE_MAP_KM * 1000.0, E.num(m, "params.erl_fig_scale_map_km", "km"))
    E.label(ax, "a", inside=True)
    lv = E.num(m, f"{ukey(u0)}.level_m")
    ax.text(0.985, 0.975, f"Sea level {lv} m\nobserved {m[ukey(u0) + '.first_year']['value']}–"
            f"{m[ukey(u0) + '.last_year']['value']}", transform=ax.transAxes, ha="right", va="top", fontsize=E.FS,
            color=E.INK, zorder=20, linespacing=1.15,
            bbox={"boxstyle": "square,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.9})

    # legend ------------------------------------------------------------------------------------------
    lines = [Line2D([], [], color=E.OBS, lw=LW, label="Observed")]
    lines += [Line2D([], [], color=E.COLOURS[p], lw=LW, ls=E.DASHES[p], label=f"{E.NAMES[p]}") for p in PRODUCTS]
    states = [Patch(fc=SEA, ec=E.MUTED, lw=0.4, label="Sea"), Patch(fc=LAND, ec=E.MUTED, lw=0.4, label="Land"),
              Patch(fc=UNKNOWN, ec=E.MUTED, lw=0.4, label="Not determined"),
              Patch(fc=OUTSIDE, ec=E.MUTED, lw=0.4, label="Outside the study domain")]
    y_leg = 1 - (TOP + map_h + G_LEG - 1.5) / H
    kw = {"fontsize": E.FS, "frameon": False, "borderaxespad": 0, "handletextpad": 0.5, "labelspacing": 0.35,
          "columnspacing": 1.0, "alignment": "left"}
    fig.legend(handles=lines, loc="upper left", bbox_to_anchor=(L / W, y_leg), ncol=2, handlelength=3.0,
               title="Shoreline at the level (lines)", **kw)
    fig.legend(handles=states, loc="upper left", bbox_to_anchor=((L + 78.0) / W, y_leg), ncol=2, handlelength=1.6,
               title="Observed state at the level (fill)", **kw)

    # (b-g) insets ----------------------------------------------------------------------------------
    y_row = TOP + map_h + G_LEG + H_LEG
    for r, u in enumerate(UNITS):
        lv = E.num(m, f"{ukey(u)}.level_m")
        yrs = f"{m[ukey(u) + '.first_year']['value']}–{m[ukey(u) + '.last_year']['value']}"
        note = ("below the −27.7 m edge: ETOPO 2022 from GEBCO lake depths" if r == 0 else
                "above the −27.7 m edge: ETOPO 2022 from FABDEM land heights")
        fig.text(L / W, 1 - (y_row + H_HEAD - 1.2) / H, f"Sea level {lv} m (observed {yrs}); {note}",
                 ha="left", va="bottom", fontsize=E.FS, color=E.INK)
        yt = y_row + H_HEAD
        for j, (k, name) in enumerate(INSETS):
            axi = E.mm_axes(fig, W, H, L + j * (iw + gap), yt, iw, ih)
            box = h["insets"][k]["box_m"]
            ext = z[f"ext_{k}"]
            sub, sext = crop(z[f"obs_{u}_{k}"], ext, box)
            axi.imshow(classes(sub), extent=sext, origin="upper", cmap=cmap, vmin=0, vmax=2, interpolation="nearest",
                       zorder=1, rasterized=True)
            E.laea_graticule(axi, fwd, (box[0], box[2], box[1], box[3]),
                             lons=np.arange(46.0, 55.0, E.GRATICULE_ZOOM_DEG),
                             lats=np.arange(44.0, 48.0, E.GRATICULE_ZOOM_DEG), inside=True)
            for p in PRODUCTS:
                ps, pext = crop(z[f"pred_{u}_{p}_{k}"], ext, box)
                shoreline(axi, ps, pext, E.COLOURS[p], E.DASHES[p], zorder=8)
            shoreline(axi, sub, sext, E.OBS, "solid", zorder=9)
            arrows = [a for a in h["arrows"] if a["unit"] == u and a["inset"] == k and "x0" in a]
            for i, a in enumerate(arrows):
                p = a["product"]
                axi.annotate("", xy=(a["x1"], a["y1"]), xytext=(a["x0"], a["y0"]), zorder=15,
                             arrowprops={"arrowstyle": "-|>,head_length=0.35,head_width=0.18", "color": E.COLOURS[p],
                                         "lw": 1.3, "shrinkA": 0, "shrinkB": 0,
                                         "path_effects": [patheffects.withStroke(linewidth=2.6, foreground="white")]})
            E.map_axes_clean(axi)
            axi.set_xlim(box[0], box[2])
            axi.set_ylim(box[1], box[3])
            E.scale_bar(axi, box[2] - 4_000 - E.SCALE_INSET_KM * 1000.0, box[3] - 10_000, E.SCALE_INSET_KM * 1000.0,
                        E.num(m, "params.erl_fig_scale_inset_km", "km"), box=True)
            letter = "bcdefg"[r * 3 + j]
            axi.text(0.012, 0.985, f"({letter}) {name}", transform=axi.transAxes, ha="left", va="top", fontsize=E.FS,
                     color=E.INK, zorder=30, bbox={"boxstyle": "square,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.9})
            # arrow lengths, one line per product, ink text beside a line sample
            lines = [(a["product"], E.num(m, f"{ukey(u)}.arrow.{k}.{E.PK[a['product']]}_km", "km")) for a in arrows]
            for i, (p, txt) in enumerate(lines):
                yy = 0.80 - i * 0.105
                axi.plot([0.03, 0.12], [yy, yy], transform=axi.transAxes, color=E.COLOURS[p], ls=E.DASHES[p], lw=LW,
                         zorder=31, solid_capstyle="butt")
                axi.text(0.14, yy, txt, transform=axi.transAxes, ha="left", va="center", fontsize=E.FS, color=E.INK,
                         zorder=31)
            axi.add_patch(Rectangle((0.015, 0.80 - (len(lines) - 1) * 0.105 - 0.06), 0.33, (len(lines) - 1) * 0.105 + 0.12,
                                    transform=axi.transAxes, fc="white", ec="none", alpha=0.9, zorder=30))
        y_row = yt + ih + 1.5
    C.save(fig, "erl_fig_shorelines")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
