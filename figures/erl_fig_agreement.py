"""ERL figure "Where the first metre emerges" (D-100, D-102; revised D-141 after workstream
N, D-123; 15 cm wide).

For every 60 m cell of the study domain, the number of the five audited grids that place it
in the seabed emerging in the first metre of fall below the 2026 level (h3.level_2026_m),
on the two surfaces of workstream N (scripts/f4_erl_nearterm_map.py, which checks every
grid's band against N's curves):
(a) the composite observed baseline (D-123 addendum 2): the observed 2026 sea that grid p
    puts above the water 1 m lower;
(b) the grids shifted by their H1 shape constant (D-070): L26 + c - 1 <= F < L26 + c.
Drawn on a 240 m display grid (the most frequent count among the 16 cells of each block).
Cells that a majority of the grids (3 of 5 or more) place in the band are hatched; the
earlier "all five" hatch was barely visible (internal review R1 minor 23). The thick black
line is the outline of the composite baseline, the observed 2025-26 sea (zone-v6 extent
where the lowest audited level was unobserved), at 0.5 of the baseline share per 240 m block;
it is not drawn within RIM_KM of the domain edge, where the D-083 rim takes the beyond-domain
state and would trace the edge, and pieces shorter than MIN_PIECE_KM are dropped. Background: the D-029 open sea of the
2025-26 low stand and land, for orientation.
(c) Per grid, the share of its 1 m band that is already dry in the observed 2025-26 extent
    (zone v6), for the grids as users apply them (uncorrected; filled) and after the shift
    (open): results/revision/n/already_dry.csv through metrics rev.n.already_dry.*.

The OpenStreetMap symbols of the earlier version are dropped: almost all "ports" are river
piers (workstream N, D-126), and the symbols drawn near any grid's 2 m band overstated the
exposure (R2 minor 14).

Numbers: results/metrics.json (rev.fig.f5.*, rev.n.already_dry.*, h3.level_2026_m).
Output: figures/out/erl_fig_agreement.{pdf,tif,png}
Usage: python figures/erl_fig_agreement.py
"""

from __future__ import annotations

import sys

import _common as C
import _erl as E
import numpy as np

EXTENT = (-272_000.0, 312_000.0, -142_000.0, 194_000.0)
SEA, LAND = "#d9e8f3", "#ecebe7"
PRODS = ("gebco_2026", "gebco_2022", "etopo_2022", "etopo1", "srtm15p_v2.7")
SURF = (("composite", "Observed 2026 sea as the baseline"), ("shifted", "Grids after the vertical shift"))
RIM_KM = 3.0          # outline not drawn this close to the domain edge (the D-083 rim is 2 km)
MIN_PIECE_KM = 5.0    # outline pieces shorter than this are not drawn


def shore_lines(field, ext, level: float = 0.5, min_len_m: float = MIN_PIECE_KM * 1000.0):
    """Contour pieces of ``field`` at ``level`` (skimage.measure.find_contours, NaN masked) in
    map metres, without pieces shorter than ``min_len_m`` (as erl_fig_then_now.iso_lines)."""
    from skimage import measure

    px = (ext[1] - ext[0]) / field.shape[1]
    ok = np.isfinite(field)
    out = []
    for c in measure.find_contours(np.where(ok, field, level - 50.0), level, mask=ok):
        xy = np.c_[ext[0] + (c[:, 1] + 0.5) * px, ext[3] - (c[:, 0] + 0.5) * px]
        if np.sum(np.hypot(*np.diff(xy, axis=0).T)) >= min_len_m:
            out.append(xy)
    return out


def main() -> int:
    import cmcrameri.cm as cmc
    import matplotlib.pyplot as plt
    from scipy import ndimage
    from matplotlib import patheffects
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    m = C.metrics()
    z = np.load(C.ROOT / "data" / "interim" / "figures" / "erl" / "nearterm_map.npz")
    ext = z["extent"]
    _, fwd = C.projection()
    domain, _ = C.study_geometries()
    sea, sea_ext = C.sea_mask()
    colours = [cmc.lajolla(v) for v in (0.92, 0.74, 0.56, 0.38, 0.2)]     # 1 light ... 5 dark (Crameri lajolla)
    cmap = ListedColormap(colours)
    norm = BoundaryNorm(np.arange(0.5, 6.0, 1.0), cmap.N)
    ny, nx = z["baseline"].shape
    xs = ext[0] + (np.arange(nx) + 0.5) * (ext[1] - ext[0]) / nx
    ys = ext[3] - (np.arange(ny) + 0.5) * (ext[3] - ext[2]) / ny
    base = z["baseline"].astype(np.float32)
    # the outline is not drawn within RIM_KM of the domain edge, where the D-083 rim takes the
    # beyond-domain state and would trace the edge; pieces shorter than MIN_PIECE_KM are dropped
    core = ndimage.binary_erosion(np.isfinite(base), iterations=int(np.ceil(RIM_KM * 1000.0 / ((ext[1] - ext[0]) / nx))))
    shore = shore_lines(np.where(core, base, np.nan), ext)

    E.style()
    W = E.DOUBLE_MM
    L, R, TOP = 9.0, 1.5, 2.0
    mw = W - L - R
    mh = mw * (EXTENT[3] - EXTENT[2]) / (EXTENT[1] - EXTENT[0])
    GAP, H_LEG, BOT = 9.0, 34.0, 2.0
    H = TOP + 2 * mh + GAP + 12.0 + H_LEG + BOT
    fig = plt.figure(figsize=(W * C.MM, H * C.MM))
    halo = [patheffects.withStroke(linewidth=2.8, foreground="white")]
    for i, (surf, title) in enumerate(SURF):
        ax = E.mm_axes(fig, W, H, L, TOP + i * (mh + GAP), mw, mh)
        ax.set_facecolor(LAND)
        ax.imshow(np.where(sea, 1.0, np.nan), extent=sea_ext, origin="upper", cmap=ListedColormap([SEA]),
                  interpolation="nearest", zorder=1, rasterized=True)
        cnt = z[f"count_{surf}"].astype(float)
        cnt[(cnt < 1) | (cnt > 5)] = np.nan
        ax.imshow(np.ma.masked_invalid(cnt), extent=ext, origin="upper", cmap=cmap, norm=norm, interpolation="nearest",
                  zorder=2, rasterized=True)
        maj = np.where((z[f"count_{surf}"] >= 3) & (z[f"count_{surf}"] <= 5), 1.0, 0.0)
        with plt.rc_context({"hatch.color": "white", "hatch.linewidth": 0.6}):
            ax.contourf(xs, ys, maj, levels=[0.5, 1.5], colors="none", hatches=["//////"], zorder=3)
        ax.contour(xs, ys, maj, levels=[0.5], colors=[E.INK], linewidths=0.35, zorder=3)
        for seg in shore:
            ax.plot(seg[:, 0], seg[:, 1], color=E.INK, lw=1.1, zorder=6, solid_capstyle="round", path_effects=halo)
        E.laea_graticule(ax, fwd, EXTENT, lons=np.arange(47, 55, E.GRATICULE_MAP_DEG),
                         lats=np.arange(44, 48, E.GRATICULE_MAP_DEG), label_lon=(i == 1))
        dx, dy = domain.exterior.xy
        ax.plot(dx, dy, color=E.MUTED, lw=0.6, zorder=4)
        E.map_axes_clean(ax)
        ax.set_xlim(EXTENT[0], EXTENT[1])
        ax.set_ylim(EXTENT[2], EXTENT[3])
        E.scale_bar(ax, EXTENT[0] + 18_000, EXTENT[2] + 16_000, E.SCALE_MAP_KM * 1000.0,
                    E.num(m, "params.erl_fig_scale_map_km", "km"))
        E.label(ax, "ab"[i], inside=True)
        any_a = E.num(m, f"rev.fig.f5.{surf}.area_km2_any", "km²")
        maj_a = E.num(m, f"rev.fig.f5.{surf}.area_km2_majority", "km²")
        ax.text(0.985, 0.975, f"{title}\nfirst metre of fall\nany grid: {any_a}\nthree or more: {maj_a}",
                transform=ax.transAxes, ha="right", va="top", fontsize=E.FS, color=E.INK, linespacing=1.2, zorder=20,
                bbox={"boxstyle": "square,pad=0.3", "fc": "white", "ec": "none", "alpha": 0.92})

    # key -------------------------------------------------------------------------------------------
    y0 = TOP + 2 * mh + GAP + 12.0
    fig.text(L / W, 1 - y0 / H, chr(10).join(["Grids that place the cell in the first metre",
                                               "(of five; hatched: three or more)"]),
             ha="left", va="top", fontsize=E.FS, color=E.INK, linespacing=1.15)
    cax = fig.add_axes([L / W, 1 - (y0 + 12.5) / H, 62.0 / W, 3.2 / H])
    for k in range(5):
        cax.add_patch(Rectangle((k, 0), 1, 1, fc=colours[k], ec="white", lw=0.8))
    with plt.rc_context({"hatch.color": "white", "hatch.linewidth": 0.6}):
        for k in (2, 3, 4):
            cax.add_patch(Rectangle((k, 0), 1, 1, fc="none", ec="white", lw=0.0, hatch="//////"))
    for k in range(5):
        cax.text(k + 0.5, 0.5, str(k + 1), ha="center", va="center", fontsize=E.FS, zorder=5,
                 color="white" if k >= 2 else E.INK,
                 bbox={"boxstyle": "square,pad=0.1", "fc": colours[k], "ec": "none"} if k >= 2 else None)
    cax.set_xlim(0, 5)
    cax.set_ylim(0, 1)
    cax.axis("off")
    handles = [Line2D([], [], color=E.INK, lw=1.1, path_effects=halo, label="Observed 2025–26 shoreline"),
               Patch(fc=SEA, ec=E.MUTED, lw=0.4, label="Open sea, 2025–26"),
               Patch(fc=LAND, ec=E.MUTED, lw=0.4, label="Land or seabed outside the band")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(L / W, 1 - (y0 + 17.5) / H), ncol=1,
               fontsize=E.FS, frameon=False, borderaxespad=0, handletextpad=0.5, labelspacing=0.3, handlelength=2.0)

    # (c) share of each band already dry ------------------------------------------------------------
    cx0, cw = L + 92.0, W - L - 92.0 - R - 2.0
    axc = E.mm_axes(fig, W, H, cx0, y0 + 2.0, cw, H_LEG - 9.0)
    for j, p in enumerate(PRODS):
        u = 100.0 * E.value(m, f"rev.n.already_dry.uncorrected.{E.PK[p]}.1.share_zv6")
        s = 100.0 * E.value(m, f"rev.n.already_dry.shifted.{E.PK[p]}.1.share_zv6")
        axc.barh(j + 0.18, u, height=0.34, color=E.COLOURS[p], ec=E.COLOURS[p], lw=0.6, zorder=3)
        axc.barh(j - 0.18, s, height=0.34, color="white", ec=E.COLOURS[p], lw=0.8, zorder=3)
    axc.set_yticks(range(len(PRODS)), [E.NAMES[p] for p in PRODS])
    axc.invert_yaxis()
    axc.tick_params(axis="y", length=0)
    axc.set_xlim(0, 100)
    axc.set_xticks([0, 25, 50, 75, 100])
    axc.set_xlabel("First-metre band already dry (%)")
    axc.grid(axis="x", color=E.GRID, lw=0.4)
    axc.set_axisbelow(True)
    E.label(axc, "c", x=-0.42, y=1.02)
    axc.legend(handles=[Patch(fc=E.INK, ec=E.INK, label="As users apply them"),
                        Patch(fc="white", ec=E.INK, label="After the shift")],
               loc="lower right", bbox_to_anchor=(1.0, 1.01), ncol=2, fontsize=E.FS, frameon=False, borderaxespad=0,
               handlelength=1.2, columnspacing=0.9)
    C.save(fig, "erl_fig_agreement")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
