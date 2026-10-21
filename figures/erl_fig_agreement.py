"""ERL figure "Where the products agree on the next metres" (D-100, D-102; 15 cm wide).

For every 60 m cell of the study domain, the number of the five audited products,
uncorrected (as users apply them: total-variant heights on the DAHITI working datum, the
H3 "uncorrected" surface of scripts/a8_emergence.py), that place it in the seabed band
that emerges between the 2026 level (h3.level_2026_m) and 1 m (a) or 2 m (b) below it:
L26 - D <= F < L26 on the product's flood-level map (flood fill from the open sea, D-070).
Each product's band area equals its emergence.csv domain estimate cell for cell
(scripts/f1_erl_display.py caspian). Drawn on a 240 m display grid (the most frequent
count among the 16 cells of each block); cells where all five agree are also hatched. No
error model is applied (D-100). Background: the open sea of the 2025-26 low stand (D-029,
results/study_area/sea_mask_1km.npz) and land, for orientation.

Exposure: OpenStreetMap ports, piers and ferry terminals; oil and gas platforms and wells;
settlements (data/raw/osm/, the layers of scripts/a8_emergence.py), drawn where they lie
within 10 km of a cell that any product places in the 2 m band (a display filter).

Numbers: results/metrics.json (erl.agree.*, h3.level_2026_m, h3.emergence.*).
Output: figures/out/erl_fig_agreement.{pdf,tif,png}
Usage: python figures/erl_fig_agreement.py
"""

from __future__ import annotations

import sys

import _common as C
import _erl as E
import numpy as np

from caspian.config import RAW

EXTENT = (-272_000.0, 312_000.0, -142_000.0, 194_000.0)
NEAR_M = E.NEAR_KM * 1000.0
LAYERS = (("ports", "o", 3.2, "Port, pier or ferry terminal"), ("platforms", "^", 4.0, "Oil or gas platform or well"),
          ("settlements", "s", 3.2, "Settlement"))
SEA, LAND = "#d9e8f3", "#ecebe7"


def main() -> int:
    import cmcrameri.cm as cmc
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    from scipy.spatial import cKDTree

    m = C.metrics()
    z = np.load(C.ROOT / "data" / "interim" / "figures" / "erl" / "agreement_main.npz")
    ext = z["extent"]
    _, fwd = C.projection()
    domain, _ = C.study_geometries()
    sea, sea_ext = C.sea_mask()
    colours = [cmc.lajolla(v) for v in (0.92, 0.74, 0.56, 0.38, 0.2)]     # 1 light ... 5 dark (Crameri lajolla)
    cmap = ListedColormap(colours)
    norm = BoundaryNorm(np.arange(0.5, 6.0, 1.0), cmap.N)

    c2 = z["count_2m"]
    ny, nx = c2.shape
    xs = ext[0] + (np.arange(nx) + 0.5) * (ext[1] - ext[0]) / nx
    ys = ext[3] - (np.arange(ny) + 0.5) * (ext[3] - ext[2]) / ny
    r, c = np.nonzero((c2 >= 1) & (c2 < 255))
    tree = cKDTree(np.c_[xs[c], ys[r]])
    feats = {}
    for name, *_ in LAYERS:
        g = gpd.read_file(RAW / "osm" / f"osm_{name}.gpkg")
        pts = g.geometry.representative_point()
        xy = np.array([fwd(p.x, p.y) for p in pts])
        d, _ = tree.query(xy)
        feats[name] = xy[d <= NEAR_M]

    E.style()
    W = E.DOUBLE_MM
    L, R, TOP = 9.0, 1.5, 2.0
    mw = W - L - R
    mh = mw * (EXTENT[3] - EXTENT[2]) / (EXTENT[1] - EXTENT[0])
    GAP, H_LEG, BOT = 9.0, 17.0, 1.0
    H = TOP + 2 * mh + GAP + 14.0 + H_LEG + BOT
    fig = plt.figure(figsize=(W * C.MM, H * C.MM))
    for i, (D, key) in enumerate(((1, "1m"), (2, "2m"))):
        ax = E.mm_axes(fig, W, H, L, TOP + i * (mh + GAP), mw, mh)
        ax.set_facecolor(LAND)
        ax.imshow(np.where(sea, 1.0, np.nan), extent=sea_ext, origin="upper", cmap=ListedColormap([SEA]),
                  interpolation="nearest", zorder=1, rasterized=True)
        cnt = z[f"count_{key}"].astype(float)
        cnt[(cnt < 1) | (cnt > 5)] = np.nan
        ax.imshow(np.ma.masked_invalid(cnt), extent=ext, origin="upper", cmap=cmap, norm=norm, interpolation="nearest",
                  zorder=2, rasterized=True)
        allf = np.where(z[f"count_{key}"] == 5, 1.0, 0.0)
        with plt.rc_context({"hatch.color": "white", "hatch.linewidth": 0.6}):
            ax.contourf(xs, ys, allf, levels=[0.5, 1.5], colors="none", hatches=["//////"], zorder=3)
        ax.contour(xs, ys, allf, levels=[0.5], colors=[E.INK], linewidths=0.35, zorder=3)
        E.laea_graticule(ax, fwd, EXTENT, lons=np.arange(47, 55, E.GRATICULE_MAP_DEG),
                         lats=np.arange(44, 48, E.GRATICULE_MAP_DEG), label_lon=(i == 1))
        dx, dy = domain.exterior.xy
        ax.plot(dx, dy, color=E.MUTED, lw=0.6, zorder=4)
        for name, mk, ms, _ in LAYERS:
            xy = feats[name]
            ax.plot(xy[:, 0], xy[:, 1], ls="none", marker=mk, ms=ms, mfc="white", mec=E.INK, mew=0.6, zorder=8)
        E.map_axes_clean(ax)
        ax.set_xlim(EXTENT[0], EXTENT[1])
        ax.set_ylim(EXTENT[2], EXTENT[3])
        E.scale_bar(ax, EXTENT[0] + 18_000, EXTENT[2] + 16_000, E.SCALE_MAP_KM * 1000.0,
                    E.num(m, "params.erl_fig_scale_map_km", "km"))
        E.label(ax, "ab"[i], inside=True)
        all_a = E.num(m, f"erl.agree.{key}.area_km2_all", "km²")
        any_a = E.num(m, f"erl.agree.{key}.area_km2_any", "km²")
        ax.text(0.985, 0.975, f"{D} m below the 2026 level\nany product: {any_a}\nall five: {all_a}",
                transform=ax.transAxes, ha="right", va="top", fontsize=E.FS, color=E.INK, linespacing=1.2, zorder=20,
                bbox={"boxstyle": "square,pad=0.3", "fc": "white", "ec": "none", "alpha": 0.92})
    # legend
    y_leg = 1 - (TOP + 2 * mh + GAP + 13.5) / H
    cax = fig.add_axes([L / W, y_leg - 4.0 / H, 62.0 / W, 3.2 / H])
    for k in range(5):
        cax.add_patch(Rectangle((k, 0), 1, 1, fc=colours[k], ec="white", lw=0.8))
        cax.text(k + 0.5, 0.5, str(k + 1), ha="center", va="center", fontsize=E.FS,
                 color="white" if k >= 2 else E.INK, zorder=5,
                 bbox={"boxstyle": "square,pad=0.1", "fc": colours[k], "ec": "none"} if k == 4 else None)
    with plt.rc_context({"hatch.color": "white", "hatch.linewidth": 0.6}):
        cax.add_patch(Rectangle((4, 0), 1, 1, fc="none", ec=colours[4], lw=0.0, hatch="//////"))
    cax.set_xlim(0, 5)
    cax.set_ylim(0, 1)
    cax.axis("off")
    fig.text(L / W, y_leg + 0.6 / H, chr(10).join(["Products that place the cell in the band",
                                                    "that emerges (of five; hatched: all five)"]),
             ha="left", va="bottom", fontsize=E.FS, color=E.INK, linespacing=1.15)
    handles = [Line2D([], [], ls="none", marker=mk, ms=ms + 1, mfc="white", mec=E.INK, mew=0.6, label=lab)
               for _, mk, ms, lab in LAYERS]
    handles += [Patch(fc=SEA, ec=E.MUTED, lw=0.4, label="Open sea, 2025–26"),
                Patch(fc=LAND, ec=E.MUTED, lw=0.4, label="Land or seabed outside the band")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=((L + 76.0) / W, y_leg + 8.0 / H), ncol=1,
               fontsize=E.FS, frameon=False, borderaxespad=0, handletextpad=0.5, labelspacing=0.3,
               title=f"Within {E.num(m, 'params.erl_fig_near_km', 'km')} of the 2 m band (any product)",
               alignment="left")
    C.save(fig, "erl_fig_agreement")
    plt.close(fig)
    print({k: len(v) for k, v in feats.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
