"""RSE figure: the North Caspian study area (double column, 190 mm).

One map: the study domain and its seven reporting sectors, each named on the map
(results/study_area/sectors.geojson); the three sectors whose observations are not
validated hatched (the validated-sector rule of PREREG Amendment 18 item 1: the waterline
DEM's mean and median errors within +-0.5 m of ICESat-2 ATL08 and ATL06 in the sector's
test blocks; results/validation/main/ATL0{6,8}_accuracy.csv, checked here); the 50 km test
blocks that hold the ICESat-2 points of the H2 population
(results/validation/main/ATL08_accuracy_blocks.csv); the seabed that emerged between the
1995-96 high stand and the 2025-26 low stand (water in the 1995-96 and not water in the
2025-26 zone-v6 extent, results/study_area/extents_v6_250m.npz); the open sea of the
2025-26 low stand (results/study_area/sea_mask_1km.npz); two cities
(data/raw/osm/osm_settlements.gpkg); international boundaries (Natural Earth 1:50 m,
cartopy); graticule and scale bar.

Journal revision (PREREG Amendment 20 item 1e; D-161):
- In the two deltaic sectors the band is drawn as emerged seabed only inside the Caspian's
  February 2000 outline (figures/_data.band_parts); landward of it the band is river-fed
  ground (delta plain, the western ilmens, the Volga-Akhtuba floodplain) and is drawn in
  grey, not interpreted as seabed.
- The regional locator of the first version is gone: the journal's guide for authors does
  not allow a location map wider than the study area's bounding box. The graticule locates
  the map.
- The key is laid out from the measured widths of its labels, so no label can run into
  the text beside it.

Drawn in the analysis projection (Lambert azimuthal equal area centred at 45.5 N, 50 E,
caspian.config.CRS_EQUAL_AREA); the scale bar is exact at the centre (scale error below
0.1% inside the domain, figures/fig_study_area.py). No result is drawn besides the
validated-sector classes, which the script recomputes from the accuracy tables.

Output: figures/out/rse_fig_study_area.{pdf,tif,png}
Usage: python figures/rse_fig_study_area.py
"""

from __future__ import annotations

import sys

import _common as C
import _data as D
import _rse as R
import numpy as np
import pandas as pd

from caspian.config import RAW, RESULTS

VAL = RESULTS / "validation" / "main"
BAND = "#e3cfa6"          # emerged seabed 1995-96 to 2025-26
TESTC = "#D55E00"         # ICESat-2 test blocks (Okabe-Ito vermilion)
HATCH = "#9a9a9a"
# label anchors (lon, lat) and alignment: cartographic placement only
LABELS = {"west": (47.02, 45.17, "center"), "volga": (47.75, 46.55, "center"),
          "interfluve": (50.66, 46.62, "center"), "ural": (51.72, 46.6, "center"),
          "northeast": (53.45, 46.55, "center"), "komsomolets": (53.95, 44.72, "center"),
          "open_shelf": (49.55, 45.05, "center")}
NAMES_MAP = {"west": "Kizlyar–Dagestan\ncoast", "volga": "Volga delta", "interfluve": "Volga–Ural\ninterfluve coast",
             "ural": "Ural delta", "northeast": "NE Kazakh coast\n(Tengiz–Kashagan)",
             "komsomolets": "Komsomolets\nBay", "open_shelf": "Open shelf"}
CITIES = {"Astrakhan": (4, -1, "left", "center"), "Atyrau": (4, 0, "left", "center")}
COUNTRIES = {"RUSSIA": (47.0, 47.17), "KAZAKHSTAN": (54.05, 47.12)}
KEY_GAP_MM = 3.0          # between a sector-class label and its explanation
KEY_PAD_MM = 1.6          # hatched margin around the "Biased or untested" label


def validated_sectors() -> list[str]:
    """Amendment 18 item 1, recomputed from the committed accuracy tables."""
    ok = None
    for src in ("ATL08", "ATL06"):
        a = pd.read_csv(VAL / f"{src}_accuracy.csv")
        d = a[(a["model"] == "dem") & (a["scope"] != "overall") & (a["n"] > 0)].set_index("scope")
        good = {s for s in d.index if abs(d.loc[s, "bias"]) <= 0.5 and abs(d.loc[s, "median"]) <= 0.5}
        ok = good if ok is None else ok & good
    return sorted(ok)


def main() -> int:
    import cartopy.feature as cfeature
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    m = C.metrics()
    val = validated_sectors()
    if sorted(val) != sorted(R.VALIDATED):
        raise SystemExit(f"validated sectors {val} differ from {R.VALIDATED}")
    proj, fwd = C.projection()
    domain, sectors = C.study_geometries()
    extent = C.map_extent(domain)
    sea, sea_ext = C.sea_mask()
    ex = np.load(RESULTS / "study_area" / "extents_v6_250m.npz")
    a, _, c0, _, e, f0 = ex["transform"]
    band = (ex["high"] == 1) & (ex["low"] == 0)
    parts = D.band_parts()
    if not np.array_equal(parts["band"], band):
        raise SystemExit("figures/_data.band_parts: another band than this figure's")
    river = parts["not_seabed"]
    seabed = band & ~river
    band_ext = [c0, c0 + a * band.shape[1], f0 + e * band.shape[0], f0]
    blocks = pd.read_csv(VAL / "ATL08_accuracy_blocks.csv")
    blocks = blocks[blocks["model"] == "dem"]
    n_blocks = int(R.value(m, "h2.atl08.n_test_blocks"))
    if len(blocks) != n_blocks:
        raise SystemExit(f"{len(blocks)} test blocks with points, metrics say {n_blocks}")
    sett = gpd.read_file(RAW / "osm" / "osm_settlements.gpkg")
    sett = sett[sett["name_en"].isin(list(CITIES))]

    R.style()
    W = R.DOUBLE_MM
    L, RGT, TOP = 8.5, 1.5, 2.0
    map_w = W - L - RGT
    map_h = map_w * (extent[3] - extent[2]) / (extent[1] - extent[0])
    G1, ROW2, BOT = 7.0, 31.0, 1.5
    H = TOP + map_h + G1 + ROW2 + BOT
    fig = plt.figure(figsize=(W * C.MM, H * C.MM))
    ax = fig.add_axes([L / W, (H - TOP - map_h) / H, map_w / W, map_h / H], projection=proj)
    ax.set_extent(extent, crs=proj)
    ax.set_facecolor(R.LAND)
    for mask, colour in ((seabed, BAND), (river, R.NOTSEA)):
        ax.imshow(np.where(mask, 1.0, np.nan), extent=band_ext, origin="upper", transform=proj,
                  cmap=ListedColormap([colour]), interpolation="nearest", zorder=1, rasterized=True)
    ax.imshow(np.where(sea, 1.0, np.nan), extent=sea_ext, origin="upper", transform=proj,
              cmap=ListedColormap([R.SEA]), interpolation="nearest", zorder=2, rasterized=True)
    bnd = cfeature.NaturalEarthFeature("cultural", "admin_0_boundary_lines_land", "50m")
    ax.add_feature(bnd, facecolor="none", edgecolor="#7a7a7a", lw=0.6, ls=(0, (4, 1.5, 1, 1.5)), zorder=3)
    gl = C.graticule(ax, size=R.FS_SMALL)
    gl.xlabel_style = gl.ylabel_style = {"size": R.FS_SMALL, "color": R.MUTED}
    hatch_geoms = [sectors[s] for s in R.NOT_VALIDATED]
    ax.add_geometries(hatch_geoms, crs=proj, facecolor="none", edgecolor=HATCH, hatch="////", lw=0, zorder=4)
    C.draw_outlines(ax, proj, domain, sectors, lw_sector=0.5, lw_domain=1.0, zorder=6)
    for _, r in blocks.iterrows():
        i, j = (int(v) for v in r["block"].split("_"))
        ax.add_patch(Rectangle((i * 50_000.0, j * 50_000.0), 50_000.0, 50_000.0, transform=proj, fc="none",
                               ec=TESTC, lw=1.0, zorder=8))
    halo = [patheffects.withStroke(linewidth=1.8, foreground="white")]
    for sid, (lon, lat, ha) in LABELS.items():
        x, y = fwd(lon, lat)
        validated = sid in R.VALIDATED
        ax.text(x, y, NAMES_MAP[sid], transform=proj, ha=ha, va="center", fontsize=R.FS,
                fontweight="bold" if validated else "normal", style="normal" if validated else "italic",
                color=R.INK, zorder=12, linespacing=1.0, multialignment="center",
                bbox={"boxstyle": "round,pad=0.25,rounding_size=0.15", "fc": "white",
                      "ec": R.INK if validated else HATCH, "lw": 0.8 if validated else 0.5,
                      "alpha": 0.95})
    for _, r in sett.iterrows():
        x, y = fwd(r.geometry.x, r.geometry.y)
        dx, dy, ha, va = CITIES[r["name_en"]]
        ax.plot(x, y, "s", ms=3.0, mfc=R.INK, mec="white", mew=0.5, transform=proj, zorder=11)
        ax.annotate(r["name_en"], (x, y), xycoords=proj._as_mpl_transform(ax), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va=va, fontsize=R.FS_SMALL, color=R.INK,
                    path_effects=halo, zorder=11)
    for txt, (lon, lat) in COUNTRIES.items():
        x, y = fwd(lon, lat)
        ax.text(x, y, txt, transform=proj, ha="center", va="center", fontsize=R.FS_SMALL, color="#6b6b6b",
                path_effects=halo, zorder=10)
    x, y = fwd(50.55, 45.75)
    ax.text(x, y, "North Caspian Sea", transform=proj, ha="center", va="center", fontsize=R.FS, style="italic",
            color="#2f5d80", zorder=10)
    R.scale_bar(ax, extent[0] + 18_000, extent[2] + 15_000, C.SCALE_KM * 1000.0,
                R.num(m, "params.fig_scale_bar_km", "km"), transform=proj, size=R.FS_SMALL)

    # key: map symbols on the left, the two sector classes on the right ---------------------------
    y_row = TOP + map_h + G1
    handles = [Patch(fc=R.SEA, ec="none", label="Open sea, 2025–26 low stand (Landsat)"),
               Patch(fc=BAND, ec="none", label="Seabed emerged between 1995–96 and 2025–26"),
               Patch(fc=R.NOTSEA, ec="none", label=R.NOTSEA_LABEL),
               Patch(fc="white", ec=TESTC, lw=1.0, label=f"ICESat-2 test block, 50 km ({n_blocks} with points)"),
               Line2D([], [], color=R.INK, lw=1.0, label="Study domain"),
               Line2D([], [], color=R.INK, lw=0.5, label="Sector boundary"),
               Line2D([], [], color="#7a7a7a", lw=0.6, ls=(0, (4, 1.5, 1, 1.5)), label="International boundary"),
               Line2D([], [], ls="none", marker="s", ms=3.0, mfc=R.INK, mec="white", label="City")]
    kw = {"frameon": False, "borderaxespad": 0, "handlelength": 1.6, "handletextpad": 0.6, "labelspacing": 0.45,
          "fontsize": R.FS, "alignment": "left"}
    leg1 = fig.legend(handles=handles[:4], loc="upper left", bbox_to_anchor=(L / W, 1 - (y_row + 0.5) / H), **kw)
    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    s = 25.4 / fig.dpi
    x_col2 = leg1.get_window_extent(rr).x1 * s + 6.0
    leg2 = fig.legend(handles=handles[4:], loc="upper left", bbox_to_anchor=(x_col2 / W, 1 - (y_row + 0.5) / H), **kw)
    fig.canvas.draw()
    sx = leg2.get_window_extent(rr).x1 * s + 9.0           # left edge of the sector-class key (mm)
    y_top = H - (y_row + 0.5)                              # top of the key row, mm from the bottom
    fig.text(sx / W, y_top / H, "Sector names", ha="left", va="top", fontsize=R.FS, color=R.INK)
    box = {"boxstyle": "round,pad=0.25,rounding_size=0.15", "fc": "white"}
    y1, y2 = y_top - 9.0, y_top - 20.5                     # centre lines of the two classes (mm)
    t1 = fig.text((sx + 1.0) / W, y1 / H, "Checked", ha="left", va="center", fontsize=R.FS, fontweight="bold",
                  color=R.INK, bbox={**box, "ec": R.INK, "lw": 0.8})
    t2 = fig.text((sx + 1.0 + KEY_PAD_MM) / W, y2 / H, "Biased or untested", ha="left", va="center", fontsize=R.FS,
                  style="italic", color=R.INK, zorder=5, bbox={**box, "ec": HATCH, "lw": 0.5})
    b1, b2 = R.text_box_mm(fig, t1), R.text_box_mm(fig, t2)
    # the hatched swatch frames the second label with an even margin; the explanations start
    # at one x, right of the wider of the two labels: nothing overlaps whatever the font renders
    fig.add_artist(Rectangle(((b2[0] - KEY_PAD_MM) / W, (b2[1] - KEY_PAD_MM) / H), (b2[2] - b2[0] + 2 * KEY_PAD_MM) / W,
                             (b2[3] - b2[1] + 2 * KEY_PAD_MM) / H, transform=fig.transFigure, fc="white", ec=HATCH,
                             hatch="////", lw=0.5, zorder=4))
    x_exp = max(b1[2], b2[2] + KEY_PAD_MM) + KEY_GAP_MM
    for yy, txt in ((y1, "waterline DEM within ±0.5 m\nof ICESat-2 (mean and median)"),
                    (y2, "mean or median error beyond ±0.5 m,\nor no ICESat-2 test block (hatched)")):
        te = fig.text(x_exp / W, yy / H, txt, ha="left", va="center", fontsize=R.FS_SMALL, color=R.INK, linespacing=1.1)
        if R.text_box_mm(fig, te)[2] > W - 0.5:
            raise SystemExit("the sector-class key runs past the right edge of the figure")
    R.save(fig, "rse_fig_study_area", W, H)
    plt.close(fig)
    print("validated sectors:", val, "| test blocks with ICESat-2 points:", n_blocks,
          "| band not drawn as seabed (250 m cells):", int(river.sum()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
