"""RSE figure: the North Caspian study area (double column, 190 mm).

a, the study domain and its seven reporting sectors, each named on the map
   (results/study_area/sectors.geojson); the three sectors whose observations are not
   validated hatched (the validated-sector rule of PREREG Amendment 18 item 1: the
   waterline DEM's mean and median errors within +-0.5 m of ICESat-2 ATL08 and ATL06 in
   the sector's test blocks; results/validation/main/ATL0{6,8}_accuracy.csv, checked
   here); the 50 km test blocks that hold the ICESat-2 points of the H2 population
   (results/validation/main/ATL08_accuracy_blocks.csv); the seabed that emerged between the
   1995-96 high stand and the 2025-26 low stand (water in the 1995-96 and not water in the
   2025-26 zone-v6 extent, results/study_area/extents_v6_250m.npz); the open sea of the
   2025-26 low stand (results/study_area/sea_mask_1km.npz); two cities
   (data/raw/osm/osm_settlements.gpkg); international boundaries (Natural Earth 1:50 m,
   cartopy); graticule and scale bar.
b, locator: the Caspian Sea and its neighbours (Natural Earth 1:50 m land, lakes and
   boundaries), with the study domain.

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
    import cartopy.crs as ccrs
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
    G1, ROW2, BOT = 7.0, 44.0, 1.5
    H = TOP + map_h + G1 + ROW2 + BOT
    fig = plt.figure(figsize=(W * C.MM, H * C.MM))
    ax = fig.add_axes([L / W, (H - TOP - map_h) / H, map_w / W, map_h / H], projection=proj)
    ax.set_extent(extent, crs=proj)
    ax.set_facecolor(R.LAND)
    ax.imshow(np.where(band, 1.0, np.nan), extent=band_ext, origin="upper", transform=proj,
              cmap=ListedColormap([BAND]), interpolation="nearest", zorder=1, rasterized=True)
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
    R.label(ax, "a", inside=True)

    # b, locator ----------------------------------------------------------------------------------
    LW_, LH_ = 50.0, 40.0
    y_row = TOP + map_h + G1
    loc_proj = ccrs.LambertAzimuthalEqualArea(central_longitude=50.0, central_latitude=42.0)
    axl = fig.add_axes([L / W, (H - y_row - LH_) / H, LW_ / W, LH_ / H], projection=loc_proj)
    axl.set_extent([38.5, 62.0, 34.5, 49.5], crs=ccrs.PlateCarree())
    axl.set_facecolor(R.SEA)
    axl.add_feature(cfeature.NaturalEarthFeature("physical", "land", "50m"), facecolor=R.LAND, edgecolor="none",
                    zorder=1)
    axl.add_feature(cfeature.NaturalEarthFeature("physical", "lakes", "50m"), facecolor=R.SEA, edgecolor="none",
                    zorder=2)
    axl.add_feature(cfeature.NaturalEarthFeature("physical", "coastline", "50m"), facecolor="none",
                    edgecolor="#8a8a8a", lw=0.3, zorder=3)
    axl.add_feature(bnd, facecolor="none", edgecolor="#9a9a9a", lw=0.35, zorder=3)
    axl.add_geometries([domain], crs=proj, facecolor="none", edgecolor=TESTC, lw=1.1, zorder=5)
    lg = axl.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, xlocs=np.arange(30, 70, 10),
                       ylocs=np.arange(30, 55, 5), color="#c8c8c8", lw=0.3, zorder=4)
    for txt, (lon, lat), kw in (("Caspian\nSea", (51.3, 41.4), {"style": "italic", "color": "#2f5d80"}),
                                ("Black Sea", (41.6, 43.4), {"style": "italic", "color": "#2f5d80"}),
                                ("Aral\nSea", (59.2, 45.4), {"style": "italic", "color": "#2f5d80"})):
        axl.text(lon, lat, txt, transform=ccrs.PlateCarree(), ha="center", va="center", fontsize=R.FS_SMALL,
                 linespacing=1.0, zorder=6, **kw)
    for s in axl.spines.values():
        s.set_edgecolor(R.MUTED)
        s.set_linewidth(0.5)
    # 500 km bar near the projection centre (50 E, 42 N), where the locator's scale is exact
    x0b, y0b = loc_proj.transform_point(53.2, 36.3, ccrs.PlateCarree())
    R.scale_bar(axl, x0b, y0b, 500_000.0, "500 km")
    R.label(axl, "b", inside=True)
    del lg

    # key -----------------------------------------------------------------------------------------
    handles = [Patch(fc=R.SEA, ec="none", label="Open sea, 2025–26 low stand (Landsat)"),
               Patch(fc=BAND, ec="none", label="Seabed emerged between 1995–96 and 2025–26"),
               Line2D([], [], color=R.INK, lw=1.0, label="Study domain"),
               Line2D([], [], color=R.INK, lw=0.5, label="Sector boundary"),
               Patch(fc="white", ec=TESTC, lw=1.0, label=f"ICESat-2 test block, 50 km ({n_blocks} with points)"),
               Line2D([], [], color="#7a7a7a", lw=0.6, ls=(0, (4, 1.5, 1, 1.5)), label="International boundary"),
               Line2D([], [], ls="none", marker="s", ms=3.0, mfc=R.INK, mec="white", label="City")]
    kw = {"frameon": False, "borderaxespad": 0, "handlelength": 1.6, "handletextpad": 0.6, "labelspacing": 0.45,
          "fontsize": R.FS, "alignment": "left"}
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=((L + LW_ + 6.0) / W, 1 - (y_row + 0.5) / H), **kw)
    sx = L + LW_ + 72.0
    fig.text(sx / W, 1 - (y_row + 0.5) / H, "Sector names", ha="left", va="top", fontsize=R.FS, color=R.INK)
    ax_k = fig.add_axes([sx / W, (H - y_row - 30.0) / H, 45.0 / W, 26.0 / H])
    ax_k.set_xlim(0, 45)
    ax_k.set_ylim(0, 26)
    ax_k.axis("off")
    ax_k.text(1.0, 21.5, "Validated", ha="left", va="center", fontsize=R.FS, fontweight="bold", color=R.INK,
              bbox={"boxstyle": "round,pad=0.25,rounding_size=0.15", "fc": "white", "ec": R.INK, "lw": 0.8})
    ax_k.text(17.0, 21.5, "waterline DEM within ±0.5 m\nof ICESat-2 (mean and median)", ha="left", va="center",
              fontsize=R.FS_SMALL, color=R.INK, linespacing=1.1)
    ax_k.add_patch(Rectangle((0.5, 6.5), 13.5, 6.0, fc="white", ec=HATCH, hatch="////", lw=0.5))
    ax_k.text(7.25, 9.5, "Not validated", ha="center", va="center", fontsize=R.FS, style="italic", color=R.INK,
              bbox={"boxstyle": "round,pad=0.25,rounding_size=0.15", "fc": "white", "ec": HATCH, "lw": 0.5})
    ax_k.text(17.0, 9.5, "mean or median error beyond ±0.5 m,\nor no ICESat-2 test block (hatched)", ha="left", va="center",
              fontsize=R.FS_SMALL, color=R.INK, linespacing=1.1)
    R.save(fig, "rse_fig_study_area", W, H)
    plt.close(fig)
    print("validated sectors:", val, "| test blocks with ICESat-2 points:", n_blocks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
