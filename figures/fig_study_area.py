"""Study-area map (Supplementary Figure; brief section 12): the domain, the seven
reporting sectors, the stage-2 computation cover and the 25 km validation blocks.

a, domain and sectors (results/study_area/sectors.geojson, D-014), the stage-2
   cover (results/study_area/zone_v6_cover_buffered.geojson: the 5 km tile cover of
   active zone v6, extended seaward by one cell; D-029, D-060) and, for orientation,
   the open sea of the 2025-26 low stand (results/study_area/sea_mask_1km.npz, the
   D-029 open-sea definition from Landsat; it is not a product under audit).
b, the 25 km blocks (results/study_area/blocks_25km.csv; PREREG section 6), coloured
   by split, with the sector outlines.

Both panels are drawn in the analysis projection (Lambert azimuthal equal area,
caspian.config.CRS_EQUAL_AREA), with longitude/latitude gridlines; the scale bar is
exact at the projection centre (the scale error of this projection is below 0.1%
inside the domain, checked below). No result is shown.

Output: figures/out/fig_study_area.{pdf,tif,png}

Usage: python figures/fig_study_area.py
"""

from __future__ import annotations

import json
import sys

import _common as C
import numpy as np
import pandas as pd

from caspian import plotting
from caspian.config import CRS_EQUAL_AREA, RESULTS

STUDY = RESULTS / "study_area"
SEA = "#d7e6f2"
COVER = "#4d4d4d"
SCALE_KM = 100          # scale-bar length (quoted in the caption via params.fig_scale_bar_km)
GRATICULE_DEG = 1      # graticule spacing (params.fig_graticule_deg)


def main() -> int:
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import pyproj
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.patches import Rectangle
    from shapely.geometry import shape
    from shapely.ops import transform

    laea = pyproj.CRS(CRS_EQUAL_AREA)
    p = laea.to_dict()
    proj = ccrs.LambertAzimuthalEqualArea(central_longitude=p["lon_0"], central_latitude=p["lat_0"],
                                          globe=ccrs.Globe(ellipse="WGS84"))
    fwd = pyproj.Transformer.from_crs("EPSG:4326", laea, always_xy=True).transform

    feats = json.loads((STUDY / "sectors.geojson").read_text(encoding="utf-8"))["features"]
    geoms = {f["properties"]["id"]: transform(fwd, shape(f["geometry"]).segmentize(0.01)) for f in feats}
    domain = geoms.pop("domain")
    cover = transform(fwd, shape(json.loads((STUDY / "zone_v6_cover_buffered.geojson")
                                            .read_text(encoding="utf-8"))["features"][0]["geometry"]))
    blocks = pd.read_csv(STUDY / "blocks_25km.csv")
    size = float(blocks["x0"].diff().abs().replace(0, np.nan).min())
    sea = np.load(STUDY / "sea_mask_1km.npz")
    a, _, c0, _, e, f0 = sea["transform"]
    ny, nx = sea["mask"].shape
    sea_extent = [c0, c0 + a * nx, f0 + e * ny, f0]

    # scale check: the LAEA point scale factors inside the domain
    lon, lat = np.meshgrid(np.linspace(46.4, 54.6, 30), np.linspace(43.9, 47.35, 20))
    fac = pyproj.Proj(laea).get_factors(lon, lat)
    worst = float(max(np.abs(np.asarray(fac.parallel_scale) - 1).max(), np.abs(np.asarray(fac.meridional_scale) - 1).max()))
    if worst > 1e-3:
        print(f"FAIL scale error {worst:.2e} too large for one scale bar")
        return 1

    x0, y0, x1, y1 = domain.bounds
    pad = 8_000
    extent = [x0 - pad, x1 + pad, y0 - pad, y1 + pad]

    C.style()
    # layout in mm: maps of width MAP_W at the extent's aspect, stacked; legends to the right
    MAP_W, LEFT, BOTTOM, GAP, TOP = 122.0, 9.0, 6.5, 9.0, 5.0
    map_h = MAP_W * (extent[3] - extent[2]) / (extent[1] - extent[0])
    W, H = C.DOUBLE_MM, BOTTOM + 2 * map_h + GAP + TOP
    fig = plt.figure(figsize=(W * C.MM, H * C.MM))
    axs = [fig.add_axes([LEFT / W, (BOTTOM + map_h + GAP) / H, MAP_W / W, map_h / H], projection=proj),
           fig.add_axes([LEFT / W, BOTTOM / H, MAP_W / W, map_h / H], projection=proj)]
    col = (LEFT + MAP_W + 6.0) / W
    top_a = (BOTTOM + 2 * map_h + GAP) / H
    sector_ids = list(geoms)
    for k, ax in enumerate(axs):
        ax.set_extent(extent, crs=proj)
        ax.imshow(np.where(sea["mask"].astype(bool), 1.0, np.nan), extent=sea_extent, origin="upper",
                  transform=proj, cmap=plt.matplotlib.colors.ListedColormap([SEA]), interpolation="nearest",
                  zorder=0, rasterized=True)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False,
                          xlocs=np.arange(46, 56, GRATICULE_DEG), ylocs=np.arange(44, 48, GRATICULE_DEG),
                          color=plotting.GRID, lw=0.4)
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {"size": 6, "color": plotting.INK_MUTED}

    # a: sectors, cover, domain
    ax = axs[0]
    ax.add_geometries([cover], crs=proj, facecolor=COVER, edgecolor="none", alpha=0.35, zorder=2)
    ax.add_geometries(list(geoms.values()), crs=proj, facecolor="none", edgecolor=plotting.INK, lw=0.5, zorder=3)
    ax.add_geometries([domain], crs=proj, facecolor="none", edgecolor=plotting.INK, lw=1.0, zorder=4)
    for i, sid in enumerate(sector_ids, start=1):
        g = geoms[sid]
        pt = g.representative_point() if sid != "open_shelf" else g.buffer(-30_000).representative_point()
        ax.text(pt.x, pt.y, str(i), transform=proj, ha="center", va="center", fontsize=6.5, fontweight="bold",
                color=plotting.INK, zorder=6,
                bbox={"boxstyle": "circle,pad=0.18", "fc": "white", "ec": plotting.INK, "lw": 0.5})
    C.scale_bar(ax, extent[0] + 20_000, extent[2] + 18_000, SCALE_KM * 1000, f"{SCALE_KM} km", transform=proj)
    plotting.panel_label(ax, "a", x=-0.065, y=1.0)

    # b: blocks by split
    ax = axs[1]
    for split, colour in C.SPLIT_COLOURS.items():
        sub = blocks[blocks["split"] == split]
        for bx, by in zip(sub["x0"], sub["y0"]):
            ax.add_patch(Rectangle((bx, by), size, size, transform=proj, facecolor=colour, edgecolor="white",
                                   lw=0.4, alpha=0.75, zorder=2))
    ax.add_geometries(list(geoms.values()), crs=proj, facecolor="none", edgecolor=plotting.INK, lw=0.5, zorder=3)
    ax.add_geometries([domain], crs=proj, facecolor="none", edgecolor=plotting.INK, lw=1.0, zorder=4)
    C.scale_bar(ax, extent[0] + 20_000, extent[2] + 18_000, SCALE_KM * 1000, f"{SCALE_KM} km", transform=proj)
    plotting.panel_label(ax, "b", x=-0.065, y=1.0)

    # legends in the right-hand column
    leg_a = [Patch(facecolor=SEA, edgecolor="none", label="Open sea, 2025–26 (Landsat)"),
             Patch(facecolor=COVER, alpha=0.35, edgecolor="none", label="Stage-2 computation cover"),
             Line2D([], [], color=plotting.INK, lw=1.0, label="Study domain"),
             Line2D([], [], color=plotting.INK, lw=0.5, label="Sector boundary")]
    fig.legend(handles=leg_a, loc="upper left", bbox_to_anchor=(col - 0.012, top_a + 0.012), handlelength=1.4,
               alignment="left")
    sector_lines = "\n".join(f"{i}  {C.SECTOR_SHORT[s]}" for i, s in enumerate(sector_ids, start=1))
    fig.text(col, top_a - 0.115, "Sectors\n" + sector_lines, ha="left", va="top", fontsize=6,
             color=plotting.INK, linespacing=1.45)
    counts = blocks["split"].value_counts()
    leg_b = [Patch(facecolor=c, alpha=0.75, edgecolor="none",
                   label=f"{C.SPLIT_NAMES[s]} ({int(counts.get(s, 0))} blocks)") for s, c in C.SPLIT_COLOURS.items()]
    fig.legend(handles=leg_b, loc="upper left", bbox_to_anchor=(col - 0.012, (BOTTOM + map_h) / H + 0.012),
               handlelength=1.4, title=f"{size / 1000:.0f} km blocks by split", title_fontproperties={"size": 6.5},
               alignment="left")

    C.save(fig, "fig_study_area")
    plt.close(fig)
    print("blocks:", counts.to_dict(), "| block size m:", size, "| max LAEA scale error:", f"{worst:.1e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
