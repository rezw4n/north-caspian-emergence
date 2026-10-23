"""RSE figure: when the North Caspian seabed emerged (double column, 190 mm).

The RSE version of figures/fig1_emergence.py: sectors named on the map, text 6 pt or more.

a, year in which each pixel of the emerged band became permanently dry
   (data/processed/dem/main/emergence/emergence_*.tif, scripts/a6_emergence.py: the
   first day after the last day on which the basin level was at or above the pixel's
   waterline-DEM height), on a 150 m display grid (nearest 30 m pixel), shown only inside
   the emerged band (water in the 1995-96 and not water in the 2025-26 zone-v6 extent;
   results/study_area/extents_v6_250m.npz); open sea of the 2025-26 low stand; sector
   outlines and names (results/study_area/sectors.geojson); OpenStreetMap ports, piers and
   ferry terminals and oil and gas platforms and wells (data/raw/osm/); two cities.
b, the daily basin level used to date emergence (data/processed/dem/main/emergence/
   level_series_daily.csv): DAHITI, with Hydroweb shifted to DAHITI in DAHITI's gaps; the
   three epochs (results/metrics.json: levels.epoch_E1_end, params.e2_end). The colour
   strip under b is the colour key of a, on the same time axis.
c, Landsat acquisitions per calendar month in the main stage-2 plan
   (data/interim/stage2/plan_main_tau9900cm_landsat.csv), the acquisitions behind every
   waterline.

The emergence dates inherit the DEM's errors (sector biases up to 1.7 m, rse_fig_dem_accuracy);
the map is a view of the record, not an independent observation.

Output: figures/out/rse_fig_emergence.{pdf,tif,png}
Usage: python figures/rse_fig_emergence.py
"""

from __future__ import annotations

import sys

import _common as C
import _data as D
import _rse as R
import numpy as np
import pandas as pd

from caspian.config import DATA, INTERIM, RAW

YEAR0, YEAR1 = 1995.0, 2027.0
CITIES = {"Astrakhan": (-4, 2, "right"), "Atyrau": (4, 2, "left")}
LABELS = {"west": (46.98, 45.2), "volga": (47.62, 46.75), "interfluve": (50.6, 46.42),
          "ural": (51.7, 46.6), "northeast": (53.62, 46.55), "komsomolets": (54.05, 44.95),
          "open_shelf": (49.55, 45.05)}
HYDRO = "#D55E00"


def _years(days):
    return 1970.0 + np.asarray(days, float) / 365.2425


def _points(gdf, fwd):
    g = gdf.geometry.representative_point()
    return np.array([fwd(p.x, p.y) for p in g])


def main() -> int:
    import cmcrameri.cm as cmc
    import geopandas as gpd
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    m = C.metrics()
    e1_end = pd.Timestamp(m["levels.epoch_E1_end"]["value"])
    e2_end = pd.Timestamp(m["params.e2_end"]["value"]) + pd.Timedelta(days=1)
    proj, fwd = C.projection()
    domain, sectors = C.study_geometries()
    extent = C.map_extent(domain)
    em = D.mosaic("emergence", float(C.MAP_RES_M))
    band = D.emerged_band(em["extent"], em["z"].shape)
    yr = np.where(band, _years(em["z"]), np.nan)
    sea, sea_ext = C.sea_mask()
    ports = gpd.read_file(RAW / "osm" / "osm_ports.gpkg")
    plats = gpd.read_file(RAW / "osm" / "osm_platforms.gpkg")
    sett = gpd.read_file(RAW / "osm" / "osm_settlements.gpkg")
    sett = sett[sett["name_en"].isin(list(CITIES))]
    p_xy, w_xy = _points(ports, fwd), _points(plats, fwd)
    lv = pd.read_csv(DATA / "processed" / "dem" / "main" / "emergence" / "level_series_daily.csv", parse_dates=["date"])
    plan = pd.read_csv(INTERIM / "stage2" / "plan_main_tau9900cm_landsat.csv", usecols=["acq", "t", "epoch"])
    acq = plan.drop_duplicates("acq").copy()
    acq["month"] = pd.to_datetime(acq["t"], format="ISO8601").dt.to_period("M")
    per_month = acq.groupby("month").size()
    months = pd.period_range(per_month.index.min(), per_month.index.max(), freq="M")
    per_month = per_month.reindex(months, fill_value=0)

    R.style()
    W = R.DOUBLE_MM
    LEFT, RGT, TOP = 11.0, 2.0, 2.0
    GAP_AB, H_B, H_STRIP, GAP_SC, H_C, BOT = 10.0, 27.0, 2.2, 5.5, 13.0, 9.0
    map_w = W - LEFT - RGT
    map_h = map_w * (extent[3] - extent[2]) / (extent[1] - extent[0])
    H = TOP + map_h + GAP_AB + H_B + H_STRIP + GAP_SC + H_C + BOT
    fig = plt.figure(figsize=(W * C.MM, H * C.MM))
    ax = fig.add_axes([LEFT / W, (H - TOP - map_h) / H, map_w / W, map_h / H], projection=proj)
    y_c = BOT
    y_s = y_c + H_C + GAP_SC
    y_b = y_s + H_STRIP
    ax_b = fig.add_axes([LEFT / W, y_b / H, map_w / W, H_B / H])
    ax_s = fig.add_axes([LEFT / W, y_s / H, map_w / W, H_STRIP / H], sharex=ax_b)
    ax_c = fig.add_axes([LEFT / W, y_c / H, map_w / W, H_C / H], sharex=ax_b)

    # a: map ------------------------------------------------------------------------------------------
    ax.set_extent(extent, crs=proj)
    ax.set_facecolor(R.LAND)
    ax.imshow(np.where(sea, 1.0, np.nan), extent=sea_ext, origin="upper", transform=proj,
              cmap=ListedColormap([R.SEA]), interpolation="nearest", zorder=1, rasterized=True)
    norm = Normalize(YEAR0, YEAR1)
    cmap = cmc.batlow
    e = em["extent"]
    ax.imshow(np.ma.masked_invalid(yr), extent=[e[0], e[1], e[2], e[3]], origin="upper", transform=proj,
              cmap=cmap, norm=norm, interpolation="nearest", zorder=2, rasterized=True)
    gl = C.graticule(ax, size=R.FS_SMALL)
    gl.xlabel_style = gl.ylabel_style = {"size": R.FS_SMALL, "color": R.MUTED}
    C.draw_outlines(ax, proj, domain, sectors, lw_sector=0.45, lw_domain=0.85)
    halo = [patheffects.withStroke(linewidth=1.8, foreground="white")]
    ax.plot(p_xy[:, 0], p_xy[:, 1], ls="none", marker="o", ms=1.8, mfc="none", mec=R.INK, mew=0.35,
            transform=proj, zorder=8)
    ax.plot(w_xy[:, 0], w_xy[:, 1], ls="none", marker="^", ms=2.6, mfc="#ffffff", mec=R.INK, mew=0.5,
            transform=proj, zorder=8)
    for _, r in sett.iterrows():
        x, y = fwd(r.geometry.x, r.geometry.y)
        dx, dy, ha = CITIES[r["name_en"]]
        ax.plot(x, y, "s", ms=3.0, mfc=R.INK, mec="white", mew=0.5, transform=proj, zorder=9)
        ax.annotate(r["name_en"], (x, y), xycoords=proj._as_mpl_transform(ax), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, fontsize=R.FS_SMALL, color=R.INK, path_effects=halo, zorder=9)
    for sid, (lon, lat) in LABELS.items():
        x, y = fwd(lon, lat)
        ax.text(x, y, R.NAME2[sid], transform=proj, ha="center", va="center", fontsize=R.FS_SMALL, color=R.INK,
                zorder=12, linespacing=1.0, multialignment="center",
                bbox={"boxstyle": "round,pad=0.2,rounding_size=0.12", "fc": "white", "ec": "none", "alpha": 0.85})
    R.scale_bar(ax, extent[0] + 18_000, extent[2] + 15_000, C.SCALE_KM * 1000.0,
                R.num(m, "params.fig_scale_bar_km", "km"), transform=proj)
    R.label(ax, "a", inside=True)
    handles = [Patch(facecolor=R.SEA, edgecolor="none", label="Open sea, 2025–26"),
               Patch(facecolor=R.LAND, edgecolor=R.MUTED, lw=0.3, label="Land, or band seabed without a date"),
               Line2D([], [], ls="none", marker="o", ms=2.8, mfc="none", mec=R.INK, mew=0.5,
                      label="Port, pier or ferry terminal (OSM)"),
               Line2D([], [], ls="none", marker="^", ms=3.2, mfc="white", mec=R.INK, mew=0.5,
                      label="Oil or gas platform or well (OSM)"),
               Line2D([], [], ls="none", marker="s", ms=3.0, mfc=R.INK, mec="white", label="City")]
    leg = ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.997, 0.004), fontsize=R.FS_SMALL,
                    handlelength=1.2, borderaxespad=0.3, labelspacing=0.3, frameon=True, framealpha=1.0,
                    edgecolor="none", facecolor="white", title="Colour: year the seabed emerged (key under b)",
                    title_fontproperties={"size": R.FS_SMALL})
    leg.set_zorder(30)

    # b: level series; colour strip; c: acquisitions per month -----------------------------------------
    t0 = pd.Timestamp("1992-07-01")
    t1 = lv["date"].max() + pd.Timedelta(days=60)
    bounds = {"E1": (t0, e1_end), "E2": (e1_end, e2_end), "E3": (e2_end, t1)}
    names = {"E1": "Rising limb (E1)", "E2": "Early fall (E2)", "E3": "Late fall (E3)"}
    shade = {"E1": "#f2f2f2", "E2": "#ffffff", "E3": "#f2f2f2"}
    for a_ in (ax_b, ax_c):
        for k, (a0, a1) in bounds.items():
            a_.axvspan(a0, a1, color=shade[k], zorder=0, lw=0)
        for x in (e1_end, e2_end):
            a_.axvline(x, color=R.MUTED, lw=0.5, ls=(0, (2, 2)), zorder=1)
    for k, (a0, a1) in bounds.items():
        ax_b.text(a0 + (a1 - a0) / 2, 1.02, names[k], transform=ax_b.get_xaxis_transform(), ha="center",
                  va="bottom", fontsize=R.FS)
    for src, col in (("dahiti", R.INK), ("hydroweb_shifted", HYDRO)):
        y = np.where(lv["source"] == src, lv["level_m"], np.nan)
        ax_b.plot(lv["date"], y, color=col, lw=0.7, zorder=3, solid_capstyle="butt")
    ax_b.set_ylabel("Level (m, DAHITI)")
    ax_b.tick_params(labelbottom=False)
    ax_b.set_yticks([-29, -28, -27, -26])
    ax_b.legend(handles=[Line2D([], [], color=R.INK, lw=0.8, label="DAHITI"),
                         Line2D([], [], color=HYDRO, lw=0.8, label="Hydroweb, shifted to DAHITI (in DAHITI gaps)")],
                loc="upper right", fontsize=R.FS_SMALL, handlelength=1.4, ncol=2, borderaxespad=0.3)
    R.fig_label(fig, W, H, "b", 1.0, H - y_b - H_B - 4.0)
    ts = pd.date_range(f"{int(YEAR0)}-01-01", f"{int(YEAR1)}-01-01", freq="MS")
    ty = ts.year + (ts.dayofyear - 1) / 365.2425
    xs = mdates.date2num(ts)
    ax_s.pcolormesh(xs, [0, 1], np.asarray(ty)[None, :-1], cmap=cmap, norm=norm, shading="flat", rasterized=True)
    ax_s.set_ylim(0, 1)
    ax_s.set_yticks([])
    for sp in ax_s.spines.values():
        sp.set_visible(False)
    ax_s.tick_params(axis="x", which="both", length=0, labelbottom=False)
    ax_s.text(mdates.date2num(t0) + 20, 0.5, "Key of a", ha="left", va="center", fontsize=R.FS_SMALL,
              color=R.INK)
    x = months.to_timestamp(how="start") + pd.Timedelta(days=15)
    ax_c.bar(x, per_month.to_numpy(), width=27, color=R.MUTED, lw=0, zorder=3)
    ax_c.set_ylabel("Acquisitions\nper month")
    ax_c.set_ylim(0, None)
    ax_c.text(0.003, 0.97, "Landsat acquisitions in the main processing plan", transform=ax_c.transAxes,
              ha="left", va="top", fontsize=R.FS_SMALL, color=R.INK, zorder=5)
    ax_c.set_xlim(t0, t1)
    ax_c.xaxis.set_major_locator(mdates.YearLocator(5))
    ax_c.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax_c.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_c.set_xlabel("Year")
    R.fig_label(fig, W, H, "c", 1.0, H - y_c - H_C - 4.5)
    R.save(fig, "rse_fig_emergence", W, H)
    plt.close(fig)
    print("dated band pixels:", int(np.isfinite(yr).sum()), "| acquisitions:", len(acq))
    return 0


if __name__ == "__main__":
    sys.exit(main())
