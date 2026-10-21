"""DAHITI -> BHS-77 offset (PREREG section 5; D-093).

delta = H_BHS-77 - H_DAHITI, estimated from the CASPCOM General Catalogue of the
Caspian Sea level (monthly means at the posts, Baltic Height System) over the years it
overlaps DAHITI. Reported for levels only: a constant offset cancels in every hindcast
and validation metric (D-070, D-074).

Fixed before the offset was computed (D-093):
  * posts: the seven CASPCOM "century" posts from which CASPCOM computes the mean sea
    level (Bulletins No. 12, 25 and 27, footnote 1): Baku, Neft Dashlary (N.KAMNI),
    Makhachkala, Fort-Shevchenko, Guvlymayak (Kuuli-Mayak), Turkmenbashi (Krasnovodsk)
    and Garabogaz / Duzlybogaz (Kara-Bogaz-Gol, file KBG);
  * annual means: the mean of 12 monthly means, only for years with all 12;
  * DAHITI averaged the same way: daily linear interpolant of the DAHITI 39 records,
    a month kept only if every day lies in a gap of at most 60 days (the caspian.levels
    default), a year only if all 12 months are kept;
  * d[post, year] = H_post - H_DAHITI; m[post] = mean over the overlap years (posts
    with at least 10 years); delta = mean of m over the posts;
  * sigma: moving-block bootstrap over calendar years (blocks of 3 years, 10,000
    replicates, seed 0), all posts sharing the drawn years.
Reported beside it, never chosen among: the median; block lengths 1 and 5; posts and
years resampled together; alternative post sets; monthly differences; DAHITI with no
gap limit; every Baltic-system post with at least 10 years.

Usage:
  python scripts/a7_dahiti_bhs_offset.py fetch   download the catalogue archive (network)
  python scripts/a7_dahiti_bhs_offset.py         compute -> results/levels/dahiti_bhs_offset.json
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import gauges as G
from caspian import io as cio
from caspian.config import RAW, RESULTS, ROOT

URL = "http://caspcom.com/wp-content/uploads/2026/07/Sea-Level-EN.zip"
PAGE = "http://www.caspcom.com/%d1%83%d1%80%d0%be%d0%b2%d0%b5%d0%bd%d1%8c/"
RAW_DIR = RAW / "gauges" / "caspcom"
ZIP = RAW_DIR / "Sea-Level-EN.zip"
PROV = RESULTS / "provenance" / "caspcom_catalogue.json"
DAHITI = RAW / "dahiti" / "dahiti_39_caspian_sea_water_level.csv"
BULLETIN_VALUES = ROOT / "refs" / "caspcom_bulletin_levels.csv"
OUT = RESULTS / "levels" / "dahiti_bhs_offset.json"
OUT_DIFFS = RESULTS / "levels" / "dahiti_bhs_annual_diffs.csv"

CENTURY = ["BAKU", "NKAMNI", "MAKH", "FORT", "KUULI", "KRASN", "KBG"]
NORTH = ["PESH", "TUL", "LAG", "KUL"]          # North Caspian posts (shallow; wind setup)
MAX_GAP_DAYS = 60.0
BLOCK, N_BOOT, SEED, MIN_YEARS = 3, 10000, 0, 10


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fetch() -> None:
    import requests

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    r = requests.get(URL, timeout=120)
    r.raise_for_status()
    ZIP.write_bytes(r.content)
    import zipfile

    with zipfile.ZipFile(ZIP) as z:
        files = [{"name": i.filename, "size": i.file_size,
                  "date": "%04d-%02d-%02d" % i.date_time[:3]} for i in z.infolist() if not i.is_dir()]
    PROV.parent.mkdir(parents=True, exist_ok=True)
    PROV.write_text(json.dumps({
        "source": "CASPCOM General Catalogue of the Caspian Sea level, monthly data (English archive)",
        "page": PAGE, "url": URL, "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": sha256(ZIP), "bytes": ZIP.stat().st_size, "files": files,
        "terms": "No licence or terms of use stated on the page; site footer 'Copyright (c) 2026 CASPCOM. "
                 "All Rights Reserved.' The archive is kept locally (data/ is git-ignored) and not redistributed.",
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {ZIP} ({ZIP.stat().st_size} bytes), {PROV}")


def dahiti_monthly(max_gap_days):
    s = cio.read_dahiti(DAHITI).data
    return G.monthly_means_daily(s["time"], s["level_m"], max_gap_days)


def annual_diffs(post_annual: pd.DataFrame, dahiti_annual: pd.DataFrame) -> pd.DataFrame:
    d = post_annual.merge(dahiti_annual[["year", "level_m"]].rename(columns={"level_m": "dahiti_m"}), on="year")
    d["d"] = d["level_m"] - d["dahiti_m"]
    return d[np.isfinite(d["d"])].reset_index(drop=True)


def monthly_diffs(post_monthly: pd.DataFrame, dahiti_m: pd.DataFrame) -> pd.DataFrame:
    """Per post and year, the mean of the monthly differences over months with both."""
    m = post_monthly.merge(dahiti_m[["year", "month", "level_m"]].rename(columns={"level_m": "dahiti_m"}),
                           on=["year", "month"])
    m["dm"] = m["level_m"] - m["dahiti_m"]
    m = m[np.isfinite(m["dm"])]
    return m.groupby(["post", "year"])["dm"].mean().rename("d").reset_index()


def estimate(diffs, posts, years, stat="mean", block=BLOCK, resample_posts=False, seed=SEED):
    po = G.post_offsets(diffs, posts, stat, MIN_YEARS)
    delta = G.combine(po["m"].to_numpy(), stat)
    reps = G.bootstrap_offset(diffs, posts, years, block, N_BOOT, seed, stat, MIN_YEARS, resample_posts)
    reps = reps[np.isfinite(reps)]
    used = po[np.isfinite(po["m"])]
    yrs = sorted(diffs[diffs["post"].isin(used["post"]) & np.isfinite(diffs["d"])]["year"].unique())
    return {"delta_m": delta, "sigma_m": float(np.std(reps, ddof=1)),
            "ci95_m": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))],
            "n_boot_valid": int(reps.size), "n_posts": int(len(used)), "posts": list(used["post"]),
            "n_years": len(yrs), "first_year": int(yrs[0]) if yrs else None,
            "last_year": int(yrs[-1]) if yrs else None,
            "n_post_years": int(used["n_years"].sum()), "stat": stat, "block_years": block,
            "resample_posts": resample_posts}


def common_drift(diffs, posts) -> dict:
    """OLS trend of the year-mean of d - m[post] (post effects removed)."""
    d = diffs[diffs["post"].isin(posts)].copy()
    d["r"] = d["d"] - d.groupby("post")["d"].transform("mean")
    y = d.groupby("year")["r"].mean()
    b = np.polyfit(y.index.to_numpy(float), y.to_numpy(float), 1)
    resid = y.to_numpy() - np.polyval(b, y.index.to_numpy(float))
    return {"trend_m_per_decade": float(b[0] * 10), "n_years": int(y.size),
            "residual_sd_m": float(np.std(resid, ddof=2))}


def bulletin_qc(post_annual: pd.DataFrame) -> list[dict]:
    """The catalogue reproduced against the levels CASPCOM printed in its bulletins."""
    tab = post_annual.pivot_table(index="year", columns="post", values="level_m")
    out = []
    with BULLETIN_VALUES.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            y, posts = int(r["year"]), r["posts"].split(";")
            have = [p for p in posts if p in tab.columns and y in tab.index and np.isfinite(tab.loc[y, p])]
            val = float(np.mean([tab.loc[y, p] for p in have])) if have else None
            out.append({"bulletin": int(r["bulletin"]), "year": y, "quantity": r["quantity"], "posts": posts,
                        "posts_in_catalogue": have, "bulletin_m": float(r["value_m"]), "catalogue_m": val,
                        "catalogue_minus_bulletin_m": (val - float(r["value_m"])) if val is not None else None,
                        "complete": len(have) == len(posts)})
    return out


def bulletin_sea_mean_offset(dahiti_annual: pd.DataFrame) -> dict:
    """Sensitivity: the 'mean level of the Caspian Sea' that CASPCOM printed in its
    bulletins (typed from the bulletin text, refs/caspcom_bulletin_levels.csv) minus
    the DAHITI annual mean of the same year. Few years, so no bootstrap."""
    da = dahiti_annual.set_index("year")["level_m"]
    rows = []
    with BULLETIN_VALUES.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            y = int(r["year"])
            if r["quantity"] == "sea_mean" and y in da.index and np.isfinite(da[y]):
                rows.append({"year": y, "bulletin": int(r["bulletin"]), "d": float(r["value_m"]) - float(da[y])})
    d = np.array([x["d"] for x in rows])
    return {"n_years": len(rows), "years": rows, "mean_m": float(d.mean()) if d.size else None,
            "sd_m": float(d.std(ddof=1)) if d.size > 1 else None}


def main(argv) -> int:
    if argv[1:2] == ["fetch"]:
        fetch()
        return 0
    prov = json.loads(PROV.read_text(encoding="utf-8"))
    if sha256(ZIP) != prov["sha256"]:
        raise SystemExit(f"{ZIP}: sha256 differs from {PROV}")
    monthly, meta = G.read_catalogue_zip(ZIP)
    baltic = sorted(p for p, m in meta.items() if m["altitude_system"] == "baltic")
    post_monthly = monthly[monthly["post"].isin(baltic)]
    post_annual = G.annual_means(post_monthly)

    dm = dahiti_monthly(MAX_GAP_DAYS)
    da = G.annual_means(dm.assign(g=0), by=("g",)).drop(columns="g")
    diffs = annual_diffs(post_annual, da)
    years = list(range(int(diffs["year"].min()), int(diffs["year"].max()) + 1))

    primary = estimate(diffs, CENTURY, years)
    two_way = estimate(diffs, CENTURY, years, resample_posts=True)
    po = G.post_offsets(diffs, CENTURY, "mean", MIN_YEARS)
    m_ok = po["m"].dropna()

    dm_free = dahiti_monthly(None)
    da_free = G.annual_means(dm_free.assign(g=0), by=("g",)).drop(columns="g")
    diffs_free = annual_diffs(post_annual, da_free)
    diffs_month = monthly_diffs(post_monthly, dm)

    six = [p for p in CENTURY if p != "KBG"]
    sens = {
        "median": estimate(diffs, CENTURY, years, stat="median"),
        "block_1": estimate(diffs, CENTURY, years, block=1),
        "block_5": estimate(diffs, CENTURY, years, block=5),
        "garabogaz_as_BEK": estimate(diffs, six + ["BEK"], years),
        "six_posts_without_garabogaz": estimate(diffs, six, years),
        "bulletin20_four_posts": estimate(diffs, ["BAKU", "MAKH", "FORT", "KRASN"], years),
        "north_caspian_posts": estimate(diffs, NORTH, years),
        "monthly_differences": estimate(diffs_month, CENTURY, years),
        "dahiti_no_gap_limit": estimate(diffs_free, CENTURY, years),
    }
    all_posts = G.post_offsets(diffs, baltic, "mean", MIN_YEARS)
    all_posts["name"] = all_posts["post"].map(lambda p: meta[p]["name"])

    n_dahiti_years = int(np.isfinite(da["level_m"]).sum())
    res = {
        "definition": "delta = H_BHS-77 - H_DAHITI (m); H_BHS-77 = H_DAHITI + delta",
        "source": {"catalogue": prov["source"], "url": prov["url"], "page": prov["page"],
                   "retrieved_utc": prov["retrieved_utc"], "sha256": prov["sha256"],
                   "dahiti_file": str(DAHITI.relative_to(ROOT)).replace("\\", "/")},
        "settings": {"posts": CENTURY, "max_gap_days": MAX_GAP_DAYS, "block_years": BLOCK, "n_boot": N_BOOT,
                     "seed": SEED, "min_years_per_post": MIN_YEARS, "stat": "mean",
                     "annual_mean": "mean of 12 monthly means (years with all 12 only), posts and DAHITI alike"},
        "primary": primary,
        "between_post_sd_m": float(m_ok.std(ddof=1)),
        "between_post_range_m": [float(m_ok.min()), float(m_ok.max())],
        "posts_and_years_bootstrap": two_way,
        "common_drift": common_drift(diffs, list(po.loc[po["m"].notna(), "post"])),
        "per_post": po.to_dict(orient="records"),
        "all_baltic_posts": all_posts.to_dict(orient="records"),
        "sensitivities": sens,
        "dahiti_complete_years": n_dahiti_years,
        "dahiti_complete_years_no_gap_limit": int(np.isfinite(da_free["level_m"]).sum()),
        "post_meta": meta,
        "datum_sign_inferred_posts": sorted(p for p, m in meta.items() if m["datum_sign_inferred"]),
        "non_baltic_posts": sorted(p for p, m in meta.items() if m["altitude_system"] != "baltic"),
        "bulletin_qc": bulletin_qc(post_annual),
        "bulletin_sea_mean_offset": bulletin_sea_mean_offset(da),
        "bulletin_sea_mean_offset_no_gap_limit": bulletin_sea_mean_offset(da_free),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    diffs.sort_values(["post", "year"]).to_csv(OUT_DIFFS, index=False, float_format="%.4f")
    p = res["primary"]
    print(f"delta = {p['delta_m']:+.3f} m, sigma (year blocks) {p['sigma_m']:.3f} m, "
          f"{p['n_posts']} posts, {p['n_years']} years ({p['first_year']}-{p['last_year']}), "
          f"{p['n_post_years']} post-years; between-post sd {res['between_post_sd_m']:.3f} m; "
          f"posts+years sigma {two_way['sigma_m']:.3f} m")
    print(f"wrote {OUT} and {OUT_DIFFS}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
