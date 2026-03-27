#!/usr/bin/env python3
"""Pull ocean data for Florida Straits near Miami (25.75N, 80.25W).

Sources:
1. NOAA OISST v2.1 via coastwatch ERDDAP for daily SST (2000-present)
   - Yearly chunks with retries and backoff
2. HYCOM expt_93.0 via OPeNDAP for subsurface profiles (~2019-present)
   - Computes MLD, OHC, 26C isotherm depth

Table: ocean_hycom (date, sst, mixed_layer_depth_m, ohc_kj_cm2, isotherm_26c_depth_m)
"""

import sqlite3
import sys
import time as time_mod
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

DB_PATH = "/Users/blakebot/blakebot/miami-project/miami_collector.db"
TARGET_LAT = 25.75
TARGET_LON = -80.25
LON_360 = 360.0 + TARGET_LON  # 279.75


def create_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ocean_hycom (
            date TEXT PRIMARY KEY,
            sst REAL,
            mixed_layer_depth_m REAL,
            ohc_kj_cm2 REAL,
            isotherm_26c_depth_m REAL
        )
    """)
    conn.commit()


# ── OISST ──────────────────────────────────────────────────

def pull_noaa_oisst():
    """Pull NOAA OISST daily SST yearly from coastwatch ERDDAP."""
    print("  Pulling NOAA OISST from coastwatch ERDDAP ...")
    base = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg.csv"
    session = requests.Session()
    session.headers.update({"User-Agent": "MiamiWeatherProject/1.0"})

    all_rows = []

    for year in range(2000, 2027):
        start = f"{year}-01-01T00:00:00Z"
        end = f"{year}-12-31T00:00:00Z" if year < 2026 else "2026-03-27T00:00:00Z"

        url = (
            f"{base}?sst[({start}):1:({end})]"
            f"[(0.0):1:(0.0)]"
            f"[({TARGET_LAT}):1:({TARGET_LAT})]"
            f"[({LON_360}):1:({LON_360})]"
        )

        success = False
        for attempt in range(5):
            try:
                if attempt > 0:
                    wait = 10 * (2 ** (attempt - 1))  # 10, 20, 40, 80
                    print(f"      Retry {attempt}, waiting {wait}s ...")
                    time_mod.sleep(wait)

                resp = session.get(url, timeout=180)
                if resp.status_code == 503:
                    print(f"    {year}: 503 Service Unavailable")
                    continue
                if resp.status_code != 200:
                    print(f"    {year}: HTTP {resp.status_code}")
                    break

                n = 0
                for line in resp.text.strip().split("\n")[2:]:
                    parts = line.split(",")
                    if len(parts) >= 5:
                        try:
                            dt_str = parts[0][:10]
                            sst = float(parts[4])
                            if -5 < sst < 40:
                                all_rows.append({"date": dt_str, "sst": sst})
                                n += 1
                        except (ValueError, IndexError):
                            continue
                print(f"    {year}: {n} days")
                success = True
                break

            except requests.exceptions.ReadTimeout:
                print(f"    {year}: timeout (attempt {attempt+1})")
            except requests.exceptions.ConnectTimeout:
                print(f"    {year}: connect timeout (attempt {attempt+1})")
            except Exception as e:
                print(f"    {year}: {e}")
                break

        if not success:
            print(f"    {year}: FAILED all attempts")

        # Pause between years to avoid rate limiting
        time_mod.sleep(3)

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["date"]) if all_rows else pd.DataFrame(columns=["date", "sst"])
    print(f"    Total OISST: {len(df)} days")
    return df


# ── HYCOM ──────────────────────────────────────────────────

def _compute_mld(profile, depths, sst):
    if sst is None:
        return None
    valid = ~np.isnan(profile)
    for di in range(1, len(depths)):
        if valid[di] and (sst - profile[di]) > 0.5:
            if di > 0 and valid[di - 1]:
                dt_prev = sst - profile[di - 1]
                dt_curr = sst - profile[di]
                if dt_curr != dt_prev:
                    frac = (0.5 - dt_prev) / (dt_curr - dt_prev)
                    return float(depths[di - 1] + frac * (depths[di] - depths[di - 1]))
            return float(depths[di])
    return None


def _compute_iso26(profile, depths, sst):
    if sst is None or sst < 26.0:
        return None
    valid = ~np.isnan(profile)
    for di in range(len(depths)):
        if valid[di] and profile[di] < 26.0:
            if di > 0 and valid[di - 1] and profile[di - 1] >= 26.0:
                frac = (profile[di - 1] - 26.0) / (profile[di - 1] - profile[di])
                return float(depths[di - 1] + frac * (depths[di] - depths[di - 1]))
            return float(depths[di])
    if valid.any() and profile[valid][-1] >= 26.0:
        return float(depths[len(profile) - 1])
    return None


def _compute_ohc(profile, depths, iso26):
    if iso26 is None:
        return None
    valid = ~np.isnan(profile)
    rho, cp = 1025.0, 3985.0
    ohc_val = 0.0
    for di in range(len(depths) - 1):
        if depths[di] >= iso26:
            break
        if valid[di] and valid[di + 1]:
            t_avg = (profile[di] + profile[di + 1]) / 2.0
            if t_avg > 26.0:
                dz_end = min(depths[di + 1], iso26)
                dz = dz_end - depths[di]
                ohc_val += rho * cp * (t_avg - 26.0) * dz
    return ohc_val / 1e7


def pull_hycom_profiles():
    """Pull HYCOM expt_93.0 subsurface temp profiles via OPeNDAP."""
    print("  Pulling HYCOM profiles via OPeNDAP ...")

    try:
        import xarray as xr
    except ImportError:
        print("    xarray not available, skipping HYCOM")
        return pd.DataFrame()

    hycom_url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0"

    try:
        ds = xr.open_dataset(hycom_url, engine="netcdf4", decode_times=False)
    except Exception as e:
        print(f"    Cannot open HYCOM: {e}")
        return pd.DataFrame()

    lats = ds["lat"].values
    lons = ds["lon"].values
    lat_idx = int(np.argmin(np.abs(lats - TARGET_LAT)))
    lon_idx = int(np.argmin(np.abs(lons - LON_360)))
    print(f"    Grid: {lats[lat_idx]:.3f}N, {lons[lon_idx]:.3f}E")

    time_vals = ds["time"].values
    epoch = datetime(2000, 1, 1)
    dates = [epoch + timedelta(hours=float(t)) for t in time_vals]
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]
    depths = ds["depth"].values

    # One profile per day
    date_to_first_idx = {}
    for i, d in enumerate(date_strs):
        if d not in date_to_first_idx:
            date_to_first_idx[d] = i

    unique_dates = sorted(date_to_first_idx.keys())
    indices = [date_to_first_idx[d] for d in unique_dates]
    print(f"    {len(unique_dates)} unique dates, {len(depths)} depth levels")
    print(f"    Range: {unique_dates[0]} to {unique_dates[-1]}")

    all_profiles = []
    chunk_size = 100

    for chunk_start in range(0, len(indices), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(indices))
        chunk_idx = indices[chunk_start:chunk_end]
        chunk_dates = unique_dates[chunk_start:chunk_end]

        try:
            temp_data = ds["water_temp"].isel(
                time=chunk_idx,
                lat=lat_idx,
                lon=lon_idx
            ).load()

            for i, date_str in enumerate(chunk_dates):
                profile = temp_data.values[i, :]
                if np.all(np.isnan(profile)):
                    continue

                sst = float(profile[0]) if not np.isnan(profile[0]) else None
                mld = _compute_mld(profile, depths, sst)
                iso26 = _compute_iso26(profile, depths, sst)
                ohc = _compute_ohc(profile, depths, iso26)

                all_profiles.append({
                    "date": date_str,
                    "sst_hycom": sst,
                    "mixed_layer_depth_m": mld,
                    "ohc_kj_cm2": ohc,
                    "isotherm_26c_depth_m": iso26,
                })

            print(f"    Chunk {chunk_start}-{chunk_end}/{len(indices)}: OK")

        except Exception as e:
            print(f"    Chunk {chunk_start}-{chunk_end} error: {e}")
            continue

    ds.close()

    df = pd.DataFrame(all_profiles)
    if not df.empty:
        df = df.drop_duplicates(subset=["date"])
    print(f"    Total HYCOM profiles: {len(df)}")
    return df


# ── Main ───────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Ocean Data Puller (SST + Subsurface)")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)
    create_table(conn)

    # 1. OISST
    oisst_df = pull_noaa_oisst()

    # 2. HYCOM
    hycom_df = pull_hycom_profiles()

    # 3. Merge
    print("\nMerging ...")
    if not oisst_df.empty:
        master = oisst_df[["date", "sst"]].copy()
    else:
        master = pd.DataFrame(columns=["date", "sst"])

    if not hycom_df.empty:
        hycom_merge = hycom_df[["date", "mixed_layer_depth_m", "ohc_kj_cm2", "isotherm_26c_depth_m"]].copy()
        if not master.empty:
            master = master.merge(hycom_merge, on="date", how="outer")
            hycom_sst = hycom_df[["date", "sst_hycom"]]
            master = master.merge(hycom_sst, on="date", how="left")
            master["sst"] = master["sst"].fillna(master.get("sst_hycom"))
            master.drop(columns=["sst_hycom"], errors="ignore", inplace=True)
        else:
            master = hycom_df.rename(columns={"sst_hycom": "sst"})
            master = master[["date", "sst", "mixed_layer_depth_m", "ohc_kj_cm2", "isotherm_26c_depth_m"]]
    else:
        for col in ["mixed_layer_depth_m", "ohc_kj_cm2", "isotherm_26c_depth_m"]:
            if col not in master.columns:
                master[col] = np.nan

    for col in ["sst", "mixed_layer_depth_m", "ohc_kj_cm2", "isotherm_26c_depth_m"]:
        if col not in master.columns:
            master[col] = np.nan

    master = master[["date", "sst", "mixed_layer_depth_m", "ohc_kj_cm2", "isotherm_26c_depth_m"]]
    master = master.sort_values("date").drop_duplicates(subset=["date"])
    master = master[(master["date"] >= "2000-01-01") & (master["date"] <= "2026-03-27")]

    print(f"  Total records: {len(master)}")
    for col in ["sst", "mixed_layer_depth_m", "ohc_kj_cm2", "isotherm_26c_depth_m"]:
        n = master[col].notna().sum()
        print(f"    {col}: {n} days")

    print("\nWriting to database ...")
    master.to_sql("ocean_hycom", conn, if_exists="replace", index=False)

    count = conn.execute("SELECT COUNT(*) FROM ocean_hycom").fetchone()[0]
    sample = conn.execute("SELECT * FROM ocean_hycom ORDER BY date DESC LIMIT 5").fetchall()

    print(f"\n{'=' * 60}")
    print(f"DONE: {count} rows in ocean_hycom")
    print("Latest 5 rows:")
    for row in sample:
        print(f"  {row}")
    print("=" * 60)

    conn.close()


if __name__ == "__main__":
    main()
