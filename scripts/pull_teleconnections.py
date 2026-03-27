#!/usr/bin/env python3
"""Pull teleconnection indices from NOAA CPC and BOM.

Sources:
  - NAO daily: ftp.cpc.ncep.noaa.gov
  - PNA daily: ftp.cpc.ncep.noaa.gov
  - ENSO ONI (Nino3.4): CPC monthly, interpolated to daily
  - MJO RMM: BOM daily phase + amplitude
  - AMO: PSL monthly, interpolated to daily
"""

import io
import re
import sqlite3
import sys
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests

DB_PATH = "/Users/blakebot/blakebot/miami-project/miami_collector.db"
START_DATE = "1990-01-01"
END_DATE = "2026-03-27"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "MiamiWeatherProject/1.0"})


def create_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS teleconnection_indices (
            date TEXT PRIMARY KEY,
            enso_oni REAL,
            nao REAL,
            pna REAL,
            mjo_phase INTEGER,
            mjo_amplitude REAL,
            amo REAL
        )
    """)
    conn.commit()


def safe_get(url, timeout=30):
    """GET with retries and short timeout."""
    for attempt in range(3):
        try:
            resp = SESSION.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            print(f"    Attempt {attempt+1} failed: {e}")
            if attempt == 2:
                raise
    return None


def pull_daily_index(url, name):
    """Pull daily NAO or PNA index from CPC FTP-style ASCII."""
    print(f"  Pulling {name} from {url} ...")
    resp = safe_get(url)

    rows = []
    for line in resp.text.strip().split("\n"):
        parts = line.split()
        if len(parts) >= 4:
            try:
                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                value = float(parts[3])
                d = date(year, month, day)
                rows.append({"date": d.isoformat(), "value": value})
            except (ValueError, IndexError):
                continue

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"    WARNING: No data parsed for {name}")
        return df

    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]
    print(f"    Got {len(df)} daily records for {name}")
    return df


def pull_nao():
    url = "https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.nao.index.b500101.current.ascii"
    return pull_daily_index(url, "NAO")


def pull_pna():
    url = "https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.pna.index.b500101.current.ascii"
    return pull_daily_index(url, "PNA")


def expand_monthly_to_daily(df_monthly, value_col="value"):
    """Expand monthly records (date=1st of month) to daily by repeating."""
    daily_rows = []
    for _, row in df_monthly.iterrows():
        d = row["date"]
        if isinstance(d, str):
            d = date.fromisoformat(d)
        if d.month == 12:
            next_month = date(d.year + 1, 1, 1)
        else:
            next_month = date(d.year, d.month + 1, 1)
        current = d
        while current < next_month:
            daily_rows.append({"date": current.isoformat(), "value": row[value_col]})
            current += timedelta(days=1)
    return pd.DataFrame(daily_rows)


def pull_enso_oni():
    """Pull monthly Nino3.4 SST anomaly and expand to daily.

    Try multiple sources in order of preference.
    """
    urls = [
        ("CPC detrend", "https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt"),
        ("PSL Nino3.4", "https://psl.noaa.gov/data/correlation/nina34.anom.data"),
    ]

    for source_name, url in urls:
        print(f"  Pulling ENSO ONI from {source_name}: {url} ...")
        try:
            resp = safe_get(url, timeout=20)
        except Exception:
            print(f"    Source {source_name} failed, trying next ...")
            continue

        text = resp.text.strip()
        lines = text.split("\n")

        if "nina34" in url or "correlation" in url:
            # PSL format: year val1 val2 ... val12 (monthly columns)
            return _parse_psl_monthly(lines, "ENSO ONI")
        else:
            # CPC detrend format: YR MON TOTAL ClimAdjust ANOM
            return _parse_cpc_enso(lines)

    print("    WARNING: All ENSO sources failed")
    return pd.DataFrame(columns=["date", "value"])


def _parse_psl_monthly(lines, name):
    """Parse PSL monthly data format (year + 12 monthly values)."""
    rows = []
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            year = int(float(parts[0]))
            if year < 1900 or year > 2030:
                continue
            for month_idx, val_str in enumerate(parts[1:], 1):
                if month_idx > 12:
                    break
                val = float(val_str)
                if val < -90 or val > 90:  # missing sentinel
                    continue
                rows.append({"date": date(year, month_idx, 1), "value": val})
        except (ValueError, IndexError):
            continue

    if not rows:
        print(f"    WARNING: No {name} data parsed")
        return pd.DataFrame(columns=["date", "value"])

    df = pd.DataFrame(rows)
    daily_df = expand_monthly_to_daily(df)
    daily_df = daily_df[(daily_df["date"] >= START_DATE) & (daily_df["date"] <= END_DATE)]
    print(f"    Got {len(daily_df)} daily records for {name}")
    return daily_df


def _parse_cpc_enso(lines):
    """Parse CPC detrend Nino3.4 format."""
    rows = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 5:
            try:
                year = int(parts[0])
                month = int(parts[1])
                anom = float(parts[4])
                rows.append({"date": date(year, month, 1), "value": anom})
            except (ValueError, IndexError):
                continue

    if not rows:
        print("    WARNING: No ENSO data parsed")
        return pd.DataFrame(columns=["date", "value"])

    df = pd.DataFrame(rows)
    daily_df = expand_monthly_to_daily(df)
    daily_df = daily_df[(daily_df["date"] >= START_DATE) & (daily_df["date"] <= END_DATE)]
    print(f"    Got {len(daily_df)} daily records for ENSO ONI")
    return daily_df


def pull_mjo():
    """Pull MJO RMM index from BOM."""
    url = "http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt"
    print(f"  Pulling MJO RMM from {url} ...")
    try:
        resp = safe_get(url, timeout=30)
    except Exception as e:
        print(f"    MJO pull failed: {e}")
        return pd.DataFrame(columns=["date", "phase", "amplitude"])

    rows = []
    for line in resp.text.strip().split("\n"):
        parts = line.split()
        if len(parts) >= 7:
            try:
                year = int(parts[0])
                month = int(parts[1])
                day = int(parts[2])
                rmm1 = float(parts[3])
                rmm2 = float(parts[4])
                phase = int(parts[5])
                amplitude = float(parts[6])
                if phase < 0 or phase > 8 or amplitude > 90 or abs(rmm1) > 90:
                    continue
                d = date(year, month, day)
                rows.append({
                    "date": d.isoformat(),
                    "phase": phase,
                    "amplitude": amplitude,
                })
            except (ValueError, IndexError):
                continue

    df = pd.DataFrame(rows)
    if df.empty:
        print("    WARNING: No MJO data parsed")
        return df

    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]
    print(f"    Got {len(df)} daily records for MJO")
    return df


def pull_amo():
    """Pull AMO monthly from PSL and expand to daily."""
    url = "https://psl.noaa.gov/data/correlation/amon.us.data"
    print(f"  Pulling AMO from {url} ...")
    try:
        resp = safe_get(url, timeout=20)
    except Exception as e:
        print(f"    AMO pull failed: {e}")
        return pd.DataFrame(columns=["date", "value"])

    lines = resp.text.strip().split("\n")
    return _parse_psl_monthly(lines, "AMO")


def main():
    print("=" * 60)
    print("Teleconnection Indices Puller")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)
    create_table(conn)

    # Pull all indices, catching errors per-source
    results = {}
    for name, func in [("nao", pull_nao), ("pna", pull_pna),
                         ("enso", pull_enso_oni), ("mjo", pull_mjo),
                         ("amo", pull_amo)]:
        try:
            results[name] = func()
        except Exception as e:
            print(f"  ERROR pulling {name}: {e}")
            results[name] = pd.DataFrame()

    nao_df = results["nao"]
    pna_df = results["pna"]
    enso_df = results["enso"]
    mjo_df = results["mjo"]
    amo_df = results["amo"]

    # Build date range
    print("\nMerging into daily records ...")
    date_range = pd.date_range(START_DATE, END_DATE, freq="D")
    master = pd.DataFrame({"date": date_range.strftime("%Y-%m-%d")})

    # Merge each index
    if not nao_df.empty:
        master = master.merge(
            nao_df.rename(columns={"value": "nao"}), on="date", how="left"
        )
    else:
        master["nao"] = np.nan

    if not pna_df.empty:
        master = master.merge(
            pna_df.rename(columns={"value": "pna"}), on="date", how="left"
        )
    else:
        master["pna"] = np.nan

    if not enso_df.empty:
        master = master.merge(
            enso_df.rename(columns={"value": "enso_oni"}), on="date", how="left"
        )
    else:
        master["enso_oni"] = np.nan

    if not mjo_df.empty:
        mjo_merge = mjo_df[["date", "phase", "amplitude"]].rename(
            columns={"phase": "mjo_phase", "amplitude": "mjo_amplitude"}
        )
        master = master.merge(mjo_merge, on="date", how="left")
    else:
        master["mjo_phase"] = np.nan
        master["mjo_amplitude"] = np.nan

    if not amo_df.empty:
        master = master.merge(
            amo_df.rename(columns={"value": "amo"}), on="date", how="left"
        )
    else:
        master["amo"] = np.nan

    # Ensure column order
    master = master[["date", "enso_oni", "nao", "pna", "mjo_phase", "mjo_amplitude", "amo"]]

    # Drop rows where ALL indices are NaN
    idx_cols = ["enso_oni", "nao", "pna", "mjo_phase", "mjo_amplitude", "amo"]
    master = master.dropna(subset=idx_cols, how="all")

    print(f"  Total daily records with at least one index: {len(master)}")
    print(f"  Coverage per index:")
    for col in idx_cols:
        non_null = master[col].notna().sum()
        print(f"    {col}: {non_null} days")

    # Write to database
    print("\nWriting to database ...")
    master.to_sql("teleconnection_indices", conn, if_exists="replace", index=False)

    # Verify
    count = conn.execute("SELECT COUNT(*) FROM teleconnection_indices").fetchone()[0]
    sample = conn.execute(
        "SELECT * FROM teleconnection_indices ORDER BY date DESC LIMIT 5"
    ).fetchall()

    print(f"\n{'=' * 60}")
    print(f"DONE: {count} rows in teleconnection_indices")
    print(f"Latest 5 rows:")
    for row in sample:
        print(f"  {row}")
    print(f"{'=' * 60}")

    conn.close()


if __name__ == "__main__":
    main()
