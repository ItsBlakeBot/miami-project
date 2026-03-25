"""Empirical sigma-remaining climatology from IEM ASOS data.

At any hour of the day, the running high/low can only move in one
direction (high up, low down).  This module provides a data-driven
estimate of how much movement remains — expressed as a standard
deviation — by querying historical KMIA ASOS data from IEM.

Usage:
    clim = SigmaClimatology("KMIA", cache_dir="analysis_data")
    sigma = clim.get_sigma_remaining(month=3, hour_lst=14, market_type="high")
    # -> ~0.65  (very little upside left at 2 PM in March)

The climatological sigma acts as a *ceiling* on model sigma.  It can
be overridden by an active changepoint (frontal passage, etc.).

Note: ASOS reports whole-degree F, so true temp is within ±0.5°F of
the reported value.  However, we do NOT apply an ASOS rounding floor
to the final sigma — the wethr envelope provides NWS-verified extremes
that don't need the ASOS rounding correction.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import statistics
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

# IEM ASOS download URL template
_IEM_URL = (
    "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
    "station={station}&data=tmpf"
    "&year1={y1}&month1={m1}&day1=1"
    "&year2={y2}&month2={m2}&day2={last_day}"
    "&tz=America%2FNew_York&format=comma&latlon=no&elev=no"
    "&missing=M&trace=T&direct=no&report_type=3"
)

# Station code mapping (IEM uses 3-letter for some ASOS)
_IEM_STATION = {"KMIA": "MIA"}


class SigmaClimatology:
    """Empirical remaining-sigma lookup by (month, hour_lst, market_type)."""

    def __init__(
        self,
        station: str = "KMIA",
        cache_dir: str | Path = "analysis_data",
        years_back: int = 6,
        shoulder_months: int = 1,
    ):
        self.station = station
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.years_back = years_back
        self.shoulder_months = shoulder_months

        # Lookup: (month, hour_lst) -> {"high": sigma, "low": sigma}
        self._table: dict[tuple[int, int], dict[str, float]] = {}

        # Raw historical remaining-move samples from IEM data, used for
        # day-of-year weighted remaining distribution estimates.
        # Each row:
        # {
        #   "year": int, "month": int, "doy": int, "hour": int,
        #   "high_remaining": float, "low_remaining": float,
        # }
        self._samples: list[dict[str, float | int]] = []

    def get_sigma_remaining(
        self,
        month: int,
        hour_lst: int,
        market_type: str,
        target_date: str | None = None,
    ) -> float:
        """Return empirical sigma-remaining for the given context.

        When `target_date` is provided, use day-of-year weighted remaining
        samples (90-day window centered on target date) so nearby calendar
        days contribute more than far-away days.
        """
        if target_date:
            profile = self.get_remaining_profile(target_date, hour_lst, market_type)
            return max(float(profile.get("sigma_remaining", 3.0)), 0.01)

        if not self._table:
            self._load_or_build(month)

        key = (month, hour_lst)
        if key in self._table:
            raw = self._table[key].get(market_type, 3.0)
        else:
            raw = 3.0  # generous fallback

        # Tiny epsilon to avoid division-by-zero in Normal CDF math.
        return max(raw, 0.01)

    @staticmethod
    def _circular_day_distance(doy_a: int, doy_b: int) -> int:
        span = 366
        delta = abs(doy_a - doy_b)
        return min(delta, span - delta)

    @staticmethod
    def _weighted_quantile(values: list[float], weights: list[float], q: float) -> float:
        pairs = sorted(zip(values, weights), key=lambda x: x[0])
        total_w = sum(weights)
        if total_w <= 0:
            return values[len(values) // 2]
        target = q * total_w
        acc = 0.0
        for v, w in pairs:
            acc += w
            if acc >= target:
                return v
        return pairs[-1][0]

    def get_remaining_profile(
        self,
        target_date: str,
        hour_lst: int,
        market_type: str,
        window_days: int = 45,
    ) -> dict[str, float]:
        """Return weighted remaining-move profile for the given date/hour.

        Output fields:
        - mean_remaining
        - sigma_remaining
        - p_lock (remaining <= 0.5°F)
        - q50_remaining, q75_remaining, q90_remaining
        - sample_count
        """
        dt = datetime.strptime(target_date, "%Y-%m-%d")
        if not self._table:
            self._load_or_build(dt.month)

        vals: list[float] = []
        wts: list[float] = []
        target_doy = dt.timetuple().tm_yday
        target_year = dt.year
        value_key = "high_remaining" if market_type == "high" else "low_remaining"

        for row in self._samples:
            if int(row.get("hour", -1)) != hour_lst:
                continue

            sample_doy = int(row.get("doy", -1))
            dist = self._circular_day_distance(sample_doy, target_doy)
            if dist > window_days:
                continue

            # Distance weighting (nearby days in seasonal cycle matter most).
            w_day = 1.0 - (dist / (window_days + 1.0))
            if w_day <= 0:
                continue

            sample_year = int(row.get("year", target_year))
            years_ago = max(0, target_year - sample_year)
            # Mild recency preference; historical structure still matters.
            w_recency = 0.9 ** years_ago

            v = float(row.get(value_key, 0.0))
            vals.append(v)
            wts.append(w_day * w_recency)

        # Fallback when samples are unavailable (e.g., cold cache load).
        if not vals:
            sigma = self.get_sigma_remaining(dt.month, hour_lst, market_type)
            return {
                "mean_remaining": 0.0,
                "sigma_remaining": sigma,
                "p_lock": 0.0,
                "q50_remaining": 0.0,
                "q75_remaining": sigma,
                "q90_remaining": 1.2816 * sigma,
                "sample_count": 0.0,
            }

        total_w = sum(wts)
        mean_v = sum(v * w for v, w in zip(vals, wts)) / max(total_w, 1e-9)
        var_v = sum(w * ((v - mean_v) ** 2) for v, w in zip(vals, wts)) / max(total_w, 1e-9)
        sigma_v = math.sqrt(max(var_v, 0.0))

        # Consider remaining <= 0.5°F as effectively "locked".
        lock_w = sum(w for v, w in zip(vals, wts) if v <= 0.5)
        p_lock = lock_w / max(total_w, 1e-9)

        q50 = self._weighted_quantile(vals, wts, 0.50)
        q75 = self._weighted_quantile(vals, wts, 0.75)
        q90 = self._weighted_quantile(vals, wts, 0.90)

        return {
            "mean_remaining": round(mean_v, 4),
            "sigma_remaining": round(max(sigma_v, 0.01), 4),
            "p_lock": round(min(max(p_lock, 0.0), 1.0), 4),
            "q50_remaining": round(q50, 4),
            "q75_remaining": round(q75, 4),
            "q90_remaining": round(q90, 4),
            "sample_count": float(len(vals)),
        }

    def _cache_path(self, month: int) -> Path:
        return self.cache_dir / f"sigma_clim_{self.station}_m{month:02d}.json"

    def _load_or_build(self, month: int) -> None:
        cache = self._cache_path(month)
        if cache.exists():
            age_days = (datetime.now() - datetime.fromtimestamp(cache.stat().st_mtime)).days
            if age_days < 30:
                self._load_cache(cache)
                # Legacy caches may not include weighted samples.
                if self._samples:
                    return
                log.info("Sigma cache %s missing samples; rebuilding with v2 format", cache.name)

        try:
            self._build_from_iem(month)
            self._save_cache(cache)
        except Exception:
            log.exception("Failed to fetch IEM data; using fallback sigma")
            if cache.exists():
                self._load_cache(cache)

    def _load_cache(self, path: Path) -> None:
        data = json.loads(path.read_text())

        # Backward-compatible cache load:
        # - old format: {"m,h": {"high":..., "low":...}}
        # - new format: {"version":2, "table": {...}, "samples": [...]} 
        if isinstance(data, dict) and "table" in data:
            table = data.get("table", {})
            samples = data.get("samples", [])
        else:
            table = data
            samples = []

        self._table = {
            (int(k.split(",")[0]), int(k.split(",")[1])): v
            for k, v in table.items()
        }
        self._samples = list(samples)

        log.info(
            "Loaded sigma climatology from %s (%d entries, %d samples)",
            path.name,
            len(self._table),
            len(self._samples),
        )

    def _save_cache(self, path: Path) -> None:
        data = {
            "version": 2,
            "table": {f"{m},{h}": v for (m, h), v in self._table.items()},
            "samples": self._samples,
        }
        path.write_text(json.dumps(data, indent=2))
        log.info("Saved sigma climatology to %s", path.name)

    def _build_from_iem(self, center_month: int) -> None:
        """Fetch IEM ASOS data for center_month ± shoulder and compute sigma tables."""
        iem_station = _IEM_STATION.get(self.station, self.station.lstrip("K"))

        # Determine months to fetch (center ± shoulder)
        months = []
        for offset in range(-self.shoulder_months, self.shoulder_months + 1):
            m = (center_month - 1 + offset) % 12 + 1
            months.append(m)

        all_obs: list[tuple[datetime, float]] = []
        now_year = datetime.now().year

        for year in range(now_year - self.years_back, now_year + 1):
            for m in months:
                try:
                    obs = self._fetch_month(iem_station, year, m)
                    all_obs.extend(obs)
                except Exception:
                    log.warning("Failed to fetch %s %d-%02d", iem_station, year, m)

        if not all_obs:
            log.warning("No IEM data retrieved; sigma table will be empty")
            return

        # Reset cache payloads before rebuild.
        self._table = {}
        self._samples = []

        # Group by climate day (midnight-midnight EST)
        day_obs: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for dt, temp in all_obs:
            day_key = dt.strftime("%Y-%m-%d")
            day_obs[day_key].append((dt.hour, temp))

        # Compute remaining upside (high) and downside (low) by hour
        high_remaining: dict[int, list[float]] = defaultdict(list)
        low_remaining: dict[int, list[float]] = defaultdict(list)

        n_days = 0
        for day_key, obs_list in day_obs.items():
            if len(obs_list) < 18:  # skip incomplete days
                continue
            n_days += 1

            final_high = max(t for _, t in obs_list)
            final_low = min(t for _, t in obs_list)

            # Running max/min through each hour
            running_max = float("-inf")
            running_min = float("inf")
            hourly_max: dict[int, float] = {}
            hourly_min: dict[int, float] = {}

            for hour, temp in sorted(obs_list):
                running_max = max(running_max, temp)
                running_min = min(running_min, temp)
                hourly_max[hour] = running_max
                hourly_min[hour] = running_min

            for hour, rmax in hourly_max.items():
                high_remaining[hour].append(final_high - rmax)
            for hour, rmin in hourly_min.items():
                low_remaining[hour].append(rmin - final_low)

            day_dt = datetime.strptime(day_key, "%Y-%m-%d")
            day_doy = day_dt.timetuple().tm_yday
            for hour in range(24):
                if hour not in hourly_max or hour not in hourly_min:
                    continue
                self._samples.append(
                    {
                        "year": day_dt.year,
                        "month": day_dt.month,
                        "doy": day_doy,
                        "hour": hour,
                        "high_remaining": round(final_high - hourly_max[hour], 4),
                        "low_remaining": round(hourly_min[hour] - final_low, 4),
                    }
                )

        # Build the sigma table: stdev of remaining potential at each hour
        for hour in range(24):
            h_vals = high_remaining.get(hour, [])
            l_vals = low_remaining.get(hour, [])

            h_sigma = statistics.stdev(h_vals) if len(h_vals) > 2 else 4.0
            l_sigma = statistics.stdev(l_vals) if len(l_vals) > 2 else 4.0

            self._table[(center_month, hour)] = {
                "high": round(h_sigma, 3),
                "low": round(l_sigma, 3),
            }

        log.info(
            "Built sigma climatology for month %d from %d days (%d obs, %d samples)",
            center_month,
            n_days,
            len(all_obs),
            len(self._samples),
        )

    @staticmethod
    def _fetch_month(station: str, year: int, month: int) -> list[tuple[datetime, float]]:
        """Fetch one month of hourly ASOS data from IEM."""
        import calendar

        last_day = calendar.monthrange(year, month)[1]
        url = _IEM_URL.format(
            station=station,
            y1=year, m1=month,
            y2=year, m2=month,
            last_day=last_day,
        )

        resp = urllib.request.urlopen(url, timeout=30)
        text = resp.read().decode()
        obs = []
        reader = csv.reader(io.StringIO(text))
        for row in reader:
            if len(row) < 3 or row[0].startswith("#") or row[0] == "station":
                continue
            try:
                dt = datetime.strptime(row[1].strip(), "%Y-%m-%d %H:%M")
                temp = float(row[2])
                obs.append((dt, temp))
            except (ValueError, IndexError):
                continue
        return obs
