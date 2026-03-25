"""DS3M Production Orchestrator v2.

Replaces both the old production orchestrator (engine/orchestrator.py)
and the shadow DS3M orchestrator (engine/ds3m/orchestrator.py).

Full pipeline per cycle:
  1. Read latest observation from DB
  2. Build feature vector (33 features)
  3. Mamba encoder step (recurrent, single timestep)
  4. Differentiable PF: predict → update → soft resample
  5. NSF density estimation → bracket probabilities
  6. Conformal calibration
  7. Bracket pricing + microstructure signals
  8. Paper trading evaluation
  9. HDP regime discovery check
  10. Dashboard data push (SSE)
  11. State persistence

Cycle cadence:
  - Full PF cycle: every 5 seconds (on obs_changed)
  - Full Mamba forward: every 60 seconds
  - Real-time KF update: on each METAR (~20 min)
  - Dashboard push: on every cycle + METAR
  - Nightly training: after CLI settlement
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from engine.ds3m.mamba_encoder import MambaEncoder, MambaConfig
from engine.ds3m.diff_particle_filter import (
    DifferentiableParticleFilter, DPFConfig, ParticleCloud,
)
from engine.ds3m.neural_spline_flow import ConditionalNSF, NSFConfig, NSFConditionBuilder
from engine.ds3m.hdp_regime import HDPRegimeManager
from engine.ds3m.feature_vector import FeatureVectorBuilder
from engine.ds3m.bracket_pricer_v2 import BracketPricerV2, Bracket
from engine.ds3m.realtime_updater import RealTimeUpdater, METARSnapshot
from engine.ds3m.conformal_calibrator import ConformalCalibrator
from engine.ds3m.skew_normal import SkewNormal

log = logging.getLogger(__name__)


class DS3MOrchestrator:
    """Production DS3M inference orchestrator.

    Manages the full inference pipeline and coordinates all components.
    """

    def __init__(
        self,
        db_path: str,
        station: str = "KMIA",
        analysis_dir: str = "analysis_data",
        device: str | None = None,
    ):
        self.db_path = db_path
        self.station = station
        self.analysis_dir = Path(analysis_dir)
        self.analysis_dir.mkdir(exist_ok=True)

        # Device selection
        if device:
            self.device = torch.device(device)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        log.info(f"DS3M device: {self.device}")

        # ── Initialize components ──────────────────────────────────
        self.mamba = MambaEncoder(MambaConfig(d_input=33))  # d_model=256, d_state=32, 8 layers
        self.mamba.to(self.device)
        self.mamba.eval()

        self.dpf = DifferentiableParticleFilter(
            DPFConfig(n_particles=500, k_regimes=5),  # d_latent=32
            d_mamba=256,
        )
        self.dpf.to(self.device)
        self.dpf.eval()

        self.nsf = ConditionalNSF(NSFConfig(d_condition=280))  # 19 + 5 + 256
        self.nsf.to(self.device)
        self.nsf.eval()

        self.regime_mgr = HDPRegimeManager()
        self.feature_builder = FeatureVectorBuilder(db_path, station)
        self.bracket_pricer = BracketPricerV2()
        self.realtime_updater = RealTimeUpdater()
        self.nsf_condition = NSFConditionBuilder(k_regimes=5, d_model=256)

        # Load conformal calibrator (existing module)
        try:
            self.conformal = ConformalCalibrator.load(
                self.analysis_dir / "conformal_state.json"
            )
        except Exception:
            self.conformal = ConformalCalibrator()

        # ── State ──────────────────────────────────────────────────
        self.cloud: ParticleCloud | None = None
        self.mamba_states: list[torch.Tensor] | None = None
        self.mamba_h: torch.Tensor | None = None  # latest hidden state

        self.current_date: str | None = None
        self.running_high: float | None = None
        self.running_low: float | None = None
        self.n_cycles: int = 0
        self.last_mamba_forward: float = 0.0

        # Dashboard data (pushed via SSE)
        self.dashboard_data: dict[str, Any] = {}
        self._dashboard_queue: asyncio.Queue | None = None

        # Load saved state if available
        self._load_state()

    # ── Dashboard Integration ─────────────────────────────────────

    def attach_dashboard(self, queue: asyncio.Queue):
        """Attach a dashboard SSE queue to receive cycle data."""
        self._dashboard_queue = queue

    def _get_station_wx_snapshot(self) -> dict:
        """Pull latest weather data for all stations, buoys, and FAWN for the map."""
        result = {}
        try:
            conn = sqlite3.connect(self.db_path, timeout=3)
            conn.row_factory = sqlite3.Row

            # ASOS stations (nearby_observations + KMIA from observations)
            cutoff = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
            rows = conn.execute(
                """SELECT stid, air_temp_f, wind_speed_mph, wind_direction_deg
                   FROM nearby_observations
                   WHERE timestamp_utc > ? AND air_temp_f IS NOT NULL
                   GROUP BY stid HAVING MAX(timestamp_utc)""",
                (cutoff,),
            ).fetchall()
            for r in rows:
                result[r["stid"]] = {
                    "temp": r["air_temp_f"],
                    "wind_speed": r["wind_speed_mph"],
                    "wind_dir": r["wind_direction_deg"],
                }

            # KMIA from main observations table
            kmia = conn.execute(
                """SELECT temperature_f, wind_speed_mph, wind_heading_deg
                   FROM observations WHERE station = 'KMIA' AND timestamp_utc > ?
                   ORDER BY timestamp_utc DESC LIMIT 1""",
                (cutoff,),
            ).fetchone()
            if kmia:
                result["KMIA"] = {
                    "temp": kmia["temperature_f"],
                    "wind_speed": kmia["wind_speed_mph"],
                    "wind_dir": kmia["wind_heading_deg"],
                }

            # NDBC buoys
            buoy_rows = conn.execute(
                """SELECT station_id, water_temp_f, air_temp_c, wind_speed_mps, wind_dir_deg
                   FROM sst_observations WHERE timestamp_utc > ?
                   GROUP BY station_id HAVING MAX(timestamp_utc)""",
                (cutoff,),
            ).fetchall()
            for b in buoy_rows:
                result[b["station_id"]] = {
                    "sst": b["water_temp_f"],
                    "wind_speed": (b["wind_speed_mps"] or 0) * 1.944,  # m/s to kt
                    "wind_dir": b["wind_dir_deg"],
                }

            # FAWN stations
            fawn_rows = conn.execute(
                """SELECT station_id, air_temp_f, wind_speed_mph, wind_direction_deg,
                          solar_radiation_wm2, soil_temp_f
                   FROM fawn_observations WHERE timestamp_utc > ?
                   GROUP BY station_id HAVING MAX(timestamp_utc)""",
                (cutoff,),
            ).fetchall()
            for f in fawn_rows:
                result["FAWN_" + str(f["station_id"])] = {
                    "temp": f["air_temp_f"],
                    "wind_speed": f["wind_speed_mph"],
                    "wind_dir": f["wind_direction_deg"],
                    "solar": f["solar_radiation_wm2"],
                    "soil_temp": f["soil_temp_f"],
                }

            conn.close()
        except Exception as e:
            log.debug(f"Station wx snapshot error: {e}")
        return result

    def _push_dashboard(self, data: dict[str, Any]):
        """Push data to the dashboard queue if attached.

        Uses drop-oldest policy when the queue is full (maxsize=10).
        """
        if not self._dashboard_queue:
            return
        try:
            self._dashboard_queue.put_nowait(data)
        except asyncio.QueueFull:
            try:
                self._dashboard_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._dashboard_queue.put_nowait(data)

    # ── Main Inference Cycle ───────────────────────────────────────

    @torch.no_grad()
    def run_cycle(self) -> dict[str, Any]:
        """Execute one full DS3M inference cycle.

        Called every 5 seconds when new observation data is available.
        Returns dashboard-ready data dict.
        """
        t0 = time.monotonic()

        # 1. Read latest observation
        obs = self._read_latest_obs()
        if obs is None:
            return self.dashboard_data

        # Check for climate day rollover
        target_date = obs.get("target_date", self.current_date)
        if target_date != self.current_date:
            self._handle_day_rollover(target_date)

        temp_f = obs["temp_f"]
        self.running_high = obs.get("running_high", self.running_high or temp_f)
        self.running_low = obs.get("running_low", self.running_low or temp_f)

        # 2. Build feature vector
        features = self.feature_builder.build_current()
        features_tensor = features.unsqueeze(0).to(self.device)  # (1, 33)

        # 3. Mamba encoder step (recurrent)
        now = time.monotonic()
        if self.mamba_states is None:
            self.mamba_states = self.mamba.init_states(1, self.device)

        # Full Mamba forward every 60s, recurrent step otherwise
        if now - self.last_mamba_forward > 60:
            seq = self.feature_builder.get_sequence(max_len=576)  # last 48h
            seq_tensor = seq.unsqueeze(0).to(self.device)  # (1, T, 33)
            h_seq = self.mamba(seq_tensor)  # (1, T, d_model)
            self.mamba_h = h_seq[0, -1, :]  # latest hidden state
            self.last_mamba_forward = now
        else:
            self.mamba_h, self.mamba_states = self.mamba.step(
                features_tensor.squeeze(0), self.mamba_states
            )
            self.mamba_h = self.mamba_h.squeeze(0)  # (d_model,)

        # 4. Differentiable PF cycle
        if self.cloud is None:
            self.cloud = self.dpf.initialize(
                mu_high=5.0, sigma_high=2.0,
                mu_low=3.0, sigma_low=2.0,
                device=self.device,
            )

        obs_remaining_high = torch.tensor(
            max(0, temp_f - (self.running_high or temp_f)),
            device=self.device, dtype=torch.float32,
        )
        obs_remaining_low = torch.tensor(
            max(0, (self.running_low or temp_f) - temp_f),
            device=self.device, dtype=torch.float32,
        )

        source_idx = {"wethr": 0, "nws": 1, "iem": 2}.get(obs.get("source", ""), 0)

        self.cloud = self.dpf.step(
            self.cloud, self.mamba_h,
            obs_remaining_high, obs_remaining_low,
            source_idx, self.regime_mgr.K,
        )

        # 5. NSF density → bracket probabilities
        brackets = self._read_active_brackets()
        regime_posterior = self.cloud.regime_posterior.detach().cpu().numpy()

        nsf_context = self.nsf_condition.build(
            particle_weights=self.cloud.weights.detach().cpu(),
            particle_values=self.cloud.remaining_high.detach().cpu(),
            regime_posterior=self.cloud.regime_posterior.detach().cpu(),
            mamba_h=self.mamba_h.detach().cpu(),
            nwp_mean=float(features[0]),  # hrrr_maxt
            nwp_spread=float(features[10]),  # model_spread
            nwp_bias=0.0,
            hour_local=float(features[22]) * 12 / 3.14159 + 12,  # approx from sin
            day_of_year=80,  # approximate
            lead_time_hours=float(features[26]),
            is_dst=bool(features[27] > 0.5),
            letkf_spread=1.0,
            letkf_n_updates=0,
            running_high=self.running_high or 85.0,
            running_low=self.running_low or 75.0,
            current_temp=temp_f,
        ).to(self.device)

        bracket_tuples = [
            (b.floor_strike, b.ceiling_strike) for b in brackets
        ]
        if bracket_tuples:
            model_probs = self.nsf.all_bracket_probs(bracket_tuples, nsf_context)
            # Convert tensor values to floats
            model_probs = {k: float(v) for k, v in model_probs.items()}
        else:
            model_probs = {}

        # 6. Conformal calibration
        calibrated_probs = {}
        for label, p in model_probs.items():
            calibrated_probs[label] = self.conformal.calibrate(p, "high")

        # 7. Bracket pricing + microstructure
        regime_display = self.regime_mgr.get_regime_display(regime_posterior)
        regime_info = self.regime_mgr.regimes.get(
            int(np.argmax(regime_posterior[:self.regime_mgr.K]))
        )

        signals = self.bracket_pricer.price_all_brackets(
            brackets, calibrated_probs,
            regime_name=regime_display.get("active_regime_name", ""),
            regime_sizing_mult=regime_info.position_sizing_mult if regime_info else 1.0,
            regime_min_edge=regime_info.min_edge_override if regime_info else None,
        )

        # 8. HDP regime discovery check
        avg_ll = float(self.cloud.log_weights.mean())
        new_regime = self.regime_mgr.check_new_regime(avg_ll)
        if new_regime:
            log.info(f"New regime discovered: {new_regime.name}")

        self.regime_mgr.update_cooccurrence(regime_posterior)

        # 9. Write results to DB
        self._write_estimates(calibrated_probs, regime_posterior)

        # 10. Build dashboard data
        self.n_cycles += 1
        cycle_ms = (time.monotonic() - t0) * 1000

        lead_time_hours = float(features[26]) if len(features) > 26 else 0.0

        self.dashboard_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cycle": self.n_cycles,
            "cycle_ms": round(cycle_ms, 1),
            "weather": {
                "temp_f": temp_f,
                "running_high": self.running_high,
                "running_low": self.running_low,
                "dewpoint_f": obs.get("dew_pt_f"),
                "wind_dir": obs.get("wind_dir_deg"),
                "wind_speed": obs.get("wind_speed_kt"),
                "pressure_hpa": obs.get("pressure_hpa"),
                "rel_humidity": obs.get("rel_humidity"),
                "visibility_mi": obs.get("visibility_mi"),
                "cloud_cover": obs.get("cloud_cover"),
                "source": obs.get("source", ""),
            },
            "model": {
                "mu_high": float(self.cloud.weighted_mean_high),
                "sigma_high": float(self.cloud.weighted_std_high),
                "mu_low": float(self.cloud.weighted_mean_low),
                "sigma_low": float(self.cloud.weighted_std_low),
                "ess": float(self.cloud.ess),
                "n_particles": self.cloud.N,
                "lead_time_hours": lead_time_hours,
            },
            "regime": {
                "active_regime": regime_display.get("active_regime", ""),
                "confidence": regime_display.get("confidence", 0.0),
                "color": regime_display.get("color", "#888888"),
                "regime_probs": regime_display.get("regime_probs", {}),
                "is_new_discovery": bool(new_regime),
            },
            "atmosphere": {
                "cape": obs.get("cape"),
                "sst": obs.get("sst"),
            },
            "innovation": {
                "regime_change_signal": bool(new_regime),
                "mean_innovation": float(self.cloud.log_weights.mean()) if self.cloud else 0.0,
                "variance_ratio": float(self.cloud.log_weights.var()) if self.cloud else 0.0,
            },
            "brackets": [
                {
                    "label": b.label,
                    "model_prob": model_probs.get(b.label, 0),
                    "conformal_prob": calibrated_probs.get(b.label, 0),
                    "yes_price": b.yes_price,
                    "no_price": b.no_price,
                    "depth": getattr(b, "yes_liquidity", 0) + getattr(b, "no_liquidity", 0),
                    "edge": calibrated_probs.get(b.label, 0) - b.yes_price,
                }
                for b in brackets
            ],
            "signals": [
                {
                    "bracket": s.bracket.label,
                    "side": s.side,
                    "order_type": s.order_type,
                }
                for s in signals
            ],
            "events": [
                {"type": e.event_type, "confidence": e.confidence, "description": e.description}
                for e in self.realtime_updater.active_events
            ],
            # Station weather data for spatial map (all stations + buoys + FAWN)
            "station_wx": self._get_station_wx_snapshot(),
        }

        # Push to dashboard SSE queue (no-op if not attached)
        self._push_dashboard(self.dashboard_data)

        # 11. State persistence (every 10 cycles)
        if self.n_cycles % 10 == 0:
            self._save_state()

        if self.n_cycles % 60 == 0:
            log.info(
                f"Cycle {self.n_cycles}: ESS={float(self.cloud.ess):.0f}, "
                f"regime={regime_display.get('active_regime', '?')}, "
                f"{len(signals)} signals, {cycle_ms:.0f}ms"
            )

        return self.dashboard_data

    # ── Co-launch with Dashboard ────────────────────────────────────

    @classmethod
    def start_with_dashboard(
        cls,
        db_path: str,
        station: str = "KMIA",
        port: int = 8050,
        cycle_secs: int = 5,
        analysis_dir: str = "analysis_data",
    ):
        """Co-launch the DS3M inference loop and dashboard server.

        Runs uvicorn in a background daemon thread and the inference loop
        on the current thread's asyncio event loop.
        """
        import threading
        from dashboard.app import configure as dashboard_configure, run_dashboard

        queue: asyncio.Queue = asyncio.Queue(maxsize=10)

        orchestrator = cls(db_path, station, analysis_dir)
        orchestrator.attach_dashboard(queue)

        dashboard_configure(db_path, queue)
        dash_thread = threading.Thread(
            target=run_dashboard,
            args=(db_path, queue, port),
            daemon=True,
        )
        dash_thread.start()
        log.info(f"Dashboard started: http://localhost:{port}")

        async def _loop():
            while True:
                try:
                    data = orchestrator.run_cycle()
                except Exception as e:
                    log.error(f"DS3M cycle error: {e}", exc_info=True)
                await asyncio.sleep(cycle_secs)

        asyncio.run(_loop())

    # ── METAR Update (inter-cycle) ─────────────────────────────────

    def handle_metar(self, metar: METARSnapshot, hour_local: float, is_dst: bool,
                     wind_925_speed: float = 0, wind_925_dir: float = 0):
        """Process a new METAR observation between full PF cycles.

        Applies the regime-conditioned KF to update distribution params
        and runs weather event detectors (sea breeze, outflow, etc.).
        """
        if self.cloud is None:
            return

        # Get regime posterior for soft KF conditioning
        regime_posterior = self.cloud.regime_posterior.detach().cpu().numpy()
        regime_display = self.regime_mgr.get_regime_display(regime_posterior)
        active_regime = regime_display.get("active_regime_name", "continental")

        # Build regime_probs dict for soft blending
        regime_probs: dict[str, float] = {}
        for idx in range(min(self.regime_mgr.K, len(regime_posterior))):
            rinfo = self.regime_mgr.regimes.get(idx)
            if rinfo:
                regime_probs[rinfo.name] = float(regime_posterior[idx])

        mu = float(self.cloud.weighted_mean_high)
        sigma = float(self.cloud.weighted_std_high)

        # Predicted temp at this hour (from particle cloud mean + running high)
        predicted_temp = (self.running_high or 85.0) + mu

        mu_new, sigma_new, alpha_new, events = self.realtime_updater.update(
            metar=metar,
            ds3m_mu=mu,
            ds3m_sigma=sigma,
            ds3m_alpha=0.0,
            predicted_hourly_temp=predicted_temp,
            hour_local=hour_local,
            is_dst=is_dst,
            regime_name=active_regime,
            regime_probs=regime_probs,
            wind_925_speed=wind_925_speed,
            wind_925_dir=wind_925_dir,
        )

        if events:
            log.info(f"METAR update: {len(events)} events, mu {mu:.2f}→{mu_new:.2f}, "
                     f"sigma {sigma:.2f}→{sigma_new:.2f}")

    # ── Day Rollover ───────────────────────────────────────────────

    def _handle_day_rollover(self, new_date: str):
        """Reset state for new climate day. Trigger nightly training."""
        log.info(f"Climate day rollover: {self.current_date} → {new_date}")

        # TODO: trigger nightly training here
        # from engine.ds3m.training_pipeline import DS3MTrainer
        # trainer = DS3MTrainer(self.mamba, self.dpf, self.nsf, self.db_path)
        # trainer.run_nightly_training()

        self.current_date = new_date
        self.running_high = None
        self.running_low = None
        self.cloud = None  # reinitialize particles
        self.mamba_states = None

    # ── DB Queries ─────────────────────────────────────────────────

    def _read_latest_obs(self) -> dict | None:
        try:
            conn = sqlite3.connect(self.db_path, timeout=3)
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """SELECT temp_f, dew_pt_f, wind_dir_deg, wind_speed_kt,
                          wethr_high_nws_f, wethr_low_nws_f, source, timestamp_utc
                   FROM observations
                   WHERE station = ?
                   ORDER BY timestamp_utc DESC LIMIT 1""",
                (self.station,),
            ).fetchone()
            conn.close()
            if row is None:
                return None
            d = dict(row)
            d["running_high"] = d.get("wethr_high_nws_f")
            d["running_low"] = d.get("wethr_low_nws_f")
            # Derive target date from timestamp
            ts = d.get("timestamp_utc", "")
            if ts:
                d["target_date"] = ts[:10]
            return d
        except Exception as e:
            log.warning(f"DB read error: {e}")
            return None

    def _read_active_brackets(self) -> list[Bracket]:
        try:
            conn = sqlite3.connect(self.db_path, timeout=3)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT ticker, event_ticker, market_type,
                          floor_strike, cap_strike, yes_price, no_price
                   FROM active_brackets
                   WHERE station = ?""",
                (self.station,),
            ).fetchall()
            conn.close()
            brackets = []
            for r in rows:
                d = dict(r)
                floor_s = d.get("floor_strike")
                cap_s = d.get("cap_strike")
                brackets.append(Bracket(
                    ticker=d["ticker"],
                    event_ticker=d.get("event_ticker", ""),
                    market_type=d.get("market_type", "high"),
                    floor_strike=float(floor_s) if floor_s else -1e6,
                    ceiling_strike=float(cap_s) if cap_s else 1e6,
                    directional="over" if cap_s is None else ("under" if floor_s is None else "range"),
                    yes_price=float(d.get("yes_price", 0)) / 100.0 if d.get("yes_price") else 0,
                    no_price=float(d.get("no_price", 0)) / 100.0 if d.get("no_price") else 0,
                ))
            return brackets
        except Exception as e:
            log.warning(f"Bracket read error: {e}")
            return []

    def _write_estimates(self, probs: dict, regime_posterior: np.ndarray):
        try:
            conn = sqlite3.connect(self.db_path, timeout=3)
            ts = datetime.now(timezone.utc).isoformat()
            for label, prob in probs.items():
                conn.execute(
                    """INSERT OR REPLACE INTO bracket_estimates
                       (station, target_date, ticker, probability, regime_posterior,
                        ess, timestamp_utc, source)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 'ds3m_v2')""",
                    (
                        self.station, self.current_date or "",
                        label, prob,
                        json.dumps({str(k): float(regime_posterior[k])
                                    for k in range(len(regime_posterior))}),
                        float(self.cloud.ess) if self.cloud else 0,
                        ts,
                    ),
                )
            conn.commit()
            conn.close()
        except Exception as e:
            log.warning(f"DB write error: {e}")

    # ── State Persistence ──────────────────────────────────────────

    def _save_state(self):
        state = {
            "n_cycles": self.n_cycles,
            "current_date": self.current_date,
            "running_high": self.running_high,
            "running_low": self.running_low,
            "regime_manager": self.regime_mgr.to_dict(),
            "realtime_kf": self.realtime_updater.kf.to_dict(),
        }
        # Save particle cloud state
        if self.cloud is not None:
            state["cloud"] = {
                "remaining_high": self.cloud.remaining_high.detach().cpu().tolist(),
                "remaining_low": self.cloud.remaining_low.detach().cpu().tolist(),
                "z": self.cloud.z.detach().cpu().tolist(),
                "regime_logits": self.cloud.regime_logits.detach().cpu().tolist(),
                "log_weights": self.cloud.log_weights.detach().cpu().tolist(),
            }
        # Save Mamba encoder weights
        torch.save(self.mamba.state_dict(), self.analysis_dir / "mamba_live.pt")
        torch.save(self.dpf.state_dict(), self.analysis_dir / "dpf_live.pt")
        torch.save(self.nsf.state_dict(), self.analysis_dir / "nsf_live.pt")

        with open(self.analysis_dir / "ds3m_v2_state.json", "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        state_path = self.analysis_dir / "ds3m_v2_state.json"
        if not state_path.exists():
            log.info("No saved state — starting fresh")
            return

        try:
            with open(state_path) as f:
                state = json.load(f)

            self.n_cycles = state.get("n_cycles", 0)
            self.current_date = state.get("current_date")
            self.running_high = state.get("running_high")
            self.running_low = state.get("running_low")

            if "regime_manager" in state:
                self.regime_mgr = HDPRegimeManager.from_dict(state["regime_manager"])

            if "realtime_kf" in state:
                self.realtime_updater.kf.from_dict(state["realtime_kf"])

            if "cloud" in state:
                c = state["cloud"]
                self.cloud = ParticleCloud(
                    remaining_high=torch.tensor(c["remaining_high"], device=self.device),
                    remaining_low=torch.tensor(c["remaining_low"], device=self.device),
                    z=torch.tensor(c["z"], device=self.device),
                    regime_logits=torch.tensor(c["regime_logits"], device=self.device),
                    log_weights=torch.tensor(c["log_weights"], device=self.device),
                )

            # Load model weights
            for name, model in [("mamba_live.pt", self.mamba),
                                ("dpf_live.pt", self.dpf),
                                ("nsf_live.pt", self.nsf)]:
                path = self.analysis_dir / name
                if path.exists():
                    model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))

            log.info(f"Restored state: cycle={self.n_cycles}, date={self.current_date}")
        except Exception as e:
            log.warning(f"State load error: {e} — starting fresh")


# ──────────────────────────────────────────────────────────────────────
# Async runner (integrates with collector's event loop)
# ──────────────────────────────────────────────────────────────────────

async def ds3m_production_loop(
    db_path: str,
    station: str = "KMIA",
    cycle_secs: int = 5,
    obs_changed_event: asyncio.Event | None = None,
    dashboard_queue: asyncio.Queue | None = None,
):
    """Main async loop for DS3M production inference.

    Integrates with the collector's event loop via obs_changed_event.
    Pushes dashboard data to the SSE queue.
    """
    orchestrator = DS3MOrchestrator(db_path, station)
    if dashboard_queue:
        orchestrator.attach_dashboard(dashboard_queue)
    log.info("DS3M production loop started")

    while True:
        try:
            if obs_changed_event:
                # Wait for new observation, with timeout
                try:
                    await asyncio.wait_for(obs_changed_event.wait(), timeout=cycle_secs)
                    obs_changed_event.clear()
                except asyncio.TimeoutError:
                    pass
            else:
                await asyncio.sleep(cycle_secs)

            data = orchestrator.run_cycle()

            if dashboard_queue and data:
                try:
                    dashboard_queue.put_nowait(data)
                except asyncio.QueueFull:
                    # Drop oldest if queue full
                    try:
                        dashboard_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    dashboard_queue.put_nowait(data)

        except Exception as e:
            log.error(f"DS3M cycle error: {e}", exc_info=True)
            await asyncio.sleep(cycle_secs)


def main():
    """CLI entrypoint: run DS3M v2 + dashboard together."""
    import argparse
    import threading

    from dashboard import configure as dashboard_configure, run_dashboard

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-24s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="DS3M v2 Production Inference")
    parser.add_argument("--db", default="miami_collector.db", help="SQLite database path")
    parser.add_argument("--station", default="KMIA")
    parser.add_argument("--port", type=int, default=8050, help="Dashboard port")
    parser.add_argument("--cycle", type=int, default=5, help="Cycle interval (seconds)")
    args = parser.parse_args()

    queue: asyncio.Queue = asyncio.Queue(maxsize=50)

    # Start dashboard in a background thread
    dashboard_configure(args.db, queue)
    dash_thread = threading.Thread(
        target=run_dashboard,
        args=(args.db, queue, args.port),
        daemon=True,
    )
    dash_thread.start()
    log.info(f"Dashboard: http://localhost:{args.port}")

    # Run DS3M loop
    asyncio.run(ds3m_production_loop(
        db_path=args.db,
        station=args.station,
        cycle_secs=args.cycle,
        dashboard_queue=queue,
    ))
