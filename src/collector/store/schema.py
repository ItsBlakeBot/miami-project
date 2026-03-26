"""SQLite schema definitions — all CREATE TABLE DDL."""

SCHEMA_VERSION = "15"

TABLES = [
    # -- Schema version tracking --
    """
    CREATE TABLE IF NOT EXISTS schema_meta (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """,

    # -- Model forecasts (daily high/low + hourly point forecasts) --
    """
    CREATE TABLE IF NOT EXISTS model_forecasts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        station TEXT NOT NULL,
        forecast_date TEXT NOT NULL,
        model TEXT NOT NULL,
        source TEXT NOT NULL,
        run_time TEXT,
        valid_time TEXT,
        forecast_high_f REAL,
        forecast_low_f REAL,
        raw_temperature_f REAL,
        run_age_hours REAL,
        fetch_time_utc TEXT,
        source_record_json TEXT,
        created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    )
    """,

    # -- Surface observations (1-min from Wethr SSE, NWS fallback) --
    # UNIQUE on (station, source, timestamp_utc) prevents duplicate NWS obs
    # while allowing Wethr 1-min obs at the same timestamps.
    """
    CREATE TABLE IF NOT EXISTS observations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        station TEXT NOT NULL,
        timestamp_utc TEXT NOT NULL,
        lst_date TEXT NOT NULL,
        temperature_f REAL,
        dew_point_f REAL,
        relative_humidity REAL,
        wind_speed_mph REAL,
        wind_direction TEXT,
        wind_gust_mph REAL,
        wind_heading_deg REAL,
        visibility_miles REAL,
        sky_cover_pct REAL,
        sky_cover_code TEXT,
        pressure_hpa REAL,
        precipitation_last_hour_mm REAL,
        wethr_high_f REAL,
        wethr_low_f REAL,
        wethr_high_nws_f REAL,
        wethr_low_nws_f REAL,
        source TEXT,
        created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        UNIQUE(station, source, timestamp_utc)
    )
    """,

    # -- Pressure levels (925/850/700/500 hPa from Open-Meteo GFS) --
    """
    CREATE TABLE IF NOT EXISTS pressure_levels (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        station TEXT NOT NULL,
        valid_time_utc TEXT NOT NULL,
        model TEXT NOT NULL,
        fetch_time_utc TEXT,
        temp_925_c REAL,
        wind_speed_925 REAL,
        wind_dir_925 REAL,
        geopotential_925 REAL,
        temp_850_c REAL,
        wind_speed_850 REAL,
        wind_dir_850 REAL,
        geopotential_850 REAL,
        temp_700_c REAL,
        wind_speed_700 REAL,
        wind_dir_700 REAL,
        geopotential_700 REAL,
        rh_850 REAL,
        rh_700 REAL,
        temp_500_c REAL,
        wind_speed_500 REAL,
        wind_dir_500 REAL,
        created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    )
    """,

    # -- Kalshi market snapshots (bracket pricing over time) --
    # Only written when best bid/ask prices actually change.
    """
    CREATE TABLE IF NOT EXISTS market_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        event_ticker TEXT,
        series_ticker TEXT,
        market_type TEXT NOT NULL,
        forecast_date TEXT,
        floor_strike REAL,
        cap_strike REAL,
        best_yes_bid_cents REAL,
        best_yes_ask_cents REAL,
        best_no_bid_cents REAL,
        best_no_ask_cents REAL,
        last_price_cents REAL,
        volume INTEGER,
        yes_bid_qty INTEGER,
        yes_ask_qty INTEGER,
        no_bid_qty INTEGER,
        no_ask_qty INTEGER,
        total_yes_depth INTEGER,
        total_no_depth INTEGER,
        spread_cents INTEGER,
        num_yes_levels INTEGER,
        num_no_levels INTEGER,
        snapshot_time TEXT NOT NULL,
        created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    )
    """,

    # -- Event settlements (CLI high/low — uses climate day, NOT issue date) --
    """
    CREATE TABLE IF NOT EXISTS event_settlements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        station TEXT NOT NULL,
        settlement_date TEXT NOT NULL,
        market_type TEXT NOT NULL,
        actual_value_f REAL,
        settlement_source TEXT,
        raw_text TEXT,
        received_at TEXT,
        created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        UNIQUE(station, settlement_date, market_type)
    )
    """,

    # -- Market settlements (which bracket won YES/NO) --
    """
    CREATE TABLE IF NOT EXISTS market_settlements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL UNIQUE,
        event_ticker TEXT,
        series_ticker TEXT,
        station TEXT,
        forecast_date TEXT,
        market_type TEXT,
        floor_strike REAL,
        cap_strike REAL,
        winning_side TEXT NOT NULL,
        settled_at TEXT,
        created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    )
    """,

    # -- Model scores (HIGH and LOW scored independently) --
    """
    CREATE TABLE IF NOT EXISTS model_scores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        station TEXT NOT NULL,
        model TEXT NOT NULL,
        score_date TEXT NOT NULL,
        market_type TEXT NOT NULL,
        mae REAL,
        bias REAL,
        rmse REAL,
        sample_count INTEGER,
        window_days INTEGER,
        created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    )
    """,

    # -- Diurnal scores (every model valid_time vs nearest obs) --
    """
    CREATE TABLE IF NOT EXISTS diurnal_scores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        station TEXT NOT NULL,
        model TEXT NOT NULL,
        forecast_date TEXT NOT NULL,
        valid_time_utc TEXT NOT NULL,
        valid_hour_local INTEGER,
        predicted_f REAL,
        observed_f REAL,
        error_f REAL,
        obs_timestamp_utc TEXT,
        created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    )
    """,

    # -- Cloud impact tracking (obs vs forecast sky cover + temp error) --
    """
    CREATE TABLE IF NOT EXISTS cloud_obs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        station TEXT NOT NULL,
        obs_time TEXT NOT NULL,
        forecast_date TEXT NOT NULL,
        obs_sky_pct REAL,
        obs_sky_code TEXT,
        forecast_sky_pct REAL,
        cloud_discrepancy_pct REAL,
        temp_error_f REAL,
        created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    )
    """,

    # -- Nearby station observations (IEM / Synoptic 5-min HF data) --
    # Spatial lead indicators: temp/wind systems approaching KMIA
    # show up at surrounding stations before reaching the airport.
    # UNIQUE on (stid, timestamp_utc) prevents duplicates when polling
    # faster than the 5-min ASOS reporting interval.
    """
    CREATE TABLE IF NOT EXISTS nearby_observations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        stid TEXT NOT NULL,
        name TEXT,
        network TEXT,
        latitude REAL,
        longitude REAL,
        distance_mi REAL,
        elevation_m REAL,
        timestamp_utc TEXT NOT NULL,
        lst_date TEXT,
        air_temp_f REAL,
        dew_point_f REAL,
        wind_speed_mph REAL,
        wind_direction_deg REAL,
        wind_gust_mph REAL,
        pressure_slp_hpa REAL,
        sky_cover_code TEXT,
        temp_delta_vs_kmia REAL,
        created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        UNIQUE(stid, timestamp_utc)
    )
    """,

    # -- Nearby station registry (discovered once, updated periodically) --
    """
    CREATE TABLE IF NOT EXISTS nearby_stations (
        stid TEXT PRIMARY KEY,
        name TEXT,
        network TEXT,
        latitude REAL,
        longitude REAL,
        distance_mi REAL,
        elevation_m REAL,
        bearing_deg REAL,
        is_active INTEGER DEFAULT 1,
        discovered_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        last_seen_at TEXT
    )
    """,

    # -- Atmospheric parameters (shortwave radiation, CAPE, BL height, PW, soil temp) --
    """
    CREATE TABLE IF NOT EXISTS atmospheric_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        station TEXT NOT NULL,
        valid_time_utc TEXT NOT NULL,
        model TEXT NOT NULL,
        fetch_time_utc TEXT,
        shortwave_radiation REAL,
        direct_radiation REAL,
        diffuse_radiation REAL,
        cape REAL,
        lifted_index REAL,
        boundary_layer_height REAL,
        precipitable_water_mm REAL,
        soil_temperature_0_7cm REAL,
        soil_moisture_0_1cm REAL,
        precipitation_mm REAL,
        rain_mm REAL,
        showers_mm REAL,
        precipitation_probability REAL,
        created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    )
    """,

    # -- Sea surface temperature observations (NDBC buoys) --
    """
    CREATE TABLE IF NOT EXISTS sst_observations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        station_id TEXT NOT NULL,
        name TEXT,
        timestamp_utc TEXT NOT NULL,
        water_temp_c REAL,
        water_temp_f REAL,
        air_temp_c REAL,
        wind_speed_mps REAL,
        wind_dir_deg REAL,
        pressure_hpa REAL,
        distance_mi REAL,
        created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        UNIQUE(station_id, timestamp_utc)
    )
    """,

    # -- FAWN observations (15-min actual sensor data from Homestead) --
    # Ground truth for solar radiation, precipitation, and soil temp
    # vs Open-Meteo forecast estimates.
    """
    CREATE TABLE IF NOT EXISTS fawn_observations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        station_id TEXT NOT NULL,
        station_name TEXT,
        timestamp_utc TEXT NOT NULL,
        air_temp_f REAL,
        air_temp_c REAL,
        dew_point_f REAL,
        relative_humidity REAL,
        wind_speed_mph REAL,
        wind_gust_mph REAL,
        wind_direction_deg REAL,
        solar_radiation_wm2 REAL,
        soil_temp_c REAL,
        soil_temp_f REAL,
        rain_mm REAL,
        rain_in REAL,
        created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        UNIQUE(station_id, timestamp_utc)
    )
    """,

    # -- Raw SSE event log --
    """
    CREATE TABLE IF NOT EXISTS sse_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        station TEXT NOT NULL,
        event_type TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        received_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    )
    """,

    # -- Bias-adjusted forecasts (per-model detail + weighted consensus) --
    """
    CREATE TABLE IF NOT EXISTS model_consensus (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        station TEXT NOT NULL,
        forecast_date TEXT NOT NULL,
        market_type TEXT NOT NULL,
        model TEXT NOT NULL,
        source TEXT NOT NULL,
        run_time TEXT,
        run_age_hours REAL,
        raw_forecast_f REAL,
        bias REAL,
        forecast_f REAL,
        mae REAL,
        skill_weight REAL,
        decay_factor REAL,
        final_weight REAL,
        sample_count INTEGER,
        consensus_forecast_f REAL,
        consensus_std_f REAL,
        n_models INTEGER,
        window_days INTEGER,
        run_time_utc TEXT,
        created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    )
    """,

    # -- Forward curves (multi-model hourly forecasts for learning) --
    """
    CREATE TABLE IF NOT EXISTS forward_curves (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        station TEXT NOT NULL,
        snapshot_time_utc TEXT NOT NULL,
        target_date TEXT NOT NULL,
        valid_hour_utc TEXT NOT NULL,
        hours_ahead INTEGER,
        nbm_temp_f REAL,
        gfs_temp_f REAL,
        ecmwf_temp_f REAL,
        hrrr_temp_f REAL,
        nam_temp_f REAL,
        model_min_f REAL,
        model_max_f REAL,
        model_spread_f REAL,
        cape REAL,
        pw_mm REAL,
        precip_prob REAL,
        precip_mm REAL,
        solar_wm2 REAL,
        temp_850_c REAL,
        temp_925_c REAL,
        wind_dir_850 REAL,
        wind_speed_850 REAL,
        created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    )
    """,

    # -- Collection run log (health monitoring) --
    """
    CREATE TABLE IF NOT EXISTS collection_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        collector TEXT NOT NULL,
        started_at TEXT NOT NULL,
        completed_at TEXT,
        status TEXT NOT NULL,
        records_collected INTEGER DEFAULT 0,
        error_text TEXT,
        created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    )
    """,

    # -- Signal events (logged each estimator run — graded after settlement) --
    # Captures what regime was detected, what signals fired, what adjustments
    # were applied, and the atmospheric conditions at the time. After the
    # climate day settles, the scorer compares our estimate vs actual to grade
    # each signal's contribution.
    """
    CREATE TABLE IF NOT EXISTS signal_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        station TEXT NOT NULL,
        target_date TEXT NOT NULL,
        market_type TEXT NOT NULL,
        timestamp_utc TEXT NOT NULL,

        -- Regime classification
        regime TEXT NOT NULL DEFAULT 'normal',

        -- What we estimated
        mu REAL NOT NULL,
        sigma REAL NOT NULL,
        forward_curve_mu REAL,
        forward_curve_sigma REAL,

        -- Signal contributions (JSON: {signal_name: shift_value})
        signal_adjustments TEXT,

        -- Atmospheric context at time of estimate
        obs_temp_f REAL,
        wind_dir_deg REAL,
        wind_speed_mph REAL,
        wind_shift_1hr_deg REAL,
        pressure_hpa REAL,
        pressure_change_1hr REAL,
        dew_point_f REAL,
        dew_change_30min REAL,
        cape REAL,
        cape_change_1hr REAL,
        temp_925_c REAL,
        temp_850_c REAL,
        wind_dir_850 REAL,

        -- Model tracking state
        best_tracking_model TEXT,
        best_tracking_error REAL,
        worst_tracking_error REAL,
        n_models_diverged INTEGER,

        created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    )
    """,

    # -- Signal scores (graded after settlement — one row per signal per day) --
    # Links a signal event to the actual outcome so we can evaluate
    # whether each signal's contribution improved or hurt the estimate.
    """
    CREATE TABLE IF NOT EXISTS signal_scores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        station TEXT NOT NULL,
        target_date TEXT NOT NULL,
        market_type TEXT NOT NULL,

        -- Actual outcome
        actual_value_f REAL NOT NULL,

        -- Our estimates at various stages
        forward_curve_mu REAL,
        final_mu REAL,
        final_sigma REAL,

        -- Regime that was active
        regime TEXT,

        -- Per-signal grading (JSON: {signal: {shift, helped: bool, error_contribution}})
        signal_grades TEXT,

        -- Summary
        forward_curve_error REAL,
        final_error REAL,
        signal_net_impact REAL,
        signals_helped INTEGER,

        created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        UNIQUE(station, target_date, market_type)
    )
    """,

    # -- Signal calibration (learned parameters — updated after each scored day) --
    # As we accumulate scored events, the calibrator adjusts signal parameters.
    # Each row is a point-in-time snapshot of the learned parameters.
    """
    CREATE TABLE IF NOT EXISTS signal_calibration (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        station TEXT NOT NULL,
        market_type TEXT NOT NULL,
        regime TEXT NOT NULL,
        signal_name TEXT NOT NULL,

        -- Learned parameters
        mean_shift REAL,
        std_shift REAL,
        hit_rate REAL,
        sample_count INTEGER,
        last_5_errors TEXT,

        updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        UNIQUE(station, market_type, regime, signal_name)
    )
    """,

    # -- Active brackets (written by trader, read by city estimator) --
    # The trader discovers available Kalshi brackets and pushes
    # their definitions here so the city can assign probabilities.
    """
    CREATE TABLE IF NOT EXISTS active_brackets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        market_type TEXT NOT NULL,
        target_date TEXT NOT NULL,
        floor_strike REAL,
        cap_strike REAL,
        settlement_floor REAL NOT NULL,
        settlement_ceil REAL NOT NULL,
        updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        UNIQUE(ticker)
    )
    """,

    # -- Bracket estimates (written by city estimator, read by trader) --
    # The city's probability assignment for each active bracket.
    # This is the sole interface between city and trader.
    """
    CREATE TABLE IF NOT EXISTS bracket_estimates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        station TEXT NOT NULL,
        target_date TEXT NOT NULL,
        market_type TEXT NOT NULL,
        ticker TEXT NOT NULL,
        probability REAL NOT NULL,
        mu REAL,
        sigma REAL,
        active_signals TEXT,
        timestamp_utc TEXT NOT NULL,
        regime_confidence REAL,
        recommended_side TEXT,
        recommended_contracts INTEGER,
        recommended_price_cents REAL,
        recommended_probability REAL,
        recommended_edge_cents REAL,
        recommended_ev_cents REAL,
        sizing_json TEXT,
        UNIQUE(ticker, timestamp_utc)
    )
    """,

    # -- Regime labels (AI or human-produced daily regime classification) --
    """
    CREATE TABLE IF NOT EXISTS regime_labels (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        station TEXT NOT NULL,
        target_date TEXT NOT NULL,
        regimes_active TEXT NOT NULL,
        path_class TEXT,
        confidence_tags TEXT,
        phase_summary TEXT,
        model_performance TEXT,
        signal_labels TEXT,
        signal_families_active TEXT,
        threshold_recommendations TEXT,
        review_path TEXT,
        review_source TEXT DEFAULT 'ai',
        created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        UNIQUE(station, target_date)
    )
    """,

    # -- HDP-Sticky shadow regime discovery (comparison/test table) --
    """
    CREATE TABLE IF NOT EXISTS regime_labels_hdp_test (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        station TEXT NOT NULL,
        target_date TEXT NOT NULL,
        n_regimes INTEGER NOT NULL,
        regime_sequence TEXT,
        regime_params TEXT,
        transition_matrix TEXT,
        phase_summary TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(station, target_date)
    )
    """,

    # -- Paper trading entries / exits --
    """
    CREATE TABLE IF NOT EXISTS paper_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        station TEXT NOT NULL,
        target_date TEXT NOT NULL,
        market_type TEXT NOT NULL,
        side TEXT NOT NULL,
        contracts INTEGER NOT NULL DEFAULT 1,
        entry_price_cents REAL NOT NULL,
        entry_time TEXT NOT NULL,
        estimated_probability REAL,
        expected_edge_cents REAL,
        expected_value_cents REAL,
        thesis_json TEXT,
        status TEXT NOT NULL DEFAULT 'open',
        exit_price_cents REAL,
        exit_time TEXT,
        exit_reason TEXT,
        realized_pnl_cents REAL,
        created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    )
    """,

    # -- Paper trade marks (mark-to-market / diagnostics over time) --
    """
    CREATE TABLE IF NOT EXISTS paper_trade_marks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_id INTEGER NOT NULL,
        ticker TEXT NOT NULL,
        mark_time TEXT NOT NULL,
        mark_price_cents REAL,
        estimated_probability REAL,
        expected_value_cents REAL,
        note TEXT,
        created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        FOREIGN KEY(trade_id) REFERENCES paper_trades(id)
    )
    """,

    # -- Paper trade settlements --
    """
    CREATE TABLE IF NOT EXISTS paper_trade_settlements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_id INTEGER NOT NULL UNIQUE,
        ticker TEXT NOT NULL,
        winning_side TEXT NOT NULL,
        settlement_price_cents REAL NOT NULL,
        settled_at TEXT,
        realized_pnl_cents REAL,
        created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        FOREIGN KEY(trade_id) REFERENCES paper_trades(id)
    )
    """,

    # -- DS3M estimates (shadow particle filter bracket probabilities) --
    """
    CREATE TABLE IF NOT EXISTS ds3m_estimates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        station TEXT NOT NULL,
        target_date TEXT NOT NULL,
        market_type TEXT NOT NULL,
        ticker TEXT NOT NULL,
        probability REAL NOT NULL,
        raw_probability REAL,
        conformal_probability REAL,
        mu REAL,
        sigma REAL,
        regime_posterior TEXT,
        n_particles INTEGER,
        ess REAL,
        timestamp_utc TEXT NOT NULL,
        UNIQUE(ticker, timestamp_utc)
    )
    """,

    # -- DS3M comparison (shadow vs production vs market vs actual) --
    """
    CREATE TABLE IF NOT EXISTS ds3m_comparison (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        station TEXT NOT NULL,
        target_date TEXT NOT NULL,
        ticker TEXT NOT NULL,
        production_probability REAL,
        ds3m_probability REAL,
        market_price_cents REAL,
        ds3m_edge REAL,
        production_edge REAL,
        actual_outcome INTEGER,
        production_crps REAL,
        ds3m_crps REAL,
        timestamp_utc TEXT NOT NULL
    )
    """,

    # -- DS3M training log (transition matrices, dynamics, metrics) --
    """
    CREATE TABLE IF NOT EXISTS ds3m_training_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        station TEXT NOT NULL,
        training_date TEXT NOT NULL,
        transition_matrix_json TEXT,
        dynamics_json TEXT,
        training_metrics_json TEXT,
        created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    )
    """,

    # -- DS3M paper trades (shadow trading decisions from DS3M probabilities) --
    """
    CREATE TABLE IF NOT EXISTS ds3m_paper_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        station TEXT NOT NULL,
        target_date TEXT NOT NULL,
        market_type TEXT NOT NULL,
        side TEXT NOT NULL,
        contracts INTEGER NOT NULL DEFAULT 1,
        entry_price_cents REAL NOT NULL,
        entry_time TEXT NOT NULL,
        ds3m_probability REAL,
        conformal_probability REAL,
        expected_edge_cents REAL,
        expected_value_cents REAL,
        regime_posterior TEXT,
        ess REAL,
        status TEXT NOT NULL DEFAULT 'open',
        exit_price_cents REAL,
        exit_time TEXT,
        exit_reason TEXT,
        realized_pnl_cents REAL,
        created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        strike_label TEXT
    )
    """,

    # -- DS3M paper trade marks (shadow mark-to-market over time) --
    """
    CREATE TABLE IF NOT EXISTS ds3m_paper_trade_marks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_id INTEGER NOT NULL,
        ticker TEXT NOT NULL,
        mark_time TEXT NOT NULL,
        mark_price_cents REAL,
        estimated_probability REAL,
        expected_value_cents REAL,
        note TEXT,
        created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        FOREIGN KEY(trade_id) REFERENCES ds3m_paper_trades(id)
    )
    """,

    # -- DS3M paper trade settlements (mirrors paper_trade_settlements) --
    """
    CREATE TABLE IF NOT EXISTS ds3m_paper_trade_settlements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_id INTEGER NOT NULL,
        ticker TEXT NOT NULL,
        winning_side TEXT NOT NULL,
        settlement_price_cents REAL NOT NULL,
        settled_at TEXT,
        realized_pnl_cents REAL,
        created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        FOREIGN KEY(trade_id) REFERENCES ds3m_paper_trades(id)
    )
    """,

    # -- RTMA-RU observations (15-min analyzed surface fields, 2.5km grid) --
    # Stores both the KMIA center point and surrounding 5x5 grid for
    # spatial context (urban heat island, land-sea breeze structure).
    """
    CREATE TABLE IF NOT EXISTS rtma_ru_observations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp_utc TEXT NOT NULL,
        lat REAL NOT NULL,
        lon REAL NOT NULL,
        temperature_2m REAL,
        dewpoint_2m REAL,
        wind_speed_10m REAL,
        wind_direction_10m REAL,
        surface_pressure REAL,
        wind_gust_10m REAL,
        cloud_cover_pct REAL,
        visibility_m REAL,
        created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        UNIQUE(timestamp_utc, lat, lon)
    )
    """,
]

VIEWS = [
    # -- Hourly aggregation of 1-min surface observations --
    # Used by train_skf.py for regime fitting.  Averages temperature,
    # dew point, pressure, and circular-mean wind direction per hour.
    """
    CREATE VIEW IF NOT EXISTS hourly_obs AS
    SELECT
        station,
        lst_date                                          AS target_date,
        CAST(strftime('%H', timestamp_utc) AS INTEGER)    AS hour_utc,
        ROUND(AVG(temperature_f), 2)                      AS temp_f,
        ROUND(AVG(dew_point_f), 2)                        AS dew_f,
        ROUND(AVG(pressure_hpa), 2)                       AS pressure_hpa,
        ROUND(
            CASE
                WHEN DEGREES(ATAN2(
                    AVG(SIN(RADIANS(wind_heading_deg))),
                    AVG(COS(RADIANS(wind_heading_deg)))
                )) < 0
                THEN DEGREES(ATAN2(
                    AVG(SIN(RADIANS(wind_heading_deg))),
                    AVG(COS(RADIANS(wind_heading_deg)))
                )) + 360
                ELSE DEGREES(ATAN2(
                    AVG(SIN(RADIANS(wind_heading_deg))),
                    AVG(COS(RADIANS(wind_heading_deg)))
                ))
            END, 1)                                       AS wind_dir
    FROM observations
    WHERE temperature_f IS NOT NULL
    GROUP BY station, lst_date, CAST(strftime('%H', timestamp_utc) AS INTEGER)
    """,
]

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_fcst_station_date ON model_forecasts(station, forecast_date)",
    "CREATE INDEX IF NOT EXISTS idx_fcst_model_date ON model_forecasts(model, forecast_date)",
    "CREATE INDEX IF NOT EXISTS idx_fcst_valid_time ON model_forecasts(station, model, valid_time)",
    "CREATE INDEX IF NOT EXISTS idx_fcst_dedupe_lookup ON model_forecasts(station, forecast_date, model, source, run_time, valid_time, id DESC)",
    "CREATE INDEX IF NOT EXISTS idx_obs_station_date ON observations(station, lst_date)",
    "CREATE INDEX IF NOT EXISTS idx_obs_timestamp ON observations(station, timestamp_utc)",
    "CREATE INDEX IF NOT EXISTS idx_plev_station_time ON pressure_levels(station, valid_time_utc)",
    "CREATE INDEX IF NOT EXISTS idx_plev_dedupe_lookup ON pressure_levels(station, model, valid_time_utc, id DESC)",
    "CREATE INDEX IF NOT EXISTS idx_mkt_snap_ticker ON market_snapshots(ticker, snapshot_time)",
    "CREATE INDEX IF NOT EXISTS idx_mkt_snap_latest ON market_snapshots(ticker, id DESC)",
    "CREATE INDEX IF NOT EXISTS idx_mkt_snap_date ON market_snapshots(forecast_date, market_type)",
    "CREATE INDEX IF NOT EXISTS idx_evt_settle ON event_settlements(station, settlement_date, market_type)",
    "CREATE INDEX IF NOT EXISTS idx_mkt_settle_date ON market_settlements(forecast_date, market_type)",
    "CREATE INDEX IF NOT EXISTS idx_model_scores ON model_scores(station, model, market_type)",
    "CREATE INDEX IF NOT EXISTS idx_diurnal_model ON diurnal_scores(station, model, forecast_date)",
    "CREATE INDEX IF NOT EXISTS idx_diurnal_hour ON diurnal_scores(station, valid_hour_local)",
    "CREATE INDEX IF NOT EXISTS idx_cloud_obs ON cloud_obs(station, forecast_date)",
    "CREATE INDEX IF NOT EXISTS idx_nearby_obs_stid ON nearby_observations(stid, timestamp_utc)",
    "CREATE INDEX IF NOT EXISTS idx_nearby_obs_date ON nearby_observations(lst_date, stid)",
    "CREATE INDEX IF NOT EXISTS idx_nearby_obs_dist ON nearby_observations(distance_mi, timestamp_utc)",
    "CREATE INDEX IF NOT EXISTS idx_nearby_obs_time ON nearby_observations(timestamp_utc)",
    "CREATE INDEX IF NOT EXISTS idx_atmos_station_time ON atmospheric_data(station, valid_time_utc)",
    "CREATE INDEX IF NOT EXISTS idx_atmos_dedupe_lookup ON atmospheric_data(station, model, valid_time_utc, id DESC)",
    "CREATE INDEX IF NOT EXISTS idx_fawn_station_time ON fawn_observations(station_id, timestamp_utc)",
    "CREATE INDEX IF NOT EXISTS idx_sst_station_time ON sst_observations(station_id, timestamp_utc)",
    "CREATE INDEX IF NOT EXISTS idx_adj_fcst_lookup ON model_consensus(station, forecast_date, market_type)",
    "CREATE INDEX IF NOT EXISTS idx_fwd_curve_lookup ON forward_curves(station, target_date, snapshot_time_utc)",
    "CREATE INDEX IF NOT EXISTS idx_sse_events ON sse_events(station, event_type, received_at)",
    "CREATE INDEX IF NOT EXISTS idx_collection_runs ON collection_runs(collector, started_at)",
    "CREATE INDEX IF NOT EXISTS idx_signal_events_lookup ON signal_events(station, target_date, market_type)",
    "CREATE INDEX IF NOT EXISTS idx_signal_events_regime ON signal_events(regime, target_date)",
    "CREATE INDEX IF NOT EXISTS idx_signal_scores_lookup ON signal_scores(station, target_date, market_type)",
    "CREATE INDEX IF NOT EXISTS idx_signal_scores_regime ON signal_scores(regime)",
    "CREATE INDEX IF NOT EXISTS idx_signal_cal_lookup ON signal_calibration(station, market_type, regime, signal_name)",
    "CREATE INDEX IF NOT EXISTS idx_active_brackets ON active_brackets(target_date, market_type)",
    "CREATE INDEX IF NOT EXISTS idx_bracket_est_lookup ON bracket_estimates(target_date, market_type, timestamp_utc)",
    "CREATE INDEX IF NOT EXISTS idx_bracket_est_ticker ON bracket_estimates(ticker, timestamp_utc)",
    "CREATE INDEX IF NOT EXISTS idx_regime_labels_lookup ON regime_labels(station, target_date)",
    "CREATE INDEX IF NOT EXISTS idx_regime_labels_regime ON regime_labels(regimes_active)",
    "CREATE INDEX IF NOT EXISTS idx_hdp_test_lookup ON regime_labels_hdp_test(station, target_date)",
    "CREATE INDEX IF NOT EXISTS idx_paper_trades_lookup ON paper_trades(status, target_date, market_type)",
    "CREATE INDEX IF NOT EXISTS idx_paper_trades_ticker ON paper_trades(ticker, status)",
    "CREATE INDEX IF NOT EXISTS idx_paper_marks_trade_time ON paper_trade_marks(trade_id, mark_time)",
    "CREATE INDEX IF NOT EXISTS idx_paper_settlements_ticker ON paper_trade_settlements(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_ds3m_est_ticker ON ds3m_estimates(ticker, timestamp_utc)",
    "CREATE INDEX IF NOT EXISTS idx_ds3m_cmp_date ON ds3m_comparison(target_date, station)",
    "CREATE INDEX IF NOT EXISTS idx_ds3m_trades_ticker ON ds3m_paper_trades(ticker, target_date)",
    "CREATE INDEX IF NOT EXISTS idx_ds3m_trades_status ON ds3m_paper_trades(status, station)",
    "CREATE INDEX IF NOT EXISTS idx_ds3m_marks_trade_time ON ds3m_paper_trade_marks(trade_id, mark_time)",
    "CREATE INDEX IF NOT EXISTS idx_ds3m_settlements_ticker ON ds3m_paper_trade_settlements(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_rtma_ru_time ON rtma_ru_observations(timestamp_utc)",
    "CREATE INDEX IF NOT EXISTS idx_rtma_ru_point ON rtma_ru_observations(lat, lon, timestamp_utc)",
]
