"""Live DS3M Dashboard — FastAPI + SSE backend.

Serves a real-time dashboard showing:
  - Current weather + temperature projection
  - Active regime with colors and confidence
  - Bracket market table (model P, market prices, liquidity, edge)
  - Current + past trades with P&L
  - DS3M internals (ESS, particle histogram, regime timeline)

Technology: FastAPI + Server-Sent Events + HTMX frontend.
No JS framework needed — pure HTMX handles real-time updates.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

log = logging.getLogger(__name__)

# Paths
DASHBOARD_DIR = Path(__file__).parent
TEMPLATES_DIR = DASHBOARD_DIR / "templates"
STATIC_DIR = DASHBOARD_DIR / "static"

app = FastAPI(title="DS3M Live Dashboard", version="2.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Shared state — set by the orchestrator
_dashboard_queue: asyncio.Queue | None = None
_db_path: str = ""
_latest_data: dict = {}


def configure(db_path: str, queue: asyncio.Queue):
    """Called by the collector runner to wire up the dashboard."""
    global _dashboard_queue, _db_path
    _dashboard_queue = queue
    _db_path = db_path


# ── Main page ──────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


# ── SSE Stream ─────────────────────────────────────────────────────

@app.get("/stream")
async def stream(request: Request):
    """Server-Sent Events endpoint — pushes real-time data to dashboard."""
    async def event_generator() -> AsyncGenerator[dict, None]:
        global _latest_data
        while True:
            if await request.is_disconnected():
                break
            if _dashboard_queue:
                try:
                    data = await asyncio.wait_for(_dashboard_queue.get(), timeout=5.0)
                    _latest_data = data
                    yield {"event": "update", "data": json.dumps(data)}
                except asyncio.TimeoutError:
                    # Send heartbeat
                    yield {"event": "heartbeat", "data": "{}"}
            else:
                await asyncio.sleep(1)
                yield {"event": "heartbeat", "data": "{}"}

    return EventSourceResponse(event_generator())


# ── REST Endpoints (for historical data) ───────────────────────────

@app.get("/api/trades/open")
async def get_open_trades():
    """Get currently open paper trades."""
    if not _db_path:
        return []
    try:
        conn = sqlite3.connect(_db_path, timeout=3)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT ticker, side, entry_price_cents, entry_probability,
                      entry_timestamp, quantity
               FROM paper_trades
               WHERE status = 'open'
               ORDER BY entry_timestamp DESC"""
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/trades/history")
async def get_trade_history(days: int = 30):
    """Get settled trades for P&L tracking."""
    if not _db_path:
        return []
    try:
        conn = sqlite3.connect(_db_path, timeout=3)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT t.ticker, t.side, t.entry_price_cents, t.entry_probability,
                      s.pnl_cents, s.settled_at, s.winning_side
               FROM paper_trade_settlements s
               JOIN paper_trades t ON t.id = s.trade_id
               WHERE s.settled_at >= date('now', ?)
               ORDER BY s.settled_at DESC""",
            (f"-{days} days",),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/regime/history")
async def get_regime_history(hours: int = 24):
    """Get regime posterior history for timeline chart."""
    if not _db_path:
        return []
    try:
        conn = sqlite3.connect(_db_path, timeout=3)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT timestamp_utc, regime_name, confidence, regime_probs
               FROM regime_history
               WHERE timestamp_utc >= datetime('now', ?)
               ORDER BY timestamp_utc""",
            (f"-{hours} hours",),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/calibration")
async def get_calibration_metrics():
    """Get conformal calibration metrics."""
    return {
        "target_coverage": 0.90,
        "empirical_coverage": 0.0,  # TODO: compute from conformal state
        "n_settlements": 0,
    }


@app.get("/api/latest")
async def get_latest():
    """Get the latest dashboard data snapshot."""
    return _latest_data


# ── Startup ────────────────────────────────────────────────────────

def run_dashboard(db_path: str, queue: asyncio.Queue, port: int = 8050):
    """Start the dashboard server (called from collector runner)."""
    import uvicorn
    configure(db_path, queue)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
