"""DS3M Live Dashboard — FastAPI + SSE real-time monitoring."""

from .app import app, configure, run_dashboard

__all__ = ["app", "configure", "run_dashboard"]
