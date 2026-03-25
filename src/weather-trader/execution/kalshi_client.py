"""Kalshi order client — stubbed for shadow/paper mode, real API for live.

Uses RSA-PSS signing for authentication.
POST /portfolio/orders for order placement.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from src.config import KalshiConfig
from src.execution.trader import TradeDecision

log = logging.getLogger(__name__)


REST_PATH_PREFIX = "/trade-api/v2"


@dataclass
class OrderResult:
    """Result of an order placement attempt."""
    success: bool
    order_id: str | None = None
    status: str = "unknown"
    fill_price_cents: int | None = None
    message: str = ""


class KalshiClient:
    """Kalshi trading client. Stubbed in shadow/paper mode."""

    def __init__(self, cfg: KalshiConfig, mode: str = "shadow"):
        self._cfg = cfg
        self._mode = mode
        self._private_key = None
        self._session = None

    async def open(self) -> None:
        """Initialize the client. Only loads keys in live mode."""
        if self._mode == "live":
            try:
                from cryptography.hazmat.primitives import serialization
                key_data = Path(self._cfg.private_key_path).read_bytes()
                self._private_key = serialization.load_pem_private_key(key_data, password=None)

                import aiohttp
                self._session = aiohttp.ClientSession()
                log.info("Kalshi client opened in LIVE mode")
            except Exception:
                log.exception("Failed to initialize Kalshi live client")
                raise
        else:
            log.info("Kalshi client opened in %s mode (no API calls)", self._mode.upper())

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        """Generate auth headers for a request."""
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        import base64

        ts_ms = int(time.time() * 1000)
        full_path = REST_PATH_PREFIX + path if not path.startswith(REST_PATH_PREFIX) else path
        message = str(ts_ms) + method.upper() + full_path
        signature = self._private_key.sign(
            message.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self._cfg.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("utf-8"),
            "KALSHI-ACCESS-TIMESTAMP": str(ts_ms),
            "Content-Type": "application/json",
        }

    async def place_order(self, decision: TradeDecision) -> OrderResult:
        """Place an order on Kalshi.

        In shadow/paper mode: logs what would happen, returns mock result.
        In live mode: sends POST /portfolio/orders to Kalshi API.
        """
        if self._mode in ("shadow", "paper"):
            log.info(
                "[%s] WOULD place %s %s on %s at %dc × %d contracts (edge=%.1f%%)",
                self._mode.upper(),
                decision.action.upper(),
                decision.side.upper(),
                decision.ticker,
                decision.price_cents,
                decision.contracts,
                decision.edge * 100,
            )
            return OrderResult(
                success=True,
                order_id=f"{self._mode}-{int(time.time())}",
                status=self._mode,
                fill_price_cents=decision.price_cents,
                message=f"{self._mode} mode — no real order placed",
            )

        # --- LIVE MODE ---
        if not self._session or not self._private_key:
            return OrderResult(success=False, message="Client not initialized for live trading")

        path = "/portfolio/orders"
        headers = self._auth_headers("POST", path)

        # Build order payload
        payload = {
            "ticker": decision.ticker,
            "side": decision.side,
            "action": decision.action,
            "count": decision.contracts,
            "type": "limit",
        }

        # Set price on the correct side
        if decision.side == "yes":
            payload["yes_price"] = decision.price_cents
        else:
            payload["no_price"] = decision.price_cents

        url = f"{self._cfg.rest_base}{path}"
        try:
            async with self._session.post(url, headers=headers, json=payload) as resp:
                if resp.status == 201:
                    data = await resp.json()
                    order = data.get("order", {})
                    return OrderResult(
                        success=True,
                        order_id=order.get("order_id"),
                        status=order.get("status", "resting"),
                        message="Order placed successfully",
                    )
                else:
                    body = await resp.text()
                    log.error("Kalshi order failed: HTTP %d — %s", resp.status, body)
                    return OrderResult(
                        success=False,
                        status="error",
                        message=f"HTTP {resp.status}: {body[:200]}",
                    )
        except Exception as e:
            log.exception("Kalshi order exception")
            return OrderResult(success=False, message=str(e))

    async def get_balance(self) -> float | None:
        """Get account balance in dollars. Returns None if not in live mode."""
        if self._mode != "live" or not self._session:
            return None

        path = "/portfolio/balance"
        headers = self._auth_headers("GET", path)
        url = f"{self._cfg.rest_base}{path}"
        try:
            async with self._session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Balance returned in cents
                    return data.get("balance", 0) / 100.0
                return None
        except Exception:
            log.exception("Failed to get Kalshi balance")
            return None

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order. Returns True on success."""
        if self._mode != "live" or not self._session:
            log.info("[%s] WOULD cancel order %s", self._mode.upper(), order_id)
            return True

        path = f"/portfolio/orders/{order_id}"
        headers = self._auth_headers("DELETE", path)
        url = f"{self._cfg.rest_base}{path}"
        try:
            async with self._session.delete(url, headers=headers) as resp:
                return resp.status in (200, 204)
        except Exception:
            log.exception("Failed to cancel order %s", order_id)
            return False
