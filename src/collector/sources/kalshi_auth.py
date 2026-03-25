"""RSA-PSS signing for Kalshi API authentication."""

from __future__ import annotations

import base64
import time
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


REST_PATH_PREFIX = "/trade-api/v2"
WS_PATH = "/trade-api/ws/v2"


def load_private_key(key_path: str | Path) -> rsa.RSAPrivateKey:
    """Load RSA private key from PEM file."""
    data = Path(key_path).read_bytes()
    key = serialization.load_pem_private_key(data, password=None)
    assert isinstance(key, rsa.RSAPrivateKey)
    return key


def sign_request(
    private_key: rsa.RSAPrivateKey,
    timestamp_ms: int,
    method: str,
    path: str,
) -> str:
    """Sign a Kalshi API request. Returns base64-encoded signature."""
    message = str(timestamp_ms) + method.upper() + path
    signature = private_key.sign(
        message.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("utf-8")


def make_rest_headers(
    key_id: str,
    private_key: rsa.RSAPrivateKey,
    method: str,
    path: str,
) -> dict[str, str]:
    """Generate auth headers for a Kalshi REST request."""
    ts_ms = int(time.time() * 1000)
    full_path = REST_PATH_PREFIX + path if not path.startswith(REST_PATH_PREFIX) else path
    sig = sign_request(private_key, ts_ms, method, full_path)
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-SIGNATURE": sig,
        "KALSHI-ACCESS-TIMESTAMP": str(ts_ms),
        "Content-Type": "application/json",
    }


def make_ws_auth_headers(
    key_id: str,
    private_key: rsa.RSAPrivateKey,
) -> dict[str, str]:
    """Generate auth headers for Kalshi WebSocket connection."""
    ts_ms = int(time.time() * 1000)
    sig = sign_request(private_key, ts_ms, "GET", WS_PATH)
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-SIGNATURE": sig,
        "KALSHI-ACCESS-TIMESTAMP": str(ts_ms),
    }
