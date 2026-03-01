"""Security utilities for Vantinel SDK.

Provides HMAC request signing, TLS enforcement, API key protection,
and anti-replay mechanisms to prevent reverse engineering and tampering.
"""

import hashlib
import hmac
import time
import os
import ctypes
from typing import Optional


def hmac_sign(api_key: str, timestamp: int, body: str) -> str:
    """Generate HMAC-SHA256 signature for request authentication.

    The signature covers the timestamp and request body to prevent
    both replay attacks and request tampering.

    Args:
        api_key: The API key used as HMAC secret
        timestamp: Unix timestamp in milliseconds
        body: JSON-serialized request body

    Returns:
        Hex-encoded HMAC-SHA256 signature
    """
    message = f"{timestamp}.{body}".encode("utf-8")
    return hmac.new(
        api_key.encode("utf-8"),
        message,
        hashlib.sha256,
    ).hexdigest()


def validate_collector_url(url: str) -> str:
    """Enforce HTTPS for non-local collector URLs.

    Localhost and private network addresses are allowed over HTTP
    for development. All other endpoints require HTTPS.

    Args:
        url: The collector URL to validate

    Returns:
        The validated (possibly upgraded) URL

    Raises:
        ValueError: If a non-local URL uses HTTP
    """
    _ALLOWED_INSECURE = (
        "http://localhost",
        "http://127.0.0.1",
        "http://0.0.0.0",
        "http://[::1]",
        "http://10.",
        "http://172.16.",
        "http://172.17.",
        "http://172.18.",
        "http://172.19.",
        "http://172.20.",
        "http://172.21.",
        "http://172.22.",
        "http://172.23.",
        "http://172.24.",
        "http://172.25.",
        "http://172.26.",
        "http://172.27.",
        "http://172.28.",
        "http://172.29.",
        "http://172.30.",
        "http://172.31.",
        "http://192.168.",
    )

    if url.startswith("https://"):
        return url

    for prefix in _ALLOWED_INSECURE:
        if url.startswith(prefix):
            return url

    raise ValueError(
        f"Collector URL must use HTTPS for non-local endpoints. "
        f"Got: {url}. Use https:// or set collector_url to a localhost address for development."
    )


def secure_zero(s: str) -> None:
    """Best-effort zeroing of a string's memory buffer.

    Python strings are immutable, so we use ctypes to overwrite
    the underlying buffer. This is not guaranteed by the runtime
    but reduces the window where secrets exist in memory.

    Args:
        s: The string to zero out
    """
    if not s:
        return
    try:
        # Get the internal buffer address of the string object
        # CPython stores string data after the object header
        str_addr = id(s)
        # PyASCIIObject header size varies; the data follows the struct.
        # For CPython 3.9+, compact ASCII strings store data at offset
        # after the object struct. We use sys.getsizeof to approximate.
        import sys
        obj_size = sys.getsizeof(s)
        buf_size = len(s)
        # Zero the tail portion (where the string characters live)
        offset = obj_size - buf_size - 1  # -1 for null terminator
        if offset > 0 and buf_size > 0:
            ctypes.memset(str_addr + offset, 0, buf_size)
    except Exception:
        # Fail silently -- this is a best-effort defense
        pass


def generate_nonce() -> str:
    """Generate a cryptographically random nonce for anti-replay.

    Returns:
        16-byte hex-encoded random nonce
    """
    return os.urandom(16).hex()


def redact_api_key(api_key: str) -> str:
    """Redact an API key for safe logging.

    Shows only the first 4 and last 4 characters.

    Args:
        api_key: The full API key

    Returns:
        Redacted string like "vant****yz12"
    """
    if len(api_key) <= 8:
        return "****"
    return f"{api_key[:4]}****{api_key[-4:]}"
