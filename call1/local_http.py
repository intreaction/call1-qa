"""Local-only HTTP client for the Call1 appliance.

Ollama model discovery and local summary generation must stay on
the loopback interface. This module provides a single opener that:

  - rejects HTTP redirects (a loopback endpoint must never bounce us to an
    external host via a Location header),
  - disables environment proxy forwarding (HTTP_PROXY/HTTPS_PROXY/ALL_PROXY),
    so call contents can never be forwarded through a corporate proxy,
  - validates the URL is a loopback http(s) endpoint with a well-formed port.

External question models use a separate opt-in client in question_models.py.
This local client remains restricted to loopback.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional
from urllib.parse import urlsplit


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Reject any redirect instead of following it."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(
            req.full_url, code, f"Redirect to non-loopback blocked: {newurl}", headers, fp
        )


def _build_opener() -> urllib.request.OpenerDirector:
    handlers = [
        urllib.request.ProxyHandler({}),  # no env proxies
        _NoRedirect(),
    ]
    return urllib.request.build_opener(*handlers)


_OPENER = _build_opener()


def validate_loopback_url(url: str) -> str:
    """Validate a loopback http(s) URL and return it normalized (no trailing /).

    Raises ValueError for non-loopback hosts, credentials, query/fragment,
    paths, or malformed ports.
    """
    if not url or not url.strip():
        raise ValueError("endpoint must not be empty")
    url = url.strip()
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise ValueError("endpoint must be an http(s) URL")
    if parts.username is not None or parts.password is not None:
        raise ValueError("endpoint must not contain credentials")
    if parts.query or parts.fragment:
        raise ValueError("endpoint must not contain a query or fragment")
    host = parts.hostname
    if host is None:
        raise ValueError("endpoint must include a host")
    host = host.strip("[]")
    if host not in ("127.0.0.1", "::1", "localhost"):
        raise ValueError("endpoint must be a loopback address (127.0.0.1, ::1, or localhost)")
    if parts.path not in ("", "/"):
        raise ValueError("endpoint must not include a path")
    # Force port parsing so a malformed port like 'badport' is rejected.
    try:
        port = parts.port
    except ValueError:
        raise ValueError("endpoint has a malformed port")
    if port is not None and not (1 <= port <= 65535):
        raise ValueError("endpoint port out of range")
    return url.rstrip("/")


def local_open(url: str, data: Optional[bytes] = None, timeout: float = 3.0) -> Dict[str, Any]:
    """Open a loopback URL with the local-only opener.

    The URL's scheme://host:port must be a validated loopback endpoint; the
    request path (e.g. /api/tags) is allowed. Returns the parsed JSON
    response. Raises on any network/HTTP error.
    """
    parts = urlsplit(url)
    base = f"{parts.scheme}://{parts.netloc}"
    validate_loopback_url(base)
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=timeout) as resp:
        return json.loads(resp.read())
