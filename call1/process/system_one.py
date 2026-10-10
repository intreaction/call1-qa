"""Local Ollama decision-model discovery for the default Contact Signals cascade. No chat/generation endpoint is used.

Laya is experimental: compatibility is checked here, not domain accuracy. Model digests are
frozen in the catalog, then checked again by the handler before and after inference.
"""
from __future__ import annotations

import re
import threading
import time
from urllib.parse import urlsplit

import httpx

ENTRY_ID = "laya-system-one"
DEFAULT_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "laya"
_CACHE = {}
_LOCK = threading.Lock()


class SystemOneUnavailable(ValueError):
    """Safe discovery error; provider bodies and transcripts never appear in it."""


def validate_url(value: str) -> str:
    try:
        parts = urlsplit(value)
        port = parts.port
    except (ValueError, TypeError):
        raise ValueError("system_one_url must be a loopback Ollama base URL") from None
    if (parts.scheme not in ("http", "https") or parts.hostname not in ("127.0.0.1", "localhost", "::1")
            or parts.username is not None or parts.password is not None or parts.query or parts.fragment
            or parts.path not in ("", "/") or port == 0):
        raise ValueError("system_one_url must be a loopback Ollama base URL without credentials or a path")
    return value.rstrip("/")


def validate_model(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"laya(?::[a-zA-Z0-9_.-]+)?", value):
        raise ValueError("system_one_model must name an installed laya tag")
    return value


def discover(url: str, model: str, *, http=None, fresh: bool = False) -> dict:
    url, model = validate_url(url), validate_model(model)
    key = (url, model)
    if not fresh and http is None:
        with _LOCK:
            cached = _CACHE.get(key)
        if cached is not None and time.monotonic() - cached[0] < 30:
            if isinstance(cached[1], str):
                raise SystemOneUnavailable(cached[1])
            return dict(cached[1])
    owns = http is None
    client = http or httpx.Client(base_url=url, trust_env=False, follow_redirects=False, timeout=2)
    try:
        version = client.get(url + "/api/version")
        version.raise_for_status()
        nums = tuple(int(n) for n in version.json()["version"].split("-")[0].split(".")[:3])
        if nums < (0, 40, 0):
            raise SystemOneUnavailable("System One requires Ollama 0.40.0 or newer")
        response = client.get(url + "/api/tags")
        response.raise_for_status()
        name = model if ":" in model else model + ":latest"
        item = next((m for m in response.json()["models"] if m.get("name") == name), None)
        if item is None:
            raise SystemOneUnavailable("the configured Laya model is not installed in Ollama")
        digest = str(item["digest"]).removeprefix("sha256:")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise SystemOneUnavailable("Ollama did not report a valid model digest")
        response = client.post(url + "/api/show", json={"model": model})
        response.raise_for_status()
        details = response.json()
        limit = details.get("model_info", {}).get("laya.context_length")
        if "decision" not in details.get("capabilities", []) or not isinstance(limit, int) or limit < 512:
            raise SystemOneUnavailable("the installed model does not advertise a supported decision context")
        configured = re.search(r"^num_ctx\s+(\d+)\s*$", details.get("parameters", ""), re.MULTILINE)
        if configured:
            limit = min(limit, int(configured.group(1)))
            if limit < 512:
                raise SystemOneUnavailable("Ollama's configured decision context is smaller than 512 tokens")
        result = {"digest": "sha256:" + digest, "context_limit": limit, "size": item.get("size")}
        if http is None:
            with _LOCK:
                _CACHE[key] = (time.monotonic(), result)
        return dict(result)
    except SystemOneUnavailable as exc:
        if http is None:
            with _LOCK:
                _CACHE[key] = (time.monotonic(), str(exc))
        raise
    except (httpx.HTTPError, ValueError, KeyError, TypeError, StopIteration):
        if http is None:
            with _LOCK:
                _CACHE[key] = (time.monotonic(), "local Ollama decision-model discovery failed")
        raise SystemOneUnavailable("local Ollama decision-model discovery failed") from None
    finally:
        if owns:
            client.close()


def catalog_entry(url: str, model: str):
    from call1.contracts.catalog import ModelPurpose
    from call1.contracts.custody import ProviderType
    from .catalog import CatalogEntry

    try:
        info = discover(url, model)
    except SystemOneUnavailable:
        info = {"digest": "unavailable", "context_limit": 512, "size": None}
    return CatalogEntry(
        entry_id=ENTRY_ID, display_name="Semantic similarity → Laya → Gemma", purposes=(ModelPurpose.SIGNAL_CATEGORY,),
        model_family="laya", model_revision=info["digest"], weights_digest=info["digest"] if info["digest"] != "unavailable" else None,
        provider_model_id=model, adapter_id="call1.signals.system_one", runtime="ollama", provider_type=ProviderType.OLLAMA,
        destination_host=urlsplit(url).hostname, endpoint_url=url, mutable_alias=True, context_limit_tokens=info["context_limit"],
        memory_bytes=info["size"], license_notice="Apache-2.0. Experimental decision scores; validate on your calls before use.")


def entry_problem(entry, *, fresh=False) -> str | None:
    try:
        info = discover(entry.endpoint_url, entry.provider_model_id, fresh=fresh)
    except SystemOneUnavailable as exc:
        return str(exc)
    if info["digest"] != entry.model_revision:
        return "Laya's installed digest changed; restart Process to refresh the catalog before selecting it"
    return None


def entry_status(entry):
    from call1.contracts.catalog import CatalogEntryStatus

    if entry_problem(entry) is not None:
        return CatalogEntryStatus.NOT_INSTALLED, []
    return CatalogEntryStatus.AVAILABLE, list(entry.purposes)
