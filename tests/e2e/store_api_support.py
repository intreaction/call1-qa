"""Helpers for the Store API end-to-end tests (``test_store_api_*.py``). No tests here.

``worker_harness(stack)`` drives a **real** Store server as a hand-rolled Process worker would:
it reuses the Store tests' ``QueueHarness`` payload builders (``tests/store/test_queue_harness.py``,
loaded by path and unchanged) on top of an ``httpx.Client`` pointed at the stack's Store and
authenticated with the stack's real service key. Use it on a ``with_process=False`` stack, so no
Process worker races the test for the jobs it creates.
"""

from __future__ import annotations

import importlib.util
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import httpx

_SOURCE = Path(__file__).resolve().parents[1] / "store" / "test_queue_harness.py"
_NAME = "call1_e2e_queue_harness"


def queue_harness_module():
    """``tests/store/test_queue_harness.py``, loaded by path (its pytest fixtures are not registered)."""
    if _NAME in sys.modules:
        return sys.modules[_NAME]
    spec = importlib.util.spec_from_file_location(_NAME, _SOURCE)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"cannot load {_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_NAME] = module
    spec.loader.exec_module(module)
    return module


def worker_harness(stack, *, headers: Optional[Dict[str, str]] = None, installation_id: Optional[str] = None):
    """A ``QueueHarness`` whose HTTP client is a real ``httpx.Client`` on ``stack.store_url``, using
    the stack's service key (or ``headers``)."""
    module = queue_harness_module()
    client = httpx.Client(base_url=stack.store_url, timeout=30.0)
    key = types.SimpleNamespace(headers=headers or stack.service_headers(), installation_id=installation_id or stack.installation_id)
    harness = module.QueueHarness(client, None, None, key, None, None)
    harness.http = client
    return harness


def poll(fetch: Callable[[], Any], done: Callable[[Any], bool], *, timeout: float = 30.0, interval: float = 0.25, what: str = "condition"):
    """Call ``fetch`` until ``done(value)``; return the value. Raises AssertionError with the last value."""
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = fetch()
        if done(last):
            return last
        time.sleep(interval)
    raise AssertionError(f"timed out after {timeout:.0f}s waiting for {what}; last value: {str(last)[:1500]}")


def error_code(response: httpx.Response) -> Optional[str]:
    try:
        return response.json().get("code")
    except ValueError:
        return None


def settled_call(stack, source: str = "call_01_compliant", **ingest_kw) -> Dict[str, Any]:
    """Ingest through Process and wait until Store reports the conversation settled. Returns the receipt."""
    receipt = stack.ingest(source, **ingest_kw)
    stack.wait_until_settled(receipt["call_id"])
    return receipt


def graph_job_types(stack, graph_id: str) -> Dict[str, str]:
    """``{job_id: job_type}`` of one job graph (read with the service key)."""
    response = stack.store_get(f"/job-graphs/{graph_id}", session="service")
    assert response.status_code == 200, response.text
    graph = response.json()
    out: Dict[str, str] = {}
    for entry in graph["jobs"]:
        job = stack.store_get(f"/jobs/{entry['job_id']}", session="service").json()
        out[entry["job_id"]] = job["job_type"]
    return out


def all_pages(get: Callable[..., httpx.Response], path: str, *, params: Optional[Dict[str, Any]] = None, limit: int = 200) -> Iterable[Dict[str, Any]]:
    """Every item of a paged listing (``items`` + ``next_page_token``)."""
    query = dict(params or {})
    query["limit"] = limit
    while True:
        response = get(path, params=query)
        assert response.status_code == 200, response.text
        body = response.json()
        yield from body["items"]
        token = body.get("next_page_token")
        if not token:
            return
        query["page_token"] = token
