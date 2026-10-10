"""The Process operator app: the ``/process/api`` routes the console uses and the console itself.

Loopback only: every request must come from and name a loopback host (``Host`` too, against DNS
rebinding). Reads need nothing more. Writes (import, retry, cancel) need the console credential
in ``X-Call1-Console-Token``, and a cross-origin ``Origin`` is refused. Retry and cancel go to
Store with Process's own service key, which must hold ``jobs:control``.

=====================================  ==========================================================
``GET  /process/api/health``           liveness and runtime state
``GET  /process/api/session``          whether a console credential exists and the sent one is valid
``GET  /process/api/overview``         Store connection and contract version, worker, slots, handlers
``GET  /process/api/conversations``    recordings this Process ingested, with Store progress
``GET  /process/api/conversations/{id}``  one conversation: progress, jobs, Evaluate link
``GET  /process/api/jobs/{id}``        a job and its attempts
``POST /process/api/jobs/{id}/retry``  ``{reason}``: manual retry (``jobs:control``)
``POST /process/api/jobs/{id}/cancel`` ``{reason, cascade}``: cancel (``jobs:control``)
``GET  /process/api/catalog``          the catalog, defaults, handlers
``POST /process/api/recordings``       multipart ``file`` (+ ``agent_id``, ``agent_display_name``,
                                       ``agent_extension``, ``agent_channel``, ``external_call_ref``):
                                       ingest a recording; a re-upload with different metadata
                                       updates the call (``metadata_updated``)
``GET  /process/api/training``         on-device training: availability, settings, time zone, next
                                       run, label counts, status, last check, active adapter,
                                       kept versions, the last 20 runs, notices
``PUT  /process/api/training/settings`` ``{enabled, schedule: {frequency, weekday, time},
                                       min_new_labels, max_duration_minutes, only_when_idle}``
``POST /process/api/training/runs``    Train now: 202 ``{run}``; 409 ``training_busy`` or
                                       ``training_unavailable``
``POST /process/api/training/runs/{id}/cancel``  idempotent cancel: ``{run}``
``GET  /process/api/training/runs``    ``?limit=50``: ``{items}``, newest first
``POST /process/api/training/active``  ``{version: "<id>" | null}``: rollback or reactivation;
                                       404 unknown version, 409 ``training_busy`` or ``stale_base``
=====================================  ==========================================================

On-device training is docs/OnDeviceTraining.md section 6.1; the service is
``training.scheduler.TrainingService``.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from call1.contracts.calls import SourceKind

from .audio import UnsupportedAudio
from .catalog import CatalogError
from .config import LOOPBACK_HOSTS, ConfigError
from .console import HEADER, ConsoleAuth
from .graph import PlanError
from .ingest import InvalidCallMetadata
from .runtime import ProcessRuntime, progress_line
from .store_client import StoreError, StoreUnavailable

log = logging.getLogger("call1.process.app")

STATIC_ROOT = Path(__file__).resolve().parent / "static"
API = "/process/api"
MAX_UPLOAD_BYTES = 4 * 1024 ** 3
PROGRESS_TIMEOUT_SECONDS = 3.0  # console display reads: one short try, no retry backoff
PROGRESS_CACHE_SIZE = 500
RUN_TIME_CACHE_SIZE = 5000
JOBS_CONTROL_HELP = ("This Process service key lacks the jobs:control scope, so the console cannot retry or cancel jobs. On the Store host, "
                     "issue a key that holds it: python -m call1.store issue-service-key --installation <name> --scope jobs:control "
                     "plus the default Process scopes (see call1/store/README.md).")


class ProcessApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details or {}


class RetryBody(BaseModel):
    reason: str = Field(default="Retried from the Process console", max_length=2000)


class CancelBody(BaseModel):
    reason: str = Field(default="Cancelled from the Process console", max_length=2000)
    cascade: bool = True


class ActivateBody(BaseModel):
    version: Optional[str] = Field(default=None, max_length=64)


def _host_of(value: str) -> str:
    try:
        return (urlsplit("//" + value).hostname or "").lower()
    except ValueError:
        return ""


def _error(status: int, code: str, message: str, details: Optional[Dict[str, Any]] = None) -> JSONResponse:
    return JSONResponse({"code": code, "message": message, "details": details or {}}, status_code=status)


def _placeholder() -> HTMLResponse:
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Call1 Process</title><style>
:root { color-scheme: light dark; --bg: #f6f8fb; --fg: #111827; --muted: #4b5563; --line: #d7dde6; }
@media (prefers-color-scheme: dark) { :root { --bg: #0b0f17; --fg: #e6e9ef; --muted: #9aa3b2; --line: #243044; } }
body { margin: 0; background: var(--bg); color: var(--fg); font-family: "IBM Plex Sans", system-ui, sans-serif; }
header { padding: 16px 24px; border-bottom: 1px solid var(--line); font-weight: 600; }
main { padding: 32px 24px; max-width: 640px; line-height: 1.5; } p { color: var(--muted); }
</style></head><body><header>Call1 Process</header><main><h1>Not built yet</h1>
<p>The Process console has not been built. Build it with <strong>npm --prefix frontend run build:process</strong>, then reload.</p>
<p>The Process API is running: <a href="/process/api/overview">/process/api/overview</a>.</p></main></body></html>"""
    return HTMLResponse(page, headers={"Cache-Control": "no-store"})


def create_app(runtime: ProcessRuntime, *, start_background: bool = True, loopback_only: bool = True, static_root: Path = STATIC_ROOT) -> FastAPI:
    config = runtime.config
    auth = ConsoleAuth(config.config_path, config.console_credential)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if start_background:
            runtime.start()
        try:
            yield
        finally:
            if start_background:
                runtime.stop()

    app = FastAPI(title="Call1 Process", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.runtime = runtime

    @app.middleware("http")
    async def loopback_guard(request: Request, call_next):
        if loopback_only:
            client = (request.client.host if request.client else "").lower()
            host = _host_of(request.headers.get("host", ""))
            if client not in LOOPBACK_HOSTS or host not in LOOPBACK_HOSTS:
                return _error(403, "loopback_only", "The Process console is served on loopback only")
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin and _host_of(urlsplit(origin).netloc) not in set(LOOPBACK_HOSTS) | {_host_of(request.headers.get("host", ""))}:
                return _error(403, "origin_not_allowed", "Cross-origin writes are refused")
        return await call_next(request)

    @app.exception_handler(ProcessApiError)
    async def _api_error(request: Request, exc: ProcessApiError):
        return _error(exc.status, exc.code, exc.message, exc.details)

    @app.exception_handler(StoreError)
    async def _store_error(request: Request, exc: StoreError):
        status = exc.status if exc.status and 400 <= exc.status < 600 else 502
        if exc.code == "insufficient_scope":
            return _error(403, "insufficient_scope", JOBS_CONTROL_HELP if request.url.path.endswith(("/retry", "/cancel")) else exc.message,
                          exc.details)
        return _error(status, exc.code, exc.message or "Store refused the request", exc.details)

    @app.exception_handler(ConfigError)
    async def _config_error(request: Request, exc: ConfigError):
        return _error(503, "not_configured", str(exc))

    def require_console(request: Request) -> None:
        if not auth.configured:
            raise ProcessApiError(503, "console_credential_missing",
                                  "No console credential exists. Run python -m call1.process console-token on this host.")
        token = request.headers.get(HEADER)
        if not token:
            bearer = request.headers.get("authorization", "")
            token = bearer[7:].strip() if bearer.lower().startswith("bearer ") else None
        if not auth.check(token):
            raise ProcessApiError(401, "console_credential_invalid", "Send the console credential in the X-Call1-Console-Token header")

    def client():
        runtime.config.require_configured()
        if runtime.client is None:
            raise ConfigError("Process has no Store connection configured")
        return runtime.client

    # --- reads -------------------------------------------------------------------------------

    @app.get(API + "/health")
    def health() -> Dict[str, Any]:
        return {"ok": True, "state": runtime.state}

    @app.get(API + "/session")
    def session(request: Request) -> Dict[str, Any]:
        token = request.headers.get(HEADER)
        return {"loopback": True, "console_credential_configured": auth.configured, "token_valid": auth.check(token) if token else False,
                "write_header": HEADER, "demo": os.environ.get("CALL1_STORE_DEMO") == "1"}

    @app.get(API + "/overview")
    def overview() -> Dict[str, Any]:
        return runtime.overview()

    # Last progress seen per conversation, served (marked stale) while Store is unreachable.
    progress_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    def _cached_progress(conversation_id: str, code: str) -> Dict[str, Any]:
        cached = progress_cache.get(conversation_id)
        if cached is None:
            return {"progress": None, "progress_line": None, "progress_error": code, "progress_stale": False}
        return dict(cached, progress_error=code, progress_stale=True)

    def _progress(conversation_id: str, outage: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """One conversation's progress for display. A single short try (no retry backoff): the
        console must answer during a Store outage. ``outage`` is shared across one listing, so
        after the first unreachable answer the remaining rows use the cache without asking."""
        if outage is not None and outage.get("code"):
            return _cached_progress(conversation_id, outage["code"])
        try:
            progress = client().get_progress(conversation_id, retry=False, timeout=PROGRESS_TIMEOUT_SECONDS)
        except StoreUnavailable as exc:
            if outage is not None:
                outage["code"] = exc.code
            return _cached_progress(conversation_id, exc.code)
        except StoreError as exc:
            return _cached_progress(conversation_id, exc.code)
        fresh = {"progress": progress.model_dump(mode="json"), "progress_line": progress_line(progress)}
        progress_cache[conversation_id] = fresh
        progress_cache.move_to_end(conversation_id)
        while len(progress_cache) > PROGRESS_CACHE_SIZE:
            progress_cache.popitem(last=False)
        return dict(fresh, progress_error=None, progress_stale=False)

    @app.get(API + "/conversations")
    def conversations(limit: int = 50) -> Dict[str, Any]:
        items = []
        outage: Dict[str, str] = {}
        for entry in runtime.ledger.list(max(1, min(limit, 200))):
            item = dict(entry, evaluate_url=config.evaluate_url(entry.get("call_id")))
            item.update(_progress(entry["conversation_id"], outage))
            items.append(item)
        return {"items": items, "total": runtime.ledger.count(), "store_unavailable": bool(outage)}

    # When each settled job last ran, keyed by (job id, updated_at): a finished job's attempts do not
    # change until a retry moves updated_at, so the console asks Store once per finished job.
    run_times: "OrderedDict[Tuple[str, str], Tuple[Optional[str], Optional[str]]]" = OrderedDict()

    def _run_window(store: Any, j: Any) -> Tuple[Optional[str], Optional[str]]:
        """``(started_at, ended_at)`` of the job's latest counted attempt, for the console's per-job
        duration. A running job's start is its lease grant; a job that never ran has neither."""
        if j.lease is not None:
            return j.lease.granted_at.isoformat(), None
        if j.status.value not in ("SUCCEEDED", "FAILED", "CANCELLED") or j.attempt_count == 0:
            return None, None
        key = (j.id, j.updated_at.isoformat())
        if key in run_times:
            return run_times[key]
        try:
            attempts = [a for a in store.list_attempts(j.id) if a.counts_as_attempt]
        except StoreError:
            return None, None
        window: Tuple[Optional[str], Optional[str]] = (None, None)
        if attempts:
            last = max(attempts, key=lambda a: a.attempt_number)
            window = (last.started_at.isoformat(), last.ended_at.isoformat() if last.ended_at else None)
        run_times[key] = window
        while len(run_times) > RUN_TIME_CACHE_SIZE:
            run_times.popitem(last=False)
        return window

    @app.get(API + "/conversations/{conversation_id}")
    def conversation(conversation_id: str) -> Dict[str, Any]:
        store = client()
        found = store.get_conversation(conversation_id)
        jobs = store.list_jobs(conversation_id=conversation_id)
        windows = {j.id: _run_window(store, j) for j in jobs}
        entry = runtime.ledger.get(conversation_id) or {}
        body = {
            "conversation": found.model_dump(mode="json"), "label": entry.get("label"), "graphs": entry.get("graphs", []),
            "evaluate_url": config.evaluate_url(found.call_id),
            "jobs": [{
                "id": j.id, "graph_id": j.graph_id, "job_type": j.job_type.value, "status": j.status.value, "priority": j.priority,
                "attempt_count": j.attempt_count, "max_attempts": j.max_attempts, "retry_generation": j.retry_generation,
                "waiting_reason": j.waiting_reason.value if j.waiting_reason else None, "error_code": j.error_code.value if j.error_code else None,
                "cancel_requested": j.cancel_requested, "blocking": [b.model_dump(mode="json") for b in j.blocking],
                "catalog_entry": j.selection.catalog_entry.entry_id if j.selection else None,
                "route_class": j.selection.route.route_class.value if j.selection else None,
                "destination_host": j.selection.route.destination_host if j.selection else None,
                "criterion_id": j.parameters.criterion_id, "result_version": j.result_version,
                "created_at": j.created_at.isoformat(), "updated_at": j.updated_at.isoformat(),
                "completed_at": j.completed_at.isoformat() if j.completed_at else None,
                "requires_job_ids": list(j.requires_job_ids), "after_job_ids": list(j.after_job_ids),
                "started_at": windows[j.id][0], "ended_at": windows[j.id][1],
            } for j in jobs],
        }
        body.update(_progress(conversation_id))
        return body

    @app.get(API + "/jobs/{job_id}")
    def job(job_id: str) -> Dict[str, Any]:
        store = client()
        found = store.get_job(job_id)
        attempts = store.list_attempts(job_id)
        conv = None
        try:
            conv = store.get_conversation(found.conversation_id)
        except StoreError:
            pass
        return {"job": found.model_dump(mode="json"), "attempts": [a.model_dump(mode="json") for a in attempts],
                "evaluate_url": config.evaluate_url(conv.call_id if conv else None)}

    @app.get(API + "/catalog")
    def catalog() -> Dict[str, Any]:
        cat = runtime.catalog
        return {"version": cat.version(), "published": runtime.catalog_published,
                "defaults": {p.value: e for p, e in sorted(cat.defaults.items(), key=lambda kv: kv[0].value)},
                "escalation_entry_id": config.escalation_entry_id, "entries": cat.describe(),
                "handlers": {"mode": runtime.registry.mode, "registered": runtime.registry.describe(),
                             "missing_job_types": [t.value for t in runtime.registry.missing()], "notes": list(runtime.registry.notes)},
                "admin_state": runtime.admin_state_source,
                "masking": runtime.masking.model_dump(mode="json")}

    # --- writes ------------------------------------------------------------------------------

    @app.get(API + "/signals/first-pass")
    def signal_first_pass() -> Dict[str, Any]:
        return runtime.signal_first_pass()

    @app.post(API + "/jobs/{job_id}/retry")
    def retry(job_id: str, request: Request, body: Optional[RetryBody] = None) -> Dict[str, Any]:
        require_console(request)
        result = client().retry_job(job_id, (body or RetryBody()).reason)
        return {"job": result.model_dump(mode="json")}

    @app.post(API + "/jobs/{job_id}/cancel")
    def cancel(job_id: str, request: Request, body: Optional[CancelBody] = None) -> Dict[str, Any]:
        require_console(request)
        body = body or CancelBody()
        result = client().cancel_job(job_id, body.reason, body.cascade)
        return {"job": result.job.model_dump(mode="json"), "cancelled_job_ids": result.cancelled_job_ids}

    @app.post(API + "/recordings", status_code=201)
    def upload_recording(request: Request, file: UploadFile = File(...), agent_id: Optional[str] = Form(default=None),
                         agent_display_name: Optional[str] = Form(default=None), agent_extension: Optional[str] = Form(default=None),
                         agent_channel: Optional[int] = Form(default=None), external_call_ref: Optional[str] = Form(default=None)) -> Dict[str, Any]:
        require_console(request)
        if runtime.ingestor is None:
            raise ConfigError("Process has no Store connection configured")
        if agent_channel is not None and agent_channel not in (0, 1):
            raise ProcessApiError(422, "validation_failed", "agent_channel is 0 or 1")
        try:
            runtime.check_store()
            result = runtime.ingestor.ingest_stream(file.file, filename=file.filename or "recording", content_type=file.content_type,
                                                    source_kind=SourceKind.API_UPLOAD, agent_id=agent_id, agent_display_name=agent_display_name,
                                                    agent_extension=agent_extension, agent_channel=agent_channel,
                                                    external_call_ref=external_call_ref, max_bytes=MAX_UPLOAD_BYTES)
        except InvalidCallMetadata as exc:
            raise ProcessApiError(422, "validation_failed", str(exc)) from None
        except UnsupportedAudio as exc:
            raise ProcessApiError(415, "unsupported_audio", str(exc)) from None
        except (PlanError, CatalogError) as exc:
            raise ProcessApiError(422, "graph_not_built", f"The recording is registered but its jobs could not be planned: {exc}") from None
        except ValueError as exc:
            raise ProcessApiError(413, "payload_too_large", str(exc)) from None
        return result.to_dict()

    @app.post(API + "/demo/recordings", status_code=201)
    def demo_recording(request: Request) -> Dict[str, Any]:
        require_console(request)
        if os.environ.get("CALL1_STORE_DEMO") != "1":
            raise ProcessApiError(404, "not_found", "Demo calls are available only in demo mode")
        if runtime.ingestor is None:
            raise ConfigError("Process has no Store connection configured")
        from .demo_call import fresh_sample
        try:
            audio = fresh_sample()
        except (OSError, ValueError):
            raise ProcessApiError(503, "demo_sample_unavailable", "The AppTek demo excerpt is unavailable") from None
        runtime.check_store()
        result = runtime.ingestor.ingest_stream(audio, filename="AppTek · stock inquiry · 16s.wav",
                                                content_type="audio/wav", source_kind=SourceKind.API_UPLOAD,
                                                agent_display_name="AppTek Demo", external_call_ref="AppTek · short stock inquiry")
        return result.to_dict()

    # --- on-device training (docs/OnDeviceTraining.md section 6.1) ---------------------------

    from .training.registry import RegistryError
    from .training.scheduler import RunNotFound, TrainingBusy, TrainingUnavailable
    from .training.settings import SettingsError

    @app.get(API + "/training")
    def training() -> Dict[str, Any]:
        return runtime.training.describe()

    @app.put(API + "/training/settings")
    async def training_settings(request: Request) -> Dict[str, Any]:
        require_console(request)
        try:
            body = await request.json()
        except ValueError:
            raise ProcessApiError(422, "validation_failed", "Send the settings as a JSON object") from None
        if not isinstance(body, dict):
            raise ProcessApiError(422, "validation_failed", "Send the settings as a JSON object")
        try:
            updated = runtime.training.update_settings(body)
        except SettingsError as exc:
            raise ProcessApiError(422, "validation_failed", exc.message, {"field": exc.field}) from None
        except TrainingUnavailable as exc:
            raise ProcessApiError(409, "training_unavailable", str(exc), {"reason": str(exc)}) from None
        return {"settings": updated.console_view()}

    @app.post(API + "/training/runs", status_code=202)
    def training_run(request: Request) -> Dict[str, Any]:
        require_console(request)
        try:
            run = runtime.training.request_run("manual")
        except TrainingBusy as exc:
            raise ProcessApiError(409, "training_busy", str(exc)) from None
        except TrainingUnavailable as exc:
            raise ProcessApiError(409, "training_unavailable", str(exc), {"reason": str(exc)}) from None
        return {"run": run}

    @app.get(API + "/training/runs")
    def training_runs(limit: int = 50) -> Dict[str, Any]:
        return {"items": runtime.training.runs(max(1, min(limit, 200)))}

    @app.post(API + "/training/runs/{run_id}/cancel")
    def training_cancel(run_id: str, request: Request) -> Dict[str, Any]:
        require_console(request)
        try:
            return {"run": runtime.training.cancel(run_id)}
        except RunNotFound:
            raise ProcessApiError(404, "not_found", "No such training run") from None

    @app.post(API + "/training/active")
    def training_activate(request: Request, body: ActivateBody) -> Dict[str, Any]:
        require_console(request)
        try:
            pointer = runtime.training.activate(body.version)
        except TrainingBusy as exc:
            raise ProcessApiError(409, "training_busy", str(exc)) from None
        except RegistryError as exc:
            status = 404 if exc.code == "not_found" else 409
            raise ProcessApiError(status, exc.code, exc.message) from None
        return {"active": pointer}

    # --- the console -------------------------------------------------------------------------

    def _index() -> Optional[Path]:
        for name in ("index.html", "process.html"):
            candidate = static_root / name
            if candidate.is_file():
                return candidate
        return None

    def _serve(path: str) -> Response:
        if path == "process" or path.startswith("process/"):
            return _error(404, "not_found", "No such route")
        if path:
            candidate = (static_root / path)
            try:
                candidate.resolve().relative_to(static_root.resolve())
            except ValueError:
                return _error(404, "not_found", "No such file")
            if candidate.is_file():
                return FileResponse(candidate)
            if "." in path.rsplit("/", 1)[-1]:
                return _error(404, "not_found", "No such file")
        index = _index()
        if index is None:
            return _placeholder()
        return FileResponse(index, headers={"Cache-Control": "no-cache"})

    @app.get("/", include_in_schema=False)
    def console_index() -> Response:
        return _serve("")

    @app.get("/{path:path}", include_in_schema=False)
    def console_page(path: str) -> Response:
        return _serve(path)

    return app


__all__ = ["create_app"]
