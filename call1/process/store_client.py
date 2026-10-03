"""The typed HTTP client Process uses to reach Store (``/store/v1``), and nothing else.

Bodies and responses are the contract models in ``call1.contracts``. The client:

* sends ``Authorization: Bearer c1sk_...`` (the Process service key) on every Store route, and
  never on an upload or download grant URL (a grant is its own capability);
* retries connection failures and 5xx/429 responses with capped exponential backoff and jitter.
  Every write Process makes is idempotent by contract (body keys, completion keys or natural
  identity), so a retried request never duplicates work;
* reads ``GET /store/v1/contract`` before any work and refuses a Store of a different major version
  (``ContractMismatch``), then uses the Store's effective ``ContractParameters``;
* turns every non-2xx response into ``StoreError`` carrying the contract error envelope
  (``code``, ``message``, ``details``, ``retryable``);
* tolerates exactly the Store dev-mode deviation that reaches Process: ``http://localhost`` grant
  URLs, which the contract's ``^https://`` pattern rejects. A grant URL must point at the Store's
  own origin in dev mode (any loopback name, same port) and be HTTPS otherwise.

Tests point it at an in-process Store by passing ``http=TestClient(create_app(...))``: no ports.
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ValidationError

from call1.contracts.artifacts import Artifact, ArtifactKind, InlineArtifactCreate, UploadCommit, UploadGrant, UploadGrantRequest
from call1.contracts.calls import Conversation, ConversationRegistered, ConversationRegistration
from call1.contracts.catalog import CatalogSnapshot
from call1.contracts.common import CONTRACT_PARAMETERS, CONTRACT_VERSION, STORE_API_PREFIX, ContractInfo, ContractParameters
from call1.contracts.jobs import (
    Attempt,
    CancelRequest,
    CancelResponse,
    ClaimRequest,
    ClaimResponse,
    CompletionReceipt,
    CompletionRequest,
    FailureReceipt,
    FailureRequest,
    HeartbeatRequest,
    HeartbeatResponse,
    Job,
    JobGraph,
    JobGraphRequest,
    JobGroupProgress,
    JobReleaseReceipt,
    JobReleaseRequest,
    ReanalysisClaimRequest,
    ReanalysisClaimResponse,
    ReanalysisReject,
    ReanalysisRequest,
    RetryRequest,
)
from call1.contracts.rubrics import RubricSnapshotRequest, RubricVersion
from call1.contracts.signals import SignalTaxonomyRecord, SignalTaxonomySnapshotRequest
from call1.contracts.training import TrainingLabelPage
from call1.contracts.usage import HardwareProfile, HardwareProfileInput, LateUsageReport, UsageRecord
from call1.contracts.vocabulary import AsrVocabularyRecord

log = logging.getLogger("call1.process.store")

M = TypeVar("M", bound=BaseModel)

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class StoreError(Exception):
    """A Store request that ended without success. ``code`` is the contract ``ErrorCode`` value
    (or ``store_unavailable`` when Store could not be reached)."""

    def __init__(self, code: str, message: str, *, status: Optional[int] = None, details: Optional[Dict[str, Any]] = None,
                 retryable: bool = False, request_id: Optional[str] = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.status = status
        self.details = dict(details or {})
        self.retryable = retryable
        self.request_id = request_id

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details, "retryable": self.retryable}


class StoreUnavailable(StoreError):
    def __init__(self, message: str) -> None:
        super().__init__("store_unavailable", message, status=503, retryable=True)


class ContractMismatch(Exception):
    """Store speaks a different major contract version; Process refuses to run against it."""

    def __init__(self, store_version: str) -> None:
        super().__init__(f"Store implements contract {store_version}; this Process speaks {CONTRACT_VERSION} "
                         "(a different major version). Upgrade the older side.")
        self.store_version = store_version


def contract_major(version: str) -> int:
    try:
        return int(str(version).split(".", 1)[0])
    except ValueError:
        return -1


class StoreClient:
    def __init__(self, base_url: str, token: Optional[str], *, http: Optional[httpx.Client] = None, timeout: float = 60.0,
                 retries: int = 4, backoff_base: float = 0.5, backoff_max: float = 8.0,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.base_url = base_url.rstrip("/")
        self._token = token
        self._own_http = http is None
        self.http = http or httpx.Client(timeout=timeout, follow_redirects=False)
        self.retries = max(0, retries)
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self._sleep = sleep
        self.contract: Optional[ContractInfo] = None
        parts = urlsplit(self.base_url)
        self._origin = (parts.scheme, _loopback_alias((parts.hostname or "").lower()), parts.port)
        self.dev_mode = parts.scheme == "http"

    # --- lifecycle ---------------------------------------------------------------------------

    def close(self) -> None:
        if self._own_http:
            self.http.close()

    @property
    def parameters(self) -> ContractParameters:
        return self.contract.parameters if self.contract is not None else CONTRACT_PARAMETERS

    def check_contract(self) -> ContractInfo:
        """``GET /store/v1/contract``: refuse a different major, keep the effective parameters."""
        data = self._json("GET", "/contract", auth=False)
        info = ContractInfo.model_validate(data)
        if contract_major(info.contract_version) != contract_major(CONTRACT_VERSION):
            raise ContractMismatch(info.contract_version)
        self.contract = info
        return info

    # --- transport ---------------------------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self.base_url}{STORE_API_PREFIX}{path}"

    def _headers(self, auth: bool, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if auth and self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        if extra:
            headers.update(extra)
        return headers

    def _backoff(self, attempt: int, retry_after: Optional[str] = None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), self.backoff_max)
            except ValueError:
                pass
        delay = min(self.backoff_max, self.backoff_base * (2 ** attempt))
        return delay * (0.5 + random.random() / 2)

    def send(self, method: str, url: str, *, headers: Dict[str, str], json: Any = None, params: Optional[Dict[str, Any]] = None,
             content: Optional[Callable[[], Union[bytes, Iterable[bytes]]]] = None, expect: Tuple[int, ...] = (200, 201, 204),
             retry: bool = True, timeout: Optional[float] = None) -> httpx.Response:
        """One logical request with retries. ``content`` is a factory so a retry resends the body.
        ``timeout`` (seconds) overrides the client default for this request."""
        attempts = self.retries + 1 if retry else 1
        last: Optional[Exception] = None
        extra: Dict[str, Any] = {"timeout": timeout} if timeout is not None else {}
        for attempt in range(attempts):
            try:
                response = self.http.request(method, url, headers=headers, json=json, params=params,
                                             content=content() if content is not None else None, **extra)
            except httpx.TransportError as exc:
                last = exc
                log.warning("Store %s %s failed (%s); attempt %d of %d", method, _path(url), type(exc).__name__, attempt + 1, attempts)
                if attempt + 1 < attempts:
                    self._sleep(self._backoff(attempt))
                    continue
                raise StoreUnavailable(f"Store is unreachable at {self.base_url} ({type(exc).__name__})") from None
            if response.status_code in expect:
                return response
            if response.status_code in RETRY_STATUSES and attempt + 1 < attempts:
                log.warning("Store %s %s answered %d; retrying", method, _path(url), response.status_code)
                self._sleep(self._backoff(attempt, response.headers.get("Retry-After")))
                continue
            raise self._error(response)
        raise StoreUnavailable(f"Store request failed: {last}")  # pragma: no cover - loop always returns or raises

    def _error(self, response: httpx.Response) -> StoreError:
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict) and isinstance(body.get("code"), str):
            return StoreError(body["code"], str(body.get("message") or ""), status=response.status_code,
                              details=body.get("details") if isinstance(body.get("details"), dict) else {},
                              retryable=bool(body.get("retryable")), request_id=body.get("request_id"))
        if response.status_code == 501:
            return StoreError("not_implemented", "Store has not implemented this route yet", status=501)
        return StoreError("unexpected_response", f"Store answered HTTP {response.status_code}", status=response.status_code,
                          retryable=response.status_code >= 500)

    def _json(self, method: str, path: str, *, body: Any = None, params: Optional[Dict[str, Any]] = None, auth: bool = True,
              retry: bool = True, timeout: Optional[float] = None) -> Any:
        payload = body.model_dump(mode="json") if isinstance(body, BaseModel) else body
        if params:
            params = {k: (v.value if hasattr(v, "value") else v) for k, v in params.items() if v is not None}
            params = {k: ("true" if v is True else "false" if v is False else v) for k, v in params.items()}
        response = self.send(method, self._url(path), headers=self._headers(auth), json=payload, params=params, retry=retry,
                             timeout=timeout)
        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    def _model(self, model: Type[M], data: Any) -> M:
        return model.model_validate(data)

    def _grant_url(self, url: str) -> str:
        """Grants point at Store. In dev mode they are ``http://localhost`` URLs on Store's own
        origin (the loopback names ``localhost``, ``127.0.0.1`` and ``::1`` count as one host, since
        Store names itself ``localhost`` whichever one Process was configured with); otherwise they
        must be HTTPS."""
        parts = urlsplit(url)
        if self.dev_mode:
            if (parts.scheme, _loopback_alias((parts.hostname or "").lower()), parts.port) != self._origin:
                raise StoreError("grant_rejected", "A dev-mode grant URL must point at the Store's own origin")
        elif parts.scheme != "https":
            raise StoreError("grant_rejected", "Grant URLs must be HTTPS")
        return url

    def _dev_tolerant(self, model: Type[M], data: Dict[str, Any], field: str = "url") -> M:
        """Validate a grant, accepting the dev-mode ``http://`` URL the contract pattern rejects."""
        try:
            return model.model_validate(data)
        except ValidationError:
            value = data.get(field)
            if not (self.dev_mode and isinstance(value, str) and value.startswith("http://")):
                raise
            checked = model.model_validate({**data, field: "https://" + value[len("http://"):]})
            return checked.model_copy(update={field: value})

    # --- status and admin state -------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """``GET /status`` as a plain dict: display only. Dev mode reports a ``localhost`` relying
        party that the contract's hostname rule rejects, so it is not validated here."""
        data = self._json("GET", "/status", auth=False, retry=False)
        return data if isinstance(data, dict) else {}

    def admin_state(self) -> Optional[Dict[str, Any]]:
        """``GET /admin/state`` (admin-state:read). ``None`` when Store has not implemented it yet
        (501), in which case the contract defaults apply."""
        try:
            data = self._json("GET", "/admin/state")
        except StoreError as exc:
            if exc.status == 501 or exc.code == "not_implemented":
                return None
            raise
        return data if isinstance(data, dict) else None

    # --- conversations and artifacts ---------------------------------------------------------

    def register_conversation(self, body: ConversationRegistration) -> ConversationRegistered:
        """``call_metadata`` goes out with only the fields the caller set (``exclude_unset``): under
        the 1.1.0 re-registration rule every field present in the body replaces the stored one, so
        sending the model's defaults would reset a known agent to "Unknown" on a re-upload."""
        payload = body.model_dump(mode="json")
        if body.call_metadata is not None:
            payload["call_metadata"] = body.call_metadata.model_dump(mode="json", exclude_unset=True)
        return self._model(ConversationRegistered, self._json("POST", "/conversations", body=payload))

    def get_conversation(self, conversation_id: str) -> Conversation:
        return self._model(Conversation, self._json("GET", f"/conversations/{conversation_id}"))

    def list_artifacts(self, conversation_id: str, *, kind: Optional[ArtifactKind] = None, slot: Optional[str] = None,
                       include_superseded: bool = False, include_unlinked: bool = False, include_draft_tests: bool = False) -> List[Artifact]:
        params: Dict[str, Any] = {"kind": kind, "slot": slot, "include_superseded": include_superseded or None,
                                  "include_unlinked": include_unlinked or None, "include_draft_tests": include_draft_tests or None}
        return [Artifact.model_validate(item) for item in self._pages(f"/conversations/{conversation_id}/artifacts", params)]

    def get_artifact(self, artifact_id: str) -> Artifact:
        return self._model(Artifact, self._json("GET", f"/artifacts/{artifact_id}"))

    def download(self, artifact: Artifact, dest: Optional[Path] = None) -> Union[bytes, Path]:
        """Fetch an artifact's bytes (``GET /artifacts/{id}/content``) and verify its checksum.
        With ``dest``, stream to that file and return the path."""
        url = self._url(f"/artifacts/{artifact.id}/content")
        headers = self._headers(True, {"Accept": "*/*"})
        for attempt in range(self.retries + 1):
            digest = hashlib.sha256()
            try:
                with self.http.stream("GET", url, headers=headers) as response:
                    if response.status_code != 200:
                        response.read()
                        if response.status_code in RETRY_STATUSES and attempt < self.retries:
                            self._sleep(self._backoff(attempt))
                            continue
                        raise self._error(response)
                    if dest is None:
                        chunks = []
                        for chunk in response.iter_bytes():
                            digest.update(chunk)
                            chunks.append(chunk)
                        data: Union[bytes, Path] = b"".join(chunks)
                    else:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with dest.open("wb") as handle:
                            for chunk in response.iter_bytes():
                                digest.update(chunk)
                                handle.write(chunk)
                        data = dest
            except httpx.TransportError as exc:
                if attempt < self.retries:
                    self._sleep(self._backoff(attempt))
                    continue
                raise StoreUnavailable(f"Store is unreachable at {self.base_url} ({type(exc).__name__})") from None
            if "sha256:" + digest.hexdigest() != artifact.checksum:
                if isinstance(data, Path):
                    data.unlink(missing_ok=True)
                raise StoreError("checksum_mismatch", f"Artifact {artifact.id} content does not match its checksum")
            return data
        raise StoreUnavailable("download failed")  # pragma: no cover

    def create_inline_artifact(self, conversation_id: str, body: InlineArtifactCreate) -> Artifact:
        return self._model(Artifact, self._json("POST", f"/conversations/{conversation_id}/artifacts", body=body))

    def upload_artifact(self, conversation_id: str, body: UploadGrantRequest, source: Union[bytes, Path]) -> Artifact:
        """Grant, PUT the bytes to the grant URL (no service key on it), then commit."""
        grant = self._dev_tolerant(UploadGrant, self._json("POST", f"/conversations/{conversation_id}/artifacts/uploads", body=body))
        url = self._grant_url(grant.url)

        def content():
            if isinstance(source, (bytes, bytearray)):
                return bytes(source)
            return _file_chunks(Path(source))

        headers = dict(grant.headers)
        headers.setdefault("Content-Type", body.content_type)
        self.send(grant.method, url, headers=headers, content=content)
        commit = UploadCommit(checksum=body.checksum, size_bytes=body.size_bytes)
        return self._model(Artifact, self._json("POST", f"/artifact-uploads/{grant.upload_id}/commit", body=commit))

    def mint_rubric_snapshot(self, conversation_id: str, rubric_id: str, version: int) -> Artifact:
        body = RubricSnapshotRequest(rubric_id=rubric_id, version=version)
        return self._model(Artifact, self._json("POST", f"/conversations/{conversation_id}/rubric-snapshots", body=body))

    def get_rubric(self, rubric_id: str) -> RubricVersion:
        return self._model(RubricVersion, self._json("GET", f"/rubrics/{rubric_id}"))

    def get_rubric_version(self, rubric_id: str, version: int) -> RubricVersion:
        return self._model(RubricVersion, self._json("GET", f"/rubrics/{rubric_id}/versions/{version}"))

    # --- Contact Signals v2 (contract 1.3.0) --------------------------------------------------

    def get_signal_taxonomy(self) -> SignalTaxonomyRecord:
        """``GET /signals/taxonomy``: the current taxonomy version and the signal settings."""
        return self._model(SignalTaxonomyRecord, self._json("GET", "/signals/taxonomy"))

    def mint_signal_taxonomy_snapshot(self, conversation_id: str, version: int) -> Artifact:
        """``POST /conversations/{id}/signal-taxonomy-snapshots``: Store copies a published version
        and the current settings into the conversation (slot ``signals:v<version>``)."""
        body = SignalTaxonomySnapshotRequest(version=version)
        return self._model(Artifact, self._json("POST", f"/conversations/{conversation_id}/signal-taxonomy-snapshots", body=body))

    # --- ASR vocabulary (contract 1.3.0, decision 33) -----------------------------------------

    def get_asr_vocabulary(self) -> AsrVocabularyRecord:
        """``GET /vocabulary`` (``getAsrVocabulary``): the effective terms dual transcription runs with,
        and whether it is active (docs/DualAsr.md section 4)."""
        return self._model(AsrVocabularyRecord, self._json("GET", "/vocabulary"))

    # --- graphs and jobs ---------------------------------------------------------------------

    def create_job_graph(self, conversation_id: str, body: JobGraphRequest) -> JobGraph:
        return self._model(JobGraph, self._json("POST", f"/conversations/{conversation_id}/job-graphs", body=body))

    def get_job_graph(self, graph_id: str) -> JobGraph:
        return self._model(JobGraph, self._json("GET", f"/job-graphs/{graph_id}"))

    def claim_jobs(self, body: ClaimRequest) -> ClaimResponse:
        # A claim whose response is lost leaves its jobs leased until the lease expires, so it is
        # not retried after the request may have reached Store.
        return self._model(ClaimResponse, self._json("POST", "/jobs/claim", body=body, retry=False))

    def heartbeat(self, job_id: str, body: HeartbeatRequest) -> HeartbeatResponse:
        return self._model(HeartbeatResponse, self._json("POST", f"/jobs/{job_id}/heartbeat", body=body))

    def complete_job(self, job_id: str, body: CompletionRequest) -> CompletionReceipt:
        return self._model(CompletionReceipt, self._json("POST", f"/jobs/{job_id}/complete", body=body))

    def fail_job(self, job_id: str, body: FailureRequest) -> FailureReceipt:
        return self._model(FailureReceipt, self._json("POST", f"/jobs/{job_id}/fail", body=body))

    def release_job(self, job_id: str, body: JobReleaseRequest) -> JobReleaseReceipt:
        return self._model(JobReleaseReceipt, self._json("POST", f"/jobs/{job_id}/release", body=body))

    def attach_late_usage(self, job_id: str, attempt_number: int, body: LateUsageReport) -> UsageRecord:
        return self._model(UsageRecord, self._json("POST", f"/jobs/{job_id}/attempts/{attempt_number}/usage", body=body))

    def retry_job(self, job_id: str, reason: str) -> Job:
        return self._model(Job, self._json("POST", f"/jobs/{job_id}/retry", body=RetryRequest(reason=reason)))

    def cancel_job(self, job_id: str, reason: str, cascade: bool = True) -> CancelResponse:
        return self._model(CancelResponse, self._json("POST", f"/jobs/{job_id}/cancel", body=CancelRequest(reason=reason, cascade=cascade)))

    def get_job(self, job_id: str) -> Job:
        return self._model(Job, self._json("GET", f"/jobs/{job_id}"))

    def list_jobs(self, *, conversation_id: Optional[str] = None, graph_id: Optional[str] = None, status: Optional[str] = None,
                  job_type: Optional[str] = None, memory_slot: Optional[str] = None, limit: Optional[int] = None,
                  retry: bool = True, timeout: Optional[float] = None) -> List[Job]:
        """Every matching job (all pages). With ``limit``, one page of at most that many: the
        on-device training start check asks ``status``, ``memory_slot=local_memory``, ``limit=1``
        (``memory_slot`` is 1.3.0)."""
        params = {"conversation_id": conversation_id, "graph_id": graph_id, "status": status, "job_type": job_type, "memory_slot": memory_slot}
        if limit is not None:
            page = self._json("GET", "/jobs", params={**params, "limit": max(1, min(int(limit), 200))}, retry=retry, timeout=timeout)
            return [Job.model_validate(item) for item in (page or {}).get("items") or []]
        return [Job.model_validate(item) for item in self._pages("/jobs", params)]

    def list_attempts(self, job_id: str) -> List[Attempt]:
        return [Attempt.model_validate(item) for item in self._pages(f"/jobs/{job_id}/attempts", {})]

    def get_progress(self, conversation_id: str, *, retry: bool = True, timeout: Optional[float] = None) -> JobGroupProgress:
        """Per-group progress. Console display reads pass ``retry=False`` and a short ``timeout``
        so a Store outage answers at once instead of after the retry backoff."""
        return self._model(JobGroupProgress, self._json("GET", f"/conversations/{conversation_id}/progress", retry=retry, timeout=timeout))

    # --- on-device training (1.3.0; docs/OnDeviceTraining.md section 2) ------------------------

    def list_training_labels(self, *, after: int = 0, limit: int = 200, kinds: Optional[Iterable[str]] = None, retry: bool = True,
                             timeout: Optional[float] = None) -> TrainingLabelPage:
        """``GET /training/labels`` (``training:read``): reviewer labels with ``seq > after``, IDs,
        enums and versions only. ``limit=0`` returns the counts only and is not audited."""
        params: Dict[str, Any] = {"after": int(after), "limit": int(limit)}
        if kinds:
            params["kinds"] = [getattr(k, "value", k) for k in kinds]
        return self._model(TrainingLabelPage, self._json("GET", "/training/labels", params=params, retry=retry, timeout=timeout))

    # --- hardware, catalog, reanalysis --------------------------------------------------------

    def upsert_hardware_profile(self, body: HardwareProfileInput) -> HardwareProfile:
        return self._model(HardwareProfile, self._json("PUT", "/hardware-profiles", body=body))

    def publish_catalog(self, body: CatalogSnapshot) -> CatalogSnapshot:
        return self._model(CatalogSnapshot, self._json("PUT", "/catalog-snapshot", body=body))

    def claim_reanalysis(self, worker_id: str, max_requests: int = 1, kinds=None) -> ReanalysisClaimResponse:
        """Claim pending requests; ``kinds`` (1.3.0) limits the claim to those kinds (None: every kind)."""
        body = ReanalysisClaimRequest(worker_id=worker_id, max_requests=max_requests, kinds=list(kinds) if kinds else None)
        return self._model(ReanalysisClaimResponse, self._json("POST", "/reanalysis-requests/claim", body=body, retry=False))

    def reject_reanalysis(self, request_id: str, claim_token: str, reason: str) -> ReanalysisRequest:
        body = ReanalysisReject(claim_token=claim_token, reason=reason)
        return self._model(ReanalysisRequest, self._json("POST", f"/reanalysis-requests/{request_id}/reject", body=body))

    # --- paging ------------------------------------------------------------------------------

    def _pages(self, path: str, params: Dict[str, Any], limit: int = 200) -> List[Any]:
        items: List[Any] = []
        token: Optional[str] = None
        for _ in range(10_000):
            page = self._json("GET", path, params={**params, "limit": limit, "page_token": token})
            items.extend(page.get("items") or [])
            token = page.get("next_page_token")
            if not token:
                return items
        return items  # pragma: no cover


def _file_chunks(path: Path, size: int = 1 << 20):
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(size)
            if not chunk:
                return
            yield chunk


LOOPBACK_NAMES = frozenset({"localhost", "127.0.0.1", "::1"})


def _loopback_alias(host: str) -> str:
    return "localhost" if host in LOOPBACK_NAMES else host


def _path(url: str) -> str:
    return urlsplit(url).path
