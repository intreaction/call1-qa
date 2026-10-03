"""The LLM transport the QA, summary and contact-signal handlers share.

Every text generation goes through ``call1.question_models.generate_text``, the pre-split registry
transport, so the Stage 0 Pro1 closure (``is_call1_operated``: customer text never goes to a
Call1-operated endpoint) and the external-model gate apply unchanged. The frozen catalog entry is
turned back into the legacy ``QuestionModel`` it came from:

* ``call1-bundled`` is the included model (source ``internal``): MLX in-process when
  ``CALL1_BACKEND=mlx``, else the loopback Ollama the legacy app used.
* a local pack (``gemma4-e4b`` ...) is a ``call1`` source with its ``local_model_id``, always MLX.

Stage 2 is appliance-only: a selection on any other route class is refused before inference
(``route_disabled``; ``route_policy_rejected`` for ``call1_confidential``, which stays closed).

Failures map to contract ``JobErrorCode``s by what the legacy transport raised, with safe details
only (never a provider body, prompt or transcript text).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from call1.adapters.mlx import use_text_adapter
from call1.process.training.registry import lora_suffix

from call1.contracts.common import ArtifactRef, canonical_digest
from call1.contracts.contents import PromptInputContent, TurnWindow
from call1.contracts.custody import RouteClass
from call1.contracts.errors import JobErrorCode
from call1.contracts.usage import TokenCount, TokenSource

from call1.process.handlers.base import HandlerError, HandlerJob, ReleaseJob, Usage
from call1.process.transcripts import estimate_tokens

from .paths import local_pack_ids, mlx_backend, peak_memory, reset_peak_memory, weights_path

_CONTEXT = ("context budget", "context limit")
_MISSING_MODEL = ("Provision the", "not installed or supported", "is no longer registered")


def question_model(job: HandlerJob):
    """The legacy ``QuestionModel`` for the job's frozen catalog entry."""
    from call1.model_catalog import LOCAL_PACKS
    from call1.models.schemas import QuestionModel, QuestionModelsSettings

    entry = job.catalog_entry
    if entry is None:
        raise HandlerError(JobErrorCode.CONFIGURATION_ERROR, "the job has no catalog entry for its model")
    if entry.entry_id in LOCAL_PACKS:
        pack = LOCAL_PACKS[entry.entry_id]
        return QuestionModel(id=entry.entry_id, name=pack["name"], source="call1", model=pack["model"], local_model_id=entry.entry_id,
                             max_tokens=entry.output_token_limit or 768)
    bundled = next(m for m in QuestionModelsSettings().models if m.source == "internal")
    if entry.legacy_question_model_id == bundled.id or entry.entry_id == bundled.id:
        return bundled
    raise HandlerError(JobErrorCode.ROUTE_DISABLED, "only the included model and local packs are served in Stage 2")


def check_route(job: HandlerJob) -> None:
    """Refuse, before inference, a route this Process does not serve in Stage 2."""
    selection = job.selection
    if selection is None:
        raise ReleaseJob("reject", JobErrorCode.CONFIGURATION_ERROR, "an LLM job needs a frozen model selection")
    route = selection.route.route_class
    if route is RouteClass.CALL1_CONFIDENTIAL:
        raise ReleaseJob("reject", JobErrorCode.ROUTE_POLICY_REJECTED, "the Pro1 route is closed until attested inference is qualified")
    if route is not RouteClass.APPLIANCE:
        raise ReleaseJob("reject", JobErrorCode.ROUTE_DISABLED, "Stage 2 serves the appliance route only")


def map_generation_error(exc: BaseException) -> HandlerError:
    """A legacy transport exception as a contract error code with safe text."""
    from call1.question_models import PRO1_UNAVAILABLE

    message = str(exc)
    if isinstance(exc, HandlerError):
        return exc
    if message == PRO1_UNAVAILABLE:
        return HandlerError(JobErrorCode.ROUTE_POLICY_REJECTED, "Pro1 inference is closed until the attested route is qualified")
    if "External model processing is disabled" in message:
        return HandlerError(JobErrorCode.ROUTE_DISABLED, "external model processing is disabled")
    if "credential is missing" in message:
        return HandlerError(JobErrorCode.CREDENTIAL_MISSING, "the model credential is not set on this Process host")
    if any(k in message for k in _CONTEXT):
        return HandlerError(JobErrorCode.CONTEXT_LIMIT_EXCEEDED, "the prompt exceeds the model's context budget")
    if isinstance(exc, ImportError) or any(k in message for k in _MISSING_MODEL) or "Model is disabled" in message:
        return HandlerError(JobErrorCode.MODEL_UNAVAILABLE, "the model is not installed or cannot be loaded on this host")
    if isinstance(exc, TimeoutError):
        return HandlerError(JobErrorCode.PROVIDER_TIMEOUT, "the model did not answer in time")
    return HandlerError(JobErrorCode.PROVIDER_ERROR, f"the model request failed ({type(exc).__name__})")


@dataclass
class Generation:
    raw: str
    seconds: float
    usage: Dict[str, int] = field(default_factory=dict)


class LlmTransport:
    """One job's generations: sends through ``generate_text``, records every request's messages
    (for the prompt digest), tokens the provider reported, and wall time."""

    def __init__(self, job: HandlerJob, *, output_limit: Optional[int] = None) -> None:
        self.job = job
        self.model = question_model(job)
        if output_limit is not None and output_limit > self.model.max_tokens:
            # ``generate_text`` clamps every request to the model's QA answer limit (768 on MLX). Batched
            # signal prompts set their own per-batch bound, already kept inside the context window.
            self.model = self.model.model_copy(update={"max_tokens": output_limit})
        entry = job.catalog_entry
        self.text_model_path: Optional[str] = None
        if entry is not None and (entry.entry_id in local_pack_ids() or (mlx_backend() and entry.runtime == "mlx")):
            path = weights_path(entry)
            self.text_model_path = str(path.resolve()) if path is not None else None
        # The on-device customer adapter (decision 28), resolved once per job: every prompt of the
        # job uses one adapter even if the active pointer changes mid-job (section 5.1).
        self.adapter_path: Optional[str] = None
        self.adapter_version: Optional[str] = None
        self._resolve_adapter()
        self.requests: List[List[Dict[str, str]]] = []
        self.tokens_in = 0
        self.tokens_out = 0
        self.reported = False
        self.seconds = 0.0
        self.peak_memory: Optional[int] = None

    def generate(self, system: str, prompt: str, *, response_schema: Optional[dict] = None, schema_name: str = "answer",
                 max_tokens: Optional[int] = None) -> Generation:
        from call1.question_models import generate_text

        self.job.check_cancelled()
        self.requests.append([{"role": "system", "content": system}, {"role": "user", "content": prompt}])
        started = time.monotonic()
        reset_peak_memory()
        try:
            with use_text_adapter(self.adapter_path):
                raw, usage = generate_text(self.model, system, prompt, False, response_schema, schema_name, max_tokens,
                                           text_model_path=self.text_model_path)
        except Exception as exc:
            self.seconds += time.monotonic() - started
            raise map_generation_error(exc) from None
        finally:
            peak = peak_memory()
            if peak is not None:
                self.peak_memory = max(self.peak_memory or 0, peak)
        elapsed = time.monotonic() - started
        self.seconds += elapsed
        usage = dict(usage or {})
        if "prompt_tokens" in usage or "completion_tokens" in usage:
            self.reported = True
            self.tokens_in += int(usage.get("prompt_tokens") or 0)
            self.tokens_out += int(usage.get("completion_tokens") or 0)
        if not isinstance(raw, str):
            raise HandlerError(JobErrorCode.PROVIDER_ERROR, "the model returned no text")
        return Generation(raw=raw, seconds=elapsed, usage=usage)

    def _resolve_adapter(self) -> None:
        """``CALL1_TEXT_ADAPTER`` stays the manual override (provenance ``+lora.env``); otherwise the
        Process's active adapter when this job's task is one it was measured on (``registry.resolve``).
        Only the included model on MLX takes an adapter; every other model runs as before."""
        from call1.adapters.mlx import INCLUDED_TEXT_MODEL
        from call1.process.training import registry

        if not self.text_model_path or Path(self.text_model_path).name != INCLUDED_TEXT_MODEL:
            return
        if os.getenv("CALL1_TEXT_ADAPTER"):
            self.adapter_version = "env"
            return
        resolved = registry.current().resolve(registry.task_for(self.job.job_type), self.text_model_path)
        if resolved is not None:
            self.adapter_path, self.adapter_version = resolved

    def model_revision(self) -> Optional[str]:
        """``<entry revision>+lora.<version>`` when an adapter served this job, else None (the frozen
        selection's revision stands). Recorded in ``AttemptProvenance`` and ``SignalStageProvenance``."""
        if not self.adapter_version:
            return None
        return lora_revision(self.job, self.adapter_version)

    # --- what the attempt records -------------------------------------------------------------

    def usage(self) -> Usage:
        """Tokens only when the provider reported them (the in-process MLX runtime does not, so
        they are recorded as unavailable, never estimated)."""
        count = (lambda n: TokenCount(count=n, source=TokenSource.PROVIDER_REPORTED)) if self.reported else (lambda n: None)
        return Usage(tokens_input=count(self.tokens_in), tokens_output=count(self.tokens_out), inference_seconds=round(self.seconds, 6) or None,
                     peak_memory_bytes=self.peak_memory)

    def tokens(self) -> tuple:
        return (self.tokens_in, self.tokens_out) if self.reported else (None, None)

    def prompt_input(self, template_id: str, template_version: str, *, window: Optional[TurnWindow] = None,
                     estimated_text: Optional[str] = None) -> PromptInputContent:
        return prompt_input(self.job, template_id, template_version, self.requests, window=window, estimated_text=estimated_text)


def lora_revision(job: HandlerJob, adapter_version: str) -> str:
    """The frozen selection's model revision with the adapter suffix ``+lora.<version>``."""
    selection = job.selection
    entry = job.catalog_entry
    base = (selection.model_revision if selection is not None else None) or getattr(entry, "model_revision", None) or "unknown"
    return f"{base}+lora.{adapter_version}"[:200]


def template_version(*parts: Any) -> str:
    """A short digest of a prompt template (system prompts, instructions, schema), so a template
    change is visible on every attempt that used it."""
    body = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return "t-" + hashlib.sha256(body).hexdigest()[:16]


def prompt_input(job: HandlerJob, template_id: str, version: str, requests: Sequence[Sequence[Dict[str, str]]], *,
                 window: Optional[TurnWindow] = None, estimated_text: Optional[str] = None) -> PromptInputContent:
    """``prompt_input.v1``: what was sent, by reference and digest only (never the text). The digest
    covers every request of the attempt in order (a malformed answer is retried once)."""
    refs: List[ArtifactRef] = [item.ref for _, item in sorted(job.inputs.items()) if item is not None]
    selection = job.selection
    text = estimated_text
    if text is None:
        text = "\n".join(message["content"] for request in requests for message in request)
    return PromptInputContent(
        template_id=template_id[:200], template_version=version[:200], prompt_digest=canonical_digest([list(r) for r in requests]),
        masked=bool(selection and selection.route.masked), transcript_window=window, inputs=refs,
        estimated_input_tokens=estimate_tokens(text) if text else 0,
    )


__all__ = ["Generation", "LlmTransport", "check_route", "lora_revision", "lora_suffix", "map_generation_error", "prompt_input", "question_model",
           "template_version"]
