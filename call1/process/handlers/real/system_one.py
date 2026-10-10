"""Laya decisions on masked seven-second Contact Signals candidate segments.

Each category is an independent binary decision, so a segment can retain two signals. The
scores are provider-reported ``noul`` values, not fitted accuracy probabilities. Gemma handles
confirmation, subcategories and extraction. Full target turns are never silently truncated.
"""
from __future__ import annotations

import json
import math
import time
from typing import Sequence

import httpx

from call1.contracts.contents import SIGNAL_NONE_OPTION
from call1.contracts.errors import JobErrorCode
from call1.contracts.jobs import JobType
from call1.contracts.usage import TokenCount, TokenSource
from call1.pipeline.signals_v2 import ChoiceRow, EngineError, RowBudget
from call1.process.handlers.base import ReleaseJob, Usage
from call1.process.system_one import ENTRY_ID, SystemOneUnavailable, discover, validate_url
from .llm import template_version

INSTRUCTIONS = "Does the target turn itself express this signal? Earlier context is background only."
NO_CRITERION = "Absent, filler, acknowledgement, or repetition."
OBJECTIVE = "A NEW, specific business request or question."
OBJECTIVE_NO = "Facts, preferences, repeats, vague fragments, or permission to ask."
TEMPLATE = template_version("signals.system_one.v1", INSTRUCTIONS, NO_CRITERION, OBJECTIVE, OBJECTIVE_NO,
                            "binary noul; full target; byte upper bound with 64 token overhead; keep_alive=0")


def byte_tokens(text: str) -> int:
    """A conservative upper bound for Laya's byte-level English tokenizer, including punctuation."""
    return len(text.encode("utf-8"))


class SystemOneClassifier:
    entry_id = ENTRY_ID
    key_orders = 1
    calibration_id = "laya-noul-unfitted-v1"
    stage1_template = TEMPLATE
    replaces_rules = True
    semantic_candidates = True
    requires_exact_revision = True

    def __init__(self, job, *, http=None):
        self.job = job
        self.entry = job.catalog_entry
        selection = job.selection
        if (job.job_type is not JobType.CONTACT_SIGNALS_CATEGORIZE or self.entry is None or selection is None
                or self.entry.entry_id != ENTRY_ID or selection.model_revision != self.entry.model_revision
                or selection.provider_model_id != self.entry.provider_model_id
                or selection.route.destination_host != self.entry.destination_host):
            raise ReleaseJob("reject", JobErrorCode.MODEL_UNAVAILABLE, "the frozen System One selection is not served here")
        self.url = validate_url(self.entry.endpoint_url)
        self.model_revision = self.entry.model_revision
        self.limit = self.entry.context_limit_tokens or 512
        self.budget = RowBudget(max_len=self.limit, head_tokens=256, count_tokens=byte_tokens)
        self.http = http
        self.owns_client = http is None
        self.input_tokens = self.output_tokens = 0
        self.reported = True
        self.seconds = 0.0
        self.requests = 0
        self.reported_model = None

    def verify(self):
        try:
            info = discover(self.url, self.entry.provider_model_id, http=self.http, fresh=True)
        except SystemOneUnavailable:
            raise EngineError("model_unavailable", "local Laya discovery failed") from None
        if info["digest"] != self.model_revision:
            raise EngineError("model_unavailable", "the local Laya digest differs from this job's frozen revision")

    def ready(self):
        try:
            self.verify()
        except EngineError as exc:
            raise ReleaseJob("reject", JobErrorCode(exc.code), exc.detail) from None

    def load(self):
        if self.http is None:
            self.http = httpx.Client(trust_env=False, follow_redirects=False, timeout=httpx.Timeout(30, connect=2))
        self.verify()

    def _request(self, row: ChoiceRow) -> dict:
        questions = {}
        for option, gloss in row.options:
            if option == SIGNAL_NONE_OPTION:
                continue
            questions[option] = {"type": "noul", "instructions": INSTRUCTIONS,
                                 "criteria": {"true": OBJECTIVE if option == "intent" else gloss,
                                              "false": OBJECTIVE_NO if option == "intent" else NO_CRITERION}}
        if not questions or len(questions) > 64:
            raise EngineError("context_limit_exceeded", "System One supports 1 to 64 category questions per segment")
        # One question is paired with the state inside the provider, not the whole question set.
        head = max(byte_tokens(json.dumps(q, ensure_ascii=False)) for q in questions.values()) + 64
        target = f"Target {row.state['speaker']}: {row.state['turn']}"
        if head + byte_tokens(target) > self.limit:
            raise EngineError("context_limit_exceeded", "a full signal segment or category gloss exceeds Laya's context; Gemma must confirm it")
        state = target
        # Only drop whole preceding segments, never the target. Restore chronological order.
        for previous in reversed(row.state.get("previous", [])):
            candidate = f"Earlier {previous['speaker']}: {previous['text']}\n" + state
            if head + byte_tokens(candidate) > self.limit:
                break
            state = candidate
        return {"model": self.entry.provider_model_id, "state": state, "questions": questions, "keep_alive": 0}

    def choose(self, rows: Sequence[ChoiceRow]):
        # Preflight all targets before inference; no half-published classification on overflow.
        payloads = [self._request(row) for row in rows]
        results = []
        for payload in payloads:
            self.job.check_cancelled()
            started = time.monotonic()
            self.requests += 1
            try:
                response = self.http.post(self.url + "/v1/systemone", json=payload)
                response.raise_for_status()
                body = response.json()
            except httpx.TimeoutException:
                self.reported = False
                raise EngineError("provider_timeout", "the local System One request timed out") from None
            except httpx.HTTPError:
                self.reported = False
                raise EngineError("provider_error", "the local System One request failed") from None
            except ValueError:
                self.reported = False
                raise EngineError("validation_rejected", "System One returned invalid JSON") from None
            finally:
                self.seconds += time.monotonic() - started
            self.job.check_cancelled()
            try:
                reported_model = body["model"]
                expected = self.entry.provider_model_id
                if reported_model not in {expected, expected + ":latest" if ":" not in expected else expected.removesuffix(":latest")}:
                    raise ValueError()
                answers = body["answers"]
                if set(answers) != set(payload["questions"]):
                    raise ValueError()
                scores = {}
                for key, answer in answers.items():
                    value = answer["noul"]
                    if (answer["type"] != "noul" or isinstance(value, bool) or not isinstance(value, (float, int))
                            or not math.isfinite(value) or not 0 <= value <= 1):
                        raise ValueError()
                    scores[key] = float(value)
                scores[SIGNAL_NONE_OPTION] = 1.0 - max(scores.values())
                usage = body.get("usage", {})
                counts = [usage.get(k) for k in ("input_tokens", "output_tokens")]
                if all(type(n) is int and n >= 0 for n in counts):
                    self.input_tokens += counts[0]
                    self.output_tokens += counts[1]
                else:
                    self.reported = False
                self.reported_model = reported_model
            except (KeyError, ValueError, TypeError, AttributeError):
                self.reported = False
                raise EngineError("validation_rejected", "System One did not return every category's bounded decision score") from None
            results.append(scores)
        self.verify()
        return results

    def usage(self):
        if not self.requests:
            return Usage()
        count = lambda n: TokenCount(count=n, source=TokenSource.PROVIDER_REPORTED) if self.reported else None
        return Usage(tokens_input=count(self.input_tokens), tokens_output=count(self.output_tokens),
                     inference_seconds=self.seconds, provider_reported_model_id=self.reported_model)

    def release(self):
        # keep_alive=0 unloads the external runner after each request, before Gemma uses MLX.
        if self.owns_client and self.http is not None:
            self.http.close()
            self.http = None
