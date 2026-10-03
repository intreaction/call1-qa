"""Question model connections and bounded, evidence-checked escalation.

Credentials are dedicated server environment references, never API values.
Remote inference is opt-in and always uses a redacted transcript.
"""
from __future__ import annotations

import ipaddress
import json
import os
import time
import urllib.request
from urllib.parse import urlsplit

from call1.models.schemas import ModelAttempt, RubricVerdict, SpeakerRole, VerdictStatus
from call1.local_http import _NoRedirect
from call1.redaction import extract_sensitive_values, mask_text_with_values

from call1.qa_output import QA_SCHEMA, QA_SYSTEM

SYSTEM = QA_SYSTEM
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


def is_external(model):
    return model.source != "internal" and not model.local_model_id and urlsplit(model.endpoint).hostname not in ("localhost", "127.0.0.1", "::1")


PRO1_UNAVAILABLE = ("Pro1 inference is unavailable: Call1 hosting is not enabled on this deployment; "
                    "the open core keeps customer transcripts on your own hardware")


CALL1_DOMAINS = ("call1.cc",)


def _call1_domains():
    configured = urlsplit(os.getenv("CALL1_PRO1_ENDPOINT") or "").hostname
    return {d.rstrip(".").lower() for d in (*CALL1_DOMAINS, configured) if d}


def is_call1_operated(model):
    """True for a remote route Call1 might operate: Pro1, a Call1-owned domain or subdomain,
    or a public IP literal (whose owner cannot be checked). Private and loopback IPs stay allowed."""
    if model.local_model_id or model.source == "internal":
        return False
    if model.source in ("pro1", "call1"):
        return True
    host = (urlsplit(model.endpoint or "").hostname or "").rstrip(".").lower()
    if not host:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return any(host == d or host.endswith("." + d) for d in _call1_domains())
    return not (address.is_private or address.is_loopback or address.is_link_local)


def generate(model, prompt, allow_external=False, synthetic=False):
    return generate_text(model, SYSTEM, prompt, allow_external, QA_SCHEMA, "qa_answer", synthetic=synthetic)


def generate_text(model, system, prompt, allow_external=False, response_schema=None,
                  schema_name="answer", max_tokens=None, synthetic=False, text_model_path=None):
    """Shared registry transport for QA and summaries; provider bodies stay private.

    `synthetic` is set only by the connection test, whose prompt holds no customer data.
    `text_model_path` names the local MLX weights for the included model or a local pack when the
    caller has already resolved them (the split Process app, from its catalog); it never applies to
    a remote model.
    """
    max_tokens = min(max_tokens, model.max_tokens) if max_tokens else model.max_tokens
    if not model.enabled:
        raise RuntimeError("Model is disabled")
    if model.local_model_id:
        from call1.model_catalog import resolve_local_pack
        from call1.adapters import get_adapter
        raw = get_adapter().generate(system, prompt, max_tokens, response_schema=response_schema,
                                     text_model_path=text_model_path or resolve_local_pack(model.local_model_id))
        return raw, {}
    if model.source == "internal" and text_model_path and os.getenv("CALL1_BACKEND") == "mlx":
        from call1.adapters import get_adapter
        raw = get_adapter().generate(system, prompt, max_tokens, response_schema=response_schema,
                                     text_model_path=text_model_path)
        return raw, {}
    if model.source == "internal":
        from call1.summarizer import _call_ollama
        raw = _call_ollama("http://127.0.0.1:11434", os.getenv("CALL1_BUNDLED_OLLAMA_MODEL", "gemma4:e2b"),
                           system, prompt, max_tokens, response_schema=response_schema)
        return raw, {}
    # Plaintext Pro1 is closed: customer text never goes to a Call1-operated endpoint
    # without attestation, whatever allow_external says.
    if is_call1_operated(model) and not synthetic:
        raise RuntimeError(PRO1_UNAVAILABLE)
    if is_external(model) and not allow_external:
        raise RuntimeError("External model processing is disabled in Models settings")
    headers = {"Content-Type": "application/json"}
    if model.api_key_env:
        key = os.getenv(model.api_key_env)
        if not key:
            raise RuntimeError("The configured model credential is missing on the server")
        headers["Authorization"] = "Bearer " + key
    body = {"model": model.model, "messages": [{"role": "system", "content": system},
            {"role": "user", "content": prompt}], "max_tokens": max_tokens, "stream": False}
    if model.json_mode:
        body["response_format"] = ({"type": "json_schema", "json_schema": {
            "name": schema_name, "strict": True, "schema": response_schema}}
            if model.schema_mode and response_schema else {"type": "json_object"})
    req = urllib.request.Request(model.endpoint + "/chat/completions", data=json.dumps(body).encode(), headers=headers)
    try:
        with _OPENER.open(req, timeout=model.timeout_seconds) as response:
            payload = response.read(2_000_001)
        if len(payload) > 2_000_000:
            raise ValueError("Response too large")
        data = json.loads(payload)
        raw = data["choices"][0]["message"]["content"]
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("Empty response")
        usage = {k: v for k, v in (data.get("usage") or {}).items()
                 if k in ("prompt_tokens", "completion_tokens", "total_tokens") and type(v) is int and v >= 0}
        return raw, usage
    except Exception:
        raise RuntimeError("Model request failed; check credentials, model ID, endpoint, and compatibility") from None


class QuestionRouter:
    def __init__(self, settings):
        self.settings = settings
        self.registry = settings.question_models
        self.models = {model.id: model for model in self.registry.models}

    def evaluate(self, criterion, check, transcript, simulated_hallucination=False):
        from call1.pipeline.evaluator import RubricEvaluator, verify_quoted_evidence
        primary = check.primary_model_id or self.registry.primary_model_id
        escalation = (self.registry.escalation_model_id if check.escalation_model_id is None
                      else None if check.escalation_model_id == "none" else check.escalation_model_id)
        attempts = []
        reason = None
        for model_id in [primary, escalation]:
            if model_id is None or (attempts and model_id == primary):
                break
            model = self.models.get(model_id)
            started = time.monotonic()
            usage = {}
            failed = False
            raw_answer = None
            # Validate quotes against exactly the same masked text supplied to the model.
            sanitized = transcript.model_copy(deep=True)
            active_check = check.model_copy(deep=True)
            external = model is not None and is_external(model)
            if external or self.settings.redaction.text:
                fields = ("pass_when", "fail_when", "not_applicable_when", "policy_context")
                guidance = [{"text": getattr(active_check, field) or ""} for field in fields]
                values = extract_sensitive_values([*transcript.turns, *guidance], external or self.settings.redaction.pii_patterns)
                for turn in sanitized.turns:
                    turn.text = mask_text_with_values(turn.text, values)
                for field in fields:
                    value = getattr(active_check, field)
                    if value:
                        setattr(active_check, field, mask_text_with_values(value, values))

            def answer(prompt):
                nonlocal usage, failed, raw_answer
                try:
                    if model is None:
                        raise RuntimeError("Selected model is no longer registered")
                    raw_answer, usage = generate(model, prompt, self.registry.allow_external)
                    return raw_answer
                except Exception as exc:
                    failed = True
                    # No provider bodies, secrets, or endpoint URLs in persisted errors.
                    message = PRO1_UNAVAILABLE if str(exc) == PRO1_UNAVAILABLE else "configured model unavailable; check Models settings"
                    raise RuntimeError(message) from None

            evaluator = RubricEvaluator(model_backend=answer)
            verdict = evaluator._check_semantic_judgement(criterion, active_check, sanitized, simulated_hallucination)
            if verdict.quoted_evidence:
                verified, turn_id, timestamp = verify_quoted_evidence(verdict.quoted_evidence, sanitized, check.speaker)
                verdict.hallucination_detected = not verified
                if verified:
                    verdict.timestamp_range = timestamp
                    verdict.speaker = next(t.speaker for t in sanitized.turns if t.turn_id == turn_id)
            attempts.append(ModelAttempt(model_id=model_id, model_name=model.name if model else model_id,
                model=(os.getenv("CALL1_BUNDLED_OLLAMA_MODEL", "gemma4:e2b")
                       if model and model.source == "internal" and os.getenv("CALL1_BACKEND") != "mlx"
                       else model.model if model else "unavailable"), source=model.source if model else "unknown",
                status=verdict.status, reasoning=verdict.reasoning, quoted_evidence=verdict.quoted_evidence,
                trigger=reason, latency_ms=int((time.monotonic() - started) * 1000), usage=usage))
            parsed = evaluator._parse_semantic_answer(raw_answer) if raw_answer is not None else None
            reason = ("provider_error" if failed else "needs_review" if parsed and parsed[0] == "needs_review"
                      else "invalid_answer" if verdict.status == VerdictStatus.FLAGGED else "always")
            if len(attempts) == 2 or not (reason in check.escalation_when or "always" in check.escalation_when):
                break
        # A second assessment that disagrees cannot silently replace a grounded verdict.
        if len(attempts) == 2 and attempts[0].status != VerdictStatus.FLAGGED and attempts[0].status != verdict.status:
            verdict = RubricVerdict(criterion_id=criterion.criterion_id, criterion_name=criterion.name,
                status=VerdictStatus.FLAGGED, confidence=0, speaker=check.speaker or SpeakerRole.AGENT,
                reasoning="Primary and escalation assessments differ or escalation is unresolved. Human review is required.")
        verdict.model_attempts = attempts
        return verdict


def test_connection(settings, model_id):
    model = next((m for m in settings.question_models.models if m.id == model_id), None)
    if model is None:
        raise ValueError("Unknown model")
    from call1.pipeline.evaluator import RubricEvaluator
    from call1.models.schemas import RubricCheck, RubricCriterion, CallTranscript, TranscriptTurn, CheckType
    check = RubricCheck(check_type=CheckType.SEMANTIC_JUDGEMENT, pass_when="The agent says hello.")
    transcript = CallTranscript(call_id="connection-test", duration_seconds=1, turns=[TranscriptTurn(
        turn_id=0, speaker=SpeakerRole.AGENT, start_time=0, end_time=1, text="Hello, welcome to Call1.")])
    evaluator = RubricEvaluator(model_backend=lambda prompt: generate(
        model, prompt, settings.question_models.allow_external, synthetic=True)[0])
    verdict = evaluator._check_semantic_judgement(RubricCriterion(criterion_id="test", name="Connection test",
        description="Synthetic connection test", weight=1), check, transcript, False)
    return {"ok": verdict.status == VerdictStatus.PASS, "message": "Connected; structured answer and evidence verified."
            if verdict.status == VerdictStatus.PASS else "Connection or answer validation failed. Check the model configuration."}
