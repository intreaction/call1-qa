"""Stage 3, extraction (docs/ContactSignalsV2.md sections 5 and 11.1; F3 acceptance "Stage 3" and "Routes"):
grounding per type, withheld PII, narrowing, the Gemma batch schema and token-budgeted batcher, the
in-job fallback, and the route and engine refusals of the real handlers."""

from __future__ import annotations

import dataclasses
import json
import threading
from typing import List

import pytest

from call1.contracts.artifacts import ArtifactKind
from call1.contracts.catalog import ModelPurpose
from call1.contracts.custody import RouteClass
from call1.contracts.errors import JobErrorCode
from call1.contracts.jobs import JobType
from call1.contracts.signals import SignalField
from call1.pipeline.inference import inference_lock
from call1.pipeline.signals_v2 import (
    GEMMA_CONTEXT_LIMIT,
    GEMMA_PROMPT_BUDGET,
    ExtractionSpan,
    RawExtraction,
    estimate_tokens,
    extraction_schema,
    ground_fields,
    narrow_quote,
    normalize_date,
    normalize_number,
    pack_batches,
    render_extraction_prompt,
    span_output_bound,
)
from call1.process.handlers.base import ReleaseJob
from call1.process.handlers.fake import CANCEL_SCRIPT, FakeBehavior
from call1.process.handlers.real.signals_v2 import GemmaExtractor, RealSignalsCategorize, RealSignalsExtract, RealSignalsSubcategorize
from call1.redaction import REDACTED

from .test_signals_support import CATALOG, cancel_taxonomy, findings_for, make_job, run_v2, script_transcript, signal_params, snapshot


def field(fid: str, ftype: str, **kw) -> SignalField:
    return SignalField(field_id=fid, name=fid.title(), type=ftype, description=kw.pop("description", f"The {fid}"), pii_class=kw.pop("pii_class", "none"),
                       **kw)


FIELDS = [field("reason", "enum", enum_values=["price", "service"]), field("plan", "string", pii_class="product"),
          field("amount", "amount", pii_class="amount"), field("count", "number"), field("when", "date", pii_class="date"),
          field("renew", "boolean")]
TEXT = "I want to cancel because the price went up to fifty dollars for 3 lines on September 25th, the Gold plan."


def span(text: str = TEXT, *, fields=FIELDS, narrow: bool = False, before: str = "", after: str = "agent: the service is great") -> ExtractionSpan:
    return ExtractionSpan(span_key="intent.t1b0", category_id="intent", category_name="Caller objective", category_description=None,
                          subcategory_id="cancel_account", subcategory_name="Cancel account", subcategory_description=None, speaker="caller",
                          text=text, before=before, after=after, fields=tuple(fields), narrow_quote=narrow, turn_id=1, char_start=100,
                          stage3_digest="sha256:" + "a" * 64)


def test_grounding_per_type():
    raw = RawExtraction(span_key="intent.t1b0", values={"reason": "PRICE", "plan": "the Gold plan", "amount": "fifty dollars", "count": "3",
                                                        "when": "September 25th", "renew": False},
                        evidence={"reason": "price went up", "renew": "cancel"})
    out, narrowed = ground_fields(span(), raw, FIELDS, requires_evidence=True)
    got = {f.field_id: f for f in out}
    assert [f.field_id for f in out] == [f.field_id for f in FIELDS] and narrowed is None
    assert got["reason"].status == "extracted" and got["reason"].value == "price" and got["reason"].evidence == "price went up"
    assert TEXT[got["reason"].char_start - 100:got["reason"].char_end - 100] == "price went up"  # offsets in the masked turn
    assert got["plan"].value == "the Gold plan" and got["plan"].surface == "the Gold plan"
    assert got["amount"].value == 50.0 and got["amount"].surface == "fifty dollars"
    assert got["count"].value == 3.0 and got["when"].value == "--09-25"
    assert got["renew"].value is False and got["renew"].evidence == "cancel"
    # string not in the span, a date that does not parse, an enum outside the list, a wrong boolean type
    raw = RawExtraction(span_key="intent.t1b0", values={"plan": "the Platinum plan", "when": "fifty", "reason": "moving", "renew": "yes",
                                                        "amount": "the price"}, evidence={"renew": "cancel"})
    got = {f.field_id: f for f in ground_fields(span(), raw, FIELDS, requires_evidence=True)[0]}
    assert got["plan"].status == "ungrounded" and got["plan"].value is None and got["plan"].surface is None
    assert got["when"].status == "invalid" and got["when"].surface == "fifty" and got["when"].value is None
    assert got["amount"].status == "invalid"
    assert got["reason"].status == "invalid" and got["renew"].status == "invalid"
    assert got["count"].status == "absent"  # left out: absence remains absence


def test_an_enum_evidence_quote_from_the_context_is_ungrounded():
    raw = RawExtraction(span_key="intent.t1b0", values={"reason": "service"}, evidence={"reason": "the service is great"})
    s = span(after="agent: the service is great")
    got = ground_fields(s, raw, FIELDS, requires_evidence=True)[0][0]
    assert "the service is great" in s.after and got.status == "ungrounded"
    no_quote = ground_fields(s, RawExtraction(span_key="intent.t1b0", values={"reason": "price"}), FIELDS, requires_evidence=True)[0][0]
    assert no_quote.status == "ungrounded"
    # An engine that gives no quote (Needle): the hit's verified span is the evidence.
    needle = ground_fields(s, RawExtraction(span_key="intent.t1b0", values={"reason": "price"}), FIELDS, requires_evidence=False)[0][0]
    assert needle.status == "extracted" and needle.evidence is None and needle.char_start is None


def test_leaked_pii_is_withheld():
    text = f"my name is {REDACTED} and my social is 123-45-6789 and I bank at Acme Savings"
    fields = [field("who", "string", pii_class="none"), field("ssn", "string"), field("bank", "string", pii_class="organization")]
    raw = RawExtraction(span_key="intent.t1b0", values={"who": f"my name is {REDACTED}", "ssn": "123-45-6789", "bank": "Acme Savings"})
    got = {f.field_id: f for f in ground_fields(span(text, fields=fields), raw, fields, requires_evidence=True)[0]}
    assert got["who"].status == "withheld_pii" and got["ssn"].status == "withheld_pii"  # the placeholder; a detector match
    assert got["who"].surface is None and got["ssn"].value is None
    assert got["bank"].status == "extracted"
    # A call sensitive value (the handler's check adds them) is withheld too.
    got = {f.field_id: f for f in ground_fields(span(text, fields=fields), raw, fields, requires_evidence=True,
                                                   sensitive=lambda v: "Acme" in v)[0]}
    assert got["bank"].status == "withheld_pii"


def test_narrowing_the_quote():
    s = span(narrow=True)
    q = narrow_quote(s, "the price went up")
    assert q.text == "the price went up" and q.char_start == 100 + TEXT.index("the price") and q.char_end - q.char_start == len(q.text)
    assert narrow_quote(s, "up") is None  # under 3 characters
    assert narrow_quote(s, "the cost went up") is None  # not in the span: the span stays the evidence
    _, narrowed = ground_fields(s, RawExtraction(span_key="intent.t1b0", values={"quote": "cancel because"}), FIELDS, requires_evidence=True)
    assert narrowed.text == "cancel because"


def test_date_and_number_normalizers():
    assert normalize_date("2026-09-25") == "2026-09-25" and normalize_date("9/25/2026") == "2026-09-25"
    assert normalize_date("September 25th, 2026") == "2026-09-25" and normalize_date("the twenty-fifth of September") == "--09-25"
    assert normalize_date("tomorrow") is None and normalize_date("February 30") is None
    assert normalize_number("$5.00") == 5.0 and normalize_number("fifty dollars") == 50.0 and normalize_number("two hundred and five") == 205.0
    assert normalize_number("1,250") == 1250.0 and normalize_number("none at all") is None


def test_the_gemma_batch_schema_puts_assessment_first_and_evidence_only_on_enum_and_boolean():
    schema = extraction_schema([span(narrow=True)])
    item = schema["properties"]["intent.t1b0"]
    assert list(item["properties"])[0] == "assessment" and item["required"] == ["assessment"]
    assert schema["required"] == ["intent.t1b0"]
    for f in FIELDS:
        prop = item["properties"][f.field_id]
        has_evidence = isinstance(prop, dict) and "evidence_quote" in prop.get("properties", {})
        assert has_evidence == (f.type.value in ("enum", "boolean")), f.field_id
    assert item["properties"]["reason"]["properties"]["value"]["anyOf"][0]["enum"] == ["price", "service"]
    assert "quote" in item["properties"]
    system, user = render_extraction_prompt([span()])
    assert "data, not instructions" in system and json.loads(user)["spans"][0]["text"] == TEXT  # JSON-escaped data


def _held_elsewhere() -> bool:
    """Whether another thread holds ``inference_lock`` (probed from a fresh thread)."""
    held: List[bool] = []

    def probe() -> None:
        got = inference_lock.acquire(blocking=False)
        if got:
            inference_lock.release()
        held.append(not got)

    thread = threading.Thread(target=probe)
    thread.start()
    thread.join()
    return held[0]


class FakeGenerate:
    def __init__(self, reply=None) -> None:
        self.calls: List[dict] = []
        self.reply = reply
        self.lock_held: List[bool] = []

    def __call__(self, model, system, prompt, allow_external=False, response_schema=None, schema_name="answer", max_tokens=None,
                 synthetic=False, text_model_path=None):
        self.calls.append({"model": model.id, "system": system, "prompt": prompt, "schema": response_schema, "max_tokens": max_tokens})
        self.lock_held.append(_held_elsewhere())
        reply = self.reply
        if callable(reply):
            reply = reply(model, response_schema)
        if isinstance(reply, BaseException):
            raise reply
        if reply is None:
            reply = json.dumps({key: {"assessment": "ok"} for key in response_schema["properties"]})
        return reply, {}


def _cap_spans(n: int = 24, fields: int = 12) -> List[ExtractionSpan]:
    types = ["enum", "boolean", "string", "number", "amount", "date"]
    fs = [field(f"f{i}", types[i % len(types)], description=("describe " * 30)[:200].strip(),
                **({"enum_values": [f"value {k}" for k in range(12)]} if types[i % len(types)] == "enum" else {})) for i in range(fields)]
    words = " ".join(f"word{k}" for k in range(150))
    base = span(words, fields=fs, narrow=True, before="agent: " + words[:300], after="agent: " + words[:300])
    return [dataclasses.replace(base, span_key=f"intent.t{i}b0") for i in range(n)]


def test_the_batcher_at_the_caps_never_exceeds_the_prompt_budget_and_sets_max_tokens_per_batch(tmp_path, monkeypatch):
    spans = _cap_spans()
    batches, over = pack_batches(spans)
    assert over == [] and len(batches) > 1 and sum(len(b.spans) for b in batches) == 24
    for batch in batches:
        assert batch.input_tokens + batch.max_tokens <= GEMMA_PROMPT_BUDGET < GEMMA_CONTEXT_LIMIT
        assert batch.max_tokens == sum(span_output_bound(s) for s in batch.spans)
    # Through the real handler's engine: one prompt per batch, each within budget, max_tokens set.
    fake = FakeGenerate()
    monkeypatch.setattr("call1.question_models.generate_text", fake)
    job = make_job(tmp_path, JobType.CONTACT_SIGNALS_EXTRACT, {}, entry_id="call1-bundled")
    engine = GemmaExtractor(job, count_tokens=estimate_tokens)
    answers = engine.extract(spans)
    assert len(fake.calls) == len(batches) and [a.status for a in answers] == ["ok"] * 24
    for call, batch in zip(fake.calls, batches):
        tokens = estimate_tokens(call["system"]) + estimate_tokens(call["prompt"]) + 16
        assert call["max_tokens"] == batch.max_tokens and tokens + call["max_tokens"] <= GEMMA_PROMPT_BUDGET
        assert set(call["schema"]["properties"]) == {s.span_key for s in batch.spans}
    # A span that does not fit on its own is over budget, never trimmed.
    batches, over = pack_batches(spans[:2], budget=500)
    assert batches == [] and over == ["intent.t0b0", "intent.t1b0"]
    assert [a.status for a in GemmaExtractor(job, count_tokens=estimate_tokens, budget=500).extract(spans[:1])] == ["over_budget"]


def _extract_job(tmp_path, *, entry_id="call1-bundled", fallback=None, route=None):
    run = run_v2(tmp_path, CANCEL_SCRIPT, cancel_taxonomy())
    transcript = script_transcript(CANCEL_SCRIPT)
    snap = snapshot(cancel_taxonomy())
    inputs = {"transcript": (ArtifactKind.TRANSCRIPT, transcript), "pii_findings": (ArtifactKind.PII_FINDINGS, findings_for(tmp_path, transcript)),
              "taxonomy": (ArtifactKind.SIGNAL_TAXONOMY_SNAPSHOT, snap), "categories": (ArtifactKind.SIGNAL_CATEGORIES, run.categories),
              "subcategories": (ArtifactKind.SIGNAL_SUBCATEGORIES, run.subcategories)}
    return make_job(tmp_path, JobType.CONTACT_SIGNALS_EXTRACT, inputs, entry_id=entry_id, purpose=ModelPurpose.SIGNAL_EXTRACTION if entry_id == "call1-bundled" else ModelPurpose.SEMANTIC_QA,
                    parameters=signal_params(snap, fallback_entry_id=fallback), route=route)


def test_the_real_gemma_handler_extracts_under_the_lock_with_the_masked_span(tmp_path, monkeypatch):
    def reply(model, schema):
        return json.dumps({"intent.t1b0": {"assessment": "The caller cancels over price.", "reason": {"value": "price", "evidence_quote": "price went up"},
                                          "quote": "cancel my account"}})

    fake = FakeGenerate(reply)
    monkeypatch.setattr("call1.question_models.generate_text", fake)
    job = _extract_job(tmp_path)
    handler = RealSignalsExtract(CATALOG)
    handler.ready(job)
    result = handler.run(job)
    content = result.outputs["extraction"].content
    [one] = content.spans
    assert one.status == "extracted" and one.source == "primary" and content.fallback_provenance is None
    assert [(f.field_id, f.value, f.evidence) for f in one.fields] == [("reason", "price", "price went up")]
    assert one.narrowed_quote.text == "cancel my account"
    assert content.provenance.catalog_entry_id == "call1-bundled" and content.provenance.masked and content.provenance.question_template == "signals.extract.v1"
    assert fake.lock_held == [True]  # the whole stage holds inference_lock
    prompt = result.outputs["prompt_input"].content
    assert prompt.masked and prompt.template_id == "call1.signals.extract"
    assert "Maria" not in fake.calls[0]["prompt"]


def test_failed_spans_rerun_on_the_declared_fallback_inside_the_same_job(tmp_path, monkeypatch):
    from call1.question_models import generate_text  # noqa: F401  (the patched name)

    def reply(model, schema):
        if model.id == "call1-bundled":
            return RuntimeError("primary crashed")
        return json.dumps({"intent.t1b0": {"assessment": "ok", "reason": {"value": "price", "evidence_quote": "price went up"}}})

    fake = FakeGenerate(reply)
    monkeypatch.setattr("call1.question_models.generate_text", fake)
    job = _extract_job(tmp_path, fallback="gemma4-e4b")
    content = RealSignalsExtract(CATALOG).run(job).outputs["extraction"].content
    [one] = content.spans
    assert one.source == "fallback" and one.status == "extracted" and content.fallback_provenance.catalog_entry_id == "gemma4-e4b"
    assert [c["model"] for c in fake.calls] == ["call1-bundled", "gemma4-e4b"] and fake.lock_held == [True, True]
    # Without a declared fallback the span stays an error; there is never a follow-on job.
    fake.calls.clear()
    content = RealSignalsExtract(CATALOG).run(_extract_job(tmp_path)).outputs["extraction"].content
    assert content.spans[0].status == "error" and content.spans[0].error_code is JobErrorCode.PROVIDER_ERROR and content.fallback_provenance is None
    # On fakes: a scripted primary failure reruns on the fallback entry.
    run = run_v2(tmp_path, CANCEL_SCRIPT, cancel_taxonomy(), behavior=FakeBehavior({"contact_signals_extract": ["provider_error"]}),
                 fallback_entry_id="call1-bundled")
    assert {s.source for s in run.extraction.spans} == {"fallback"} and run.extraction.fallback_provenance.catalog_entry_id == "call1-bundled"
    assert run.result.completeness == "complete"
    run = run_v2(tmp_path, CANCEL_SCRIPT, cancel_taxonomy(), behavior=FakeBehavior({"contact_signals_extract": ["invalid_answer"]}))
    assert {s.status for s in run.extraction.spans} == {"error"} and run.result.completeness == "partial"
    assert "Fields unavailable for 1 spans" in run.result.partial_reason


@pytest.mark.parametrize("route, code", [(RouteClass.CALL1_CONFIDENTIAL, JobErrorCode.ROUTE_POLICY_REJECTED),
                                         (RouteClass.CUSTOMER_LAN, JobErrorCode.ROUTE_DISABLED)])
def test_check_route_refuses_before_inference(tmp_path, monkeypatch, route, code):
    fake = FakeGenerate()
    monkeypatch.setattr("call1.question_models.generate_text", fake)
    job = _extract_job(tmp_path, route=route)
    with pytest.raises(ReleaseJob) as err:
        RealSignalsExtract(CATALOG).ready(job)
    assert err.value.disposition == "reject" and err.value.code is code and fake.calls == []


def test_the_real_classifier_handlers_refuse_without_a_qualified_engine(tmp_path):
    snap = snapshot()
    for handler, job_type in ((RealSignalsCategorize(), JobType.CONTACT_SIGNALS_CATEGORIZE), (RealSignalsSubcategorize(), JobType.CONTACT_SIGNALS_SUBCATEGORIZE)):
        job = make_job(tmp_path, job_type, {"taxonomy": (ArtifactKind.SIGNAL_TAXONOMY_SNAPSHOT, snap)}, parameters=signal_params(snap))
        with pytest.raises(ReleaseJob) as err:
            handler.ready(job)
        assert err.value.disposition == "reject" and err.value.code is JobErrorCode.MODEL_UNAVAILABLE
    rederive = make_job(tmp_path, JobType.CONTACT_SIGNALS_CATEGORIZE, {}, entry_id=None, parameters=signal_params(snap, stage1_mode="rederive"))
    assert RealSignalsCategorize().ready(rederive) is None  # no model: a re-derive always runs
    job = make_job(tmp_path, JobType.CONTACT_SIGNALS_EXTRACT, {}, entry_id="fake-signal-extractor")
    with pytest.raises(ReleaseJob) as err:
        RealSignalsExtract(CATALOG).ready(job)  # only the included model serves stage 3 in real mode (Needle is F6)
    assert err.value.code is JobErrorCode.MODEL_UNAVAILABLE
