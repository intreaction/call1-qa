"""Contact Signals v2 on the real handler registry and the real catalog, end to end against an
in-process Store, preserving the historical all-Gemma route with Gemma behind a scripted ``generate_text`` (team decision 24: the included model,
``call1-bundled``, runs all three stages). No GPU, no weights, no ports.

* Ingest with the Store setting ``pipeline: v2`` mints the taxonomy snapshot and plans the cascade on
  ``call1-bundled`` for every stage (no v1 pass, no ``pipeline_note``); the real categorize,
  subcategorize and extract handlers send constrained-JSON prompts through ``LlmTransport``, and the
  merge publishes a v2 result with the subcategory and the grounded field.
* Every prompt the model sees is masked, even with ``mask_model_text: off``.
* A field edit reruns only stage 3 on Gemma; a threshold-only edit re-derives stage 1 with no model.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from call1.contracts.contents import SIGNAL_NONE_OPTION, SIGNAL_OTHER_OPTION
from call1.contracts.jobs import JobStatus, JobType
from call1.contracts.signals import SignalSettings, SignalTaxonomy
from call1.process.handlers.fake import CANCEL_SCRIPT
from call1.process.handlers.real import signals_v2 as real_signals
from call1.redaction import REDACTED

from .test_real_handlers import FFMPEG
from .test_real_handlers_store import ScriptedLlm, _real_runtime
from .conftest import SAMPLE, write_headers
from .test_signals_store import configure
from .test_signals_support import REASON, cancel_taxonomy

V = "/store/v1"
V2_TYPES = {JobType.CONTACT_SIGNALS_CATEGORIZE, JobType.CONTACT_SIGNALS_SUBCATEGORIZE, JobType.CONTACT_SIGNALS_EXTRACT}
V1_TYPES = {JobType.CONTACT_SIGNALS_LIFECYCLE, JobType.CONTACT_SIGNALS_RESOLUTION}
SCRIPT = [(role.value, text) for role, text in CANCEL_SCRIPT]
DIGITS = "1234"


class ScriptedGemma(ScriptedLlm):
    """The QA and summary answers of ``ScriptedLlm``, plus the three Contact Signals v2 schemas answered
    the way Gemma should: stage 1 picks ``intent`` on the caller's cancel line and "none" elsewhere,
    stage 2 picks ``cancel_account``, stage 3 extracts ``reason: price`` and narrows the quote."""

    def __call__(self, model, system, prompt, allow_external=False, response_schema=None, schema_name="answer", max_tokens=None,
                 synthetic=False, text_model_path=None):
        if schema_name not in ("signal_choice", "signal_extraction"):
            return super().__call__(model, system, prompt, allow_external, response_schema, schema_name, max_tokens, synthetic, text_model_path)
        self.calls[schema_name] += 1
        self.prompts.append((schema_name, prompt))
        self.models = getattr(self, "models", []) + [model.id]
        body = json.loads(prompt)
        if "targets" in body:
            return json.dumps({target["id"]: {"form": "agent_request"}
                               for target in body["targets"]}), {}
        if schema_name == "signal_extraction":
            return json.dumps({span["span_id"]: {"assessment": "The caller cancels because the price rose.",
                                                 "reason": {"value": "price", "evidence_quote": "the price went up"},
                                                 "quote": "cancel my account"} for span in body["spans"]}), {}
        if "spans" in body:  # stage 2
            answer = {}
            for span in body["spans"]:
                options = [o["id"] for o in span["options"]]
                choice = "cancel_account" if "cancel" in span["span"] and "cancel_account" in options else SIGNAL_OTHER_OPTION
                answer[span["id"]] = {"assessment": "The caller asks to cancel.", "fits": "yes", "choice": choice, "speech_act": "request", "objective_status": "new_request"}
            return json.dumps(answer), {}
        answer = {}
        for segment in body["segments"]:
            hit = "cancel my account" in segment["text"] and "intent" in segment["options"]
            answer[segment["id"]] = {"labels": ["intent"] if hit else [SIGNAL_NONE_OPTION]}
        return json.dumps(answer), {}


@pytest.fixture
def gemma_runtime(make_runtime, monkeypatch, tmp_path):
    def build(**overrides):
        runtime = _real_runtime(make_runtime, monkeypatch, tmp_path, script=SCRIPT, name=f"gemma-{len(overrides)}", legacy_signals=False, **overrides)
        llm = ScriptedGemma()
        monkeypatch.setattr("call1.question_models.generate_text", llm)
        runtime.test_llm = llm  # type: ignore[attr-defined]
        return runtime
    return build


def _jobs(runtime, conversation_id):
    return runtime.client.list_jobs(conversation_id=conversation_id)


def _failed(jobs):
    return {(j.job_type.value, j.status.value, j.error_code, j.error_detail) for j in jobs if j.status is not JobStatus.SUCCEEDED}


def _signal_prompts(llm):
    return [(kind, prompt) for kind, prompt in llm.prompts if kind in ("signal_choice", "signal_extraction")]


@FFMPEG
@pytest.mark.parametrize("mask_model_text", ["store", "off"])
def test_real_ingest_on_pipeline_v2_runs_every_stage_on_gemma_and_publishes_the_hit(gemma_runtime, store_http, session, mask_model_text):
    configure(store_http, session, cancel_taxonomy(), SignalSettings(pipeline="v2"))
    runtime = gemma_runtime(mask_model_text=mask_model_text)
    assert runtime.registry.mode == "real" and runtime.registry.missing() == []
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE, agent_id="agent-9")
    worker.drain()
    jobs = _jobs(runtime, result.conversation_id)
    assert _failed(jobs) == set()
    types = {j.job_type for j in jobs}
    assert V2_TYPES <= types and not V1_TYPES & types
    for job in jobs:
        if job.job_type in V2_TYPES:
            assert job.selection.catalog_entry.entry_id == "call1-bundled", job.job_type
            assert job.selection.route.masked and job.parameters.signals is not None
            assert any(i.role == "pii_findings" for i in job.inputs)
    merge = next(j for j in jobs if j.job_type is JobType.CONTACT_SIGNALS_MERGE)
    assert "pipeline_note" not in merge.parameters.extra
    adapters = {j.job_type: runtime.client.list_attempts(j.id)[0].provenance.adapter_id for j in jobs if j.job_type in V2_TYPES}
    assert adapters == {JobType.CONTACT_SIGNALS_CATEGORIZE: "call1.signals.categorize",
                        JobType.CONTACT_SIGNALS_SUBCATEGORIZE: "call1.signals.subcategorize",
                        JobType.CONTACT_SIGNALS_EXTRACT: "call1.signals.extract"}

    llm = runtime.test_llm
    assert llm.calls["signal_choice"] >= 2 and llm.calls["signal_extraction"] == 1
    assert not any(kind.startswith("contact_signals_") for kind in llm.calls)  # no v1 pass ran
    assert set(llm.models) == {"call1-bundled"}
    for _, prompt in _signal_prompts(llm):  # always masked (decision 22, Q12), whatever mask_model_text says
        assert DIGITS not in prompt
    assert any(REDACTED in prompt for _, prompt in _signal_prompts(llm))

    reviewer = session()
    detail = store_http.get(f"{V}/calls/{result.call_id}", headers=reviewer.read_headers).json()
    assert {g["kind"]: g["state"] for g in detail["results"]}["contact_signals"] == "available"
    signals = store_http.get(f"{V}/calls/{result.call_id}/contact-signals", headers=reviewer.read_headers).json()
    assert signals["pipeline"] == "v2" and signals["completeness"] == "complete"
    assert [s["stage"] for s in signals["stages"]] == ["categorize", "subcategorize", "extract"]
    assert {s["provenance"]["catalog_entry_id"] for s in signals["stages"]} == {"call1-bundled"}
    templates = {s["stage"]: s["provenance"]["question_template"] for s in signals["stages"]}
    assert templates["categorize"] == real_signals.STAGE1_TEMPLATE_VERSION and templates["subcategorize"] == real_signals.STAGE2_TEMPLATE_VERSION
    [intent] = [h for h in signals["signals"] if h["category_id"] == "intent"]
    assert intent["subcategory_id"] == "cancel_account" and intent["quote"] == "cancel my account" and intent["quote_narrowed"]
    assert [(f["field_id"], f["value"]) for f in intent["fields"]] == [("reason", "price")]


def _reanalyse(runtime, store_http, reviewer, call_id, key, clock):
    created = store_http.post(f"{V}/calls/{call_id}/reanalysis-requests", json={"kind": "contact_signals"},
                              headers={**write_headers(reviewer), "Idempotency-Key": key})
    assert created.status_code == 201, created.text
    clock.advance(1)
    assert runtime.reanalysis.poll_once() == 1, runtime.reanalysis.last_error


@FFMPEG
def test_real_reanalysis_on_v2_reruns_only_the_outdated_stages_on_gemma(gemma_runtime, store_http, session, clock):
    configure(store_http, session, cancel_taxonomy(), SignalSettings(pipeline="v2"))
    runtime = gemma_runtime()
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE, agent_id="agent-9")
    worker.drain()
    assert _failed(_jobs(runtime, result.conversation_id)) == set()
    reviewer = session()
    llm = runtime.test_llm

    # A field description edit: stage 3 only, on Gemma; stages 1 and 2 are carried forward.
    before = {j.id for j in _jobs(runtime, result.conversation_id)}
    calls = Counter(llm.calls)
    edited = cancel_taxonomy().model_dump(mode="json")
    intent = next(c for c in edited["categories"] if c["category_id"] == "intent")
    intent["subcategories"][0]["fields"] = [REASON.model_copy(update={"description": "Why the caller is leaving"}).model_dump(mode="json")]
    configure(store_http, session, SignalTaxonomy.model_validate(edited))
    _reanalyse(runtime, store_http, reviewer, result.call_id, "rq-gemma-field-0001", clock)
    worker.drain()
    new = [j for j in _jobs(runtime, result.conversation_id) if j.id not in before]
    assert _failed(new) == set()
    assert {j.job_type for j in new} == {JobType.CONTACT_SIGNALS_EXTRACT, JobType.CONTACT_SIGNALS_MERGE}
    assert next(j for j in new if j.job_type is JobType.CONTACT_SIGNALS_EXTRACT).selection.catalog_entry.entry_id == "call1-bundled"
    assert llm.calls["signal_extraction"] == calls["signal_extraction"] + 1 and llm.calls["signal_choice"] == calls["signal_choice"]
    signals = store_http.get(f"{V}/calls/{result.call_id}/contact-signals", headers=reviewer.read_headers).json()
    assert signals["version"] == 2 and signals["completeness"] == "complete"
    assert [(f["field_id"], f["value"]) for h in signals["signals"] if h["category_id"] == "intent" for f in h["fields"]] == [("reason", "price")]

    # A stage-1 threshold edit: categorize re-derives from the stored Gemma scores; no model call at all.
    before = {j.id for j in _jobs(runtime, result.conversation_id)}
    calls = Counter(llm.calls)
    edited = SignalTaxonomy.model_validate(edited).model_dump(mode="json")
    next(c for c in edited["categories"] if c["category_id"] == "intent")["threshold"] = 0.8  # still below the 0.9 first pick
    configure(store_http, session, SignalTaxonomy.model_validate(edited))
    _reanalyse(runtime, store_http, reviewer, result.call_id, "rq-gemma-threshold-0001", clock)
    worker.drain()
    new = [j for j in _jobs(runtime, result.conversation_id) if j.id not in before]
    assert _failed(new) == set()
    assert {j.job_type for j in new} == {JobType.CONTACT_SIGNALS_CATEGORIZE, JobType.CONTACT_SIGNALS_MERGE}  # no new span
    [categorize] = [j for j in new if j.job_type is JobType.CONTACT_SIGNALS_CATEGORIZE]
    assert categorize.parameters.signals.stage1_mode == "rederive" and categorize.selection is None
    assert Counter(llm.calls) == calls
    signals = store_http.get(f"{V}/calls/{result.call_id}/contact-signals", headers=reviewer.read_headers).json()
    assert signals["version"] == 3 and [h["subcategory_id"] for h in signals["signals"] if h["category_id"] == "intent"] == ["cancel_account"]


# --- the real handlers on Gemma, without Store ---------------------------------------------------


class GemmaStub:
    """A ``generate_text`` stand-in for the three v2 schemas: stage 1 picks the planted ``maria``
    category on the caller's digits line and ``intent`` on the fee question, stage 2 takes the first
    listed kind, stage 3 answers each span with an assessment only."""

    def __init__(self) -> None:
        self.prompts: list = []
        self.reported = {"prompt_tokens": 100, "completion_tokens": 10}

    def __call__(self, model, system, prompt, allow_external=False, response_schema=None, schema_name="answer", max_tokens=None,
                 synthetic=False, text_model_path=None):
        self.prompts.append((schema_name, model.id, system, prompt))
        body = json.loads(prompt)
        if "targets" in body:
            return json.dumps({target["id"]: {"form": "agent_request"}
                               for target in body["targets"]}), {}
        if schema_name == "signal_extraction":
            return json.dumps({span["span_id"]: {"assessment": "Read the span."} for span in body["spans"]}), dict(self.reported)
        if "spans" in body:
            return json.dumps({span["id"]: {"assessment": "It fits.", "fits": "yes", "choice": span["options"][0]["id"]}
                               for span in body["spans"]}), dict(self.reported)
        answer = {}
        for segment in body["segments"]:
            labels = [SIGNAL_NONE_OPTION]
            if "maria" in segment["options"] and "digits" in segment["text"]:
                labels = ["maria"]
            elif "intent" in segment["options"] and "question about" in segment["text"]:
                labels = ["intent"]
            answer[segment["id"]] = {"labels": labels}
        return json.dumps(answer), dict(self.reported)


def _gemma_run(tmp_path, monkeypatch, script, taxonomy):
    from call1.process.handlers import build_registry
    from call1.process.handlers.real import signals_v2 as real_v2

    from . import test_signals_support as support

    for job_type in list(support.STAGE_ENTRY):
        monkeypatch.setitem(support.STAGE_ENTRY, job_type, "call1-bundled")
    monkeypatch.setattr(real_v2, "_device", lambda: "cpu")
    stub = GemmaStub()
    monkeypatch.setattr("call1.question_models.generate_text", stub)
    registry = build_registry("real", config=None, catalog=support.CATALOG)
    return support.run_v2(tmp_path, script, taxonomy, registry=registry), stub


def test_gemma_prompts_never_carry_a_raw_sensitive_value_even_one_planted_in_the_taxonomy(tmp_path, monkeypatch):
    """Section 11.2 and 9.4 item 2 on the real engine: the caller's name is masked in every segment,
    context line, option legend, stage-2 question and stage-3 field, including a name an admin pasted
    into a category's name, gloss, description and a subcategory."""
    from call1.contracts.signals import SignalField, SignalSubcategory
    from call1.process.handlers.fake import CALLER_NAME_SCRIPT

    from .test_signals_support import custom_category, with_categories

    note = SignalField(field_id="note", name="Maria Lopez note", type="string", description="What Maria Lopez asks", pii_class="none")
    sub = SignalSubcategory(subcategory_id="maria_sub", name="About Maria Lopez", gloss="Maria Lopez reads digits", examples=["digits"],
                            description="Maria Lopez details", fields=[note], narrow_quote=True)
    planted = custom_category("maria", name="Maria Lopez calls", gloss="Maria Lopez asks for help", speaker="CALLER",
                              examples=["last four digits"], description="Anything Maria Lopez says").model_copy(update={"subcategories": [sub]})
    run, stub = _gemma_run(tmp_path, monkeypatch, CALLER_NAME_SCRIPT, with_categories([planted]))
    kinds = Counter(kind for kind, *_ in stub.prompts)
    assert kinds["signal_choice"] >= 2 and kinds["signal_extraction"] >= 1 and {m for _, m, *_ in stub.prompts} == {"call1-bundled"}
    blob = json.dumps(stub.prompts)
    assert "Maria" not in blob and "Lopez" not in blob and REDACTED in blob
    [hit] = [h for h in run.result.signals if h.category_id == "maria"]
    assert hit.subcategory_id == "maria_sub" and "Maria" not in hit.quote
    assert {s.provenance.catalog_entry_id for s in run.result.stages if s.provenance} == {"call1-bundled"}
    # Stage 1 and 2 report the model's usage like stage 3 (tokens only because the stub reports them).
    stage1_prompts = [prompt for kind, _, _, prompt in stub.prompts if kind == "signal_choice" and "segments" in json.loads(prompt)]
    usage = run.results["categorize"].usage
    assert usage.tokens_input is not None and usage.tokens_input.count == 100 * len(stage1_prompts)
    assert run.results["subcategorize"].usage.tokens_output is not None


@pytest.mark.parametrize("job_type", [JobType.CONTACT_SIGNALS_CATEGORIZE, JobType.CONTACT_SIGNALS_SUBCATEGORIZE])
def test_the_gemma_classifier_stages_refuse_confidential_and_non_appliance_routes_before_inference(tmp_path, monkeypatch, job_type):
    from call1.contracts.catalog import RouteClass
    from call1.contracts.errors import JobErrorCode
    from call1.process.handlers.base import ReleaseJob
    from call1.process.handlers.real.signals_v2 import RealSignalsCategorize, RealSignalsSubcategorize

    from .test_signals_support import make_job, signal_params, snapshot

    stub = GemmaStub()
    monkeypatch.setattr("call1.question_models.generate_text", stub)
    handler = RealSignalsCategorize() if job_type is JobType.CONTACT_SIGNALS_CATEGORIZE else RealSignalsSubcategorize()
    snap = snapshot()
    assert handler.ready(make_job(tmp_path, job_type, {}, entry_id="call1-bundled", parameters=signal_params(snap))) is None
    for route, code in ((RouteClass.CALL1_CONFIDENTIAL, JobErrorCode.ROUTE_POLICY_REJECTED), (RouteClass.CUSTOMER_LAN, JobErrorCode.ROUTE_DISABLED)):
        job = make_job(tmp_path, job_type, {}, entry_id="call1-bundled", parameters=signal_params(snap), route=route)
        with pytest.raises(ReleaseJob) as err:
            handler.ready(job)
        assert err.value.disposition == "reject" and err.value.code is code
    assert stub.prompts == []
