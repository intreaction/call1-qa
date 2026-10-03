"""Contact Signals v2 through a real in-process Store on fake handlers (docs/ContactSignalsV2.md
sections 8.1 and 7.5): ingest mints the taxonomy snapshot and runs the cascade, an enrichment failure
dead-blocks every v2 job and the merge until it is retried, and a contact-signals reanalysis after a
field edit reruns only stage 3."""

from __future__ import annotations

from typing import Optional

from call1.contracts.common import ReviewerRole
from call1.contracts.jobs import JobStatus, JobType
from call1.contracts.signals import SignalSettings, SignalTaxonomy
from call1.process.handlers.fake import FakeBehavior

from .conftest import SAMPLE, write_headers
from .test_signals_support import REASON, cancel_taxonomy

V = "/store/v1"
V2_TYPES = {JobType.CONTACT_SIGNALS_CATEGORIZE, JobType.CONTACT_SIGNALS_SUBCATEGORIZE, JobType.CONTACT_SIGNALS_EXTRACT}


def configure(store_http, session, taxonomy: Optional[SignalTaxonomy] = None, settings: Optional[SignalSettings] = None) -> dict:
    admin = session(ReviewerRole.ADMIN)
    record = store_http.get(f"{V}/signals/taxonomy", headers=admin.read_headers)
    assert record.status_code == 200, record.text
    record = record.json()
    if taxonomy is not None:
        saved = store_http.put(f"{V}/signals/taxonomy", json={"taxonomy": taxonomy.model_dump(mode="json"),
                                                              "expected_record_version": record["record_version"]}, headers=write_headers(admin))
        assert saved.status_code == 200, saved.text
        record = saved.json()
    if settings is not None:
        saved = store_http.put(f"{V}/signals/settings", json={"settings": settings.model_dump(mode="json"),
                                                              "expected_record_version": record["record_version"]}, headers=write_headers(admin))
        assert saved.status_code == 200, saved.text
        record = saved.json()
    return record


def _jobs(runtime, conversation_id):
    return runtime.client.list_jobs(conversation_id=conversation_id)


def _state(store_http, reviewer, call_id) -> str:
    detail = store_http.get(f"{V}/calls/{call_id}", headers=reviewer.read_headers).json()
    return {g["kind"]: g["state"] for g in detail["results"]}["contact_signals"]


def test_ingest_on_v2_runs_the_cascade_and_publishes_the_hits(make_runtime, store_http, session):
    configure(store_http, session, cancel_taxonomy(), SignalSettings(pipeline="v2"))
    runtime = make_runtime(behavior=FakeBehavior({"asr": ["script:cancel"]}))
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE)
    worker.drain()
    jobs = _jobs(runtime, result.conversation_id)
    types = {j.job_type for j in jobs}
    assert V2_TYPES <= types and not {JobType.CONTACT_SIGNALS_LIFECYCLE, JobType.CONTACT_SIGNALS_RESOLUTION} & types
    assert {j.status for j in jobs} == {JobStatus.SUCCEEDED}
    for job in jobs:
        if job.job_type in V2_TYPES:
            assert job.selection.route.masked and job.parameters.signals is not None
    reviewer = session(ReviewerRole.REVIEWER)
    signals = store_http.get(f"{V}/calls/{result.call_id}/contact-signals", headers=reviewer.read_headers).json()
    assert signals["pipeline"] == "v2" and signals["completeness"] == "complete" and signals["taxonomy"]["version"] == 2
    intent = next(h for h in signals["signals"] if h["category_id"] == "intent")
    assert intent["subcategory_id"] == "cancel_account" and intent["quote_narrowed"] and intent["quote"] == "cancel my account"
    assert [(f["field_id"], f["value"]) for f in intent["fields"]] == [("reason", "price")]
    assert [s["stage"] for s in signals["stages"]] == ["categorize", "subcategorize", "extract"]


def test_an_enrichment_failure_dead_blocks_every_v2_job_until_it_is_retried(make_runtime, store_http, session):
    configure(store_http, session, None, SignalSettings(pipeline="v2"))
    runtime = make_runtime(behavior=FakeBehavior({"enrichment": ["fail:configuration_error"]}))
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE)
    worker.drain()
    jobs = _jobs(runtime, result.conversation_id)
    enrichment = next(j for j in jobs if j.job_type is JobType.ENRICHMENT)
    assert enrichment.status is JobStatus.FAILED
    for job in jobs:
        if job.job_type in V2_TYPES or job.job_type is JobType.CONTACT_SIGNALS_MERGE:
            assert job.status is JobStatus.BLOCKED and any(b.dead for b in job.blocking), job.job_type
    reviewer = session(ReviewerRole.REVIEWER)
    assert _state(store_http, reviewer, result.call_id) == "failed"
    runtime.client.retry_job(enrichment.id, "the privacy filter is back")
    worker.drain()
    assert {j.status for j in _jobs(runtime, result.conversation_id)} == {JobStatus.SUCCEEDED}
    assert _state(store_http, reviewer, result.call_id) == "available"


def test_a_contact_signals_reanalysis_after_a_field_edit_reruns_only_stage_three(make_runtime, store_http, session, clock):
    configure(store_http, session, cancel_taxonomy(), SignalSettings(pipeline="v2"))
    runtime = make_runtime(behavior=FakeBehavior({"asr": ["script:cancel"]}))
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE)
    worker.drain()
    before = {j.id for j in _jobs(runtime, result.conversation_id)}
    edited = cancel_taxonomy().model_dump(mode="json")
    intent = next(c for c in edited["categories"] if c["category_id"] == "intent")
    intent["subcategories"][0]["fields"] = [REASON.model_copy(update={"description": "Why the caller is leaving"}).model_dump(mode="json")]
    record = configure(store_http, session, SignalTaxonomy.model_validate(edited))
    reviewer = session(ReviewerRole.REVIEWER)
    created = store_http.post(f"{V}/calls/{result.call_id}/reanalysis-requests", json={"kind": "contact_signals"},
                              headers={**write_headers(reviewer), "Idempotency-Key": "rq-signals-0001"})
    assert created.status_code == 201, created.text
    clock.advance(1)
    assert runtime.reanalysis.poll_once() == 1, runtime.reanalysis.last_error
    worker.drain()
    new = [j for j in _jobs(runtime, result.conversation_id) if j.id not in before]
    assert {j.job_type for j in new} == {JobType.CONTACT_SIGNALS_EXTRACT, JobType.CONTACT_SIGNALS_MERGE}
    extract = next(j for j in new if j.job_type is JobType.CONTACT_SIGNALS_EXTRACT)
    assert extract.parameters.signals.span_keys == ["intent.t1b0"] and extract.status is JobStatus.SUCCEEDED
    signals = store_http.get(f"{V}/calls/{result.call_id}/contact-signals", headers=reviewer.read_headers).json()
    assert signals["taxonomy"]["version"] == record["current"]["version"] and signals["version"] == 2
    carried = {s["stage"]: s["spans_carried_forward"] for s in signals["stages"]}
    assert carried["categorize"] > 0 and carried["subcategorize"] > 0  # stages 1 and 2 were reused, not rerun
