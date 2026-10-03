"""Process features p1-p4 and p13 (inventory area "process"), end to end against real Store and
Process server processes: ingest and idempotency, the job graph, the worker (claims, slots,
heartbeat cancel), the fake-handler pipeline, and the import boundary of the running server.

Each test is named after its inventory feature ID. A failing test here is a finding: the assertion
message says what the acceptance criterion expected.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import Counter

import pytest

from .process_helpers import (
    SecondProcess,
    attempts_of,
    by_type,
    clean_env,
    fixed_copy,
    graph,
    job,
    jobs_of,
    mono_copy,
    one,
    rubric,
    run_python,
    sha256_file,
    validate_outputs,
    wait_job,
    wait_job_of_type,
)

pytestmark = pytest.mark.e2e

ALL_JOB_TYPES = {
    "validation_vad", "asr", "speaker_attribution", "acoustic_tone", "text_sentiment", "embeddings", "enrichment",
    "qa_deterministic", "qa_criterion", "qa_escalation", "qa_scorecard", "summary_segment", "summary_synthesis",
    "summary_assembly", "contact_signals_lifecycle", "contact_signals_resolution", "contact_signals_merge",
}
MODEL_BACKED = {"asr", "speaker_attribution", "acoustic_tone", "text_sentiment", "embeddings", "qa_criterion", "qa_escalation",
                "summary_segment", "summary_synthesis", "contact_signals_lifecycle", "contact_signals_resolution"}


def _first_json(text: str) -> dict:
    start = text.index("{")
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    return value


# --- p1: ingest -----------------------------------------------------------------------------------


def test_p1_api_ingest_registers_uploads_snapshots_and_is_idempotent(stack):
    """POST /process/api/recordings registers the conversation, uploads the source, mints the
    rubric snapshot, builds the graph and returns the Evaluate URL; the same bytes again return the
    same conversation and graph."""
    path = fixed_copy(stack)
    digest = sha256_file(path)
    first = stack.ingest(path, unique=False, agent_id="agent-p1", external_call_ref="ref-p1")
    assert first["conversation_created"] is True and first["graph_created"] is True, first
    assert first["evaluate_url"] == f"{stack.store_url}/#/calls/{first['call_id']}", first["evaluate_url"]
    assert isinstance(first["jobs"], int) and first["jobs"] > 0, first

    again = stack.ingest(path, unique=False, agent_id="agent-p1", external_call_ref="ref-p1")
    for key in ("conversation_id", "call_id", "graph_id", "evaluate_url"):
        assert again[key] == first[key], f"repeat ingest changed {key}: {first[key]!r} -> {again[key]!r}"
    # ``jobs`` is len(graph.jobs) at the time of the receipt (call1/process/ingest.py). The graph
    # legitimately grows once ASR completes and its summary follow-on jobs are added, so a repeat
    # that lands after ASR sees a larger count: same graph, never fewer jobs.
    assert again["jobs"] >= first["jobs"], f"repeat ingest reported fewer jobs: {first['jobs']} -> {again['jobs']}"
    assert again["conversation_created"] is False and again["graph_created"] is False, again

    conv = stack.store_get(f"/conversations/{first['conversation_id']}", session="service").json()
    assert conv["call_id"] == first["call_id"]
    assert conv["source"]["content_digest"] == digest, "Store's source identity is the upload's SHA-256"
    assert conv["call_metadata"]["agent_id"] == "agent-p1" and conv["call_metadata"]["external_call_ref"] == "ref-p1"

    artifacts = stack.store_get(f"/conversations/{first['conversation_id']}/artifacts", session="service", params={"limit": 200}).json()["items"]
    sources = [a for a in artifacts if a["kind"] == "source_audio"]
    assert len(sources) == 1, f"one source_audio even after a repeat ingest, got {len(sources)}"
    assert sources[0]["checksum"] == digest and sources[0]["linked"] is True
    snapshots = [a for a in artifacts if a["kind"] == "rubric_snapshot"]
    assert len(snapshots) >= 1, "ingest mints a rubric snapshot"

    initial = graph(stack, first["graph_id"])
    assert initial["reason"] == "ingest"
    assert len([j for j in initial["jobs"]]) >= first["jobs"]
    jobs = jobs_of(stack, first["conversation_id"])
    for qa in by_type(jobs, "qa_criterion") + by_type(jobs, "qa_scorecard"):
        pinned = [i for i in qa["inputs"] if i["role"] == "rubric"]
        assert pinned and pinned[0]["artifact"]["artifact_id"] in {s["id"] for s in snapshots}, qa["inputs"]

    listed = stack.process_get("/conversations", params={"limit": 200}).json()["items"]
    assert [i["conversation_id"] for i in listed].count(first["conversation_id"]) == 1, "the ledger lists a recording once"
    stack.wait_until_settled(first["call_id"])


def test_p1_cli_ingest_waits_and_repeats_idempotently(stack):
    """``python -m call1.process ingest FILE`` does the same as the API, and --wait follows the
    graph to settled; the same file again returns the same conversation and graph.

    (Store dedups a conversation on ``SourceReference.dedup_identity`` = ``<source kind>:<digest>``,
    so the same bytes uploaded through the console (``api_upload``) after a CLI import
    (``local_import``) become a second call. That is the contract's rule, not asserted here.)"""
    path = fixed_copy(stack)
    proc = stack.process_cli("ingest", str(path), "--agent-id", "agent-p1-cli", "--external-ref", "cli-ref", "--wait", "60")
    receipt = _first_json(proc.stdout)
    assert receipt["conversation_created"] is True and receipt["evaluate_url"] == f"{stack.store_url}/#/calls/{receipt['call_id']}"
    assert '"settled": true' in proc.stdout, f"--wait did not report settled:\n{proc.stdout}\n{proc.stderr}"
    again = _first_json(stack.process_cli("ingest", str(path), "--agent-id", "agent-p1-cli", "--external-ref", "cli-ref").stdout)
    assert (again["conversation_id"], again["graph_id"]) == (receipt["conversation_id"], receipt["graph_id"])
    assert again["conversation_created"] is False and again["graph_created"] is False
    conv = stack.store_get(f"/conversations/{receipt['conversation_id']}", session="service").json()
    assert conv["source"]["kind"] == "local_import", conv["source"]
    listed = stack.process_get("/conversations", params={"limit": 200}).json()["items"]
    assert [i["conversation_id"] for i in listed].count(receipt["conversation_id"]) == 1


def test_p1_unsupported_upload_is_refused_without_registering(stack):
    before = stack.process_get("/overview").json()["conversations"]
    body = stack.ingest(__file__, unique=False, filename="notes.txt", content_type="text/plain", expect_status=None)
    assert body.get("code") == "unsupported_audio", body
    assert stack.process_get("/overview").json()["conversations"] == before


# --- p2: the job graph ----------------------------------------------------------------------------


def _check_graph(stack, receipt, *, mono: bool):
    stack.wait_until_settled(receipt["call_id"])
    rub = rubric(stack)
    criteria = rub["definition"]["criteria"]
    semantic = [c for c in criteria if c["check"]["check_type"] == "semantic_judgement"]
    deterministic = [c for c in criteria if c["check"]["check_type"] != "semantic_judgement"]

    g = graph(stack, receipt["graph_id"])
    refs = {j["ref"]: j for j in g["jobs"]}
    expected = {"vad", "asr", "tone", "sentiment", "enrichment", "embeddings", "scorecard", "summary", "cs-lifecycle", "cs-resolution", "cs-merge"}
    expected |= {f"qa-{i}-{re.sub(r'[^a-z0-9_-]+', '-', c['criterion_id'].lower()).strip('-')}" for i, c in enumerate(semantic)}
    if deterministic:
        expected.add("qa-det")
    if mono:
        expected.add("speaker")
    initial = {r for r in refs if not r.startswith("sum-")}
    assert initial == expected, f"graph refs: missing {sorted(expected - initial)}, unexpected {sorted(initial - expected)}"
    assert receipt["jobs"] == len(expected), f"receipt jobs {receipt['jobs']} != planned {len(expected)}"

    jobs = {j["id"]: j for j in jobs_of(stack, receipt["conversation_id"])}
    by_ref = {ref: jobs[r["job_id"]] for ref, r in refs.items()}
    id_of = {ref: r["job_id"] for ref, r in refs.items()}
    assert all(j["idempotency_key"].startswith(f"ingest.{receipt['conversation_id']}.") for ref, j in by_ref.items() if ref in expected)

    def requires(ref):
        return set(by_ref[ref]["requires_job_ids"])

    speaker = {id_of["speaker"]} if mono else set()
    assert requires("vad") == set() and requires("asr") == set()
    if mono:
        assert requires("speaker") == {id_of["asr"]}
    for ref in ("tone", "sentiment", "embeddings"):
        assert requires(ref) == {id_of["asr"]} | speaker, (ref, requires(ref))
    # Contract 1.2.0: enrichment writes the PII findings (it reads the attribution on mono calls), and
    # every masked text-model job (the appliance default) pins them, so it also requires enrichment.
    assert requires("enrichment") == {id_of["asr"]} | speaker
    pii = {id_of["enrichment"]}
    qa_refs = [r for r in expected if r.startswith("qa-")]
    snapshot_ids = set()
    for ref in qa_refs:
        if ref == "qa-det":
            continue
        assert requires(ref) == {id_of["asr"]} | speaker | pii, (ref, requires(ref))
        rubric_input = [i for i in by_ref[ref]["inputs"] if i["role"] == "rubric"]
        assert len(rubric_input) == 1 and rubric_input[0]["artifact"], f"{ref} pins the rubric snapshot as input 'rubric'"
        snapshot_ids.add(rubric_input[0]["artifact"]["artifact_id"])
        assert by_ref[ref]["parameters"]["criterion_id"] in {c["criterion_id"] for c in semantic}
    assert len(snapshot_ids) == 1, f"every QA job pins the same snapshot: {snapshot_ids}"
    snap = stack.store_get(f"/artifacts/{snapshot_ids.pop()}", session="service").json()
    assert snap["kind"] == "rubric_snapshot" and snap["labels"]["rubric_id"] == rub["ref"]["rubric_id"]
    assert requires("scorecard") == {id_of[r] for r in qa_refs}, "the scorecard requires every assessment"
    assert {i["role"] for i in by_ref["scorecard"]["inputs"]} >= {f"assessment:{c['criterion_id']}" for c in semantic}
    assert id_of["asr"] in requires("summary")
    assert id_of["enrichment"] in requires("summary")
    assert requires("cs-lifecycle") == {id_of["asr"]} | speaker | pii and requires("cs-resolution") == {id_of["asr"]} | speaker | pii
    merge = by_ref["cs-merge"]
    assert set(merge["after_job_ids"]) == {id_of["cs-lifecycle"], id_of["cs-resolution"]}, "the merge uses after edges on both passes"
    assert {i["role"] for i in merge["inputs"] if i.get("optional")} == {"pass:lifecycle:0", "pass:resolution:0"}, merge["inputs"]

    for ref, j in by_ref.items():
        if j["job_type"] in MODEL_BACKED:
            assert j["selection"] is not None, f"{ref} ({j['job_type']}) freezes a catalog selection"
    # Follow-ons: the ASR completion added the summary segment job(s), bound to the assembly.
    all_jobs = list(jobs.values())
    segments = by_type(all_jobs, "summary_segment")
    assert segments, "ASR completion adds summary segment jobs"
    assert {i["role"] for i in by_ref["summary"]["inputs"]} >= {f"segment:{s['parameters']['segment']['index']}" for s in segments} \
        or any(i["role"] == "synthesis" for i in by_ref["summary"]["inputs"])
    return all_jobs


def test_p2_stereo_graph_covers_every_stage_with_edges_and_pinned_rubric(stack):
    receipt = stack.ingest("call_01_compliant", agent_id="agent-p2")
    jobs = _check_graph(stack, receipt, mono=False)
    assert not by_type(jobs, "speaker_attribution"), "stereo recordings get no speaker stage"


def test_p2_mono_graph_adds_speaker_attribution(stack):
    receipt = stack.ingest(mono_copy(stack), unique=False, agent_id="agent-p2-mono")
    jobs = _check_graph(stack, receipt, mono=True)
    assert len(by_type(jobs, "speaker_attribution")) == 1


# --- p3: the worker --------------------------------------------------------------------------------


def test_p3_two_workers_racing_never_share_a_claim(stack_factory):
    """A second ``process serve`` (same Store and installation, its own worker ID) races the first
    over four recordings: every job is claimed once, runs once and succeeds."""
    private = stack_factory(name="p3-race")
    second = SecondProcess(private, "racer").start()
    try:
        receipts = [private.ingest("call_01_compliant", agent_id=f"agent-p3-{n}") for n in range(4)]
        for receipt in receipts:
            private.wait_until_settled(receipt["call_id"])
    finally:
        second.stop()
    workers = Counter()
    for receipt in receipts:
        for j in jobs_of(private, receipt["conversation_id"]):
            assert j["status"] == "SUCCEEDED", (j["job_type"], j["status"], j.get("error_code"))
            assert j["claim_count"] == 1 and j["attempt_count"] == 1, f"{j['job_type']} {j['id']} claimed {j['claim_count']}x"
            attempts = attempts_of(private, j["id"])
            assert [a["status"] for a in attempts] == ["succeeded"], attempts
            workers[attempts[0]["worker_id"]] += 1
    assert set(workers) == {f"e2e-{private.name}", second.worker_id}, f"both workers should have run jobs: {dict(workers)}"


def test_p3_resource_slots_bound_concurrency(stack_factory):
    """Pools come from the config (mlx is always 1); a pool never runs more jobs than its size."""
    private = stack_factory(name="p3-slots", process_config={"slots": {"mlx": 1, "torch": 1, "cpu_io": 2, "outbound": 2}},
                            fake_behavior={"qa_criterion": ["hold:0.8"] * 8, "asr": ["hold:0.6"] * 2})
    pools = {p["pool"]: p for p in private.process_get("/overview").json()["worker"]["pools"]}
    assert {k: v["size"] for k, v in pools.items()} == {"mlx": 1, "torch": 1, "cpu_io": 2}, pools
    assert set(pools["torch"]["job_types"]) == {"acoustic_tone", "text_sentiment", "contact_signals_categorize", "contact_signals_subcategorize"}

    seen = []
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            overview = private.process_get("/overview").json()
            running = overview["worker"]["running"]
            seen.append(Counter(r["pool"] for r in running))
            time.sleep(0.05)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        receipts = [private.ingest("call_01_compliant", agent_id=f"agent-p3s-{n}") for n in range(2)]
        for receipt in receipts:
            private.wait_until_settled(receipt["call_id"])
    finally:
        stop.set()
        sampler.join(5)
    assert any(sum(c.values()) for c in seen), "the sampler saw no running job"
    for counts in seen:
        for pool, n in counts.items():
            assert n <= pools[pool]["size"], f"pool {pool} ran {n} jobs at once (size {pools[pool]['size']})"

    # mlx is the one shared slot: a config asking for two is refused at start-up.
    config = json.loads(private.process_config_path.read_text())
    config["slots"] = {"mlx": 2}
    bad = private.dir / "bad-slots.json"
    bad.write_text(json.dumps(config))
    proc = run_python(["-m", "call1.process", "serve"], clean_env(CALL1_PROCESS_CONFIG=str(bad), CALL1_PROCESS_HANDLERS="fake"), private.dir, timeout=30)
    assert proc.returncode != 0 and "mlx" in (proc.stdout + proc.stderr), (proc.returncode, proc.stderr[-800:])


def test_p3_heartbeat_cancel_stops_the_handler_promptly(stack_factory):
    private = stack_factory(name="p3-cancel", store_parameters={"heartbeat_interval_seconds": 5, "lease_duration_seconds": 30},
                            fake_behavior={"acoustic_tone": ["hold:60"]})
    receipt = private.ingest("call_01_compliant", agent_id="agent-p3c")
    tone = wait_job_of_type(private, receipt["conversation_id"], "acoustic_tone", lambda j: j["status"] == "RUNNING")
    started = time.monotonic()
    response = private.process_post(f"/jobs/{tone['id']}/cancel", {"reason": "e2e: stop the held tone job", "cascade": False})
    assert response.status_code == 200, response.text
    done = wait_job(private, tone["id"], lambda j: j["status"] in ("CANCELLED", "FAILED", "SUCCEEDED"), timeout=20, what="cancel")
    elapsed = time.monotonic() - started
    assert done["status"] == "CANCELLED", done["status"]
    assert elapsed <= 5 + 5, f"the handler stopped {elapsed:.1f}s after the cancel (heartbeat interval 5s; the hold was 60s)"
    attempts = attempts_of(private, tone["id"])
    assert attempts[-1]["status"] == "cancelled" and attempts[-1]["error_code"] == "cancelled", attempts[-1]
    running = private.process_get("/overview").json()["worker"]["running"]
    assert tone["id"] not in {r.get("job_id") for r in running}, running
    progress = private.wait_until_settled(receipt["call_id"])
    assert {g["kind"]: g["state"] for g in progress["groups"]}["transcript"] == "available"


# --- p4: fake handlers --------------------------------------------------------------------------


def test_p4_fake_handlers_complete_every_job_type_with_schema_valid_outputs(stack_factory):
    """Every job type runs end to end on fake handlers: a stereo call and a mono call, with
    two-turn summary segments (pairwise synthesis) and an escalation model, REG-01 flagged."""
    private = stack_factory(name="p4-fake", process_config={"summary_batch_turns": 2, "escalation_entry_id": "gemma4-e4b"},
                            fake_behavior={"qa_criterion:REG-01": ["needs_review"]})
    stereo = private.ingest("call_01_compliant", agent_id="agent-p4")
    mono = private.ingest(mono_copy(private), unique=False, agent_id="agent-p4-mono")
    seen = set()
    for receipt in (stereo, mono):
        progress = private.wait_until_settled(receipt["call_id"])
        states = {g["kind"]: g["state"] for g in progress["groups"]}
        assert set(states.values()) == {"available"}, states
        for j in jobs_of(private, receipt["conversation_id"]):
            assert j["status"] == "SUCCEEDED", (j["job_type"], j["status"], j.get("error_code"), j.get("error_detail"))
            validate_outputs(private, j)
            attempt = attempts_of(private, j["id"])[-1]
            if j["job_type"] not in ("embeddings", "qa_scorecard", "summary_assembly", "contact_signals_merge"):
                assert attempt["provenance"]["adapter_id"] == f"fake.{j['job_type']}", attempt["provenance"]
            seen.add(j["job_type"])
    missing = ALL_JOB_TYPES - seen - {"qa_deterministic"}
    assert not missing, f"job types that never ran: {sorted(missing)}"
    if any(c["check"]["check_type"] != "semantic_judgement" for c in rubric(private)["definition"]["criteria"]):
        assert "qa_deterministic" in seen


# --- p13: the import boundary of the running server ----------------------------------------------

FORBIDDEN = re.compile(r"\|\s*(call1\.(store|db|ingest))(\.|\s*$)")


def _forbidden_imports(log_text: str):
    return sorted({m.group(1) + (m.group(3) or "") for line in log_text.splitlines() if "import time:" in line
                   for m in [FORBIDDEN.search(line)] if m})


def _imported_call1(log_text: str):
    return {line.rsplit("|", 1)[-1].strip() for line in log_text.splitlines() if "import time:" in line and "call1." in line}


def test_p13_running_process_never_imports_store_db_or_ingest(stack_factory):
    """``PYTHONPROFILEIMPORTTIME=1`` makes the real ``process serve`` log every module it imports,
    including lazy imports during ingest and job execution. None may be call1.store, call1.db or
    call1.ingest (fake mode, then real mode's handler package)."""
    private = stack_factory(name="p13-fake", process_env={"PYTHONPROFILEIMPORTTIME": "1"})
    receipt = private.ingest("call_01_compliant", agent_id="agent-p13")
    private.wait_until_settled(receipt["call_id"])
    private.process_get("/overview"), private.process_get("/catalog"), private.process_get(f"/conversations/{receipt['conversation_id']}")
    log = (private.logs_dir / "process.log").read_text(errors="replace")
    assert "call1.process.worker" in _imported_call1(log), "the import log was not captured"
    assert _forbidden_imports(log) == [], f"Process imported {_forbidden_imports(log)}"

    real = stack_factory(name="p13-real", handlers="real", with_process=True, process_env={"PYTHONPROFILEIMPORTTIME": "1"})
    real.process_get("/catalog")
    log = (real.logs_dir / "process.log").read_text(errors="replace")
    imported = _imported_call1(log)
    assert any(m.startswith("call1.process.handlers.real") for m in imported), sorted(m for m in imported if "handlers" in m)
    assert _forbidden_imports(log) == [], f"Process (real handlers) imported {_forbidden_imports(log)}"
