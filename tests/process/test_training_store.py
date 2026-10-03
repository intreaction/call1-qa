"""On-device training against the in-process Store (docs/OnDeviceTraining.md section 7.5, step 2):
a recording runs through the fake-handler pipeline, a reviewer overrides a QA verdict through
Store's review API, and Train now pages the label through ``listTrainingLabels``, replays its
source job and artifacts with the Process key, and reaches a decision. Fake trainer and fake
generator only; no MLX, torch or real model runs."""

from __future__ import annotations

from call1.contracts.common import ReviewerRole, ServiceScope
from call1.pipeline.signals_v2 import estimate_tokens

from .conftest import PROCESS_SCOPES, SAMPLE, write_headers
from .training_support import fake_base, loose_settings

V = "/store/v1"


def test_train_now_reads_real_labels_and_sources_from_store(make_runtime, mint_key, store_http, session, store, tmp_path):
    runtime = make_runtime(key=mint_key(list(PROCESS_SCOPES) + [ServiceScope.TRAINING_READ, ServiceScope.JOBS_CONTROL]))
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE)
    worker.drain()
    reviewer = session(ReviewerRole.REVIEWER)
    review = store_http.get(f"{V}/calls/{result.call_id}/review", headers=reviewer.read_headers).json()
    body = {"status": "FAIL", "reason_code": "model_misread_evidence", "reviewer_notes": "not what the policy asks", "evaluation_version": 1,
            "expected_version": review["review_version"]}
    response = store_http.post(f"{V}/calls/{result.call_id}/verdicts/REG-01", json=body, headers=write_headers(reviewer))
    assert response.status_code == 200, response.text

    base = fake_base(tmp_path)
    training = runtime.training
    training.base_model = base
    runtime.adapters.base = base
    training._count_tokens = estimate_tokens
    training.settings = loose_settings()
    view = training.describe()
    assert view["labels"] == {"total": 1, "new_since_last_run": 1, "error": None} and view["available"] is True

    run = training.request_run()
    record = training.wait(timeout=60)
    assert record["run_id"] == run["run_id"] and record["status"] in ("promoted", "skipped", "rejected"), record
    assert record["labels"]["qa_verdict"] == 1 and record["label_cursor"]["to"] == 1
    skipped = record["skipped"]
    assert "source_unavailable" not in skipped and "no_pii_findings" not in skipped, skipped
    produced = record["examples"]["train"] + record["examples"]["valid"] + record["examples"]["eval_items"]
    assert produced + sum(skipped.values()) == 1
    assert worker.claims_paused is None
    # Store saw the label reads (audited, counts only) and nothing else about the run
    with store.connection() as conn:
        reads = [dict(r) for r in conn.execute("SELECT * FROM audit_events WHERE action = 'training_labels_read' ORDER BY sequence").fetchall()]
    assert reads and "not what the policy asks" not in str(reads)
