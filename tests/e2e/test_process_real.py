"""Process feature p5 (inventory area "process"): the real handlers on the real models, end to end
through real Store and Process servers. Opt-in: ``CALL1_REAL_MODELS=1`` (Apple Silicon, weights in
data/models); skipped otherwise by the e2e conftest.

    CALL1_REAL_MODELS=1 .venv-local/bin/python -m pytest tests/e2e/test_process_real.py -m real_models -s -p no:warnings
"""

from __future__ import annotations

import re

import pytest

from .process_helpers import attempts_of, jobs_of, validate_outputs
from .stack import REPO

pytestmark = pytest.mark.e2e

CODE_STAGES = {"embeddings", "qa_scorecard", "contact_signals_merge"}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


@pytest.mark.real_models
def test_p5_real_handlers_run_the_full_pipeline_on_call_01(stack, admin_session):
    overview = stack.process_get("/overview").json()
    assert overview["handlers"]["mode"] == "real", overview["handlers"]
    receipt = stack.ingest("call_01_compliant", agent_id="agent-p5-real")
    progress = stack.wait_until_settled(receipt["call_id"])
    states = {g["kind"]: g["state"] for g in progress["groups"]}
    assert set(states.values()) == {"available"}, states

    jobs = jobs_of(stack, receipt["conversation_id"])
    bad = [(j["job_type"], j["parameters"].get("criterion_id"), j["status"], j.get("error_code"), j.get("error_detail")) for j in jobs if j["status"] != "SUCCEEDED"]
    assert not bad, bad
    content = {}
    for j in jobs:
        parsed = validate_outputs(stack, j)
        content.setdefault(j["job_type"], []).append(parsed)
        attempt = attempts_of(stack, j["id"])[-1]
        adapter = attempt["provenance"]["adapter_id"]
        assert not adapter.startswith("fake."), f"{j['job_type']} ran the fake adapter {adapter}"

    transcript = content["asr"][0]["transcript"]
    turns = {t.turn_id: t for t in transcript.turns}
    assert len(turns) >= 4 and all(t.text.strip() for t in turns.values())
    assert {t.speaker.value for t in turns.values()} >= {"AGENT", "CALLER"}, "the stereo call labels both sides"
    whole = _norm(" ".join(t.text for t in turns.values()))
    assert "recorded" in whole or "calling" in whole, whole[:300]

    sentiment = content["text_sentiment"][0]["text_sentiment"]
    assert {r.turn_id for r in sentiment.turns} == set(turns)
    tone = content["acoustic_tone"][0]["tone_blocks"]
    assert tone.model_dump()  # schema-valid, parsed above

    # Parity: the pre-split text sentiment function on the same transcript gives the same labels.
    from call1.pipeline.sentiment import LocalSentimentModels, analyze_text_sentiment
    from call1.process.handlers.real.convert import legacy_turns

    legacy = legacy_turns(transcript)
    analyze_text_sentiment(legacy, LocalSentimentModels(text_path=str(REPO / "data" / "models" / "roberta-sentiment")))
    for turn, row in zip(legacy, sorted(sentiment.turns, key=lambda r: r.turn_id)):
        assert row.label is None or row.label.value == turn.text_sentiment_label, (turn.turn_id, row.label, turn.text_sentiment_label)
        if row.score is not None and turn.text_sentiment is not None:
            assert abs(row.score - float(turn.text_sentiment)) < 1e-3, (turn.turn_id, row.score, turn.text_sentiment)

    # Citation checks: every quoted QA evidence is in the cited turn; summary citations name real turns.
    for parsed in content["qa_criterion"]:
        assessment = parsed["assessment"]
        if assessment.quoted_evidence:
            cited = turns.get(assessment.quote_turn_id) if assessment.quote_turn_id is not None else None
            haystack = _norm(cited.text if cited else " ".join(t.text for t in turns.values()))
            assert _norm(assessment.quoted_evidence) in haystack, (assessment.criterion_id, assessment.quoted_evidence)
    summary = content["summary_assembly"][0]["summary"]
    assert summary.narrative.strip()
    for citation in summary.citations:
        assert set(citation.turn_ids) <= set(turns), citation
    signals = content["contact_signals_merge"][0]["contact_signals"]
    for signal in signals.signals:
        assert _norm(signal.quote) in whole, (signal.label, signal.quote)

    card = content["qa_scorecard"][0]["scorecard"]
    criteria = {c["criterion_id"] for c in stack.store_get("/rubrics/call1_standard_v2", session="service").json()["definition"]["criteria"]}
    assert {v.criterion_id for v in card.verdicts} == criteria
    evaluation = admin_session.get(f"/calls/{receipt['call_id']}/evaluation")
    assert evaluation.status_code == 200, evaluation.text
