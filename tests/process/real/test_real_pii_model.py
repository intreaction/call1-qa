"""The real PII masking model (opt-in: ``CALL1_REAL_MODELS=1`` and the ``openai/privacy-filter`` weights
under ``data/models/openai-privacy-filter``, fetched with ``python -m call1.pii_model download``).
MPS when available, else CPU.

The demo PII call (``call_05_pii_heavy``) carries an SSN, a phone and a card number but no caller
name, email or address, so the test adds one clearly synthetic caller turn with those (and a date)
to the call's reference transcript. Run with ``-s`` to see load time, latency and memory.

    CALL1_REAL_MODELS=1 .venv-local/bin/python -m pytest tests/process/real/test_real_pii_model.py -m real_models -s -p no:warnings
"""

from __future__ import annotations

import json
import logging
import os
import resource
import time
from types import SimpleNamespace

import pytest

from call1 import pii_model
from call1.process.handlers.real import masking
from call1.redaction import mask_text_with_values

from ..conftest import REPO

WEIGHTS = REPO / "data" / "models" / pii_model.MODEL_DIRECTORY

pytestmark = [
    pytest.mark.real_models,
    pytest.mark.skipif(os.getenv("CALL1_REAL_MODELS") != "1", reason="set CALL1_REAL_MODELS=1 to run on the real models"),
    pytest.mark.skipif(not pii_model.weights_installed(WEIGHTS), reason=f"the PII model weights are not installed at {WEIGHTS}"),
]

SYNTHETIC_CALLER = ("For the record my name is Daniel Okafor, my email is daniel.okafor@gmail.com, and I live at "
                    "1420 West Cedar Lane, Tempe, Arizona 85281. I opened the account on May 12, 2019.")


def _call_05_turns():
    manifest = json.loads((REPO / "sample_audio" / "manifest.json").read_text())
    turns = sorted(manifest["call_05_pii_heavy"]["turns"], key=lambda t: t["start_time"])
    turns = [{"speaker": t["speaker"].upper(), "text": t["text"]} for t in turns]
    return turns[:2] + [{"speaker": "CALLER", "text": SYNTHETIC_CALLER}] + turns[2:]


def _job(turns):
    """A stand-in HandlerJob over ``turns``: no pinned findings, so masking runs the model."""
    from call1.contracts.common import ArtifactRef
    from call1.contracts.contents import TranscriptContent, TranscriptTurnContent

    transcript = TranscriptContent(duration_seconds=float(len(turns)), is_redacted=False, turns=[
        TranscriptTurnContent(turn_id=i, speaker=t["speaker"], start_time=float(i), end_time=float(i) + 0.9, text=t["text"])
        for i, t in enumerate(turns)])
    item = SimpleNamespace(ref=ArtifactRef(artifact_id="art_call05", checksum="sha256:" + "5" * 64), content=lambda: transcript)
    return SimpleNamespace(job=SimpleNamespace(conversation_id="call_05_pii_heavy"), transcript=lambda: transcript,
                           input=lambda role: None, require=lambda role: item, log=logging.getLogger("test"),
                           call_metadata=lambda: SimpleNamespace(agent_display_name="Samantha"), check_cancelled=lambda: None)


@pytest.fixture
def privacy_filter(monkeypatch):
    monkeypatch.setenv("CALL1_PII_MODEL_BACKEND", "privacy-filter")
    monkeypatch.setenv("CALL1_PII_MODEL_PATH", str(WEIGHTS))
    masking.clear_cache()
    yield
    masking.clear_cache()


def test_call_05_masks_the_caller_and_keeps_the_agent_name_and_dates(privacy_filter):
    import torch

    turns = _call_05_turns()
    job = _job(turns)

    started = time.monotonic()
    detector = pii_model.detector("privacy-filter")
    load = time.monotonic() - started
    try:
        started = time.monotonic()
        spans = detector.detect([t["text"] for t in turns])
        latency = time.monotonic() - started
        peak_gpu = torch.mps.driver_allocated_memory() if detector.device == "mps" else 0
    finally:
        detector.release()
    print(f"\nprivacy-filter on {detector.device}: load {load:.2f} s, call_05 inference {latency:.2f} s, "
          f"peak MPS driver {peak_gpu / 2**30:.2f} GiB, max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GiB")
    print("spans:", sorted({(s.label, s.text) for row in spans for s in row}))

    values = masking.sensitive_values(turns, job=job)
    masked = [mask_text_with_values(t["text"], values) for t in turns]
    text = "\n".join(masked)
    for leaked in ("Daniel", "Okafor", "daniel.okafor@gmail.com", "Cedar Lane", "85281",
                   "442-89-1099", "480-555-0199", "4111-2222-3333-4444"):
        assert leaked not in text, leaked
    assert "My name is Samantha" in masked[0] and masked[1].startswith("Hi Samantha")
    assert "May 12, 2019" in masked[2]


def test_call_05_findings_are_the_enrichment_output_store_masks_with(privacy_filter):
    """Contract 1.2.0: the enrichment job's ``pii_findings`` on the real model carry the caller's
    name, email and address (never the agent's name or a date), bound to the transcript revision,
    and the values Store derives from them mask the caller turn."""
    from call1.contracts.contents import PiiFindingsContent

    turns = _call_05_turns()
    findings = masking.pii_findings(_job(turns))
    PiiFindingsContent.model_validate(findings.model_dump(mode="json"))
    assert findings.detector == pii_model.MODEL_REPOSITORY and findings.detector_revision == pii_model.MODEL_REVISION
    assert findings.transcript.artifact_id == "art_call05"
    spans = [s for t in findings.turns for s in t.spans]
    print("findings:", sorted({(s.category, s.text) for s in spans}))
    assert all(s.category != "private_date" for s in spans)
    assert not any("Samantha" in s.text for s in spans)
    for turn in findings.turns:
        for span in turn.spans:
            assert turns[turn.turn_id]["text"][span.start:span.end] == span.text
    values = {s.text for s in spans}
    caller = mask_text_with_values(SYNTHETIC_CALLER, values)
    for leaked in ("Daniel", "Okafor", "daniel.okafor@gmail.com", "Cedar Lane"):
        assert leaked not in caller, (leaked, caller)
    assert "May 12, 2019" in caller
