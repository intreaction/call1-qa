"""The model PII layer end to end through the fake stack (contract 1.2.0, team decision 19).

The fake ASR's ``caller_name`` script has the caller say "my name is Maria Lopez". The labelled stub
PII detector flags the name in the ``enrichment`` job's ``pii_findings``, and Store masks it on every
reviewer read, while the agent's own introduction ("this is Sam") stays visible.
"""

from __future__ import annotations

import pytest

from .store_api_support import poll

pytestmark = pytest.mark.e2e


def _no_name(text: str) -> bool:
    return "Maria" not in text and "Lopez" not in text


def test_stub_detector_masks_a_caller_name_in_stores_reviewer_reads(stack_factory):
    private = stack_factory(name="pii", fake_behavior={"asr": ["caller_name"]})
    reviewer = private.user("reviewer")
    receipt = private.ingest("call_01_compliant", agent_id="agent-pii")
    call_id = receipt["call_id"]
    private.wait_until_settled(call_id)

    detail = reviewer.get(f"/calls/{call_id}")
    assert detail.status_code == 200, detail.text
    groups = {g["kind"]: g for g in detail.json()["results"]}
    for kind in ("transcript", "qa", "summary", "contact_signals"):  # the masked jobs' pii_findings edges all resolved
        assert groups[kind]["state"] == "available", groups[kind]

    view = poll(lambda: reviewer.get(f"/calls/{call_id}/transcript").json(), lambda v: not v.get("text_withheld", True),
                timeout=20, what="transcript text to be served")
    texts = [t["text"] for t in view["turns"]]
    caller = next(t for t in texts if "question about a fee" in t)
    assert _no_name(caller) and "[REDACTED]" in caller, caller
    assert all(_no_name(t) for t in texts), texts
    assert any("this is Sam" in t for t in texts), texts  # the agent's own name is kept

    summary = reviewer.get(f"/calls/{call_id}/summary").json()
    assert _no_name(summary["narrative"]) and all(_no_name(k) for k in summary["key_points"])
    hits = reviewer.post("/search/semantic", json={"query": "question about a fee", "call_id": call_id, "top_k": 20}).json()["results"]
    assert hits and all(_no_name(h["text"]) for h in hits), hits
    audio = reviewer.get(f"/calls/{call_id}/audio")
    assert audio.status_code == 200, audio.text[:500]
