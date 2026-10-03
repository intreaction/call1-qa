"""End to end on the fake-handler stack: Contact Signals v2 with rules detection (contract 1.4.0,
docs/SignalsEmbeddings.md). A seed taxonomy gives intent a rules recipe and caller_confirms_resolved a
rules recipe with a Gemma check; the other categories stay on the (fake) classifier. After
``python -m call1.store signals-detection rules`` a new call's hits carry their "why"."""

from __future__ import annotations

import json

from call1.contracts.signals import SignalTaxonomySave, builtin_signal_taxonomy

from .store_api_support import poll


def _seed(path):
    data = builtin_signal_taxonomy().model_dump(mode="json")
    phrase = {"op": "rule", "rule": {"rule_id": "lexicon", "params": {"type": "phrase"}}}
    recipes = {
        "intent": {"engine": "rules", "filter": {"op": "all", "children": [phrase]},
                   "lexicon": {"syntax": "words", "phrases": ["question about"]}, "lexicon_weight": 0.5, "threshold": 0.4},
        "caller_confirms_resolved": {"engine": "rules", "filter": {"op": "all", "children": [phrase]},
                                     "lexicon": {"syntax": "words", "phrases": ["makes sense"]}, "lexicon_weight": 0.5, "threshold": 0.4,
                                     "check": "gemma"},
    }
    for c in data["categories"]:
        if c["category_id"] in recipes:
            c["recipe"] = recipes[c["category_id"]]
    body = {"taxonomy": data, "expected_record_version": 1, "notes": "e2e rules seed"}
    SignalTaxonomySave.model_validate(body)
    path.write_text(json.dumps(body))
    return path


def test_rules_detection_end_to_end_on_fake_handlers(stack_factory, tmp_path):
    private = stack_factory(name="rules", signals_pipeline="v2", signals_seed=str(_seed(tmp_path / "seed.json")))
    private.store_cli("signals-detection", "rules")
    reviewer = private.user("reviewer")
    receipt = private.ingest("call_01_compliant", agent_id="agent-rules")
    call_id = receipt["call_id"]
    private.wait_until_settled(call_id)
    body = poll(lambda: reviewer.get(f"/calls/{call_id}/contact-signals").json(), lambda b: b.get("pipeline") == "v2",
                timeout=20, what="the v2 contact signals")
    assert body["completeness"] == "complete", body.get("partial_reason")
    hits = {h["category_id"]: h for h in body["signals"]}
    assert "intent" in hits and hits["intent"]["why"]["category_source"] == "rules", body["signals"]
    rule = hits["intent"]["why"]["rule"]
    assert rule["lexicon_match"] and rule["outcomes"][0]["type"] == "phrase" and rule["score"] >= rule["threshold"]
    assert hits["intent"]["why"]["subcategory_source"] == "rules"
    if "caller_confirms_resolved" in hits:  # checked by the fake stage 2
        assert hits["caller_confirms_resolved"]["why"]["check"] == "confirmed"
    others = [h for h in body["signals"] if h["category_id"] not in ("intent", "caller_confirms_resolved")]
    assert all(h["why"]["category_source"] == "gemma" and h["why"]["rule"] is None for h in others)
