"""Shared builders for the Contact Signals v2 Store tests (no tests here).

``TAXONOMY`` is the built-ins plus a "Cancel account" subcategory under intent (with an enum
``reason`` field) and a custom agent category ``upsell``. ``v2_content`` builds a valid v2
``contact_signals`` result whose hit IDs follow ``signal_hit_id``; ``v1_content`` a v1 one.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from call1.contracts.contents import (
    ContactSignalsContent,
    ContactSignalsPassOutcome,
    ContactSignalView,
    ExtractedFieldView,
    SegmentationSummary,
    SignalSpanView,
    SignalStageOutcome,
    SignalTaxonomyRef,
    SpeakerRole,
    short_digest,
    signal_hit_id,
)
from call1.contracts.signals import (
    SignalCategory,
    SignalField,
    SignalSubcategory,
    SignalTaxonomy,
    builtin_signal_taxonomy,
    category_digest,
    stage1_digest,
    subcategory_digest,
    taxonomy_digest,
)

V = "/store/v1"
NOW = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
TRANSCRIPT_CHECKSUM = "sha256:" + "ab" * 32

REASON = SignalField(field_id="reason", name="Reason", type="enum", description="Why the caller wants to cancel.",
                     enum_values=["price", "service quality", "moving", "other"], pii_class="none")
CANCEL = SignalSubcategory(subcategory_id="cancel_account", name="Cancel account", gloss="Caller wants to cancel their account",
                           examples=["cancel"], fields=[REASON])
BILLING = SignalSubcategory(subcategory_id="billing_question", name="Billing question", gloss="Asks about a bill or charge")
UPSELL = SignalCategory(category_id="upsell", builtin=False, name="Upsell attempt", gloss="Agent offers an add-on or upgrade", speaker="AGENT")


def taxonomy_with(*, intent_subs: Sequence[SignalSubcategory] = (CANCEL, BILLING), custom: Sequence[SignalCategory] = (UPSELL,),
                  intent_threshold: Optional[float] = None) -> SignalTaxonomy:
    base = builtin_signal_taxonomy()
    categories = []
    for c in base.categories:
        if c.category_id == "intent":
            c = c.model_copy(update={"subcategories": list(intent_subs), "threshold": intent_threshold})
        categories.append(c)
    return SignalTaxonomy.model_validate({"categories": [c.model_dump() for c in categories] + [c.model_dump() for c in custom]})


TAXONOMY = taxonomy_with()


def save_body(taxonomy: SignalTaxonomy, record_version: int, notes: Optional[str] = None) -> Dict:
    return {"taxonomy": taxonomy.model_dump(mode="json"), "expected_record_version": record_version, "notes": notes}


def hit(taxonomy: SignalTaxonomy, category_id: str, turn_id: int, *, block: int = 0, sub: Optional[str] = None, confidence: float = 0.9,
        quote: str = "I want to cancel my account", start: float = 2.0, end: float = 4.0, speaker: str = "CALLER",
        fields: Sequence[Dict] = (), preview: bool = False) -> ContactSignalView:
    cat = taxonomy.category(category_id)
    subcat = next((s for s in cat.subcategories if s.subcategory_id == sub), None)
    hit_id = (f"{category_id}.preview.t{turn_id}b{block}" if preview
              else signal_hit_id(category_id, category_digest(cat), TRANSCRIPT_CHECKSUM, turn_id, block))
    return ContactSignalView(
        id=hit_id, kind=category_id if cat.builtin else "custom", label=cat.name, start=start, end=end, speaker=speaker, quote=quote,
        turn_id=turn_id, char_start=0, char_end=len(quote), confidence=confidence, category_id=category_id,
        category_digest=short_digest(category_digest(cat)), category_confidence=0.95,
        subcategory_id=sub, subcategory_label=(subcat.name if subcat else ("Other" if sub == "other" else None)),
        subcategory_digest=short_digest(subcategory_digest(subcat)) if subcat else None,
        subcategory_confidence=0.8 if sub else None,
        span=SignalSpanView(block=block, first_window=block * 4, last_window=block * 4, timing="interpolated", context_start=max(0.0, start - 2),
                            context_end=end + 2),
        fields=[ExtractedFieldView.model_validate(f) for f in fields],
    )


def reason_field(value: Optional[str] = "price") -> Dict:
    if value is None:
        return {"field_id": "reason", "type": "enum", "status": "absent", "name": "Reason"}
    return {"field_id": "reason", "type": "enum", "status": "extracted", "value": value, "name": "Reason"}


def v2_content(taxonomy: SignalTaxonomy, version: int, hits: List[ContactSignalView], *, extract: Optional[bool] = None,
               completeness: str = "complete", partial_reason: Optional[str] = None) -> ContactSignalsContent:
    extract = any(h.fields for h in hits) if extract is None else extract
    stages = [SignalStageOutcome(stage="categorize", included=True), SignalStageOutcome(stage="subcategorize", included=completeness == "complete")]
    if extract:
        stages.append(SignalStageOutcome(stage="extract", included=True))
    return ContactSignalsContent(
        completeness=completeness, partial_reason=partial_reason, signals=hits, passes=[], transcript_fingerprint=TRANSCRIPT_CHECKSUM,
        generated_at=NOW, pipeline="v2", taxonomy=SignalTaxonomyRef(version=version, digest=taxonomy_digest(taxonomy)), stages=stages,
        segmentation=SegmentationSummary(segmenter_version="seg-1", window_seconds=7.0, segments=4, scored_segments=4, skipped_unattributed=0,
                                         skipped_system=0, interpolated_turns=2),
        stage1_digests={s: stage1_digest(taxonomy, SpeakerRole(s)) for s in ("AGENT", "CALLER")},
    )


def v1_content(kinds: Sequence[tuple] = (("intent", 1, "I want to cancel my account"),)) -> ContactSignalsContent:
    signals = [ContactSignalView(id=f"s{i}", kind=kind, label=kind.title(), start=2.0 * turn, end=2.0 * turn + 2, speaker="CALLER", quote=quote,
                                 turn_id=turn, confidence=0.9) for i, (kind, turn, quote) in enumerate(kinds)]
    return ContactSignalsContent(completeness="complete", signals=signals, passes=[
        ContactSignalsPassOutcome(pass_kind="lifecycle", included=True), ContactSignalsPassOutcome(pass_kind="resolution", included=True)],
        transcript_fingerprint=TRANSCRIPT_CHECKSUM, generated_at=NOW)


def put_taxonomy(client, session, taxonomy: SignalTaxonomy, *, expect: int = 200):
    current = client.get(f"{V}/signals/taxonomy", headers=session.read_headers).json()
    response = client.put(f"{V}/signals/taxonomy", json=save_body(taxonomy, current["record_version"]), headers=session.headers)
    assert response.status_code == expect, response.text
    return response.json()


def put_alert(client, session, rule_id: str, condition: Dict, *, name: Optional[str] = None, enabled: bool = True, expect: int = 200):
    existing = next((r for r in client.get(f"{V}/signals/alert-rules", headers=session.read_headers).json()["items"] if r["rule_id"] == rule_id), None)
    body = {"rule": {"rule_id": rule_id, "name": name or rule_id.replace("-", " ").title(), "condition": condition, "enabled": enabled},
            "expected_record_version": existing["record_version"] if existing else 0}
    response = client.put(f"{V}/signals/alert-rules/{rule_id}", json=body, headers=session.headers)
    assert response.status_code == expect, response.text
    return response.json()
