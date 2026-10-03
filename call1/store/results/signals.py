"""Contact Signals v2 in the results area: projection, alert evaluation, reads, feedback and the
preview helpers (contract 1.3.0; docs/ContactSignalsV2.md sections 7.4, 7.6, 9.2, 9.3 and 9.5).

**Projection.** ``project`` turns one published ``contact_signals`` version into projection rows
(``results_signal_outcomes``, ``results_signal_hits``, ``results_signal_hit_fields``) and sets
``results_calls.signals_version``. v1 results project too, with ``category_id = kind`` and no
subcategory, so built-in alerts, filters, metrics and queue rules work before any v2 engine
ships. It is replay-safe: a version at or below the projected one is skipped. **No text is copied**:
no quotes, and only enum and boolean field values (from the admin's own enum list); string,
number, amount and date fields record presence only. Every text read passes the Masker.

**Alerts** are evaluated at read time with one predicate, ``alert_condition_sql``, over the
projection tables: the view's ``alerts``, the call-list fields and filters, metrics, the queue
facts and the ``signal_alert`` events all use it, so an edited rule applies at once to every
projected call with no model run. A rule whose node is inactive matches nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

from call1.contracts.calls import ContactSignalsView
from call1.contracts.contents import (
    SIGNAL_OTHER_OPTION,
    ContactSignalsContent,
    ResultKind,
    SignalFieldType,
)
from call1.contracts.errors import ErrorCode
from call1.contracts.events import SIGNAL_FEEDBACK_STATUS, Actor, AuditAction, ChangeKind, signal_alert_fired_status
from call1.contracts.signals import (
    SignalAlertMatch,
    SignalAlertRuleRecord,
    SignalHitFeedback,
    SignalHitFeedbackSave,
    SignalPreviewDiff,
    SignalTaxonomy,
    SignalTaxonomyStatus,
    signal_taxonomy_status,
)

from .. import audit, db, feed
from ..db import StoreConnection
from ..errors import StoreError, not_found
from . import content, signal_store
from .masking import Masker

INTENT = "intent"
MAX_ALERT_HITS = 20
"""``SignalAlertMatch.hit_ids`` lists at most 20 hits."""


# --- the alert predicate ------------------------------------------------------------------------


def alert_condition_sql(rule: SignalAlertRuleRecord, alias: str = "h") -> Tuple[str, list]:
    """The SQL condition over hit rows (``alias``) that a rule's condition matches. The one
    predicate behind every alert read. Callers exclude rules whose node is inactive."""
    cond = rule.condition
    where = [f"{alias}.category_id = ?"]
    args: list = [cond.category_id]
    if cond.subcategory_id is not None:
        where.append(f"{alias}.subcategory_id = ?")
        args.append(cond.subcategory_id)
    if cond.min_confidence is not None:
        where.append(f"{alias}.confidence >= ?")
        args.append(cond.min_confidence)
    if cond.field_id is not None:
        sub = (f"SELECT 1 FROM results_signal_hit_fields f WHERE f.call_id = {alias}.call_id AND f.signals_version = {alias}.signals_version "
               f"AND f.hit_id = {alias}.hit_id AND f.field_id = ? AND f.status = 'extracted'")
        args.append(cond.field_id)
        if isinstance(cond.field_equals, bool):
            sub += " AND f.value_bool = ?"
            args.append(1 if cond.field_equals else 0)
        elif cond.field_equals is not None:
            sub += " AND f.value_enum = ?"
            args.append(cond.field_equals)
        where.append(f"EXISTS ({sub})")
    return " AND ".join(where), args


def alert_matches(conn: StoreConnection, rule: SignalAlertRuleRecord, call_id: str, version: Optional[int]) -> List[str]:
    """Hit IDs of one projected version that ``rule`` matches (empty for an inactive node or disabled rule)."""
    if version is None or not rule.enabled or not rule.node_active:
        return []
    cond, args = alert_condition_sql(rule)
    rows = conn.execute(f"SELECT h.hit_id FROM results_signal_hits h WHERE h.call_id = ? AND h.signals_version = ? AND {cond} "
                        "ORDER BY h.start, h.hit_id", (call_id, version, *args)).fetchall()
    return [r["hit_id"] for r in rows]


def matching_rules(conn: StoreConnection, call_id: str, version: Optional[int],
                   rules: Optional[Sequence[SignalAlertRuleRecord]] = None) -> Dict[str, List[str]]:
    """rule_id -> matching hit IDs, for every enabled rule on an active node that matches."""
    if version is None:
        return {}
    out: Dict[str, List[str]] = {}
    for rule in (rules if rules is not None else signal_store.active_alert_rules(conn)):
        hits = alert_matches(conn, rule, call_id, version)
        if hits:
            out[rule.rule_id] = hits
    return out


def call_alert_filter_sql(rule: Optional[SignalAlertRuleRecord], calls_alias: str = "results_calls") -> Tuple[str, list]:
    """A WHERE fragment over ``results_calls`` for calls the rule matches now (``listCalls?signal_alert=``)."""
    if rule is None or not rule.enabled or not rule.node_active:
        return "0", []
    cond, args = alert_condition_sql(rule)
    return (f"EXISTS (SELECT 1 FROM results_signal_hits h WHERE h.call_id = {calls_alias}.call_id "
            f"AND h.signals_version = {calls_alias}.signals_version AND {cond})"), args


@dataclass(frozen=True)
class AlertFacts:
    """What the review queue needs about a call's current signals (``ScoreFacts`` additions)."""

    signals_version: Optional[int] = None
    alerts: FrozenSet[str] = frozenset()
    reasons: Mapping[str, str] = field(default_factory=dict)
    """rule_id -> 'Cancel account (caller, 0:42)': the rule's name and its first matching hit."""


def _clock(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 60}:{total % 60:02d}"


def alert_facts(conn: StoreConnection, call_id: str) -> AlertFacts:
    row = conn.execute("SELECT signals_version FROM results_calls WHERE call_id = ?", (call_id,)).fetchone()
    version = row["signals_version"] if row is not None else None
    if version is None:
        return AlertFacts()
    rules = signal_store.active_alert_rules(conn)
    names = {r.rule_id: r.name for r in rules}
    matched = matching_rules(conn, call_id, version, rules)
    reasons = {}
    for rule_id, hits in matched.items():
        hit = conn.execute("SELECT speaker, start FROM results_signal_hits WHERE call_id = ? AND signals_version = ? AND hit_id = ?",
                           (call_id, version, hits[0])).fetchone()
        reasons[rule_id] = f"{names[rule_id]} ({hit['speaker'].lower()}, {_clock(hit['start'])})"
    return AlertFacts(signals_version=int(version), alerts=frozenset(matched), reasons=reasons)


# --- projection ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Projected:
    call_id: str
    version: int
    fired: Tuple[str, ...]
    """Enabled rules that match this version and did not match the previous projected version."""


def project(conn: StoreConnection, call_id: str, version: int, checksum: str) -> Optional[Projected]:
    """Project one ``contact_signals`` version (the caller's transaction). None when an equal or
    newer version is already projected (replay or out-of-order)."""
    row = conn.execute("SELECT signals_version FROM results_calls WHERE call_id = ?", (call_id,)).fetchone()
    if row is None:
        return None
    previous = row["signals_version"]
    if previous is not None and int(previous) >= version:
        return None
    signals = content.read_model(conn, checksum, ContactSignalsContent)
    before = set(matching_rules(conn, call_id, previous))
    now = db.ts(conn.now())
    conn.execute(
        "INSERT OR REPLACE INTO results_signal_outcomes (call_id, signals_version, pipeline, taxonomy_version, completeness, hit_count, projected_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (call_id, version, signals.pipeline, signals.taxonomy.version if signals.taxonomy else None, signals.completeness, len(signals.signals), now),
    )
    # One row per hit: a multi-segment hit (decision 25) counts once, under its anchor's hit ID, and
    # its row spans start → span_end. Alerts, metrics and filters read these rows.
    for hit in signals.signals:
        category_id = hit.category_id or hit.kind.value
        conn.execute(
            "INSERT OR IGNORE INTO results_signal_hits (call_id, signals_version, hit_id, category_id, subcategory_id, turn_id, speaker, start, \"end\", "
            "confidence, category_digest, subcategory_digest) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (call_id, version, hit.id, category_id, hit.subcategory_id, hit.turn_id, hit.speaker.value, hit.start,
             hit.span_end if hit.span_end is not None else hit.end, hit.confidence,
             hit.category_digest, hit.subcategory_digest),
        )
        for f in hit.fields:
            extracted = f.status == "extracted"
            value_enum = f.value if extracted and f.type is SignalFieldType.ENUM else None
            value_bool = (1 if f.value else 0) if extracted and f.type is SignalFieldType.BOOLEAN else None
            conn.execute(
                "INSERT OR IGNORE INTO results_signal_hit_fields (call_id, signals_version, hit_id, field_id, field_type, status, value_enum, value_bool) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (call_id, version, hit.id, f.field_id, f.type.value, f.status, value_enum, value_bool),
            )
    conn.execute("UPDATE results_calls SET signals_version = ?, updated_at = ? WHERE call_id = ?", (version, now, call_id))
    after = matching_rules(conn, call_id, version)
    return Projected(call_id=call_id, version=version, fired=tuple(sorted(set(after) - before)))


def append_fired_events(conn: StoreConnection, projected: Projected, conversation_id: str) -> None:
    for rule_id in projected.fired:
        feed.append(conn, ChangeKind.SIGNAL_ALERT, projected.call_id, projected.version, signal_alert_fired_status(rule_id),
                    conversation_id=conversation_id, call_id=projected.call_id)


# --- reads ----------------------------------------------------------------------------------------


def _mask_fields(hit: dict, masker: Masker) -> None:
    for f in hit.get("fields", []):
        if f.get("type") in (SignalFieldType.STRING.value, SignalFieldType.DATE.value) and isinstance(f.get("value"), str):
            f["value"] = masker.text(f["value"]) or f["value"]
        for key in ("surface", "evidence"):
            if f.get(key):
                f[key] = masker.text(f[key]) or f[key]


def masked_content_data(signals: ContactSignalsContent, masker: Masker) -> dict:
    """``model_dump`` of a contact_signals content with every text masked: quotes (a multi-segment
    hit's part quotes too, decision 25), string and date field values, surface text and evidence
    (``[REDACTED]`` while the findings are pending)."""
    data = signals.model_dump(mode="json")
    if masker.enabled:
        for hit in data["signals"]:
            hit["quote"] = masker.text(hit["quote"]) or ""
            for part in hit.get("parts") or []:
                part["quote"] = masker.text(part["quote"]) or ""
            _mask_fields(hit, masker)
    return data


def taxonomy_status(conn: StoreConnection, signals: ContactSignalsContent) -> SignalTaxonomyStatus:
    current = signal_store.current_version(conn)
    scored = None
    if signals.pipeline == "v2" and signals.taxonomy is not None and signals.taxonomy.version is not None:
        scored = signal_store.get_version(conn, signals.taxonomy.version)
    return signal_taxonomy_status(scored, current, dict(signals.stage1_digests))


def _feedback(row) -> SignalHitFeedback:
    return SignalHitFeedback(
        call_id=row["call_id"], hit_id=row["hit_id"], category_verdict=row["category_verdict"], subcategory_id=row["subcategory_id"],
        subcategory_digest=row["subcategory_digest"], subcategory_verdict=row["subcategory_verdict"],
        corrected_subcategory_id=row["corrected_subcategory_id"], note=row["note"], account_id=row["account_id"],
        feedback_version=int(row["feedback_version"]), updated_at=db.parse_ts(row["updated_at"]),
    )


_HIT_SPAN_SUFFIX = re.compile(r"\.t\d+b\d+$")


def feedback_hit_ids(hits: Iterable) -> List[str]:
    """The hit IDs whose feedback a published result shows: every hit's own ID, and for a merged
    multi-segment hit (decision 25) the ID each of its parts had as a separate hit (the anchor's ID
    with that part's turn and block; section 6.3). Feedback saved before a merge absorbed the hit
    stays readable, as the judgement of an earlier segmentation, instead of vanishing."""
    out: List[str] = []
    for hit in hits:
        out.append(hit.id)
        if _HIT_SPAN_SUFFIX.search(hit.id):
            out.extend(_HIT_SPAN_SUFFIX.sub(f".t{part.turn_id}b{part.block}", hit.id) for part in (getattr(hit, "parts", None) or []))
    return out


def feedback_for(conn: StoreConnection, call_id: str, hit_ids: Iterable[str]) -> List[SignalHitFeedback]:
    wanted = set(hit_ids)
    rows = conn.execute("SELECT * FROM results_signal_feedback WHERE call_id = ? ORDER BY hit_id", (call_id,)).fetchall()
    return [_feedback(r) for r in rows if r["hit_id"] in wanted]


def view(conn: StoreConnection, call_id: str, conversation_id: str, masker: Masker, pub, *, admin: bool = False) -> ContactSignalsView:
    """The published contact signals, masked, with the 1.3.0 read-time context."""
    from ..queue import api as queue_api

    signals = content.read_model(conn, pub.checksum, ContactSignalsContent)
    data = masked_content_data(signals, masker)
    row = conn.execute("SELECT signals_version FROM results_calls WHERE call_id = ?", (call_id,)).fetchone()
    projected = row["signals_version"] if row is not None else None
    names = signal_store.rule_names(conn)
    alerts = []
    if projected is not None and int(projected) == pub.version:
        for rule_id, hits in sorted(matching_rules(conn, call_id, pub.version).items()):
            alerts.append(SignalAlertMatch(rule_id=rule_id, name=names[rule_id], hit_ids=hits[:MAX_ALERT_HITS]))
    return ContactSignalsView(
        **data, call_id=call_id, artifact_id=pub.artifact_id, version=pub.version,
        taxonomy_status=taxonomy_status(conn, signals),
        feedback=feedback_for(conn, call_id, feedback_hit_ids(signals.signals)),
        alerts=alerts,
        text_withheld=masker.withheld,
        comparison_preview_id=queue_api.latest_compare_preview(conn, call_id) if admin else None,
    )


# --- call list fields and filters ---------------------------------------------------------------


@dataclass
class ListContext:
    """Per-request cache for the call list: the current taxonomy and the active rules."""

    taxonomy: SignalTaxonomy
    rules: List[SignalAlertRuleRecord]

    @classmethod
    def load(cls, conn: StoreConnection) -> "ListContext":
        return cls(signal_store.current_version(conn).taxonomy, signal_store.active_alert_rules(conn))

    def category_active(self, category_id: str) -> bool:
        c = self.taxonomy.category(category_id)
        return c is not None and c.active

    def subcategory_active(self, category_id: str, subcategory_id: Optional[str]) -> bool:
        if subcategory_id is None:
            return False
        if subcategory_id == SIGNAL_OTHER_OPTION:
            return self.category_active(category_id)
        c = self.taxonomy.category(category_id)
        return c is not None and c.active and any(s.subcategory_id == subcategory_id and s.active for s in c.subcategories)


def list_fields(conn: StoreConnection, call_row, ctx: ListContext) -> Dict[str, List[str]]:
    """``CallListItem.signal_categories``, ``caller_needs`` and ``signal_alerts`` (current hits on
    active nodes, enabled rules)."""
    version = call_row["signals_version"]
    if version is None:
        return {"signal_categories": [], "caller_needs": [], "signal_alerts": []}
    rows = conn.execute("SELECT DISTINCT category_id, subcategory_id FROM results_signal_hits WHERE call_id = ? AND signals_version = ?",
                        (call_row["call_id"], version)).fetchall()
    order = {c.category_id: i for i, c in enumerate(ctx.taxonomy.categories)}
    categories = sorted({r["category_id"] for r in rows if ctx.category_active(r["category_id"])}, key=lambda c: (order.get(c, 999), c))
    needs = sorted({r["subcategory_id"] for r in rows if r["category_id"] == INTENT and ctx.subcategory_active(INTENT, r["subcategory_id"])})
    alerts = sorted(matching_rules(conn, call_row["call_id"], version, ctx.rules))
    return {"signal_categories": categories, "caller_needs": needs, "signal_alerts": alerts}


def list_filters(conn: StoreConnection, *, category: Optional[str], subcategory: Optional[str], alert: Optional[str]) -> Tuple[List[str], list]:
    """WHERE fragments over ``results_calls`` for the 1.3.0 ``listCalls`` filters."""
    where: List[str] = []
    args: list = []
    if category is not None or subcategory is not None:
        cond = ["h.call_id = results_calls.call_id", "h.signals_version = results_calls.signals_version"]
        if category is not None:
            cond.append("h.category_id = ?")
            args.append(category)
        if subcategory is not None:
            cond.append("h.subcategory_id = ?")
            args.append(subcategory)
        where.append("EXISTS (SELECT 1 FROM results_signal_hits h WHERE " + " AND ".join(cond) + ")")
    if alert is not None:
        rule = signal_store.get_alert_rule(conn, alert)
        sql, rule_args = call_alert_filter_sql(rule)
        where.append(sql)
        args.extend(rule_args)
    return where, args


# --- feedback -------------------------------------------------------------------------------------


def save_feedback(conn: StoreConnection, call_id: str, hit_id: str, body: SignalHitFeedbackSave, *, account_id: str, actor: Actor,
                  conversation_id: str) -> SignalHitFeedback:
    """``saveSignalHitFeedback`` (the caller's transaction). The save replaces the verdicts; Store
    records the subcategory the verdict judged (ID and digest) from the current hit."""
    call = conn.execute("SELECT signals_version FROM results_calls WHERE call_id = ?", (call_id,)).fetchone()
    hit = None
    if call is not None and call["signals_version"] is not None:
        hit = conn.execute("SELECT * FROM results_signal_hits WHERE call_id = ? AND signals_version = ? AND hit_id = ?",
                           (call_id, call["signals_version"], hit_id)).fetchone()
    if hit is None:
        raise not_found("Signal hit in the call's current contact signals", call_id=call_id, hit_id=hit_id)
    existing = conn.execute("SELECT * FROM results_signal_feedback WHERE call_id = ? AND hit_id = ?", (call_id, hit_id)).fetchone()
    current = int(existing["feedback_version"]) if existing else 0
    if body.expected_feedback_version != current:
        raise StoreError(ErrorCode.CONFLICT, "The feedback changed since you read it", details={"current_version": current, "reason": "feedback_version"})
    subcategory_id = subcategory_digest = None
    if body.subcategory_verdict is not None:
        if hit["subcategory_id"] is None:
            raise StoreError(ErrorCode.VALIDATION_FAILED, "This hit has no subcategory to judge", details={"field": "subcategory_verdict", "reason": "no_subcategory"})
        subcategory_id, subcategory_digest = hit["subcategory_id"], hit["subcategory_digest"]
    if body.corrected_subcategory_id is not None and body.corrected_subcategory_id != SIGNAL_OTHER_OPTION:
        category = signal_store.current_version(conn).taxonomy.category(hit["category_id"])
        if category is None or not any(s.subcategory_id == body.corrected_subcategory_id for s in category.subcategories):
            raise StoreError(ErrorCode.VALIDATION_FAILED, "The corrected subcategory is not one of this category's",
                             details={"field": "corrected_subcategory_id", "reason": "unknown_subcategory"})
    now = db.ts(conn.now())
    conn.execute(
        "INSERT INTO results_signal_feedback (call_id, hit_id, category_verdict, subcategory_id, subcategory_digest, subcategory_verdict, "
        "corrected_subcategory_id, note, account_id, feedback_version, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(call_id, hit_id) DO UPDATE SET category_verdict = excluded.category_verdict, subcategory_id = excluded.subcategory_id, "
        "subcategory_digest = excluded.subcategory_digest, subcategory_verdict = excluded.subcategory_verdict, "
        "corrected_subcategory_id = excluded.corrected_subcategory_id, note = excluded.note, account_id = excluded.account_id, "
        "feedback_version = excluded.feedback_version, updated_at = excluded.updated_at",
        (call_id, hit_id, body.category_verdict, subcategory_id, subcategory_digest, body.subcategory_verdict, body.corrected_subcategory_id,
         body.note, account_id, current + 1, now),
    )
    audit.append(conn, actor=actor, action=AuditAction.SIGNAL_HIT_REVIEWED, target_kind="call", target_id=call_id,
                 details={"call_id": call_id, "hit_id": hit_id[:200], "category_verdict": body.category_verdict,
                          "subcategory_verdict": body.subcategory_verdict, "feedback_version": current + 1})
    feed.append(conn, ChangeKind.REVIEW, call_id, current + 1, SIGNAL_FEEDBACK_STATUS, conversation_id=conversation_id, call_id=call_id)
    row = conn.execute("SELECT * FROM results_signal_feedback WHERE call_id = ? AND hit_id = ?", (call_id, hit_id)).fetchone()
    # On-device training (1.3.0, decision 28): the label log row for a v2 hit, in this transaction.
    # Clearing both verdicts appends a withdrawn row. The note is never copied.
    from . import training_labels

    training_labels.record_signal_feedback(conn, call_id=call_id, conversation_id=conversation_id, hit_id=hit_id,
                                           signals_version=int(call["signals_version"]), feedback_row=row)
    return _feedback(row)


# --- previews: masking and the diff against the published signals ------------------------------


def _hit_key(hit, *, v1: bool) -> Tuple:
    category = hit.category_id or hit.kind.value
    return (category, hit.turn_id) if v1 else (category, hit.turn_id, hit.span.block if hit.span else None)


def _segment_keys(hit, *, v1: bool) -> List[Tuple]:
    """Every (category, turn[, block]) a hit covers: its anchor, then each part of a merged
    multi-segment hit (decision 25, section 6.5). Parts share the anchor's category."""
    anchor = _hit_key(hit, v1=v1)
    keys = [anchor]
    for part in getattr(hit, "parts", None) or []:
        key = (anchor[0], part.turn_id) if v1 else (anchor[0], part.turn_id, part.block)
        if key not in keys:
            keys.append(key)
    return keys


def _subcategory(hit) -> Optional[str]:
    """No subcategory and 'Other' read the same in a diff: a category's hits turn into 'Other' when
    it gains its first subcategory, which is not a change to the signal."""
    sub = hit.subcategory_id
    return None if sub in (None, SIGNAL_OTHER_OPTION) else sub


def _fields_signature(hit) -> Tuple:
    return tuple(sorted((f.field_id, f.status, repr(f.value)) for f in hit.fields))


def preview_diff(published: Optional[ContactSignalsContent], preview: ContactSignalsContent) -> SignalPreviewDiff:
    """A preview result against the call's published contact signals. A preview hit pairs with the
    published hit that covers the same segment: category, turn and block (category and turn against
    a v1 result, which has no blocks), counting every part of a merged hit, not just its anchor. So
    a span that became a part of a merged hit is not reported as removed. A paired hit is
    relabelled (its subcategory changed), else fields_changed, else segments_changed (the set of
    segments it covers changed: a merge formed, split or grew)."""
    from call1.contracts.signals import BUILTIN_SIGNAL_CATEGORIES

    v1 = published is None or published.pipeline == "v1"
    old_hits = list(published.signals) if published else []
    old_by_key: Dict[Tuple, Any] = {}
    for h in old_hits:
        for key in _segment_keys(h, v1=v1):
            old_by_key.setdefault(key, h)
    added, relabelled, fields_changed, segments_changed = [], [], [], []
    covered = set()
    for h in preview.signals:
        keys = _segment_keys(h, v1=v1)
        olds: List[Any] = []
        for key in keys:
            o = old_by_key.get(key)
            if o is not None and all(o is not x for x in olds):
                olds.append(o)
        if not olds:
            added.append(h)
            continue
        covered.update(id(o) for o in olds)
        o = olds[0]
        if _subcategory(h) != _subcategory(o):
            relabelled.append(h.id)
        elif _fields_signature(h) != _fields_signature(o):
            fields_changed.append(h.id)
        elif len(olds) > 1 or set(keys) != set(_segment_keys(o, v1=v1)):
            segments_changed.append(h.id)
    removed = [o for o in old_hits if id(o) not in covered]
    builtin = [x.id for x in added + removed if _hit_key(x, v1=v1)[0] in BUILTIN_SIGNAL_CATEGORIES]
    return SignalPreviewDiff(added=sorted(x.id for x in added), removed=sorted(x.id for x in removed), relabelled=sorted(relabelled),
                             fields_changed=sorted(fields_changed), segments_changed=sorted(segments_changed), builtin_changed=sorted(builtin))


def published_content(conn: StoreConnection, conversation_id: str) -> Optional[ContactSignalsContent]:
    from . import records

    pub = records.latest_publication(conn, conversation_id, ResultKind.CONTACT_SIGNALS)
    return None if pub is None else content.read_model(conn, pub.checksum, ContactSignalsContent)
