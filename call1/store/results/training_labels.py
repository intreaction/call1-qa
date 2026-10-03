"""The on-device training label log (contract 1.3.0, team decision 28; docs/OnDeviceTraining.md section 2).

Reviewers correct the machine in Evaluate. Each correction is also appended here, in the label
write's own transaction, so a rolled-back write appends nothing:

- ``overrideVerdict`` appends one ``qa_verdict`` row. It judged the scorecard publication at
  ``evaluation_version``.
- ``saveSignalHitFeedback`` appends one ``signal_hit`` row for a v2 hit (one whose published
  content names a ``category_id``). It judged the call's current ``contact_signals`` publication;
  the anchor's and the parts' ``(turn_id, block)`` are read from it at save time. A save that clears
  both verdicts appends a ``withdrawn`` row.
- ``correctSpeaker`` appends one ``speaker_role`` row. It judged the model's ``speaker_attribution``
  (the newest one not written by a reviewer correction, else the newest), and that artifact's
  cluster for the turn is resolved into the row.

**No text.** A row holds IDs, enums and versions only: no transcript text, quote, reviewer note,
signal note or reviewer identity. Process rebuilds the text from the source artifacts and masks it
with the engines' own masking, on the device.

**Reads** (``listTrainingLabels``, ``training:read``) resolve each row's source job and source
artifacts at read time through ``call1.store.queue.api`` (section 2.3). A source that can no longer
be resolved leaves ``sources: []`` and ``source_job_id: null``; Process skips the label as
``source_unavailable``. ``pii_findings`` is the newest indexed ``pii_findings`` artifact made from
the label's transcript revision, omitted when there is none. Draft-test and preview artifacts are
never sources: only published results and the results index are consulted, and draft slots are
skipped.

**Backfill.** Labels written before this log existed are imported once at Store start
(``backfill``), guarded by the ``store_meta`` key ``training_labels_backfilled``: every verdict
override, every speaker-correction history row and one row per current signal feedback row (only
the latest state exists for those), in timestamp order.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from pydantic import ValidationError

from call1.contracts.artifacts import Artifact, ArtifactKind
from call1.contracts.contents import (
    ContactSignalsContent,
    ContactSignalView,
    PiiFindingsContent,
    ResultKind,
    SpeakerAttributionContent,
    VerdictStatus,
)
from call1.contracts.jobs import Job, JobType
from call1.contracts.reviews import OverrideReasonCode, ReviewHistoryKind
from call1.contracts.training import (
    TRAINING_SOURCE_ROLE_KINDS,
    TRAINING_SOURCE_ROLES,
    QaVerdictLabel,
    SignalHitLabel,
    SignalSpanAt,
    SpeakerRoleLabel,
    TrainingLabel,
    TrainingLabelKind,
    TrainingLabelPage,
    TrainingLabelQuery,
    TrainingSourceRef,
    speaker_subject_key,
    training_label_subject,
)

from .. import db
from ..db import StoreConnection
from ..queue import api as queue_api
from . import content, records

log = logging.getLogger("call1.store.results.training")

META_BACKFILLED = "training_labels_backfilled"
"""``store_meta`` key: the one-time import of labels written before the log existed has run."""

MAX_SPANS = 64
"""``SignalHitLabel.spans`` holds at most 64 segments (the anchor first)."""

_SPAN_SUFFIX = re.compile(r"\.t(\d+)b(\d+)$")

LabelModel = Union[QaVerdictLabel, SignalHitLabel, SpeakerRoleLabel]
_MODEL: Dict[TrainingLabelKind, type] = {
    TrainingLabelKind.QA_VERDICT: QaVerdictLabel,
    TrainingLabelKind.SIGNAL_HIT: SignalHitLabel,
    TrainingLabelKind.SPEAKER_ROLE: SpeakerRoleLabel,
}
_FIELD = {TrainingLabelKind.QA_VERDICT: "qa", TrainingLabelKind.SIGNAL_HIT: "signal", TrainingLabelKind.SPEAKER_ROLE: "speaker"}

# Roles taken from the source job's own resolved inputs, per kind (section 2.3). pii_findings is
# never taken from a job's inputs: it is always the newest findings for the label's transcript.
_INPUT_ROLES: Dict[TrainingLabelKind, Tuple[str, ...]] = {
    TrainingLabelKind.QA_VERDICT: ("transcript", "speaker_attribution", "rubric", "enrichment"),
    TrainingLabelKind.SIGNAL_HIT: ("transcript", "speaker_attribution", "taxonomy", "stage:categorize", "stage:subcategorize", "stage:extract"),
    TrainingLabelKind.SPEAKER_ROLE: ("transcript",),
}
_SOURCE_JOB_TYPE: Dict[TrainingLabelKind, JobType] = {
    TrainingLabelKind.QA_VERDICT: JobType.QA_CRITERION,
    TrainingLabelKind.SIGNAL_HIT: JobType.CONTACT_SIGNALS_MERGE,
    TrainingLabelKind.SPEAKER_ROLE: JobType.SPEAKER_ATTRIBUTION,
}


# --- building and appending rows ---------------------------------------------------------------


def _key(kind: TrainingLabelKind, label: LabelModel) -> str:
    if kind is TrainingLabelKind.QA_VERDICT:
        return label.criterion_id
    if kind is TrainingLabelKind.SIGNAL_HIT:
        return label.hit_id
    return speaker_subject_key(label.turn_id, label.apply_to_cluster, label.speaker_cluster)


def append(conn: StoreConnection, *, kind: TrainingLabelKind, call_id: str, conversation_id: str, source_artifact_id: str,
           label: LabelModel, recorded_at: Optional[datetime] = None) -> Optional[int]:
    """Append one row in the caller's transaction and return its ``seq``.

    The row is checked as the ``TrainingLabel`` it will read back as (with no sources), so every
    stored row can always be served. A row that would not validate is a Store bug: it is logged and
    not written, and the reviewer's own write still commits (a training label never fails a review)."""
    kind = TrainingLabelKind(kind)
    withdrawn = kind is TrainingLabelKind.SIGNAL_HIT and label.cleared
    subject = training_label_subject(kind, call_id, _key(kind, label))
    at = recorded_at or conn.now()
    try:
        TrainingLabel(seq=1, kind=kind, subject=subject, call_id=call_id, conversation_id=conversation_id, recorded_at=at,
                      withdrawn=withdrawn, **{_FIELD[kind]: label})
    except ValidationError as exc:
        log.warning("training label not logged (%s for call %s): %s", kind.value, call_id, exc.errors()[:1])
        return None
    cur = conn.execute(
        "INSERT INTO results_training_labels (kind, subject, call_id, conversation_id, source_artifact_id, label_json, withdrawn, recorded_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (kind.value, subject, call_id, conversation_id, source_artifact_id, db.dumps(label), 1 if withdrawn else 0, db.ts(at)),
    )
    return int(cur.lastrowid)


def record_verdict_override(conn: StoreConnection, *, call_id: str, conversation_id: str, override_id: str, criterion_id: str,
                            evaluation_version: int, original_status: Union[VerdictStatus, str], status: Union[VerdictStatus, str],
                            reason_code: Optional[Union[OverrideReasonCode, str]], recorded_at: Optional[datetime] = None) -> Optional[int]:
    """The ``qa_verdict`` row of one ``overrideVerdict`` (the caller's transaction). The judged
    artifact is the scorecard publication at ``evaluation_version``."""
    pub = records.publication(conn, conversation_id, ResultKind.QA, evaluation_version)
    if pub is None:
        log.warning("training label not logged: call %s has no QA publication v%s", call_id, evaluation_version)
        return None
    try:
        label = QaVerdictLabel(override_id=override_id, criterion_id=criterion_id, evaluation_version=evaluation_version,
                               original_status=VerdictStatus(original_status), status=VerdictStatus(status),
                               reason_code=OverrideReasonCode(reason_code) if reason_code else None)
    except (ValidationError, ValueError) as exc:
        log.warning("training label not logged (qa_verdict for call %s): %s", call_id, exc)
        return None
    return append(conn, kind=TrainingLabelKind.QA_VERDICT, call_id=call_id, conversation_id=conversation_id,
                  source_artifact_id=pub.artifact_id, label=label, recorded_at=recorded_at)


def _hit_spans(hit: ContactSignalView) -> List[SignalSpanAt]:
    """The anchor's (turn_id, block), then each part's, in order, each once (at most 64)."""
    anchor: Optional[Tuple[int, int]] = None
    if hit.turn_id is not None and hit.span is not None:
        anchor = (hit.turn_id, hit.span.block)
    else:
        found = _SPAN_SUFFIX.search(hit.id)
        if found:
            anchor = (int(found.group(1)), int(found.group(2)))
    ordered: List[Tuple[int, int]] = [anchor] if anchor is not None else []
    for part in hit.parts:
        ordered.append((part.turn_id, part.block))
    seen: List[Tuple[int, int]] = []
    for pair in ordered:
        if pair not in seen:
            seen.append(pair)
    return [SignalSpanAt(turn_id=t, block=b) for t, b in seen[:MAX_SPANS]]


def record_signal_feedback(conn: StoreConnection, *, call_id: str, conversation_id: str, hit_id: str, signals_version: int, feedback_row,
                           recorded_at: Optional[datetime] = None) -> Optional[int]:
    """The ``signal_hit`` row of one ``saveSignalHitFeedback`` (the caller's transaction), for a v2
    hit only. ``feedback_row`` is the saved ``results_signal_feedback`` row; clearing both verdicts
    makes the row ``withdrawn``."""
    pub = records.publication(conn, conversation_id, ResultKind.CONTACT_SIGNALS, signals_version)
    if pub is None:
        return None
    signals = content.read_model(conn, pub.checksum, ContactSignalsContent)
    hit = next((h for h in signals.signals if h.id == hit_id), None)
    if hit is None or hit.category_id is None:
        return None  # a v1 hit: nothing on-device training can rebuild
    spans = _hit_spans(hit)
    try:
        label = SignalHitLabel(
            hit_id=hit_id, category_id=hit.category_id, signals_version=signals_version, spans=spans,
            category_verdict=feedback_row["category_verdict"], subcategory_id=feedback_row["subcategory_id"],
            subcategory_digest=feedback_row["subcategory_digest"], subcategory_verdict=feedback_row["subcategory_verdict"],
            corrected_subcategory_id=feedback_row["corrected_subcategory_id"], feedback_version=int(feedback_row["feedback_version"]),
        )
    except ValidationError as exc:
        log.warning("training label not logged (signal_hit %s for call %s): %s", hit_id[:80], call_id, exc.errors()[:1])
        return None
    return append(conn, kind=TrainingLabelKind.SIGNAL_HIT, call_id=call_id, conversation_id=conversation_id,
                  source_artifact_id=pub.artifact_id, label=label, recorded_at=recorded_at)


def judged_attribution(conn: StoreConnection, conversation_id: str, turn_id: int, *,
                       at_or_before: Optional[str] = None) -> Optional[Tuple[str, Optional[str]]]:
    """(artifact ID, the turn's speaker cluster) of the attribution a speaker correction judged: the
    newest indexed ``speaker_attribution`` not written by a reviewer correction (the model's), else
    the newest one. ``at_or_before`` (storage timestamp) limits the search for the backfill."""
    sql = "SELECT artifact_id, checksum, slot FROM results_artifacts WHERE conversation_id = ? AND kind = ?"
    args: List[Any] = [conversation_id, ArtifactKind.SPEAKER_ATTRIBUTION.value]
    if at_or_before is not None:
        sql += " AND committed_at <= ?"
        args.append(at_or_before)
    rows = [r for r in conn.execute(sql + " ORDER BY version DESC", args).fetchall() if not str(r["slot"]).startswith("draft:")]
    chosen = None
    for row in rows:
        attribution = content.read_model(conn, row["checksum"], SpeakerAttributionContent)
        if attribution.method != "reviewer_correction":
            chosen = (row, attribution)
            break
    if chosen is None and rows:
        chosen = (rows[0], content.read_model(conn, rows[0]["checksum"], SpeakerAttributionContent))
    if chosen is None:
        return None
    row, attribution = chosen
    cluster = next((a.speaker_cluster for a in attribution.assignments if a.turn_id == turn_id), None)
    return row["artifact_id"], (cluster or None)


def record_speaker_correction(conn: StoreConnection, *, call_id: str, conversation_id: str, turn_id: int, speaker: str, apply_to_cluster: bool,
                              reanalysis_request_id: str, recorded_at: Optional[datetime] = None,
                              at_or_before: Optional[str] = None) -> Optional[int]:
    """The ``speaker_role`` row of one ``correctSpeaker`` (the caller's transaction). No row when
    the call has no speaker attribution to judge."""
    judged = judged_attribution(conn, conversation_id, turn_id, at_or_before=at_or_before)
    if judged is None:
        return None
    artifact_id, cluster = judged
    try:
        label = SpeakerRoleLabel(turn_id=turn_id, speaker=speaker, apply_to_cluster=bool(apply_to_cluster),
                                 speaker_cluster=cluster[:200] if cluster else None, reanalysis_request_id=reanalysis_request_id)
    except ValidationError as exc:
        log.warning("training label not logged (speaker_role for call %s): %s", call_id, exc.errors()[:1])
        return None
    return append(conn, kind=TrainingLabelKind.SPEAKER_ROLE, call_id=call_id, conversation_id=conversation_id,
                  source_artifact_id=artifact_id, label=label, recorded_at=recorded_at)


# --- backfill --------------------------------------------------------------------------------------


def backfill(conn: StoreConnection) -> int:
    """Import labels written before the log existed, once (``store_meta`` guard), in timestamp order.
    Returns the rows appended. A label whose judged artifact cannot be found is skipped."""
    with db.transaction(conn):
        if db.get_meta(conn, META_BACKFILLED) is not None:
            return 0
        pending: List[Tuple[str, int, Any]] = []  # (timestamp, source order, thunk)
        order = 0
        for row in conn.execute("SELECT o.*, c.conversation_id AS conv FROM results_verdict_overrides o "
                                "JOIN results_calls c ON c.call_id = o.call_id ORDER BY o.created_at, o.id").fetchall():
            pending.append((row["created_at"], order, ("qa", row)))
            order += 1
        for row in conn.execute("SELECT h.*, c.conversation_id AS conv FROM results_review_history h JOIN results_calls c ON c.call_id = h.call_id "
                                "WHERE h.kind = ? ORDER BY h.created_at, h.id", (ReviewHistoryKind.SPEAKER_CORRECTION.value,)).fetchall():
            pending.append((row["created_at"], order, ("speaker", row)))
            order += 1
        for row in conn.execute("SELECT f.*, c.conversation_id AS conv, c.signals_version AS current_version FROM results_signal_feedback f "
                                "JOIN results_calls c ON c.call_id = f.call_id ORDER BY f.updated_at, f.call_id, f.hit_id").fetchall():
            pending.append((row["updated_at"], order, ("signal", row)))
            order += 1
        appended = 0
        for stamp, _, (what, row) in sorted(pending, key=lambda item: (item[0], item[1])):
            try:
                seq = _backfill_one(conn, what, row, stamp)
            except Exception as exc:  # one unreadable label never blocks Store start
                log.warning("training label backfill skipped a %s label of call %s: %s", what, row["call_id"], exc)
                seq = None
            appended += 1 if seq is not None else 0
        conn.execute("INSERT INTO store_meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                     (META_BACKFILLED, db.ts(conn.now())))
        if appended:
            log.info("training label log: imported %d earlier labels", appended)
        return appended


def _backfill_one(conn: StoreConnection, what: str, row, stamp: str) -> Optional[int]:
    at = db.parse_ts(stamp)
    if what == "qa":
        return record_verdict_override(conn, call_id=row["call_id"], conversation_id=row["conv"], override_id=row["id"],
                                       criterion_id=row["criterion_id"], evaluation_version=int(row["evaluation_version"]),
                                       original_status=row["original_status"], status=row["status"], reason_code=row["reason_code"],
                                       recorded_at=at)
    if what == "speaker":
        payload = db.loads(row["payload_json"]) or {}
        if payload.get("turn_id") is None or not payload.get("reanalysis_request_id"):
            return None
        return record_speaker_correction(conn, call_id=row["call_id"], conversation_id=row["conv"], turn_id=int(payload["turn_id"]),
                                         speaker=payload.get("speaker"), apply_to_cluster=bool(payload.get("apply_to_cluster")),
                                         reanalysis_request_id=payload["reanalysis_request_id"], recorded_at=at, at_or_before=stamp)
    found = conn.execute("SELECT MAX(signals_version) AS v FROM results_signal_hits WHERE call_id = ? AND hit_id = ? AND signals_version <= ?",
                         (row["call_id"], row["hit_id"], row["current_version"] or 0)).fetchone()
    if found is None or found["v"] is None:
        return None
    return record_signal_feedback(conn, call_id=row["call_id"], conversation_id=row["conv"], hit_id=row["hit_id"], signals_version=int(found["v"]),
                                  feedback_row=row, recorded_at=at)


# --- reads -----------------------------------------------------------------------------------------


@dataclass
class _Resolver:
    """Resolves source jobs and artifacts through ``queue.api`` for one page, with small caches."""

    conn: StoreConnection
    jobs: Dict[str, Optional[Job]] = field(default_factory=dict)
    artifacts: Dict[str, Optional[Artifact]] = field(default_factory=dict)
    findings: Dict[Tuple[str, str], Optional[Artifact]] = field(default_factory=dict)

    def job(self, job_id: Optional[str]) -> Optional[Job]:
        if job_id is None:
            return None
        if job_id not in self.jobs:
            self.jobs[job_id] = queue_api.get_job(self.conn, job_id)
        return self.jobs[job_id]

    def artifact(self, artifact_id: Optional[str]) -> Optional[Artifact]:
        if artifact_id is None:
            return None
        if artifact_id not in self.artifacts:
            self.artifacts[artifact_id] = queue_api.get_artifact(self.conn, artifact_id)
        return self.artifacts[artifact_id]

    def pii_findings(self, conversation_id: str, transcript_checksum: str) -> Optional[Artifact]:
        """The newest indexed ``pii_findings`` made from this transcript revision (the lookup reviewer
        reads use, ``results/masking.py``), or None."""
        key = (conversation_id, transcript_checksum)
        if key not in self.findings:
            found = None
            for row in self.conn.execute("SELECT artifact_id, checksum, slot FROM results_artifacts WHERE conversation_id = ? AND kind = ? "
                                         "ORDER BY version DESC", (conversation_id, ArtifactKind.PII_FINDINGS.value)).fetchall():
                if str(row["slot"]).startswith("draft:"):
                    continue
                findings = content.read_model(self.conn, row["checksum"], PiiFindingsContent)
                if findings.transcript.checksum == transcript_checksum:
                    found = self.artifact(row["artifact_id"])
                    break
            self.findings[key] = found
        return self.findings[key]


def _ref(role: str, art: Optional[Artifact]) -> Optional[TrainingSourceRef]:
    if art is None or art.kind not in TRAINING_SOURCE_ROLE_KINDS.get(role, frozenset()):
        return None
    return TrainingSourceRef(role=role, artifact_id=art.id, checksum=art.checksum, kind=art.kind)


def _job_output(resolver: _Resolver, job: Job, role: str) -> Optional[Artifact]:
    out = next((o for o in job.outputs if o.role == role), None)
    return resolver.artifact(out.artifact_id) if out is not None else None


def resolve_sources(resolver: _Resolver, kind: TrainingLabelKind, conversation_id: str, source_artifact_id: str,
                    label: LabelModel) -> Tuple[Optional[str], List[TrainingSourceRef]]:
    """(source_job_id, sources) per docs/OnDeviceTraining.md section 2.3, or (None, []) when the
    source can no longer be resolved or lacks a required role."""
    judged = resolver.artifact(source_artifact_id)
    if judged is None or judged.producing_job_id is None:
        return None, []
    refs: Dict[str, TrainingSourceRef] = {}

    def add(role: str, art: Optional[Artifact]) -> None:
        ref = _ref(role, art)
        if ref is not None and role not in refs:
            refs[role] = ref

    if kind is TrainingLabelKind.QA_VERDICT:
        scorecard_job = resolver.job(judged.producing_job_id)
        if scorecard_job is None:
            return None, []
        pinned = {i.role: i for i in scorecard_job.resolved_inputs}
        assessment_in = pinned.get(f"assessment:{label.criterion_id}")
        assessment = resolver.artifact(assessment_in.artifact_id) if assessment_in is not None else None
        source = resolver.job(assessment.producing_job_id) if assessment is not None else None
        if source is None:
            return None, []
        escalation_in = pinned.get(f"escalation:{label.criterion_id}")
        add("escalation_assessment", resolver.artifact(escalation_in.artifact_id) if escalation_in is not None else None)
        add("assessment", _job_output(resolver, source, "assessment"))
        add("prompt_input", _job_output(resolver, source, "prompt_input"))
    else:
        source = resolver.job(judged.producing_job_id)
        if source is None:
            return None, []
        add("contact_signals" if kind is TrainingLabelKind.SIGNAL_HIT else "speaker_attribution", judged)
    if source.job_type is not _SOURCE_JOB_TYPE[kind]:
        return None, []
    for resolved in source.resolved_inputs:
        if resolved.role in _INPUT_ROLES[kind]:
            add(resolved.role, resolver.artifact(resolved.artifact_id))
    transcript = refs.get("transcript")
    if transcript is not None:
        findings = resolver.pii_findings(conversation_id, transcript.checksum)
        add("pii_findings", findings)
        if kind is TrainingLabelKind.SPEAKER_ROLE and findings is not None:
            enrichment_job = resolver.job(findings.producing_job_id)
            if enrichment_job is not None:
                add("enrichment", _job_output(resolver, enrichment_job, "enrichment"))
    required, _optional = TRAINING_SOURCE_ROLES[kind]
    if not required <= set(refs):
        return None, []
    return source.id, list(refs.values())


def _item(resolver: _Resolver, row) -> TrainingLabel:
    kind = TrainingLabelKind(row["kind"])
    label = _MODEL[kind].model_validate(db.loads(row["label_json"]))
    withdrawn = bool(row["withdrawn"])
    base = dict(seq=int(row["seq"]), kind=kind, subject=row["subject"], call_id=row["call_id"], conversation_id=row["conversation_id"],
                recorded_at=db.parse_ts(row["recorded_at"]), withdrawn=withdrawn, **{_FIELD[kind]: label})
    if withdrawn:
        return TrainingLabel(**base)
    try:
        job_id, sources = resolve_sources(resolver, kind, row["conversation_id"], row["source_artifact_id"], label)
    except Exception as exc:  # an unreadable source artifact: Process skips the label (source_unavailable)
        log.warning("training label %s: sources unresolved: %s", row["seq"], exc)
        job_id, sources = None, []
    try:
        return TrainingLabel(**base, source_job_id=job_id, sources=sources)
    except ValidationError as exc:
        log.warning("training label %s: resolved sources rejected: %s", row["seq"], exc.errors()[:1])
        return TrainingLabel(**base)


def _kinds_clause(kinds: Optional[Sequence[TrainingLabelKind]]) -> Tuple[str, list]:
    if not kinds:
        return "", []
    values = sorted({TrainingLabelKind(k).value for k in kinds})
    return " AND kind IN (" + ", ".join("?" for _ in values) + ")", values


def high_water(conn: StoreConnection) -> int:
    row = conn.execute("SELECT MAX(seq) AS s FROM results_training_labels").fetchone()
    return int(row["s"] or 0)


def list_labels(conn: StoreConnection, query: TrainingLabelQuery) -> TrainingLabelPage:
    """One page of the log (inside the caller's read snapshot). ``limit`` 0 returns the counts only.

    When ``after`` is beyond the log's ``high_water`` (a cursor from another Store dataset) the page
    is empty and ``next_after`` is ``high_water``, so the cursor can never pass the log."""
    clause, args = _kinds_clause(query.kinds)
    top = high_water(conn)
    count = int(conn.execute(f"SELECT COUNT(*) AS n FROM results_training_labels WHERE seq > ?{clause}", (query.after, *args)).fetchone()["n"])
    items: List[TrainingLabel] = []
    if query.limit > 0 and count:
        rows = conn.execute(f"SELECT * FROM results_training_labels WHERE seq > ?{clause} ORDER BY seq LIMIT ?",
                            (query.after, *args, query.limit)).fetchall()
        resolver = _Resolver(conn)
        items = [_item(resolver, row) for row in rows]
    next_after = items[-1].seq if items else min(query.after, top)
    return TrainingLabelPage(items=items, next_after=next_after, count_after=count, high_water=top)


def audit_details(query: TrainingLabelQuery, page: TrainingLabelPage) -> Dict[str, Any]:
    """``training_labels_read`` details: the cursor, the item count and counts per kind; never content."""
    details: Dict[str, Any] = {"after": query.after, "next_after": page.next_after, "items": len(page.items),
                               "withdrawn": sum(1 for i in page.items if i.withdrawn)}
    for kind in TrainingLabelKind:
        details[kind.value] = sum(1 for i in page.items if i.kind is kind)
    return details


__all__ = [
    "META_BACKFILLED", "append", "record_verdict_override", "record_signal_feedback", "record_speaker_correction", "judged_attribution",
    "backfill", "resolve_sources", "list_labels", "high_water", "audit_details",
]
