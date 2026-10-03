"""Projection hooks: how the queue area's transactions update the results area's projections.

The **queue owner calls** these, always inside its own open transaction (``conn.in_transaction``),
at the step ``jobs.COMPLETION_TRANSACTION_STEPS`` names. The signatures are the coupling. Hooks
write only results-owned tables, use ``db.transaction(conn)`` (a SAVEPOINT inside the caller's
transaction), ``feed.append`` and ``audit.append``, and never commit, roll back or open another
connection. An exception aborts the caller's transaction.

What a completion projects (never for a draft-test graph):

* every linked output goes into the results artifact index (``results_artifacts``), which the
  views compose from;
* ``validation_vad`` fills the call record's media fields, ``acoustic_tone`` and
  ``text_sentiment`` its averages;
* ``embeddings`` replace the conversation's search vectors, whatever their scheme (search ranks
  only the scheme of Store's configured embedder, ``search.py``);
* a completion that carries ``result`` records the group's new version (the publishing artifact's
  slot version) and returns it for the receipt. A QA version also refreshes the call's score
  columns and verdict rows and applies the stale-write rule to the review and the review queue;
* a contact_signals version (1.3.0, v1 or v2) is projected into the signal tables
  (``signals.project``, replay-safe), then ``review_queue.on_new_signals`` creates the items of
  alert-targeting rules and a ``signal_alert`` event is appended per newly matching rule.
"""

from __future__ import annotations

import array
from typing import Optional, Sequence

from call1.contracts.artifacts import ArtifactKind
from call1.contracts.calls import Conversation
from call1.contracts.contents import (
    AudioValidationContent,
    EmbeddingsContent,
    QaScorecardContent,
    ResultKind,
    TextSentimentContent,
    ToneBlocksContent,
    VadMetricsContent,
)
from call1.contracts.errors import ErrorCode
from call1.contracts.events import CALL_METADATA_UPDATED_STATUS, ChangeKind
from call1.contracts.jobs import JOB_TYPE_RULES

from .. import db, feed
from ..db import StoreConnection
from ..errors import StoreError
from ..hooks import CompletedJob, FailedJob, LinkedOutput, ProjectionOutcome
from ..queue import api as queue_api
from . import content, records, review_queue, review_state, rubric_store, signals


def on_conversation_registered(conn: StoreConnection, conversation: Conversation) -> None:
    """``registerConversation`` created a conversation (``created: true`` only). For a call
    conversation, results creates its call record so the call lists as Analyzing at once."""
    if conversation.call_id is None:
        return None
    with db.transaction(conn):
        if records.ensure_call(conn, conversation):
            feed.append(conn, ChangeKind.CALL, conversation.call_id, 0, "registered", conversation_id=conversation.id,
                        call_id=conversation.call_id)
    return None


def on_call_metadata_updated(conn: StoreConnection, conversation: Conversation, updated_fields: Sequence[str]) -> None:
    """``registerConversation`` re-registered a known source identity and changed its call metadata
    (contract 1.1.0, ``calls.merge_call_metadata``). Refresh every projection that shows it (the
    call record, which the call list, call detail and escalations read, and the call's review-queue
    items) and append the ``call`` change event with status ``metadata_updated``. Nothing is
    reprocessed: no result, review version or review-queue state changes. A conversation without a
    call has no projection and no call event."""
    if conversation.call_id is None or conversation.call_metadata is None or not updated_fields:
        return None
    with db.transaction(conn):
        if records.call_row(conn, conversation.call_id) is None:
            records.ensure_call(conn, conversation)  # a call registered before results existed: project it now
        else:
            records.refresh_call_metadata(conn, conversation.call_id, conversation.call_metadata)
        feed.append(conn, ChangeKind.CALL, conversation.call_id, None, CALL_METADATA_UPDATED_STATUS, conversation_id=conversation.id,
                    call_id=conversation.call_id)
    return None


def _call_row(conn: StoreConnection, conversation_id: str, call_id: Optional[str]):
    row = records.call_row_by_conversation(conn, conversation_id)
    if row is None and call_id is not None:
        conversation = queue_api.get_conversation(conn, conversation_id)
        if conversation is not None and records.ensure_call(conn, conversation):
            feed.append(conn, ChangeKind.CALL, conversation.call_id, 0, "registered", conversation_id=conversation.id, call_id=conversation.call_id)
        row = records.call_row_by_conversation(conn, conversation_id)
    return row


def _index_output(conn: StoreConnection, completion: CompletedJob, output: LinkedOutput) -> None:
    art = output.artifact
    if art.version is None or art.conversation_id is None:
        return
    conn.execute(
        "INSERT OR IGNORE INTO results_artifacts (artifact_id, conversation_id, kind, slot, version, checksum, content_type, job_id, graph_id, committed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (art.id, art.conversation_id, art.kind.value, art.slot, art.version, art.checksum, art.content_type, completion.job.job_id,
         completion.job.graph_id, db.ts(completion.completed_at)),
    )


def _update_call(conn: StoreConnection, call_id: str, **fields) -> None:
    assignments = ", ".join(f"{name} = ?" for name in fields)
    conn.execute(f"UPDATE results_calls SET {assignments}, updated_at = ? WHERE call_id = ?", (*fields.values(), db.ts(conn.now()), call_id))


def _project_supporting(conn: StoreConnection, completion: CompletedJob, call_row) -> bool:
    """Call-record fields and the search index from non-publishing outputs. True when the call record changed."""
    changed = False
    conversation_id = completion.job.conversation_id
    for output in completion.outputs:
        art = output.artifact
        if call_row is not None and art.kind is ArtifactKind.VALIDATION_REPORT and art.slot == "":
            report = content.read_model(conn, art.checksum, AudioValidationContent)
            _update_call(conn, call_row["call_id"], duration_seconds=report.duration_seconds, channels=report.channels,
                         channel_layout=report.channel_layout.value, sample_rate=report.sample_rate, codec=report.codec)
            changed = True
        elif call_row is not None and art.kind is ArtifactKind.VAD_METRICS and art.slot == "":
            vad = content.read_model(conn, art.checksum, VadMetricsContent)
            _update_call(conn, call_row["call_id"], silence_ratio=vad.silence_ratio, overtalk_duration=vad.overtalk_duration)
            changed = True
        elif call_row is not None and art.kind is ArtifactKind.TONE_BLOCKS and art.slot == "":
            tone = content.read_model(conn, art.checksum, ToneBlocksContent)
            _update_call(conn, call_row["call_id"], avg_agent_tone=tone.avg_agent_tone)
            changed = True
        elif call_row is not None and art.kind is ArtifactKind.TEXT_SENTIMENT and art.slot == "":
            sentiment = content.read_model(conn, art.checksum, TextSentimentContent)
            _update_call(conn, call_row["call_id"], avg_caller_sentiment=sentiment.avg_caller_sentiment)
            changed = True
        elif art.kind is ArtifactKind.EMBEDDINGS and art.slot == "" and call_row is not None:
            embeddings = content.read_model(conn, art.checksum, EmbeddingsContent)
            # Every scheme is indexed (the newest embeddings replace the call's vectors); search ranks
            # only the configured embedder's scheme and counts the rest as needing re-embedding.
            conn.execute("DELETE FROM results_search_vectors WHERE conversation_id = ?", (conversation_id,))
            conn.executemany(
                "INSERT OR REPLACE INTO results_search_vectors (conversation_id, call_id, turn_id, artifact_id, scheme, dimensions, vector) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(conversation_id, call_row["call_id"], tv.turn_id, art.id, embeddings.scheme, len(tv.vector),
                  array.array("f", tv.vector).tobytes()) for tv in embeddings.turn_vectors],
            )
    return changed


def _publish(conn: StoreConnection, completion: CompletedJob, call_row) -> Optional[int]:
    result = completion.result
    if result is None:
        return None
    rule = JOB_TYPE_RULES[completion.job.job_type]
    if rule.publishes is not result.kind:
        raise StoreError(ErrorCode.VALIDATION_FAILED, "This job type does not publish that result group",
                         details={"job_type": completion.job.job_type.value, "result_kind": result.kind.value})
    kind = records.PUBLISHED_KIND[result.kind]
    published = [o.artifact for o in completion.outputs if o.artifact.kind is kind and o.artifact.slot == ""]
    if len(published) != 1 or published[0].version is None:
        raise StoreError(ErrorCode.VALIDATION_FAILED, "A publishing completion links exactly one artifact in the group's slot",
                         details={"result_kind": result.kind.value})
    art = published[0]
    conversation_id = completion.job.conversation_id
    call_id = call_row["call_id"] if call_row is not None else completion.job.call_id
    committed = db.ts(completion.completed_at)
    conn.execute(
        "INSERT OR IGNORE INTO results_group_versions (conversation_id, kind, version, call_id, artifact_id, checksum, job_id, graph_id, "
        "graph_created_at, state, partial_reason, committed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (conversation_id, result.kind.value, art.version, call_id, art.id, art.checksum, completion.job.job_id, completion.job.graph_id,
         db.ts(completion.job.graph_created_at), result.state.value, result.partial_reason, committed),
    )
    if result.kind is ResultKind.QA and call_row is not None:
        _publish_qa(conn, call_row, art.version, art.checksum, committed)
    if result.kind is ResultKind.CONTACT_SIGNALS and call_row is not None:
        _publish_signals(conn, call_row["call_id"], art.version, art.checksum)
    feed.append(conn, ChangeKind.RESULT, art.id, art.version, f"{result.kind.value}:{result.state.value}", conversation_id=conversation_id,
                call_id=call_id)
    return art.version


def _publish_qa(conn: StoreConnection, call_row, version: int, checksum: str, committed: str) -> None:
    call_id = call_row["call_id"]
    current = call_row["evaluation_version"]
    if current is not None and int(current) >= version:
        return  # replayed or out-of-order projection: keep the newest
    scorecard = content.read_model(conn, checksum, QaScorecardContent)
    low_confidence = review_queue.is_low_confidence(scorecard)
    _update_call(conn, call_id, evaluation_version=version, rubric_id=scorecard.rubric.rubric_id, rubric_version=scorecard.rubric.rubric_version,
                 overall_score=scorecard.overall_score, passed=1 if scorecard.passed else 0,
                 critical_failure=1 if scorecard.critical_failure else 0,
                 requires_human_review=1 if scorecard.requires_human_review else 0, low_confidence=1 if low_confidence else 0,
                 evaluated_at=db.ts(scorecard.evaluated_at))
    conn.executemany(
        "INSERT OR REPLACE INTO results_verdicts (call_id, evaluation_version, criterion_id, criterion_name, status, confidence) VALUES (?, ?, ?, ?, ?, ?)",
        [(call_id, version, v.criterion_id, v.criterion_name, v.status.value, v.confidence) for v in scorecard.verdicts],
    )
    review_state.on_new_evaluation(conn, call_id, call_row["conversation_id"], version, scorecard.requires_human_review)
    _, transcript = records.transcript_content(conn, call_row["conversation_id"])
    category = None
    if scorecard.rubric.rubric_version is not None:
        rubric = rubric_store.get_version(conn, scorecard.rubric.rubric_id, scorecard.rubric.rubric_version)
        category = rubric.definition.category.value if rubric else None
    review_queue.on_new_evaluation(conn, review_queue.ScoreFacts(
        call_id=call_id,
        conversation_id=call_row["conversation_id"],
        evaluation_version=version,
        agent_id=call_row["agent_id"],
        agent_display_name=call_row["agent_display_name"],
        agent_extension=call_row["agent_extension"],
        overall_score=scorecard.overall_score,
        critical_failure=scorecard.critical_failure,
        low_confidence=low_confidence,
        dispute_signal=review_queue.dispute_signal(transcript),
        duration_seconds=call_row["duration_seconds"],
        rubric_category=category,
        **_signal_facts(conn, call_id),
    ))


def _signal_facts(conn: StoreConnection, call_id: str) -> dict:
    facts = signals.alert_facts(conn, call_id)
    return {"signals_version": facts.signals_version, "signal_alerts": facts.alerts, "signal_alert_reasons": dict(facts.reasons)}


def _score_facts_now(conn: StoreConnection, call_id: str) -> Optional[review_queue.ScoreFacts]:
    """ScoreFacts of the call's current evaluation (from the projected score columns), or None."""
    row = records.call_row(conn, call_id)
    if row is None or row["evaluation_version"] is None:
        return None
    _, transcript = records.transcript_content(conn, row["conversation_id"])
    category = None
    if row["rubric_id"] is not None and row["rubric_version"] is not None:
        rubric = rubric_store.get_version(conn, row["rubric_id"], int(row["rubric_version"]))
        category = rubric.definition.category.value if rubric else None
    return review_queue.ScoreFacts(
        call_id=call_id, conversation_id=row["conversation_id"], evaluation_version=int(row["evaluation_version"]), agent_id=row["agent_id"],
        agent_display_name=row["agent_display_name"], agent_extension=row["agent_extension"], overall_score=row["overall_score"],
        critical_failure=bool(row["critical_failure"]), low_confidence=bool(row["low_confidence"]),
        dispute_signal=review_queue.dispute_signal(transcript), duration_seconds=row["duration_seconds"], rubric_category=category,
        **_signal_facts(conn, call_id),
    )


def _publish_signals(conn: StoreConnection, call_id: str, version: int, checksum: str) -> None:
    """Project a contact_signals version (1.3.0), then the queue hook and the alert events."""
    projected = signals.project(conn, call_id, version, checksum)
    if projected is None:
        return  # replayed or out-of-order: keep the newest
    facts = _score_facts_now(conn, call_id)
    if facts is not None:
        review_queue.on_new_signals(conn, facts)
    signals.append_fired_events(conn, projected, records.call_row(conn, call_id)["conversation_id"])


def project_existing_signals(conn: StoreConnection, *, limit: int = 1000) -> int:
    """``python -m call1.store project-signals``: project the newest published contact_signals
    version of calls published before 1.3.0 (or otherwise not projected yet). Idempotent and
    bounded (``limit`` calls per run). Returns the number of calls projected."""
    rows = conn.execute(
        "SELECT c.call_id, g.version, g.checksum FROM results_calls c JOIN results_group_versions g ON g.conversation_id = c.conversation_id "
        "AND g.kind = ? AND g.version = (SELECT MAX(version) FROM results_group_versions WHERE conversation_id = c.conversation_id AND kind = ?) "
        "WHERE c.signals_version IS NULL OR c.signals_version < g.version ORDER BY c.created_at, c.call_id LIMIT ?",
        (ResultKind.CONTACT_SIGNALS.value, ResultKind.CONTACT_SIGNALS.value, limit)).fetchall()
    done = 0
    for row in rows:
        with db.transaction(conn):
            _publish_signals(conn, row["call_id"], int(row["version"]), row["checksum"])
        done += 1
    return done


def apply_completion(conn: StoreConnection, completion: CompletedJob) -> ProjectionOutcome:
    """Step 7 of the completion transaction, after the outputs are linked and the job SUCCEEDED.

    A draft test (``completion.job.draft_test_request_id``) projects nothing: no call-record change,
    no review-queue items, no change to the call's result groups. Otherwise it indexes the outputs,
    updates the call record, publishes ``completion.result`` and returns the published version.
    """
    if completion.job.draft_test_request_id is not None:
        return ProjectionOutcome()
    with db.transaction(conn):
        call_row = _call_row(conn, completion.job.conversation_id, completion.job.call_id)
        for output in completion.outputs:
            _index_output(conn, completion, output)
        changed = _project_supporting(conn, completion, call_row)
        version = _publish(conn, completion, call_row)
        if call_row is not None and (changed or completion.job.job_type.value == "speaker_attribution"):
            feed.append(conn, ChangeKind.CALL, call_row["call_id"], None, "updated", conversation_id=call_row["conversation_id"],
                        call_id=call_row["call_id"])
    return ProjectionOutcome(result_version=version)


def on_job_failed(conn: StoreConnection, failure: FailedJob) -> None:
    """A job ended an attempt or its life without a result (/fail, lease expiry, reject release,
    admission rejection, cancel). Results caches nothing about it: the derived state and
    ``failure_code`` are computed on read from the queue's job graph (``records.result_groups``).
    It tells Evaluate through the change feed. Draft tests and supporting jobs (embeddings) are ignored."""
    job = failure.job
    group = records.GROUP_OF_JOB_TYPE.get(job.job_type)
    if job.draft_test_request_id is not None or group is None:
        return None
    with db.transaction(conn):
        status = f"{group.value}:job_{failure.status.value.lower()}"
        feed.append(conn, ChangeKind.RESULT, job.call_id or job.conversation_id, None, status, conversation_id=job.conversation_id,
                    call_id=job.call_id)
    return None
