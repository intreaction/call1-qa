"""Call records, published result-group versions, the linked-artifact index, and the read views.

Everything here reads and writes only results-owned tables (``040_results.sql``). The queue area
is reached through ``call1.store.queue.api`` (conversations, source audio, group stakes, pending
work); artifact bytes through ``content.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from call1.contracts.artifacts import ArtifactKind
from call1.contracts.calls import (
    CallListItem,
    CallMetadata,
    CallRecordView,
    ContactSignalsView,
    Conversation,
    EvaluationView,
    PublisherState,
    ResultGroup,
    SummaryView,
    TranscriptReplacementView,
    TranscriptTurnView,
    TranscriptView,
    VocabularyCorrectionView,
    derive_result_state,
    result_state_inputs,
)
from call1.contracts.contents import (
    AudioChannelLayout,
    EnrichmentContent,
    PiiFindingsContent,
    QaScorecardContent,
    ResultKind,
    ResultState,
    SpeakerAttributionContent,
    SummaryContent,
    TextSentimentContent,
    ToneBlocksContent,
    TranscriptContent,
    TranscriptReplacement,
    TranscriptTurnContent,
    VadMetricsContent,
    VocabularyCorrection,
)
from call1.contracts.jobs import JOB_TYPE_RULES, JobType

from .. import db
from ..db import StoreConnection
from ..queue import api as queue_api
from . import content
from .masking import REDACTED, Masker, findings_match, masking_enabled

# The artifact kind each group's publisher writes (one output per publishing job type).
PUBLISHED_KIND: Dict[ResultKind, ArtifactKind] = {}
for _rule in JOB_TYPE_RULES.values():
    if _rule.publishes is not None:
        (_only,) = _rule.outputs.values()
        PUBLISHED_KIND[_rule.publishes] = _only

GROUP_OF_JOB_TYPE: Dict[JobType, Optional[ResultKind]] = {t: r.group for t, r in JOB_TYPE_RULES.items()}


# --- call records ----------------------------------------------------------------------------


def ensure_call(conn: StoreConnection, conversation: Conversation) -> bool:
    """Create the call record of a call conversation (idempotent). True when it was created."""
    if conversation.call_id is None:
        return False
    meta = conversation.call_metadata
    now = db.ts(conn.now())
    cur = conn.execute(
        "INSERT OR IGNORE INTO results_calls (call_id, conversation_id, agent_id, agent_display_name, agent_extension, external_call_ref, "
        "caller_reference, recorded_at, created_at, legacy_call, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            conversation.call_id,
            conversation.id,
            (meta.agent_id if meta else None) or "Unknown",
            meta.agent_display_name if meta else None,
            meta.agent_extension if meta else None,
            meta.external_call_ref if meta else None,
            meta.caller_reference if meta else None,
            db.ts(meta.recorded_at) if meta and meta.recorded_at else None,
            db.ts(conversation.created_at),
            1 if conversation.source.legacy_ingest_job_id is not None else 0,
            now,
        ),
    )
    conn.execute(
        "INSERT OR IGNORE INTO results_review_state (call_id, review_version, staleness, escalation_status, updated_at) VALUES (?, 0, 'current', 'NONE', ?)",
        (conversation.call_id, now),
    )
    return cur.rowcount > 0


def refresh_call_metadata(conn: StoreConnection, call_id: str, meta: CallMetadata) -> bool:
    """Project a call's updated ``CallMetadata`` (contract 1.1.0 re-registration) into the call
    record and every review-queue item of the call, resolved ones included (their agent fields are
    "from the call's current CallMetadata"). Items keep their ``item_version``: this is not a queue
    transition. True when the call record exists."""
    now = db.ts(conn.now())
    cur = conn.execute(
        "UPDATE results_calls SET agent_id = ?, agent_display_name = ?, agent_extension = ?, external_call_ref = ?, caller_reference = ?, "
        "recorded_at = ?, updated_at = ? WHERE call_id = ?",
        (meta.agent_id or "Unknown", meta.agent_display_name, meta.agent_extension, meta.external_call_ref, meta.caller_reference,
         db.ts(meta.recorded_at) if meta.recorded_at else None, now, call_id),
    )
    conn.execute(
        "UPDATE results_review_items SET agent_id = ?, agent_display_name = ?, agent_extension = ? WHERE call_id = ?",
        (meta.agent_id or "Unknown", meta.agent_display_name, meta.agent_extension, call_id),
    )
    return cur.rowcount > 0


def call_row(conn: StoreConnection, call_id: str):
    return conn.execute("SELECT * FROM results_calls WHERE call_id = ?", (call_id,)).fetchone()


def call_row_by_conversation(conn: StoreConnection, conversation_id: str):
    return conn.execute("SELECT * FROM results_calls WHERE conversation_id = ?", (conversation_id,)).fetchone()


def call_record_view(row) -> CallRecordView:
    return CallRecordView(
        call_id=row["call_id"],
        conversation_id=row["conversation_id"],
        agent_id=row["agent_id"],
        agent_display_name=row["agent_display_name"],
        agent_extension=row["agent_extension"],
        duration_seconds=row["duration_seconds"],
        channels=row["channels"],
        channel_layout=AudioChannelLayout(row["channel_layout"]) if row["channel_layout"] else None,
        sample_rate=row["sample_rate"],
        codec=row["codec"],
        silence_ratio=row["silence_ratio"],
        overtalk_duration=row["overtalk_duration"],
        avg_agent_tone=row["avg_agent_tone"],
        avg_caller_sentiment=row["avg_caller_sentiment"],
        created_at=db.parse_ts(row["created_at"]),
        recorded_at=db.parse_ts(row["recorded_at"]),
        legacy_call=bool(row["legacy_call"]),
    )


def review_version_of(conn: StoreConnection, call_id: str) -> int:
    row = conn.execute("SELECT review_version FROM results_review_state WHERE call_id = ?", (call_id,)).fetchone()
    return int(row["review_version"]) if row else 0


# --- the linked-artifact index ---------------------------------------------------------------


@dataclass(frozen=True)
class IndexedArtifact:
    artifact_id: str
    kind: ArtifactKind
    slot: str
    version: int
    checksum: str
    content_type: str
    job_id: Optional[str]
    graph_id: Optional[str]


def _indexed(row) -> IndexedArtifact:
    return IndexedArtifact(row["artifact_id"], ArtifactKind(row["kind"]), row["slot"], int(row["version"]), row["checksum"],
                           row["content_type"], row["job_id"], row["graph_id"])


def latest_artifact(conn: StoreConnection, conversation_id: str, kind: ArtifactKind, slot: str = "") -> Optional[IndexedArtifact]:
    row = conn.execute(
        "SELECT * FROM results_artifacts WHERE conversation_id = ? AND kind = ? AND slot = ? ORDER BY version DESC LIMIT 1",
        (conversation_id, kind.value, slot),
    ).fetchone()
    return None if row is None else _indexed(row)


# --- published group versions ----------------------------------------------------------------


@dataclass(frozen=True)
class Publication:
    kind: ResultKind
    version: int
    artifact_id: str
    checksum: str
    graph_id: Optional[str]
    graph_created_at: Optional[str]
    state: ResultState
    partial_reason: Optional[str]
    committed_at: str


def _publication(row) -> Publication:
    return Publication(ResultKind(row["kind"]), int(row["version"]), row["artifact_id"], row["checksum"], row["graph_id"],
                       row["graph_created_at"], ResultState(row["state"]), row["partial_reason"], row["committed_at"])


def latest_publication(conn: StoreConnection, conversation_id: str, kind: ResultKind) -> Optional[Publication]:
    row = conn.execute(
        "SELECT * FROM results_group_versions WHERE conversation_id = ? AND kind = ? ORDER BY version DESC LIMIT 1",
        (conversation_id, kind.value),
    ).fetchone()
    return None if row is None else _publication(row)


def publication(conn: StoreConnection, conversation_id: str, kind: ResultKind, version: int) -> Optional[Publication]:
    row = conn.execute(
        "SELECT * FROM results_group_versions WHERE conversation_id = ? AND kind = ? AND version = ?",
        (conversation_id, kind.value, version),
    ).fetchone()
    return None if row is None else _publication(row)


def latest_publications(conn: StoreConnection, conversation_id: str) -> Dict[ResultKind, Publication]:
    rows = conn.execute(
        "SELECT g.* FROM results_group_versions g JOIN (SELECT kind, MAX(version) AS v FROM results_group_versions "
        "WHERE conversation_id = ? GROUP BY kind) m ON g.kind = m.kind AND g.version = m.v WHERE g.conversation_id = ?",
        (conversation_id, conversation_id),
    ).fetchall()
    return {ResultKind(r["kind"]): _publication(r) for r in rows}


# --- derived result states -------------------------------------------------------------------


def result_groups(conn: StoreConnection, conversation_id: str) -> List[ResultGroup]:
    """Every result group with ``calls.derive_result_state`` over the queue's stakes.

    ``failure_code`` and ``reanalysis_request_id`` come from the queue area, which owns the job
    graph and the requests (``queue.api.failure_codes``, ``reanalysis_request_for_group``);
    ``failure_code`` is shown only while the derived rule says the newest publisher ended without
    a result after the published version."""
    stakes = queue_api.group_stakes(conn, conversation_id)
    pending = queue_api.reanalysis_pending_groups(conn, conversation_id)
    failures = queue_api.failure_codes(conn, conversation_id)
    published = latest_publications(conn, conversation_id)
    groups: List[ResultGroup] = []
    for kind in ResultKind:
        pub = published.get(kind)
        inputs = result_state_inputs(
            list(stakes.get(kind, [])),
            pub.state if pub else None,
            db.parse_ts(pub.graph_created_at) if pub and pub.graph_created_at else None,
            kind in pending,
        )
        ended = inputs.newest_publisher is PublisherState.ENDED_WITHOUT_RESULT and (pub is None or inputs.newest_publisher_is_newer_than_published)
        state = derive_result_state(inputs)
        partial_reason = pub.partial_reason if pub and pub.state is ResultState.PARTIAL else None
        if kind is ResultKind.TRANSCRIPT and pub is not None and state in (ResultState.AVAILABLE, ResultState.PARTIAL):
            # Contract 1.2.0 (decision 19): the text is withheld until the PII findings of this
            # revision exist, so the group reads 'partial' with the reason (fail closed).
            withheld = pii_withheld_reason(conn, conversation_id, pub)
            if withheld is not None:
                state, partial_reason = ResultState.PARTIAL, withheld
        groups.append(ResultGroup(
            kind=kind,
            state=state,
            version=pub.version if pub else None,
            artifact_id=pub.artifact_id if pub else None,
            produced_by_graph_id=pub.graph_id if pub else None,
            committed_at=db.parse_ts(pub.committed_at) if pub else None,
            partial_reason=partial_reason,
            failure_code=failures.get(kind) if ended else None,
            reanalysis_request_id=queue_api.reanalysis_request_for_group(conn, conversation_id, kind) if kind in pending else None,
        ))
    return groups


def group_state(groups: Sequence[ResultGroup], kind: ResultKind) -> ResultState:
    for group in groups:
        if group.kind is kind:
            return group.state
    return ResultState.DISABLED


def call_list_item(conn: StoreConnection, row, signal_context=None) -> CallListItem:
    from . import signals

    groups = result_groups(conn, row["conversation_id"])
    signal_fields = signals.list_fields(conn, row, signal_context or signals.ListContext.load(conn))
    review = conn.execute("SELECT review_version, escalation_status FROM results_review_state WHERE call_id = ?", (row["call_id"],)).fetchone()
    escalation = review["escalation_status"] if review else "NONE"
    return CallListItem(
        call_id=row["call_id"],
        conversation_id=row["conversation_id"],
        agent_id=row["agent_id"],
        agent_display_name=row["agent_display_name"],
        agent_extension=row["agent_extension"],
        duration_seconds=row["duration_seconds"],
        channel_layout=AudioChannelLayout(row["channel_layout"]) if row["channel_layout"] else None,
        created_at=db.parse_ts(row["created_at"]),
        transcript_state=group_state(groups, ResultKind.TRANSCRIPT),
        qa_state=group_state(groups, ResultKind.QA),
        summary_state=group_state(groups, ResultKind.SUMMARY),
        rubric_id=row["rubric_id"],
        overall_score=row["overall_score"],
        passed=None if row["passed"] is None else bool(row["passed"]),
        critical_failure=None if row["critical_failure"] is None else bool(row["critical_failure"]),
        requires_human_review=bool(row["requires_human_review"]),
        review_status=escalation if escalation and escalation != "NONE" else None,
        review_version=int(review["review_version"]) if review else 0,
        contact_signals_state=group_state(groups, ResultKind.CONTACT_SIGNALS),
        **signal_fields,
    )


# --- content of the current artifacts --------------------------------------------------------


def transcript_content(conn: StoreConnection, conversation_id: str) -> Tuple[Optional[Publication], Optional[TranscriptContent]]:
    pub = latest_publication(conn, conversation_id, ResultKind.TRANSCRIPT)
    if pub is None:
        return None, None
    return pub, content.read_model(conn, pub.checksum, TranscriptContent)


def enrichment_content(conn: StoreConnection, conversation_id: str) -> Optional[EnrichmentContent]:
    art = latest_artifact(conn, conversation_id, ArtifactKind.ENRICHMENT)
    return content.read_model(conn, art.checksum, EnrichmentContent) if art else None


def pii_findings_content(conn: StoreConnection, conversation_id: str) -> Optional[PiiFindingsContent]:
    """The newest linked ``pii_findings`` (contract 1.2.0). Callers check which transcript it covers."""
    art = latest_artifact(conn, conversation_id, ArtifactKind.PII_FINDINGS)
    return content.read_model(conn, art.checksum, PiiFindingsContent) if art else None


def masker_of(conn: StoreConnection, conversation_id: str, pub: Optional[Publication], transcript: Optional[TranscriptContent]) -> Masker:
    """The call's masker: rule values plus the PII findings of this transcript revision, or
    withheld (fail closed) while those findings are missing."""
    if not masking_enabled():
        return Masker.disabled()
    if pub is None or transcript is None:
        return Masker.for_call(None, None)
    return Masker.for_call(transcript, enrichment_content(conn, conversation_id), pii_findings_content(conn, conversation_id), pub.checksum)


def masker_for(conn: StoreConnection, conversation_id: str) -> Masker:
    pub, transcript = transcript_content(conn, conversation_id)
    return masker_of(conn, conversation_id, pub, transcript)


PII_PENDING_REASON = "Transcript text withheld until PII masking finishes for this revision"
PII_MISSING_REASON = "Transcript text withheld: PII masking did not finish for this revision (retry it, or request a full reanalysis)"


def pii_withheld_reason(conn: StoreConnection, conversation_id: str, pub: Publication) -> Optional[str]:
    """Why the published transcript's text is withheld (None when it is not): reviewer reads are
    masked and no ``pii_findings`` were made from this transcript revision yet."""
    if not masking_enabled():
        return None
    if findings_match(pii_findings_content(conn, conversation_id), pub.checksum):
        return None
    return PII_PENDING_REASON if queue_api.job_type_active(conn, conversation_id, JobType.ENRICHMENT) else PII_MISSING_REASON


def _tone_polarity_by_turn(tone: Optional[ToneBlocksContent]) -> Dict[int, float]:
    """Speech-weighted acoustic polarity (2 * valence - 1) of the scored blocks covering each turn."""
    if tone is None:
        return {}
    sums: Dict[int, Tuple[float, float]] = {}
    for block in tone.blocks:
        if block.status.value != "SCORED" or block.valence is None:
            continue
        weight = block.speech_seconds or 1.0
        polarity = 2 * max(0.0, min(1.0, float(block.valence))) - 1
        for turn_id in block.turn_ids:
            total, weights = sums.get(turn_id, (0.0, 0.0))
            sums[turn_id] = (total + polarity * weight, weights + weight)
    return {turn_id: total / weights for turn_id, (total, weights) in sums.items() if weights > 0}


def speaker_assignments(conn: StoreConnection, conversation_id: str) -> Tuple[Optional[IndexedArtifact], Dict[int, Tuple[str, Optional[str], Optional[float]]]]:
    art = latest_artifact(conn, conversation_id, ArtifactKind.SPEAKER_ATTRIBUTION)
    if art is None:
        return None, {}
    attribution = content.read_model(conn, art.checksum, SpeakerAttributionContent)
    return art, {a.turn_id: (a.speaker.value, a.speaker_cluster, a.confidence) for a in attribution.assignments}


def transcript_view(conn: StoreConnection, call_id: str, conversation_id: str) -> Optional[TranscriptView]:
    pub, transcript = transcript_content(conn, conversation_id)
    if pub is None or transcript is None:
        return None
    masker = masker_of(conn, conversation_id, pub, transcript)
    attribution_art, assignments = speaker_assignments(conn, conversation_id)
    sentiment_pub = latest_publication(conn, conversation_id, ResultKind.TEXT_SENTIMENT)
    sentiment = content.read_model(conn, sentiment_pub.checksum, TextSentimentContent) if sentiment_pub else None
    tone_pub = latest_publication(conn, conversation_id, ResultKind.TONE)
    tone = content.read_model(conn, tone_pub.checksum, ToneBlocksContent) if tone_pub else None
    vad_art = latest_artifact(conn, conversation_id, ArtifactKind.VAD_METRICS)
    vad = content.read_model(conn, vad_art.checksum, VadMetricsContent) if vad_art else None

    by_turn = {t.turn_id: t for t in sentiment.turns} if sentiment else {}
    tone_by_turn = _tone_polarity_by_turn(tone)
    turns: List[TranscriptTurnView] = []
    for turn in transcript.turns:
        speaker, cluster, _conf = assignments.get(turn.turn_id, (turn.speaker.value, turn.speaker_cluster, None))
        scored = by_turn.get(turn.turn_id)
        score = scored.score if scored else None
        divergence = None
        if score is not None and turn.turn_id in tone_by_turn:
            divergence = round(abs(tone_by_turn[turn.turn_id] - score), 2)
        words = [w.model_dump(mode="json") for w in turn.word_timestamps] if turn.word_timestamps else None
        turns.append(TranscriptTurnView(
            turn_id=turn.turn_id,
            speaker=speaker,
            speaker_cluster=cluster,
            start_time=turn.start_time,
            end_time=max(turn.end_time, turn.start_time),
            text="" if masker.withheld else masker.text(turn.text) or "",
            channel=turn.channel,
            confidence=turn.confidence,
            word_timestamps=masker.words(words, turn.text),
            text_sentiment=score,
            text_sentiment_label=scored.label if scored else None,
            sentiment_divergence=divergence,
        ))
    return TranscriptView(
        call_id=call_id,
        artifact_id=pub.artifact_id,
        version=pub.version,
        speaker_attribution_version=attribution_art.version if attribution_art else None,
        is_redacted=bool(transcript.is_redacted or masker.enabled),
        text_withheld=masker.withheld,
        duration_seconds=transcript.duration_seconds,
        turns=turns,
        tone_blocks=list(tone.blocks) if tone else [],
        vad_metrics=vad.model_dump(include={"total_speech_duration", "total_silence_duration", "silence_ratio", "overtalk_duration", "overtalk_ratio"}) if vad else None,
        avg_agent_tone=tone.avg_agent_tone if tone else None,
        avg_caller_sentiment=sentiment.avg_caller_sentiment if sentiment else None,
        vocabulary_correction=(vocabulary_correction_view(transcript.vocabulary_correction, transcript, turns, masker)
                               if transcript.vocabulary_correction is not None else None),
    )


# --- vocabulary corrections (1.3.0, decision 33; docs/DualAsr.md section 8) --------------------


def _holds_digit(value: str) -> bool:
    return any(ch.isdigit() or ch.isnumeric() for ch in value)


def _replacement_view(r: TranscriptReplacement, raw: Optional[TranscriptTurnContent], shown: Optional[TranscriptTurnView],
                      masker: Masker) -> Optional[TranscriptReplacementView]:
    """One replacement as a reviewer may see it, or None when it must be withheld: its turn or words
    cannot be found, one of its words is masked in the view, or its characters overlap a masked span
    (fail closed). ``heard`` passes the call's masking and is null when that changed it or it holds a
    digit; the offsets are mapped into the masked view text and null unless they land exactly."""
    if raw is None or shown is None:
        return None
    words = shown.word_timestamps
    if words is not None:
        if r.word_end > len(words) or any(w.word.strip() == REDACTED for w in words[r.word_start:r.word_end]):
            return None
    text = raw.text or ""
    if not (0 <= r.char_start < r.char_end <= len(text)):
        return None
    spans = masker.spans(text)
    if any(s < r.char_end and r.char_start < e for s, e in spans):
        return None
    shift = sum(len(REDACTED) - (e - s) for s, e in spans if e <= r.char_start)
    char_start, char_end = r.char_start + shift, r.char_end + shift
    if shown.text[char_start:char_end] != text[r.char_start:r.char_end]:
        char_start = char_end = None
    heard = masker.text(r.heard)
    return TranscriptReplacementView(
        turn_id=r.turn_id, word_start=r.word_start, word_end=r.word_end, char_start=char_start, char_end=char_end, term=r.term,
        source=r.source, heard=r.heard if heard == r.heard and not _holds_digit(r.heard) else None,
        start_time=r.start_time, end_time=r.end_time,
    )


def vocabulary_correction_view(correction: VocabularyCorrection, transcript: TranscriptContent, turns: Sequence[TranscriptTurnView],
                               masker: Masker) -> VocabularyCorrectionView:
    """``TranscriptView.vocabulary_correction``: status, the safe note and the replacements a reviewer
    may see. While the text is withheld every replacement is withheld; otherwise each one that
    overlaps masking is left out and counted in ``withheld_count``. ``candidate_text`` (Whisper's
    words) is never sent."""
    total = len(correction.replacements)
    if masker.withheld:
        return VocabularyCorrectionView(status=correction.status, note=correction.note, replacement_count=total, withheld_count=total)
    raw_turns = {t.turn_id: t for t in transcript.turns}
    shown_turns = {t.turn_id: t for t in turns}
    shown: List[TranscriptReplacementView] = []
    for r in correction.replacements:
        view = _replacement_view(r, raw_turns.get(r.turn_id), shown_turns.get(r.turn_id), masker)
        if view is not None:
            shown.append(view)
    return VocabularyCorrectionView(status=correction.status, note=correction.note, replacement_count=total,
                                    withheld_count=total - len(shown), replacements=shown)


def masked_scorecard_data(scorecard: QaScorecardContent, masker: Masker) -> dict:
    data = scorecard.model_dump(mode="json")
    if not masker.enabled:
        return data
    for verdict in data["verdicts"]:
        verdict["quoted_evidence"] = masker.text(verdict.get("quoted_evidence"))
        verdict["reasoning"] = masker.text(verdict.get("reasoning")) or ""
        for attempt in verdict.get("model_attempts", []):
            attempt["quoted_evidence"] = masker.text(attempt.get("quoted_evidence"))
            attempt["reasoning"] = masker.text(attempt.get("reasoning")) or ""
    data["escalation_reasons"] = [masker.text(r) or "" for r in data.get("escalation_reasons", [])]
    return data


def evaluation_view(conn: StoreConnection, call_id: str, conversation_id: str, version: Optional[int] = None,
                    masker: Optional[Masker] = None) -> Optional[EvaluationView]:
    pub = (publication(conn, conversation_id, ResultKind.QA, version) if version is not None
           else latest_publication(conn, conversation_id, ResultKind.QA))
    if pub is None:
        return None
    scorecard = content.read_model(conn, pub.checksum, QaScorecardContent)
    masker = masker or masker_for(conn, conversation_id)
    return EvaluationView(**masked_scorecard_data(scorecard, masker), call_id=call_id, artifact_id=pub.artifact_id, version=pub.version)


def summary_view(conn: StoreConnection, call_id: str, conversation_id: str) -> Optional[SummaryView]:
    pub = latest_publication(conn, conversation_id, ResultKind.SUMMARY)
    if pub is None:
        return None
    summary = content.read_model(conn, pub.checksum, SummaryContent)
    data = summary.model_dump(mode="json")
    masker = masker_for(conn, conversation_id)
    if masker.enabled:
        data["narrative"] = masker.text(data["narrative"]) or ""
        data["key_points"] = [masker.text(k) or "" for k in data["key_points"]]
        for highlight in data["rubric_highlights"]:
            highlight["note"] = masker.text(highlight["note"]) or ""
    return SummaryView(**data, call_id=call_id, artifact_id=pub.artifact_id, version=pub.version)


def contact_signals_view(conn: StoreConnection, call_id: str, conversation_id: str, *, admin: bool = False) -> Optional[ContactSignalsView]:
    """The published contact signals, masked (quotes, and since 1.3.0 field values, surface text and
    evidence), with the 1.3.0 read-time context (``signals.view``)."""
    from . import signals

    pub = latest_publication(conn, conversation_id, ResultKind.CONTACT_SIGNALS)
    if pub is None:
        return None
    return signals.view(conn, call_id, conversation_id, masker_for(conn, conversation_id), pub, admin=admin)
