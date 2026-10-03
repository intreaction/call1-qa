"""Contact Signals extraction for contact-centre call milestones.

Extracts the full lifecycle of milestone signals grounded in exact transcript
quotes using the local Gemma LLM via MLX:
  - intent: Caller states what they want to accomplish or why they are calling.
  - issue: Caller describes the problem, discrepancy, fee, or dispute prompting contact.
  - friction: Caller describes an obstacle, repeated failure, lockout, or frustration.
  - fix_proposed: Agent proposes a fix or workaround, not yet completed.
  - agent_reports_completed: Agent explicitly reports completing an action.
  - caller_confirms_resolved: Caller explicitly confirms problem is resolved / fix works.
  - caller_reports_unresolved: Caller explicitly reports problem persists / fix failed.
  - deferred: Agent defers work, promises callback, or notes later action required.

Strict provenance: every observation must cite an exact verbatim quote from the
target speaker. Missing evidence is not a negative extraction; absence remains absence.
No heuristic or regex fallback is used.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from typing import Any, Iterator, Optional

from call1.models.schemas import SpeakerRole, TranscriptTurn

logger = logging.getLogger(__name__)

ALL_SIGNALS: dict[str, tuple[str, str, str]] = {
    "intent": (
        "Caller objective",
        "caller",
        "Caller explicitly states what they want to accomplish or why they are calling.",
    ),
    "issue": (
        "Reported issue",
        "caller",
        "Caller describes the specific problem, discrepancy, fee, or dispute that prompted contact.",
    ),
    "friction": (
        "Friction point",
        "caller",
        "Caller describes an obstacle, repeated failure, lockout, or frustration in this or prior contacts.",
    ),
    "fix_proposed": (
        "Proposed fix",
        "agent",
        "Agent proposes a fix, workaround, reset, or adjustment, not yet completed.",
    ),
    "agent_reports_completed": (
        "Agent completed",
        "agent",
        "Agent explicitly reports completing an action, updating records, or verifying status.",
    ),
    "caller_confirms_resolved": (
        "Caller confirmed",
        "caller",
        "Caller explicitly confirms problem is resolved, question answered, or fix works.",
    ),
    "caller_reports_unresolved": (
        "Still unresolved",
        "caller",
        "Caller explicitly says the problem is unresolved, unsatisfied, or escalating.",
    ),
    "deferred": (
        "Deferred",
        "agent",
        "Agent defers work, promises callback, or states a later review/specialist timeline is required.",
    ),
}

# Backwards-compatible alias for tests and callers written against the
# resolution-only signal set.
RESOLUTIONS = ALL_SIGNALS

RESOLUTION_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": [
                            "intent",
                            "issue",
                            "friction",
                            "fix_proposed",
                            "agent_reports_completed",
                            "caller_confirms_resolved",
                            "caller_reports_unresolved",
                            "deferred",
                        ],
                    },
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "evidence_turn": {"type": "integer"},
                    "evidence_quote": {"type": "string", "maxLength": 500},
                    "rationale": {"type": "string", "maxLength": 500},
                },
                "required": ["kind", "confidence", "evidence_turn", "evidence_quote"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["observations"],
    "additionalProperties": False,
}

LIFECYCLE_SCHEMA = {
    "type": "object",
    "properties": {
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["intent", "issue", "friction"]},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "evidence_turn": {"type": "integer"},
                    "evidence_quote": {"type": "string", "maxLength": 500},
                },
                "required": ["kind", "confidence", "evidence_turn", "evidence_quote"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["observations"],
    "additionalProperties": False,
}

RESOLUTION_PASS_SCHEMA = {
    "type": "object",
    "properties": {
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": [
                            "fix_proposed",
                            "agent_reports_completed",
                            "caller_confirms_resolved",
                            "caller_reports_unresolved",
                            "deferred",
                        ],
                    },
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "evidence_turn": {"type": "integer"},
                    "evidence_quote": {"type": "string", "maxLength": 500},
                },
                "required": ["kind", "confidence", "evidence_turn", "evidence_quote"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["observations"],
    "additionalProperties": False,
}

LIFECYCLE_SYSTEM = (
    "You extract contact-centre lifecycle milestone signals from a transcript.\n"
    "Identify all of these caller lifecycle signals that are explicitly supported by the transcript:\n"
    "  - intent: Caller explicitly states what they want to accomplish or why they are calling (target speaker: caller)\n"
    "  - issue: Caller describes the specific problem, discrepancy, fee, or dispute that prompted contact (target speaker: caller)\n"
    "  - friction: Caller describes an obstacle, repeated failure, lockout, or frustration in this or prior contacts (target speaker: caller)\n\n"
    "Rules:\n"
    "1. For each observation, evidence_quote MUST copy a contiguous, exact phrase from one line of the transcript. Never paraphrase or repair it.\n"
    "2. The speaker of evidence_turn MUST match caller.\n"
    "3. Only explicit support creates an observation. If none occurred, return {\"observations\": []}.\n"
    "4. Return strictly valid JSON adhering to schema."
)

RESOLUTION_SYSTEM = (
    "You extract contact-centre resolution and outcome milestone signals from a transcript.\n"
    "Identify all of these resolution signals that are explicitly supported by the transcript:\n"
    "  - fix_proposed: Agent proposes a fix, workaround, reset, or adjustment, not yet completed (target speaker: agent)\n"
    "  - agent_reports_completed: Agent explicitly reports completing an action, updating records, or verifying status (target speaker: agent)\n"
    "  - caller_confirms_resolved: Caller explicitly confirms problem is resolved, question answered, or fix works (target speaker: caller)\n"
    "  - caller_reports_unresolved: Caller explicitly says problem is unresolved, unsatisfied, or escalating (target speaker: caller)\n"
    "  - deferred: Agent defers work, promises callback, or states later review is needed (target speaker: agent)\n\n"
    "Rules:\n"
    "1. For each observation, evidence_quote MUST copy a contiguous, exact phrase from one line of the transcript.\n"
    "2. The speaker of evidence_turn MUST match target speaker.\n"
    "3. Only explicit support creates an observation. If none occurred, return {\"observations\": []}.\n"
    "4. Return strictly valid JSON adhering to schema."
)

RUBRIC_SYSTEM = (
    "You extract contact-centre call milestone signals from a transcript.\n"
    "Identify all of these eight milestone signals that are explicitly supported by the transcript:\n"
    "  - intent: Caller explicitly states what they want to accomplish or why they are calling (target speaker: caller)\n"
    "  - issue: Caller describes the specific problem, discrepancy, fee, or dispute that prompted contact (target speaker: caller)\n"
    "  - friction: Caller describes an obstacle, repeated failure, lockout, or frustration in this or prior contacts (target speaker: caller)\n"
    "  - fix_proposed: Agent proposes a fix, workaround, reset, or adjustment, not yet completed (target speaker: agent)\n"
    "  - agent_reports_completed: Agent explicitly reports completing an action, updating records, or verifying status (target speaker: agent)\n"
    "  - caller_confirms_resolved: Caller explicitly confirms problem is resolved, question answered, or fix works (target speaker: caller)\n"
    "  - caller_reports_unresolved: Caller explicitly says the problem is unresolved, unsatisfied, or escalating (target speaker: caller)\n"
    "  - deferred: Agent defers work, promises callback, or states a later review/specialist timeline is required (target speaker: agent)\n\n"
    "Rules:\n"
    "1. For each observation, evidence_quote MUST copy a contiguous, exact phrase from one line of the transcript. Never paraphrase or repair it.\n"
    "2. The speaker of evidence_turn MUST match the target speaker of that signal kind. A caller signal may never cite an agent turn and vice versa.\n"
    "3. Only explicit support creates an observation. If no milestone events occurred, return an empty observations array: {\"observations\": []}.\n"
    "4. Extract every qualifying milestone, including several observations of the same kind when they appear.\n"
    "5. Return strictly valid JSON adhering to the schema."
)


PASS_MAX_TOKENS = 512

PASS_DEFINITIONS: dict[str, tuple[str, str, dict]] = {
    "lifecycle": (
        LIFECYCLE_SYSTEM,
        "Extract all caller lifecycle milestone signals (intent, issue, friction) supported by exact quotes.",
        LIFECYCLE_SCHEMA,
    ),
    "resolution": (
        RESOLUTION_SYSTEM,
        "Extract all resolution and outcome milestone signals supported by exact quotes.",
        RESOLUTION_PASS_SCHEMA,
    ),
}
"""The two extraction passes: system prompt, closing instruction and response schema. Shared by
``extract_contact_signals`` and the split Process app's per-pass jobs."""


def build_transcript_block(turns: list[TranscriptTurn]) -> tuple[dict[int, TranscriptTurn], str]:
    """The turns by ID and the numbered transcript block both passes are prompted with."""
    lines = []
    turns_by_id = {}
    for idx, t in enumerate(turns):
        tid = t.turn_id if t.turn_id is not None else idx
        turns_by_id[tid] = t
        spk = (t.speaker.value if hasattr(t.speaker, "value") else str(t.speaker)).lower()
        lines.append(f"[Turn {tid}] {spk.capitalize()}: {t.text.strip()}")
    return turns_by_id, "\n".join(lines)


def pass_prompt(pass_name: str, turn_count: int, transcript_block: str) -> tuple[str, str, dict]:
    """(system prompt, user prompt, response schema) for one pass."""
    system, instruction, schema = PASS_DEFINITIONS[pass_name]
    user = f"Transcript extract ({turn_count} turns):\n{transcript_block}\n\n{instruction}"
    return system, user, schema


def parse_pass_observations(raw_json: str, turns_by_id: dict[int, TranscriptTurn], call_id: str = "") -> Iterator[dict[str, Any]]:
    """Quote-verified signals from one pass's JSON answer, yielded in answer order. Raises on output
    that is not JSON or not an observations object; observations whose quote, turn or speaker do
    not verify are dropped."""
    parsed = json.loads(raw_json)
    for obs in parsed.get("observations", []):
        kind = obs.get("kind")
        evidence_turn = obs.get("evidence_turn")
        evidence_quote = obs.get("evidence_quote")
        confidence = obs.get("confidence", 0.9)

        if evidence_turn in turns_by_id and evidence_quote and kind:
            sig = _validate_and_build_signal(
                call_id=call_id,
                kind=kind,
                turn=turns_by_id[evidence_turn],
                raw_quote=evidence_quote,
                confidence=confidence,
            )
            if sig:
                yield sig


def compute_transcript_fingerprint(turns: list[TranscriptTurn]) -> str:
    """Compute deterministic SHA-256 fingerprint over turn sequence."""
    payload = [
        (
            t.turn_id if t.turn_id is not None else idx,
            (t.speaker.value if hasattr(t.speaker, "value") else str(t.speaker)).lower(),
            t.text.strip(),
        )
        for idx, t in enumerate(turns)
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def compute_report_sha256(call_id: str, signals: list[dict[str, Any]]) -> str:
    """Compute SHA-256 fingerprint over extracted signals."""
    content = json.dumps([call_id, signals], sort_keys=True, default=str)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _align_quote_timestamps(
    turn: TranscriptTurn, char_start: int, char_end: int
) -> tuple[float, float]:
    """Calculate exact start/end timestamps from word timestamps if present, or interpolate."""
    words = turn.word_timestamps or []
    if not words:
        dur = max(0.1, turn.end_time - turn.start_time)
        text_len = max(1, len(turn.text))
        s = turn.start_time + (char_start / text_len) * dur
        e = turn.start_time + (char_end / text_len) * dur
        return round(s, 2), round(max(s + 0.1, e), 2)

    char_pos = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    for w in words:
        w_text = w.word.strip()
        w_start = turn.text.find(w_text, char_pos)
        if w_start == -1:
            w_start = char_pos
        w_end = w_start + len(w_text)
        char_pos = w_end

        if w_end >= char_start and (start_time is None):
            start_time = w.start_time
        if w_start <= char_end:
            end_time = w.end_time

    s = start_time if start_time is not None else turn.start_time
    e = end_time if end_time is not None else turn.end_time
    return round(s, 2), round(max(s + 0.1, e), 2)


def _validate_and_build_signal(
    call_id: str,
    kind: str,
    turn: TranscriptTurn,
    raw_quote: str,
    confidence: float,
) -> Optional[dict[str, Any]]:
    """Validate quote provenance against turn text and return structured signal."""
    if kind not in ALL_SIGNALS:
        return None

    label, expected_speaker, _ = ALL_SIGNALS[kind]
    turn_speaker = (turn.speaker.value if hasattr(turn.speaker, "value") else str(turn.speaker)).lower()
    if turn_speaker != expected_speaker:
        return None

    quote = raw_quote.strip()
    if not quote or len(quote) < 3:
        return None

    pos = turn.text.find(quote)
    if pos == -1:
        # Quote does not appear verbatim in turn text: reject to enforce provenance
        return None

    char_start = pos
    char_end = pos + len(quote)
    start_t, end_t = _align_quote_timestamps(turn, char_start, char_end)
    signal_id = f"cs_{call_id or 'call'}_{uuid.uuid4().hex[:8]}"

    return {
        "id": signal_id,
        "kind": kind,
        "label": label,
        "start": start_t,
        "end": end_t,
        "speaker": turn_speaker,
        "quote": quote,
        "turn_id": turn.turn_id if turn.turn_id is not None else 0,
        "char_start": char_start,
        "char_end": char_end,
        "review_status": "detected",
        "confidence": round(float(confidence), 3),
    }


def extract_contact_signals(
    turns: list[TranscriptTurn],
    call_id: str = "",
) -> tuple[list[dict[str, Any]], str, str, bool]:
    """Extract lifecycle contact signals with strict quote verification.

    Returns:
      (signals, transcript_fingerprint, report_sha256, is_partial)
    """
    transcript_fp = compute_transcript_fingerprint(turns)
    is_partial = False
    signals: list[dict[str, Any]] = []

    if not turns:
        report_sha = compute_report_sha256(call_id, [])
        return [], transcript_fp, report_sha, False

    # Check if MLX backend is available
    if os.getenv("CALL1_BACKEND") != "mlx":
        # Machine learning inference unavailable in non-MLX mode; do not substitute regex
        report_sha = compute_report_sha256(call_id, [])
        return [], transcript_fp, report_sha, True

    from call1.adapters import get_adapter

    adapter = get_adapter()
    text_model_path = os.getenv("CALL1_MLX_TEXT_PATH")

    turns_by_id, transcript_block = build_transcript_block(turns)
    seen_spans = set()
    for pass_name in ("lifecycle", "resolution"):
        sys_prompt, user_prompt, resp_schema = pass_prompt(pass_name, len(turns), transcript_block)
        try:
            raw_json = adapter.generate(
                system=sys_prompt,
                prompt=user_prompt,
                max_tokens=PASS_MAX_TOKENS,
                response_schema=resp_schema,
                text_model_path=text_model_path,
            )
            for sig in parse_pass_observations(raw_json, turns_by_id, call_id):
                span_key = (sig["kind"], sig["turn_id"], sig["char_start"], sig["char_end"])
                if span_key not in seen_spans:
                    seen_spans.add(span_key)
                    signals.append(sig)
        except Exception as exc:
            logger.warning("MLX Contact Signals extraction pass failed: %s", exc)
            is_partial = True

    signals.sort(key=lambda s: s["start"])
    report_sha = compute_report_sha256(call_id, signals)
    return signals, transcript_fp, report_sha, is_partial
