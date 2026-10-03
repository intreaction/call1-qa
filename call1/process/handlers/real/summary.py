"""The summary: one job per transcript segment, pairwise then final syntheses, and the assembly.

This is ``call1.summarizer.generate_summary`` split along its own seams, with its prompts, its
strict answer parsing and one structural retry (``parse_summary_answer``,
``SUMMARY_RETRY_INSTRUCTIONS``), and its citation check (``_validate_key_points_grounded``: a key
point survives only if it cites an existing turn it shares at least two words with):

* ``summary_segment``: the numbered ``[turn] SPEAKER: text`` lines of its window (long turns split
  at 2,500 bytes, keeping their turn number), the chunk prompt ("segment i of n", or "the full call
  transcript" for a single segment), and the key points checked against the segment's own lines.
  A single-segment summary with no surviving key point fails, as before the split.
* ``summary_synthesis``: the pre-split reduction over its parts, pairwise (a JSON list of the two
  parts) or final ("SEGMENT i:" blocks), with the synthesis prompt.
* ``summary_assembly`` (replacing the code stage in real mode): the final citation check against the
  whole call, and the pre-split fallback to evenly spaced, re-checked source key points when a lossy
  synthesis drops its citations. No key point surviving is a failure, never an unchecked summary.

Generation goes through ``call1.question_models.generate_text`` (see ``llm.py``).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Set

from call1.contracts.contents import (
    SummaryCitation,
    SummaryContent,
    SummarySegmentContent,
    SummarySynthesisContent,
    TranscriptContent,
    TranscriptTurnContent,
    TurnWindow,
)
from call1.contracts.errors import JobErrorCode
from call1.contracts.jobs import JobType

from call1.process.handlers.base import Handler, HandlerError, HandlerJob, HandlerResult, Output
from call1.process.transcripts import plan_segments, turns_in

from .convert import legacy_turns
from .llm import LlmTransport, check_route, template_version
from .masking import enrichment, mask, route_masked, sensitive_values

ADAPTER_VERSION = "1"
NO_KEY_POINTS = "No key points survived citation checking."


def _summary_max_tokens() -> int:
    """The pre-split summary output budget (768 on MLX, else 400; ``SummaryProviderSettings``)."""
    from call1.models.schemas import SummaryProviderSettings

    return SummaryProviderSettings().max_tokens


def _schema():
    from call1.summarizer import SUMMARY_SCHEMA

    return SUMMARY_SCHEMA


def numbered_lines(turns: Sequence[TranscriptTurnContent], texts: Optional[Dict[int, str]] = None) -> List[str]:
    """``[turn_id] SPEAKER: text`` per turn, with the real turn IDs the citations use."""
    return [f"[{t.turn_id}] {t.speaker.value}: {(texts or {}).get(t.turn_id, t.text)}" for t in turns]


def masked_texts(job: HandlerJob, transcript: TranscriptContent) -> Optional[Dict[int, str]]:
    """Masked text per turn when the route is masked (values from the whole call)."""
    if not route_masked(job):
        return None
    return masked_texts_for(job, transcript)


def masked_texts_for(job: HandlerJob, transcript: TranscriptContent) -> Dict[int, str]:
    """Every turn masked with the call's sensitive values: its numeric entities (the pinned
    ``enrichment`` input, or the same extractor run here), the PII patterns and the PII model."""
    turns = legacy_turns(transcript, enrichment(job, compute=True))
    values = sensitive_values(turns, job=job)
    return {t.turn_id: mask(t.text, values) for t in turns}


def generate_validated(transport: LlmTransport, system: str, prompt: str) -> Dict[str, Any]:
    """One summary answer, strictly parsed; a malformed answer is retried once with the tighter
    instruction, exactly as ``call1.summarizer._generate_one``. Still malformed: validation_rejected."""
    from call1.summarizer import SUMMARY_RETRY_INSTRUCTIONS, parse_summary_answer

    last = "Model returned unparseable output."
    for instruction in SUMMARY_RETRY_INSTRUCTIONS:
        answer = transport.generate(system + instruction, prompt, response_schema=_schema(), schema_name="call_summary",
                                    max_tokens=_summary_max_tokens())
        try:
            return parse_summary_answer(answer.raw)
        except RuntimeError as exc:
            last = str(exc)
    raise HandlerError(JobErrorCode.VALIDATION_REJECTED, last, outputs={}, usage=transport.usage())


def citations_for(key_points: Sequence[str], allowed: Set[int]) -> List[SummaryCitation]:
    """A key-point citation for every point whose ``(turn N)`` references name allowed turns."""
    from call1.summarizer import _extract_turn_refs

    out = []
    for index, point in enumerate(key_points):
        refs = sorted({r for r in _extract_turn_refs(point) if r in allowed})
        if refs:
            out.append(SummaryCitation(claim="key_point", index=index, turn_ids=refs))
    return out


def _fail(job: HandlerJob, transport: LlmTransport, exc: HandlerError, template_id: str, version: str,
          window: Optional[TurnWindow] = None) -> HandlerError:
    exc.outputs = {"prompt_input": Output(transport.prompt_input(template_id, version, window=window))}
    exc.usage = transport.usage()
    return exc


class RealSummarySegment(Handler):
    job_type = JobType.SUMMARY_SEGMENT
    adapter_id = "call1.summarizer.segment"
    adapter_version = ADAPTER_VERSION

    def __init__(self, batch_turns: int = 60) -> None:
        self.batch_turns = batch_turns

    def ready(self, job: HandlerJob) -> None:
        check_route(job)

    def run(self, job: HandlerJob) -> HandlerResult:
        from call1.summarizer import SYSTEM_PROMPT, _build_chunk_prompt, _validate_key_points_grounded, bound_turn_lines

        spec = job.parameters.segment
        transcript = job.transcript()
        window = spec.window if spec is not None else None
        index = spec.index if spec is not None else 0
        count = int(job.parameters.extra.get("segment_count") or 0) or len(plan_segments(transcript, self.batch_turns))
        turns = turns_in(transcript, window)
        version = template_version(SYSTEM_PROMPT, _schema(), "segment")
        transport = LlmTransport(job)
        if not turns:
            raise HandlerError(JobErrorCode.VALIDATION_REJECTED, "No transcript content to summarize.")
        lines = numbered_lines(turns, masked_texts(job, transcript))
        chunk = bound_turn_lines(lines)
        prompt = _build_chunk_prompt(chunk, [], index, count)
        try:
            result = generate_validated(transport, SYSTEM_PROMPT, prompt)
        except HandlerError as exc:
            raise _fail(job, transport, exc, "call1.summary.segment", version, window) from None
        # Multi-segment: each segment's points are checked against its own lines, so a model cannot
        # cite another segment; a single segment is checked against the whole call's lines.
        key_points = _validate_key_points_grounded(result["key_points"], lines if count == 1 else chunk)
        if count == 1 and not key_points:
            raise _fail(job, transport, HandlerError(JobErrorCode.VALIDATION_REJECTED, NO_KEY_POINTS), "call1.summary.segment", version, window)
        allowed = {t.turn_id for t in turns}
        content = SummarySegmentContent(segment_index=index, window=window or TurnWindow(turn_start=turns[0].turn_id, turn_end=turns[-1].turn_id),
                                        narrative=result["narrative"], key_points=key_points, citations=citations_for(key_points, allowed))
        return HandlerResult(outputs={"segment": Output(content),
                                      "prompt_input": Output(transport.prompt_input("call1.summary.segment", version, window=window))},
                             usage=transport.usage())


def _part_index(role: str) -> int:
    try:
        return int(role.split(":", 1)[1])
    except (IndexError, ValueError):
        return 0


class RealSummarySynthesis(Handler):
    job_type = JobType.SUMMARY_SYNTHESIS
    adapter_id = "call1.summarizer.synthesis"
    adapter_version = ADAPTER_VERSION

    def ready(self, job: HandlerJob) -> None:
        check_route(job)

    def run(self, job: HandlerJob) -> HandlerResult:
        from call1.summarizer import SYNTHESIS_PROMPT

        parts = [item.content() for _, item in sorted(job.inputs_with_prefix("part:").items(), key=lambda kv: _part_index(kv[0]))]
        if not parts:
            raise HandlerError(JobErrorCode.INPUT_UNAVAILABLE, "a synthesis needs its parts")
        final = bool(job.parameters.extra.get("final"))
        results = [{"narrative": p.narrative, "key_points": list(p.key_points)} for p in parts]  # type: ignore[union-attr]
        if final:
            prompt = "\n\n".join(f"SEGMENT {i + 1}:\n" + json.dumps(r, ensure_ascii=False) for i, r in enumerate(results))
        else:
            prompt = json.dumps(results, ensure_ascii=False)
        version = template_version(SYNTHESIS_PROMPT, _schema(), "final" if final else "pairwise")
        transport = LlmTransport(job)
        try:
            result = generate_validated(transport, SYNTHESIS_PROMPT, prompt)
        except HandlerError as exc:
            raise _fail(job, transport, exc, "call1.summary.synthesis", version) from None
        indexes: List[int] = []
        for part in parts:
            if isinstance(part, SummarySegmentContent):
                indexes.append(part.segment_index)
            else:
                indexes.extend(part.segment_indexes)  # type: ignore[union-attr]
        # Citations name the turns the points cite; the assembly checks them against the call.
        cited = {r for point in result["key_points"] for r in _refs(point)}
        content = SummarySynthesisContent(segment_indexes=sorted(set(indexes)), final=final, narrative=result["narrative"],
                                          key_points=list(result["key_points"]), citations=citations_for(result["key_points"], cited))
        return HandlerResult(outputs={"synthesis": Output(content), "prompt_input": Output(transport.prompt_input("call1.summary.synthesis", version))},
                             usage=transport.usage())


def _refs(point: str) -> List[int]:
    from call1.summarizer import _extract_turn_refs

    return _extract_turn_refs(point)


class RealSummaryAssembly(Handler):
    """The final summary with the pre-split whole-call citation check (replaces the code stage's
    turn-existence check in real mode)."""

    job_type = JobType.SUMMARY_ASSEMBLY
    adapter_id = "call1.summarizer.assembly"
    adapter_version = ADAPTER_VERSION

    def run(self, job: HandlerJob) -> HandlerResult:
        from call1.summarizer import _validate_key_points_grounded

        transcript = job.transcript()
        segments: List[SummarySegmentContent] = [item.content() for _, item in sorted(  # type: ignore[misc]
            job.inputs_with_prefix("segment:").items(), key=lambda kv: _part_index(kv[0]))]
        if not segments:
            raise HandlerError(JobErrorCode.CONFIGURATION_ERROR, "no summary segments were planned for this call")
        synthesis_input = job.input("synthesis")
        synthesis: Optional[SummarySynthesisContent] = synthesis_input.content() if synthesis_input else None  # type: ignore[assignment]
        if len(segments) > 1 and synthesis is None:
            raise HandlerError(JobErrorCode.CONFIGURATION_ERROR, "a multi-segment summary needs its final synthesis")
        extra = job.parameters.extra
        route_class = str(extra.get("route_class") or "appliance")
        # The planner records the summary route's masking flag (graphs planned before it did: the
        # contract default, masked off the appliance route only).
        masked = bool(extra["masked"]) if "masked" in extra else route_class != "appliance"
        texts = masked_texts_for(job, transcript) if masked else None
        lines = numbered_lines(transcript.turns, texts)
        source = synthesis if synthesis is not None and len(segments) > 1 else segments[0]
        key_points = _validate_key_points_grounded(list(source.key_points), lines)
        from_segments = False
        if not key_points and len(segments) > 1:
            # Synthesis may lose citation IDs: keep already checked source points across the call,
            # including the last, without manufacturing citations for synthesized claims.
            unique = list(dict.fromkeys(p for seg in segments for p in seg.key_points))
            count = min(6, len(unique))
            chosen = [unique[round(i * (len(unique) - 1) / (count - 1))] for i in range(count)] if count > 1 else unique
            key_points = _validate_key_points_grounded(chosen, lines)
            from_segments = bool(key_points)
        if not key_points:
            raise HandlerError(JobErrorCode.VALIDATION_REJECTED, NO_KEY_POINTS)
        content = SummaryContent(
            narrative=source.narrative, key_points=key_points, rubric_highlights=[],
            citations=citations_for(key_points, {t.turn_id for t in transcript.turns}),
            grounding={"redacted_input": masked, "sanitized": masked, "model_narrative_only": True, "rubric_judgments_deterministic": True,
                       "full_call_coverage": True, "chunked": len(segments) > 1, "key_points_citations_checked": True,
                       "key_points_from_source_segments": from_segments},
            route_class=route_class, catalog_entry_id=str(extra.get("summary_entry") or "unknown"),
            generated_at=datetime.now(timezone.utc), segments=len(segments),
        )
        return HandlerResult(outputs={"summary": Output(content)})


__all__ = ["RealSummaryAssembly", "RealSummarySegment", "RealSummarySynthesis", "numbered_lines"]
