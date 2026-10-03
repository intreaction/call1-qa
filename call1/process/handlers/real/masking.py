"""Transcript masking for model inputs on a masked route (the frozen ``RouteRecord.masked``).

The values masked are the pre-split ones (``call1.redaction``): privacy-sensitive numeric entities
from the ``enrichment`` input (account and phone numbers) plus the PII patterns (SSN, card, phone,
account, PIN), collected once over the whole call so a value repeated without its entity is still
masked. It applies when a job's frozen selection says ``route.masked``. The planner sets that on
every route class in ``MaskingSettings.masked_route_classes`` and, on the appliance route, for the
text-model purposes (QA, summaries, contact signals) while Store's text masking is on
(``mask_reviewer_reads``, the legacy ``redaction.text``; on by default), as the pre-split app did.
The Process config's ``mask_model_text`` (``on``/``off``) overrides the appliance rule.

The model-based PII layer (team decision 19, ``call1.pii_model``) adds names, addresses, emails,
URLs and secrets that the number rules never catch: ``sensitive_values(..., job=job)`` takes the
union of the rule-based values with ``openai/privacy-filter``'s spans over the call's turns, minus
dates and the agent's own name (``agent_display_name`` plus agent self-introductions). The model runs
once per transcript revision, in the ``enrichment`` job, which writes the filtered spans as its
``pii_findings`` output (contract 1.2.0; ``pii_findings``); Store masks reviewer reads and mutes
audio with the same findings. Masked text-model jobs pin that output (input role
``pii_findings``) and use it when it was made from their own transcript input; otherwise (graphs
planned before the findings existed) they run the model here. Either way the model is loaded
inside the job and released after it, and its raw spans are kept per call and transcript text in a
small in-process cache (the agent-name filter applies per job), so other jobs of the same call,
including ``speaker_attribution`` before the roles are known, do not run it again. A real install without the weights
fails the job closed with ``model_unavailable``; fake-handler mode and CI use the labelled stub
(``CALL1_PII_MODEL_BACKEND=stub``).

Positional masking (team decision 22): ``sensitive_values`` returns a ``SensitiveValues`` set that
also carries per-turn spans masked *by position*, which ``mask`` applies to the turn they belong
to. They are the digits inside a card read-out window (``call1.redaction.read_out_digit_spans``,
so a CVV split into "seven" / "Two four" is masked without hiding every "seven" in the call) and
the PII findings that are not strong identifiers (``call1.pii_model.strong_identifier``); findings
made only of common words are dropped. Store masks reviewer reads the same way.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from call1.contracts.contents import EnrichmentContent, TranscriptContent, TurnEnrichment
from call1.contracts.errors import JobErrorCode

from call1.process.handlers.base import HandlerError, HandlerJob

log = logging.getLogger("call1.process.handlers.real.masking")


def route_masked(job: HandlerJob) -> bool:
    selection = job.selection
    return bool(selection is not None and selection.route.masked)


def enrichment(job: HandlerJob, compute: Optional[bool] = None) -> Optional[EnrichmentContent]:
    """The call's numeric entities: the pinned ``enrichment`` input when the job has one (a masked
    non-appliance route pins it). Otherwise, when the job masks (``compute``, default: the route is
    masked), the same deterministic extractor the ``enrichment`` stage runs, over the job's
    transcript, so appliance text masking needs no extra graph edge. None when neither applies."""
    item = job.input("enrichment")
    if item is not None:
        return item.content()  # type: ignore[return-value]
    if not (route_masked(job) if compute is None else compute):
        return None
    transcript = job.input("transcript")
    if transcript is None:
        return None
    return extract_enrichment(transcript.content())  # type: ignore[arg-type]


def extract_enrichment(transcript: TranscriptContent) -> EnrichmentContent:
    """``RealEnrichment``'s output for a transcript (``extract_numeric_references`` per turn)."""
    from call1.pipeline.numeric_extractor import extract_numeric_references

    from .convert import contract_entities

    return EnrichmentContent(turns=[TurnEnrichment(turn_id=t.turn_id, numeric_entities=contract_entities(
        extract_numeric_references(t.text, turn_start=t.start_time, turn_end=t.end_time))) for t in transcript.turns])


class SensitiveValues(set):
    """The values to mask by value (a plain ``set`` of strings, as before) plus ``positions``: the
    spans masked by position in the call's turns (``call1.redaction.TurnPositions``): the digits
    inside a card read-out window and the PII model's findings that are not strong identifiers
    (team decision 22). ``mask`` applies both; a turn text gets its own spans only."""

    def __init__(self, values: Iterable[str] = (), positions=None) -> None:
        from call1.redaction import TurnPositions

        super().__init__(values)
        self.positions = positions if positions is not None else TurnPositions.empty()


def sensitive_values(turns: Sequence, extra_texts: Iterable[str] = (), job: Optional[HandlerJob] = None) -> SensitiveValues:
    """The values to mask: the number rules over the turns and ``extra_texts``, plus (with a
    ``job``) the PII model's strong identifiers over the turns. The result also carries the
    positional spans of the turns (``SensitiveValues.positions``): read-out window digits, and
    (with a ``job``) every kept PII finding."""
    from call1.redaction import TurnPositions, extract_sensitive_values, read_out_digit_spans

    values = extract_sensitive_values([*turns, *({"text": text} for text in extra_texts if text)], True)
    texts = [str(_turn_field(t, "text") or "") for t in turns]
    spans = read_out_digit_spans(turns)
    if job is not None:
        found, unplaced = _positional_findings(model_findings(job, turns), texts)
        values |= unplaced
        for index, turn_spans in enumerate(found):
            spans[index] = spans[index] + turn_spans
    return SensitiveValues(values, TurnPositions(texts, spans))


def _positional_findings(findings: Sequence[Sequence], texts: Sequence[str]) -> Tuple[List[List[Tuple[int, int]]], Set[str]]:
    """(positional spans per turn, values to mask by value) for a call's kept findings: every
    finding that fits its turn text is masked there by position; strong identifiers
    (``pii_model.strong_identifier``) and any finding whose offsets do not fit its text are also
    masked by value."""
    from call1 import pii_model

    cased = pii_model.transcript_cased(texts)
    spans: List[List[Tuple[int, int]]] = [[] for _ in texts]
    values: Set[str] = set()
    for index, (text, turn_findings) in enumerate(zip(texts, findings)):
        for span in turn_findings:
            if not pii_model.keep_span(span):
                continue
            if 0 <= span.start < span.end <= len(text) and text[span.start:span.end] == span.text:
                spans[index].append((span.start, span.end))
            else:
                values.add(span.text)
            if pii_model.strong_identifier(span, cased):
                values.add(span.text)
    return spans, values


def _turn_field(turn, name: str):
    return turn.get(name) if isinstance(turn, dict) else getattr(turn, name, None)


def _is_agent(turn) -> bool:
    speaker = _turn_field(turn, "speaker")
    return str(getattr(speaker, "value", speaker) or "").upper() == "AGENT"


_CACHE_SIZE = 16
_cache: "OrderedDict[Tuple[str, str], List[list]]" = OrderedDict()
_cache_lock = threading.Lock()
_model_lock = threading.Lock()


def _backend(backend: Optional[str] = None) -> str:
    from call1 import pii_model

    try:
        return backend or pii_model.configured_backend()
    except pii_model.PiiConfigError as exc:
        raise HandlerError(JobErrorCode.MODEL_UNAVAILABLE, str(exc)) from None


def masked_spans(job: HandlerJob, turns: Sequence, *, backend: Optional[str] = None) -> List[list]:
    """The PII model's masked spans per turn (``pii_model.PiiSpan`` lists, aligned with ``turns``):
    masked categories only, never a span naming only the agent, never a span of only common words
    (``pii_model.filter_spans``). The model runs once per call and transcript text (the in-process
    cache holds its raw spans); the agent-name filter is applied per request, so jobs that see the
    same text with different speaker labels (``speaker_attribution`` before the roles are known,
    ``enrichment`` after) share one detection. Raises ``HandlerError`` (``model_unavailable``) when
    the configured model cannot run: masking never skips silently."""
    from call1 import pii_model

    texts = [str(_turn_field(t, "text") or "") for t in turns]
    if not any(texts):
        return [[] for _ in texts]
    metadata = job.call_metadata()
    agent_tokens = pii_model.agent_name_tokens(getattr(metadata, "agent_display_name", None),
                                               (text for turn, text in zip(turns, texts) if _is_agent(turn)))
    backend = _backend(backend)
    return pii_model.filter_spans(_raw_spans(job, texts, backend), agent_tokens)


def _raw_spans(job: HandlerJob, texts: List[str], backend: str) -> List[list]:
    """The detector's unfiltered spans for ``texts``, from the per-call cache or one model run."""
    digest = hashlib.sha256("\x1e".join([backend, "\x1f", *texts]).encode()).hexdigest()
    key = (job.job.conversation_id, digest)
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return [list(spans) for spans in _cache[key]]
    with _model_lock:  # one load at a time; a second job of the same call waits, then hits the cache
        with _cache_lock:
            if key in _cache:
                return [list(spans) for spans in _cache[key]]
        spans = _detect(job, texts, backend)
    with _cache_lock:
        _cache[key] = spans
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return [list(s) for s in spans]


def _pinned_findings(job: HandlerJob):
    """The job's pinned ``pii_findings`` content when it was made from the job's own
    ``transcript`` input (same checksum); None otherwise (no input, or another revision)."""
    item = job.input("pii_findings")
    transcript = job.input("transcript")
    if item is None or transcript is None:
        return None
    findings = item.content()
    if findings.transcript.checksum != transcript.artifact.checksum:  # type: ignore[attr-defined]
        job.log.warning("pii_findings input is for another transcript revision; running the PII model here")
        return None
    return findings


def findings_spans(job: HandlerJob, turns: Sequence) -> Optional[List[list]]:
    """The pinned ``pii_findings`` as ``pii_model.PiiSpan`` lists aligned with ``turns`` (matched
    by ``turn_id``, else by position), common-word findings dropped; None when there is no pinned
    input for the job's transcript revision."""
    from call1 import pii_model

    findings = _pinned_findings(job)
    if findings is None:
        return None
    by_id = {turn.turn_id: turn.spans for turn in findings.turns}  # type: ignore[attr-defined]
    out: List[list] = []
    for index, turn in enumerate(turns):
        turn_id = _turn_field(turn, "turn_id")
        rows = by_id.get(index if turn_id is None else turn_id, [])
        out.append([span for span in (pii_model.PiiSpan(str(r.category), r.start, r.end, r.text) for r in rows)
                    if pii_model.keep_span(span)])
    return out


def findings_values(job: HandlerJob) -> Optional[Set[str]]:
    """The values the job's pinned ``pii_findings`` input masks *by value* (its strong identifiers,
    ``pii_model.strong_identifier``), when it was made from the job's own ``transcript`` input;
    None otherwise (no input, or another revision). Its other findings are masked by position
    (``sensitive_values``)."""
    from call1 import pii_model

    findings = _pinned_findings(job)
    if findings is None:
        return None
    spans = [[pii_model.PiiSpan(str(s.category), s.start, s.end, s.text) for s in turn.spans] for turn in findings.turns]  # type: ignore[attr-defined]
    transcript = job.input("transcript").content()  # type: ignore[union-attr]
    cased = pii_model.transcript_cased(t.text for t in transcript.turns)
    return {span.text for turn_spans in spans for span in turn_spans if pii_model.strong_identifier(span, cased)}


def model_findings(job: HandlerJob, turns: Sequence, *, backend: Optional[str] = None) -> List[list]:
    """The PII model's kept findings per turn (decision 19): the pinned ``pii_findings`` input
    (made once per transcript revision by the ``enrichment`` job) when it matches the job's
    transcript, else the model run here through the in-process cache (graphs planned before the
    findings existed, or a job with no findings input)."""
    pinned = findings_spans(job, turns)
    if pinned is not None:
        return pinned
    return masked_spans(job, turns, backend=backend)


def model_values(job: HandlerJob, turns: Sequence, *, backend: Optional[str] = None) -> Set[str]:
    """The PII model's values to mask by value for a call's turns: its strong identifiers, plus
    any finding that does not fit its turn text (so it cannot be masked by position)."""
    texts = [str(_turn_field(t, "text") or "") for t in turns]
    return _positional_findings(model_findings(job, turns, backend=backend), texts)[1]


def _detect(job: HandlerJob, texts: List[str], backend: str) -> List[list]:
    """One model run over ``texts``: the raw spans (``masked_spans`` filters them per request)."""
    from call1 import pii_model
    from call1.pipeline.inference import inference_lock

    job.check_cancelled()
    started = time.monotonic()
    with inference_lock:
        try:
            detector = pii_model.detector(backend)
        except pii_model.PiiModelUnavailable as exc:
            raise HandlerError(JobErrorCode.MODEL_UNAVAILABLE, f"PII masking model unavailable: {exc}") from None
        try:
            spans = detector.detect(texts)
        finally:
            detector.release()
            del detector
    spans = [list(turn_spans) for turn_spans in spans]
    log.info("PII model (%s): %d spans over %d turns in %.2f s", backend, sum(len(k) for k in spans), len(texts),
             time.monotonic() - started)
    return spans


def pii_findings(job: HandlerJob, *, backend: Optional[str] = None):
    """The ``pii_findings`` output of the ``enrichment`` job: the masked spans of the job's
    transcript input (speaker labels from its ``speaker_attribution`` input when present, so agent
    self-introductions are recognized), bound to that transcript revision by reference."""
    from call1 import pii_model
    from call1.contracts.contents import PiiFindingsContent, PiiSpanContent, TurnPiiFindings

    transcript = job.transcript()
    backend = _backend(backend)
    spans = masked_spans(job, transcript.turns, backend=backend)
    turns = [TurnPiiFindings(turn_id=turn.turn_id, spans=[
        PiiSpanContent(start=s.start, end=s.end, category=s.label, text=s.text[:2000]) for s in turn_spans])
        for turn, turn_spans in zip(transcript.turns, spans)]
    stub = backend == pii_model.STUB_BACKEND
    return PiiFindingsContent(transcript=job.require("transcript").ref,
                              detector=pii_model.STUB_BACKEND if stub else pii_model.MODEL_REPOSITORY,
                              detector_revision=pii_model.STUB_REVISION if stub else pii_model.MODEL_REVISION, turns=turns)


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def mask(text: Optional[str], values: Set[str]) -> str:
    """``text`` masked with ``values``; with a ``SensitiveValues`` also by position (a turn text
    gets its own spans; any other text the spans of the turn it repeats)."""
    from call1.redaction import mask_text

    return mask_text(text, values, getattr(values, "positions", None))


def mask_turns(turns: List, values: Set[str]) -> List:
    """Copies of legacy turns with masked text (word timestamps dropped: they are raw)."""
    out = []
    for turn in turns:
        text = mask(turn.text, values)
        masked = bool(values) or bool(getattr(values, "positions", None))
        out.append(turn.model_copy(update={"text": text, "raw_text": text, "word_timestamps": None if masked else turn.word_timestamps}))
    return out


__all__ = ["SensitiveValues", "clear_cache", "enrichment", "extract_enrichment", "findings_spans", "findings_values", "mask", "mask_turns",
           "masked_spans", "model_findings", "model_values", "pii_findings", "route_masked", "sensitive_values"]
