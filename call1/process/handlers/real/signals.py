"""Contact-signal passes: the lifecycle pass (intent, issue, friction) and the resolution pass
(fix proposed, agent completed, caller confirmed, still unresolved, deferred), one job each.

Each wraps one pass of ``call1.pipeline.contact_signals.extract_contact_signals``: the same
numbered ``[Turn N] Speaker: text`` block, system prompt, response schema and 512-token budget, and
the same strict provenance (``parse_pass_observations``): an observation survives only when its
quote is an exact substring of the cited turn and that turn's speaker is the signal kind's target
speaker. There is no heuristic fallback. A pass whose answer is not the observations JSON fails
(``validation_rejected``) and the merge publishes a partial result naming it, as the pre-split
pipeline marked a failed pass partial. Generation goes through ``generate_text`` (see ``llm.py``);
on a non-MLX host the included model is the loopback Ollama rather than no extraction at all.

Signal IDs are deterministic (kind, turn, character offset), so a rerun of the same answer produces
the same content; the merge prefixes them with the pass kind.
"""

from __future__ import annotations

from typing import List

from call1.contracts.contents import ContactSignalKind, ContactSignalPass, ContactSignalsPassContent, ContactSignalView, SpeakerRole
from call1.contracts.errors import JobErrorCode
from call1.contracts.jobs import JobType

from call1.process.handlers.base import Handler, HandlerError, HandlerJob, HandlerResult, Output
from call1.process.transcripts import turns_in

from .convert import legacy_turns
from .llm import LlmTransport, check_route, template_version
from .masking import enrichment, mask, route_masked, sensitive_values

ADAPTER_VERSION = "1"


def _signal(sig: dict) -> ContactSignalView:
    kind = ContactSignalKind(sig["kind"])
    return ContactSignalView(
        id=f"{kind.value}-t{sig['turn_id']}-c{sig['char_start']}"[:200], kind=kind, label=str(sig["label"])[:200], start=float(sig["start"]),
        end=float(sig["end"]), speaker=SpeakerRole(str(sig["speaker"]).upper()), quote=sig["quote"], turn_id=int(sig["turn_id"]),
        char_start=int(sig["char_start"]), char_end=int(sig["char_end"]), confidence=max(0.0, min(1.0, float(sig["confidence"]))),
    )


class _RealPass(Handler):
    pass_kind: ContactSignalPass
    adapter_version = ADAPTER_VERSION

    @property
    def adapter_id(self) -> str:  # type: ignore[override]
        return f"call1.contact_signals.{self.pass_kind.value}"

    def ready(self, job: HandlerJob) -> None:
        check_route(job)

    def run(self, job: HandlerJob) -> HandlerResult:
        from call1.pipeline.contact_signals import PASS_MAX_TOKENS, build_transcript_block, parse_pass_observations, pass_prompt

        transcript = job.transcript()
        window = job.parameters.window
        turns = legacy_turns(transcript, enrichment(job))
        if route_masked(job):
            values = sensitive_values(turns, job=job)
            turns = [t.model_copy(update={"text": mask(t.text, values), "raw_text": None, "word_timestamps": None}) for t in turns]
        if window is not None:
            wanted = {t.turn_id for t in turns_in(transcript, window)}
            turns = [t for t in turns if t.turn_id in wanted]
        name = self.pass_kind.value
        turns_by_id, block = build_transcript_block(turns)
        system, user, schema = pass_prompt(name, len(turns), block)
        version = template_version(system, schema, name)
        transport = LlmTransport(job)
        signals: List[ContactSignalView] = []
        if turns:
            try:
                answer = transport.generate(system, user, response_schema=schema, schema_name=f"contact_signals_{name}", max_tokens=PASS_MAX_TOKENS)
            except HandlerError as exc:
                exc.usage = transport.usage()
                raise
            try:
                found = list(parse_pass_observations(answer.raw, turns_by_id))
            except (ValueError, TypeError, AttributeError):
                raise HandlerError(JobErrorCode.VALIDATION_REJECTED, "the model's answer was not the observations JSON",
                                   usage=transport.usage()) from None
            seen = set()
            for sig in found:
                key = (sig["kind"], sig["turn_id"], sig["char_start"], sig["char_end"])
                if key in seen:
                    continue
                seen.add(key)
                signals.append(_signal(sig))
            signals.sort(key=lambda s: (s.start, s.id))
        content = ContactSignalsPassContent(pass_kind=self.pass_kind, window=window, signals=signals)
        return HandlerResult(outputs={"pass": Output(content),
                                      "prompt_input": Output(transport.prompt_input(f"call1.contact_signals.{name}", version, window=window))},
                             usage=transport.usage())


class RealLifecyclePass(_RealPass):
    job_type = JobType.CONTACT_SIGNALS_LIFECYCLE
    pass_kind = ContactSignalPass.LIFECYCLE


class RealResolutionPass(_RealPass):
    job_type = JobType.CONTACT_SIGNALS_RESOLUTION
    pass_kind = ContactSignalPass.RESOLUTION


__all__ = ["RealLifecyclePass", "RealResolutionPass"]
