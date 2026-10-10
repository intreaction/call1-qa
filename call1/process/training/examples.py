"""Training examples and held-out evaluation items, built with the engines' own prompt builders and
masking (docs/OnDeviceTraining.md sections 1 and 4.4).

Rules every example follows:

* **One prompt builder per engine, shared with the handler.** Stage 1 and stage 2 use
  ``signals_v2.render_batch`` over the rows ``stage1_rows``/``stage2_rows`` build inside
  ``SignalContext``; QA uses ``qa.qa_prompt``; speaker roles use ``media.roles_prompt``. A prompt
  change in an engine changes the training data too.
* **Masked text only.** The replayed job carries the pinned ``pii_findings`` of its transcript
  revision; a label without them is skipped (``no_pii_findings``). QA is always rebuilt masked,
  even when the call's own QA ran unmasked. Training never runs the PII model.
* **Answers** are ``json.dumps(answer, ensure_ascii=False, separators=ANSWER_SEPARATORS)`` with keys
  in schema order, and every answer must parse back to the intended label through the engine's own
  parser (``GemmaSegmentClassifier._scores``, ``RubricEvaluator._parse_semantic_answer``,
  ``speaker_roles.parse_roles``). One that does not is a builder bug (``BuilderBug``).
* **Budget.** An example's size is ``count(system) + count(user) + count(assistant) + 16``. mlx_lm
  truncates from the end, which would cut the answer, so nothing over ``max_seq_length`` is written:
  stage-1 chunks split 8 -> 4 -> 2 -> 1 rows on the fixed grid, then drop a single row's oldest
  ``earlier`` entries, then drop it (``too_long``); stage 2 splits 4 -> 2 -> 1 spans, then drops;
  QA and speaker prompts are used whole or dropped.

Held-out calls produce ``EvalItem``s instead, over the prompts production sends (all of a call's
stage-1 rows packed by ``pack_rows`` with the production budgets, the labelled stage-2 spans packed
the same way, ``qa_prompt``, ``roles_prompt``).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from call1.contracts.common import canonical_digest
from call1.contracts.contents import SIGNAL_NONE_OPTION, SIGNAL_NOT_OPTION, SIGNAL_OTHER_OPTION, VerdictStatus, short_digest, signal_span_key
from call1.contracts.reviews import OverrideReasonCode
from call1.contracts.signals import SignalTaxonomy, category_digest, subcategory_digest
from call1.contracts.training import EXCLUDED_QA_REASON_CODES, TrainingLabel, TrainingLabelKind

from .replay import Sources, has_findings, source_map

log = logging.getLogger("call1.process.training.examples")

ANSWER_SEPARATORS = (", ", ": ")
"""The spacing of every assistant message: ``json.dumps``'s defaults, pinned in one place (check L2
compares it with what the base model emits under the outlines constraint)."""
TEMPLATE_OVERHEAD_TOKENS = 16
STAGE1_CHUNK = 8
STAGE2_CHUNK = 4
QA_MAX_TOKENS = 768
ASSESSMENT_CHARS = 120
ROLES_ASSESSMENT_CHARS = 200
TASKS = ("signal_stage1", "signal_stage2", "qa_verdict", "speaker_roles")

SIGNAL_INPUT_ROLES = ("transcript", "speaker_attribution", "pii_findings", "taxonomy", "stage:categorize", "stage:subcategorize")
QA_INPUT_ROLES = ("transcript", "speaker_attribution", "rubric", "enrichment", "pii_findings")
SPEAKER_INPUT_ROLES = ("transcript", "pii_findings", "enrichment")
MODEL_ROLE_CONFIDENCES = (0.8, 0.6)
"""A mono call's clusters named by the model (0.8) or the first-speaker fallback (0.6)."""

VERDICT_ANSWER = {VerdictStatus.PASS: "pass", VerdictStatus.FAIL: "fail", VerdictStatus.NOT_APPLICABLE: "not_applicable",
                  VerdictStatus.FLAGGED: "needs_review"}
QA_TEMPLATES = {
    "pass": "The transcript establishes every applicable requirement of this criterion.",
    "fail": "The transcript establishes a violation of this criterion.",
    "not_applicable": "The transcript establishes the criterion's configured exception.",
    "needs_review": "The evidence is insufficient or ambiguous for this criterion.",
}
QA_NOT_IN_TRANSCRIPT = " The required behavior is not in the transcript."
ROLE_OF_SPEAKER = {"AGENT": "agent", "CALLER": "caller", "UNKNOWN": "neither"}


class BuilderBug(AssertionError):
    """An example whose answer does not parse back to its label through the engine's parser."""


def finding_texts(job) -> Set[str]:
    """The text of every kept finding in the job's pinned ``pii_findings`` (strong or not)."""
    from call1 import pii_model

    item = job.input("pii_findings")
    if item is None:
        return set()
    try:
        findings = item.content()
    except Exception:
        return set()
    out: Set[str] = set()
    for turn in getattr(findings, "turns", None) or []:
        for span in turn.spans:
            text = str(span.text or "").strip()
            if text and pii_model.keep_span(pii_model.PiiSpan(str(span.category), span.start, span.end, text)):
                out.add(text)
    return out


def mentions_any(text: str, values: Iterable[str]) -> bool:
    """Whether ``text`` contains any of ``values`` as a whole word or phrase, ignoring case."""
    import re

    lowered = text.casefold()
    for value in values:
        needle = value.casefold()
        if needle and re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", lowered):
            return True
    return False


def answer_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=ANSWER_SEPARATORS)


def cut_on_word(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:")


def split_of(installation_id: str, call_id: str) -> str:
    """``h = sha256(installation_id + ":" + call_id) % 100``: ``eval`` below 20, ``valid`` below 30,
    else ``train``. A call stays on its side forever."""
    h = int(hashlib.sha256(f"{installation_id}:{call_id}".encode("utf-8")).hexdigest(), 16) % 100
    return "eval" if h < 20 else "valid" if h < 30 else "train"


# --- what the builders produce -------------------------------------------------------------------


@dataclass
class Example:
    task: str
    call_id: str
    seqs: List[int]
    system: str
    user: str
    assistant: str
    tokens: int
    split: str = "train"

    @property
    def messages(self) -> List[Dict[str, str]]:
        return [{"role": "system", "content": self.system}, {"role": "user", "content": self.user},
                {"role": "assistant", "content": self.assistant}]

    @property
    def prompt_digest(self) -> str:
        return canonical_digest(self.messages[:2])

    @property
    def request_digest(self) -> str:
        """What a production attempt records as ``prompt_input.prompt_digest`` for these messages."""
        return canonical_digest([self.messages[:2]])

    @property
    def digest(self) -> str:
        return canonical_digest(self.messages)

    @property
    def newest(self) -> int:
        return max(self.seqs) if self.seqs else 0


@dataclass
class EvalPrompt:
    """One prompt the generation worker answers, once per adapter."""

    id: str
    task: str
    system: str
    user: str
    schema: Dict[str, Any]
    max_tokens: int
    draft: Callable[[], Any]
    """A neutral answer object for the scripted (fake) generator to edit."""

    def record(self) -> Dict[str, Any]:
        return {"id": self.id, "system": self.system, "user": self.user, "schema": self.schema, "max_tokens": self.max_tokens}


@dataclass
class EvalItem:
    """One held-out label: which prompts it reads, how its answer is judged, and how the scripted
    generator writes a right or wrong answer for it."""

    task: str
    call_id: str
    seq: int
    prompt_ids: List[str]
    judge: Callable[[Dict[str, Optional[str]]], Tuple[bool, bool]]
    """answers by prompt ID -> (correct, invalid)."""
    script: Callable[[Dict[str, Any], bool], None]
    """Edit the draft answers so this item comes out right (True) or wrong (False)."""


@dataclass
class BuildResult:
    examples: List[Example] = field(default_factory=list)
    prompts: Dict[str, EvalPrompt] = field(default_factory=dict)
    items: List[EvalItem] = field(default_factory=list)
    skipped: Counter = field(default_factory=Counter)
    notes: Counter = field(default_factory=Counter)

    def skip(self, reason: str, n: int = 1) -> None:
        self.skipped[reason] += n


# --- the builder ---------------------------------------------------------------------------------


class ExampleBuilder:
    """Builds every current label's examples (training and validation calls) or held-out items
    (evaluation calls). ``sources`` reads jobs and artifacts; ``taxonomy`` is the current published
    taxonomy (``getSignalTaxonomy``), which decides whether a signal label is outdated."""

    def __init__(self, sources: Sources, *, installation_id: str, count_tokens: Callable[[str], int], max_seq_length: int = 2600,
                 taxonomy: Optional[SignalTaxonomy] = None, prompt_budget: Optional[int] = None) -> None:
        from call1.pipeline.signals_v2 import GEMMA_PROMPT_BUDGET

        from ..handlers.real.signals_v2 import gemma_row_budget

        self.sources = sources
        self.installation_id = installation_id
        self.count = count_tokens
        self.max_seq_length = int(max_seq_length)
        self.taxonomy = taxonomy
        self.prompt_budget = prompt_budget or GEMMA_PROMPT_BUDGET
        self.row_budget = gemma_row_budget(count_tokens)
        self.result = BuildResult()

    # --- entry point -----------------------------------------------------------------------------

    def build(self, labels: Sequence[TrainingLabel], cancelled: Callable[[], bool] = lambda: False) -> BuildResult:
        from ..store_client import StoreError, StoreUnavailable

        qa = [label for label in labels if label.kind is TrainingLabelKind.QA_VERDICT]
        signal: Dict[Tuple[str, str], List[TrainingLabel]] = defaultdict(list)
        speaker: Dict[str, List[TrainingLabel]] = defaultdict(list)
        for label in labels:
            if not label.sources or not label.source_job_id:
                self.result.skip("source_unavailable")
                continue
            if label.kind is TrainingLabelKind.SIGNAL_HIT:
                signal[(label.call_id, label.source_job_id)].append(label)
            elif label.kind is TrainingLabelKind.SPEAKER_ROLE:
                speaker[label.call_id].append(label)
        qa = [label for label in qa if label.sources and label.source_job_id]
        work: List[Tuple[Callable[..., None], Any, int]] = []
        work += [(self._qa, label, 1) for label in qa]
        work += [(self._signal_group, group, len(group)) for group in signal.values()]
        work += [(self._speaker_call, group, len(group)) for group in speaker.values()]
        for fn, arg, n in work:
            if cancelled():
                break
            try:
                fn(arg)
            except BuilderBug:
                raise
            except StoreUnavailable:
                raise
            except StoreError as exc:
                if exc.code in ("insufficient_scope", "unauthorized", "forbidden"):
                    raise
                self.result.skip("source_unavailable", n)
            except _Skip as skip:
                self.result.skip(skip.reason, n)
        return self.result

    # --- shared ----------------------------------------------------------------------------------

    def split(self, call_id: str) -> str:
        return split_of(self.installation_id, call_id)

    def size(self, system: str, user: str, assistant: str) -> int:
        return self.count(system) + self.count(user) + self.count(assistant) + TEMPLATE_OVERHEAD_TOKENS

    def _example(self, task: str, call_id: str, seqs: Iterable[int], system: str, user: str, assistant: str) -> Optional[Example]:
        tokens = self.size(system, user, assistant)
        if tokens > self.max_seq_length:
            return None
        example = Example(task=task, call_id=call_id, seqs=sorted(set(seqs)), system=system, user=user, assistant=assistant, tokens=tokens,
                          split=self.split(call_id))
        self.result.examples.append(example)
        return example

    def _prompt(self, task: str, system: str, user: str, schema: Dict[str, Any], max_tokens: int, draft: Callable[[], Any]) -> EvalPrompt:
        pid = f"p{len(self.result.prompts) + 1:05d}"
        prompt = EvalPrompt(id=pid, task=task, system=system, user=user, schema=schema, max_tokens=max_tokens, draft=draft)
        self.result.prompts[pid] = prompt
        return prompt

    # --- Contact Signals (stages 1 and 2) --------------------------------------------------------

    def _signal_group(self, labels: List[TrainingLabel]) -> None:
        from call1.pipeline.signals_v2 import fire, span_text, stage1_rows, stage2_rows

        from ..handlers.signal_stages import SignalContext, _same_grid

        first = labels[0]
        job = self.sources.job(str(first.source_job_id))
        hjob = self.sources.handler_job(job, first.sources, SIGNAL_INPUT_ROLES)
        if hjob.input("stage:categorize") is None or hjob.input("taxonomy") is None:
            raise _Skip("source_unavailable")
        if not has_findings(hjob):
            raise _Skip("no_pii_findings")
        ctx = SignalContext(hjob)
        categories = hjob.require("stage:categorize").content()
        if not _same_grid(categories, ctx.segmentation.segments):  # type: ignore[arg-type]
            raise _Skip("grid_mismatch")
        sub_item = hjob.input("stage:subcategorize")
        decisions = {d.span_key: d for d in sub_item.content().decisions} if sub_item is not None else {}  # type: ignore[attr-defined]
        by_key = {s.span_key: s for s in categories.spans}  # type: ignore[attr-defined]
        segments = list(categories.segments)  # type: ignore[attr-defined]
        call_id = first.call_id

        judged: List[_HitJudgement] = []
        for label in sorted(labels, key=lambda l: l.seq):
            try:
                judged.append(self._judge_hit(label, ctx.taxonomy, by_key, segments))
            except _Skip as skip:
                self.result.skip(skip.reason)
        if not judged:
            return

        plan = stage1_rows(ctx.segmentation.segments, ctx.taxonomy, budget=self.row_budget, mask=ctx.mask)
        rows = plan.rows
        split = self.split(call_id)

        # stage 2 rows: every labelled span, in call order
        position = {s.index: i for i, s in enumerate(segments)}
        spans: List[Tuple[int, Any, _HitJudgement]] = []
        for hit in judged:
            for ref in hit.span_refs:
                item = span_text(ref, segments, ctx.masked_turns)
                if item is None:
                    continue
                first_seg = next((s for s in segments if s.turn_id == ref.turn_id and s.window == ref.first_window), None)
                spans.append((position.get(first_seg.index, 0) if first_seg is not None else 0, item, hit))
        spans.sort(key=lambda t: t[0])
        rows2 = stage2_rows([item for _, item, _ in spans], ctx.taxonomy, segments, ctx.masked_turns, budget=self.row_budget, mask=ctx.mask)
        hit_of_row = {item.span.span_key: hit for _, item, hit in spans}

        if split == "eval":
            self._signal_eval(call_id, rows, rows2, judged, hit_of_row, decisions)
            return

        # --- stage 1 targets: the artifact's own picks, then the dismissals
        scores = {s.index: s.probabilities for s in categories.scores}  # type: ignore[attr-defined]
        targets: Dict[str, List[str]] = {}
        for row in rows:
            options = [o for o, _ in row.options]
            picks = [c for c in fire(scores.get(int(row.key), {}), categories.thresholds) if c in options]  # type: ignore[attr-defined]
            targets[row.key] = picks
        labelled: Dict[str, Set[int]] = defaultdict(set)
        for hit in judged:
            for key in hit.row_keys:
                labelled[key].add(hit.label.seq)
            if hit.dismissed:
                for key in hit.row_keys:
                    if key in targets:
                        targets[key] = [c for c in targets[key] if c != hit.category_id]
        for start in range(0, len(rows), STAGE1_CHUNK):
            chunk = rows[start:start + STAGE1_CHUNK]
            if any(row.key in labelled for row in chunk):
                self._stage1_fit(call_id, chunk, targets, labelled)

        # --- stage 2 targets
        targets2: Dict[str, Tuple[str, str, str]] = {}
        seqs2: Dict[str, int] = {}
        for row in rows2:
            hit = hit_of_row.get(row.key)
            if hit is None:
                continue
            target = self._stage2_target(row, hit, decisions.get(row.key))
            if target is None:
                self.result.skip("not_in_options")
                continue
            targets2[row.key] = target
            seqs2[row.key] = hit.label.seq
        kept = [row for row in rows2 if row.key in targets2]
        for start in range(0, len(kept), STAGE2_CHUNK):
            self._stage2_fit(call_id, kept[start:start + STAGE2_CHUNK], targets2, seqs2)

    def _judge_hit(self, label: TrainingLabel, snapshot: SignalTaxonomy, by_key, segments) -> "_HitJudgement":
        sig = label.signal
        assert sig is not None
        current = self.taxonomy.category(sig.category_id) if self.taxonomy is not None else None
        parts = sig.hit_id.split(".")
        embedded = parts[1] if len(parts) >= 3 else ""
        if current is None or not current.active or short_digest(category_digest(current)) != embedded:
            raise _Skip("outdated")
        category = snapshot.category(sig.category_id)
        if category is None:
            raise _Skip("outdated")
        sub_verdict = sig.subcategory_verdict
        if sub_verdict is not None and sig.subcategory_digest and sig.subcategory_id not in (None, SIGNAL_OTHER_OPTION):
            node = next((s for s in category.subcategories if s.subcategory_id == sig.subcategory_id), None)
            if node is None or short_digest(subcategory_digest(node)) != sig.subcategory_digest:
                sub_verdict = None  # it judged an earlier subcategory (ContactSignalsV2 section 6.3)
                if sig.category_verdict is None:
                    raise _Skip("stale_subcategory")
        refs = []
        row_keys: List[str] = []
        for at in sig.spans:
            ref = by_key.get(signal_span_key(sig.category_id, at.turn_id, at.block))
            if ref is None:
                continue
            refs.append(ref)
            row_keys += [str(s.index) for s in segments
                         if s.turn_id == ref.turn_id and s.block == ref.block and ref.first_window <= s.window <= ref.last_window]
        if not refs:
            raise _Skip("grid_mismatch")
        choice = None
        if sub_verdict == "confirmed":
            choice = sig.subcategory_id
        elif sub_verdict == "corrected":
            choice = sig.corrected_subcategory_id
        return _HitJudgement(label=label, category_id=sig.category_id, category_name=category.name, dismissed=sig.category_verdict == "dismissed",
                             choice=choice, span_refs=refs, row_keys=row_keys)

    def _stage1_answer(self, chunk, targets: Dict[str, List[str]]) -> str:
        return answer_json({row.key: {"labels": list(targets.get(row.key) or []) or [SIGNAL_NONE_OPTION]} for row in chunk})

    def _stage1_fit(self, call_id: str, chunk, targets, labelled) -> None:
        from ..handlers.real.signals_v2 import render_batch

        if not any(row.key in labelled for row in chunk):
            return
        system, user, _schema, _max = render_batch(chunk, False)
        answer = self._stage1_answer(chunk, targets)
        if self.size(system, user, answer) <= self.max_seq_length:
            self._check_stage1(chunk, answer, targets)
            seqs = {s for row in chunk for s in labelled.get(row.key, ())}
            self._example("signal_stage1", call_id, seqs, system, user, answer)
            return
        if len(chunk) > 1:
            half = (len(chunk) + 1) // 2
            self._stage1_fit(call_id, chunk[:half], targets, labelled)
            self._stage1_fit(call_id, chunk[half:], targets, labelled)
            return
        row = chunk[0]
        previous = list(row.state.get("previous") or [])
        while previous:
            previous = previous[1:]  # the oldest earlier entry first
            trimmed = replace(row, state={**dict(row.state), "previous": previous})
            system, user, _schema, _max = render_batch([trimmed], False)
            if self.size(system, user, answer) <= self.max_seq_length:
                self._check_stage1([trimmed], answer, targets)
                self._example("signal_stage1", call_id, labelled.get(row.key, ()), system, user, answer)
                return
        self.result.skip("too_long")

    @staticmethod
    def _check_stage1(chunk, answer: str, targets) -> None:
        from ..handlers.real.signals_v2 import GemmaSegmentClassifier

        parsed = json.loads(answer)
        for row in chunk:
            scores = GemmaSegmentClassifier._scores(row, parsed[row.key], False)
            picked = [c for c, p in sorted(scores.items(), key=lambda kv: -kv[1]) if c != SIGNAL_NONE_OPTION and p >= 0.5]
            if picked != list(targets.get(row.key) or []):
                raise BuilderBug(f"stage-1 answer for row {row.key} parses to {picked}, not {targets.get(row.key)}")

    def _stage2_target(self, row, hit: "_HitJudgement", decision) -> Optional[Tuple[str, str, str]]:
        """(assessment, fits, choice) per section 1.3, or None when the choice is not offered."""
        options = dict(row.options)
        speaker = row.state.get("speaker") or "speaker"
        if hit.dismissed:
            name = row.question  # "Which kind of <name> is this?" / "Is this <name>?": use the offered "Not <name>" text
            not_text = options.get(SIGNAL_NOT_OPTION, "")
            name = not_text[4:] if not_text.startswith("Not ") else name
            return cut_on_word(f"The {speaker}'s own words do not show {name}.", ASSESSMENT_CHARS), "no", SIGNAL_OTHER_OPTION
        choice = hit.choice
        if choice is None:
            if decision is not None and decision.decision == "subcategory" and decision.subcategory_id in options:
                choice = decision.subcategory_id
            else:
                choice = SIGNAL_OTHER_OPTION
        if choice not in options or choice == SIGNAL_NOT_OPTION:
            return None
        return cut_on_word(f"The {speaker}'s own words show {options[choice]}.", ASSESSMENT_CHARS), "yes", choice

    def _stage2_fit(self, call_id: str, chunk, targets2, seqs2) -> None:
        from ..handlers.real.signals_v2 import render_batch

        if not chunk:
            return
        system, user, _schema, _max = render_batch(chunk, True)
        answer = answer_json({row.key: _stage2_answer(row.key, *targets2[row.key])
                              for row in chunk})
        if self.size(system, user, answer) <= self.max_seq_length:
            self._check_stage2(chunk, answer, targets2)
            self._example("signal_stage2", call_id, [seqs2[row.key] for row in chunk], system, user, answer)
            return
        if len(chunk) > 1:
            half = (len(chunk) + 1) // 2
            self._stage2_fit(call_id, chunk[:half], targets2, seqs2)
            self._stage2_fit(call_id, chunk[half:], targets2, seqs2)
            return
        self.result.skip("too_long")

    @staticmethod
    def _check_stage2(chunk, answer: str, targets2) -> None:
        from ..handlers.real.signals_v2 import GemmaSegmentClassifier

        parsed = json.loads(answer)
        for row in chunk:
            scores = GemmaSegmentClassifier._scores(row, parsed[row.key], True)
            chosen = max(scores.items(), key=lambda kv: kv[1])[0]
            _, fits, choice = targets2[row.key]
            want = SIGNAL_NOT_OPTION if fits == "no" else choice
            if chosen != want:
                raise BuilderBug(f"stage-2 answer for span {row.key} parses to {chosen}, not {want}")

    def _signal_eval(self, call_id: str, rows, rows2, judged: List["_HitJudgement"], hit_of_row, decisions) -> None:
        from ..handlers.real.signals_v2 import pack_rows, render_batch

        # stage 1: every row of the call, packed as production packs it
        row_prompt: Dict[str, str] = {}
        row_of: Dict[str, Any] = {}
        for batch in pack_rows(rows, False, self.count, self.prompt_budget):
            system, user, schema, max_tokens = render_batch(batch, False)
            keys = [row.key for row in batch]
            prompt = self._prompt("signal_stage1", system, user, schema, max_tokens,
                                  lambda keys=keys: {k: {"labels": [SIGNAL_NONE_OPTION]} for k in keys})
            for row in batch:
                row_prompt[row.key] = prompt.id
                row_of[row.key] = row
        for hit in judged:
            keys = [k for k in hit.row_keys if k in row_prompt]
            if not keys:
                continue
            pids = sorted({row_prompt[k] for k in keys})
            self.result.items.append(EvalItem(task="signal_stage1", call_id=call_id, seq=hit.label.seq, prompt_ids=pids,
                                              judge=_stage1_judge(hit.category_id, hit.dismissed, keys, row_prompt, row_of),
                                              script=_stage1_script(hit.category_id, hit.dismissed, keys, row_prompt, row_of)))

        # stage 2: the labelled spans, packed as production packs them
        labelled = [row for row in rows2 if row.key in hit_of_row]
        for batch in pack_rows(labelled, True, self.count, self.prompt_budget):
            system, user, schema, max_tokens = render_batch(batch, True)
            keys = [row.key for row in batch]
            prompt = self._prompt("signal_stage2", system, user, schema, max_tokens,
                                  lambda keys=keys: {k: _stage2_answer(k, "The span.", "yes", SIGNAL_OTHER_OPTION) for k in keys})
            for row in batch:
                hit = hit_of_row[row.key]
                want: Optional[str]
                if hit.dismissed:
                    want = SIGNAL_NOT_OPTION
                else:
                    want = hit.choice if hit.choice in dict(row.options) else None
                self.result.items.append(EvalItem(task="signal_stage2", call_id=call_id, seq=hit.label.seq, prompt_ids=[prompt.id],
                                                  judge=_stage2_judge(row, prompt.id, want), script=_stage2_script(row, prompt.id, want)))

    # --- QA verdict overrides --------------------------------------------------------------------

    def _qa(self, label: TrainingLabel) -> None:
        from call1.contracts.rubrics import CheckType
        from call1.pipeline.evaluator import verify_quoted_evidence
        from call1.qa_output import QA_SCHEMA

        from ..handlers.real.masking import mask
        from ..handlers.real.qa import qa_inputs, qa_prompt

        q = label.qa
        assert q is not None
        if q.reason_code in EXCLUDED_QA_REASON_CODES:
            raise _Skip("excluded_reason")
        src = source_map(label.sources)
        job = self.sources.job(str(label.source_job_id))
        hjob = self.sources.handler_job(job, label.sources, QA_INPUT_ROLES)
        if hjob.input("rubric") is None or hjob.input("transcript") is None:
            raise _Skip("source_unavailable")
        criterion = hjob.criterion()
        if criterion.check.check_type is not CheckType.SEMANTIC_JUDGEMENT:
            raise _Skip("not_model")
        prompt_input = self.sources.content(src["prompt_input"]) if "prompt_input" in src else None
        if prompt_input is not None and prompt_input.template_id.endswith(".sectioned"):
            raise _Skip("sectioned_qa")  # Multi-request evidence synthesis cannot be rebuilt as one training prompt.
        if prompt_input is not None and prompt_input.prompt_digest == canonical_digest([]):  # type: ignore[attr-defined]
            raise _Skip("not_model")  # a gate FLAG: the primary never called a model
        if not has_findings(hjob):
            raise _Skip("no_pii_findings")
        system, user, transcript, check = qa_prompt(hjob, masked=True,
                                                  compact=bool(prompt_input and prompt_input.template_id.endswith(".compact")))
        digest = canonical_digest([[{"role": "system", "content": system}, {"role": "user", "content": user}]])
        if prompt_input is not None and prompt_input.masked and digest != prompt_input.prompt_digest:  # type: ignore[attr-defined]
            self.result.notes["template_drift"] += 1
        verdict = VERDICT_ANSWER[q.status]
        speaker = check.speaker

        if self.split(label.call_id) == "eval":
            self._qa_eval(label, system, user, QA_SCHEMA, transcript, speaker, verdict)
            return

        _legacy, _check, _transcript, values = qa_inputs(hjob, masked=True)
        values = values or set()
        assessment_text = QA_TEMPLATES[verdict]
        if verdict == "fail" and q.reason_code is OverrideReasonCode.EVIDENCE_NOT_IN_TRANSCRIPT:
            assessment_text += QA_NOT_IN_TRANSCRIPT
        quote: Optional[str] = None
        reasoning = assessment_text
        escalation = self.sources.content(src["escalation_assessment"]) if "escalation_assessment" in src else None
        primary = self.sources.content(src["assessment"]) if "assessment" in src else None
        if escalation is not None and escalation.status is q.status and escalation.quoted_evidence:  # type: ignore[attr-defined]
            candidate = mask(escalation.quoted_evidence, values)  # type: ignore[attr-defined]
            if verify_quoted_evidence(candidate, transcript, speaker)[0]:
                quote = candidate
                # The quote verified against the masked transcript, but the escalation's own words did
                # not: on an unmasked route, or where a weak finding (masked by position only in its
                # turn) reappears in them, a name could survive the by-value mask. Those fall back to
                # the template assessment.
                written = cut_on_word(mask(escalation.reasoning, values).strip(), 600)  # type: ignore[attr-defined]
                unmasked_route = prompt_input is not None and not prompt_input.masked  # type: ignore[attr-defined]
                if written and not unmasked_route and not mentions_any(written, finding_texts(hjob)):
                    reasoning = written
                else:
                    self.result.notes["qa_reasoning_templated"] += 1
        if quote is None and verdict != "needs_review" and primary is not None and primary.quoted_evidence:  # type: ignore[attr-defined]
            candidate = mask(primary.quoted_evidence, values)  # type: ignore[attr-defined]
            if verify_quoted_evidence(candidate, transcript, speaker)[0]:
                quote = candidate
        if quote is None:
            if verdict != "needs_review":
                raise _Skip("no_evidence")
            quote = ""
        answer = answer_json({"assessment": reasoning, "verdict": verdict, "quote": quote})
        _check_qa(answer, verdict, quote, reasoning)
        if self._example("qa_verdict", label.call_id, [label.seq], system, user, answer) is None:
            self.result.skip("too_long")
            self.result.notes["qa_too_long"] += 1

    def _qa_eval(self, label: TrainingLabel, system: str, user: str, schema, transcript, speaker, verdict: str) -> None:
        from call1.pipeline.evaluator import verify_quoted_evidence

        permitted = [t.text for t in transcript.turns if speaker is None or t.speaker == speaker]
        good_quote = permitted[0] if permitted else ""
        prompt = self._prompt("qa_verdict", system, user, schema, QA_MAX_TOKENS,
                              lambda: {"assessment": QA_TEMPLATES["needs_review"], "verdict": "needs_review", "quote": ""})

        def judge(answers: Dict[str, Optional[str]]) -> Tuple[bool, bool]:
            from call1.pipeline.evaluator import RubricEvaluator

            raw = answers.get(prompt.id)
            parsed = RubricEvaluator._parse_semantic_answer(raw) if raw is not None else None
            if parsed is None:
                return False, True
            got, quote, _assessment = parsed
            if got != verdict:
                return False, False
            if got in ("pass", "fail", "not_applicable") and not verify_quoted_evidence(quote, transcript, speaker)[0]:
                return False, False  # production would FLAG it
            return True, False

        def script(drafts: Dict[str, Any], correct: bool) -> None:
            if correct:
                drafts[prompt.id] = {"assessment": QA_TEMPLATES[verdict], "verdict": verdict, "quote": "" if verdict == "needs_review" else good_quote}
            else:
                wrong = "needs_review" if verdict != "needs_review" else "pass"
                drafts[prompt.id] = {"assessment": QA_TEMPLATES[wrong], "verdict": wrong, "quote": ""}

        self.result.items.append(EvalItem(task="qa_verdict", call_id=label.call_id, seq=label.seq, prompt_ids=[prompt.id], judge=judge, script=script))

    # --- speaker corrections ---------------------------------------------------------------------

    def _speaker_call(self, labels: List[TrainingLabel]) -> None:
        from call1.pipeline.speaker_roles import ROLE_AGENT, ROLE_CALLER, parse_roles

        from ..handlers.real.media import roles_prompt

        labels = sorted(labels, key=lambda l: l.seq)
        live = []
        for label in labels:
            assert label.speaker is not None
            if label.speaker.apply_to_cluster and label.speaker.speaker_cluster:
                live.append(label)
            else:
                self.result.skip("turn_only")
        if not live:
            return
        base = None
        for label in live:
            job = self.sources.job(str(label.source_job_id))
            if job.parameters.speaker_correction is None:
                base = (label, job)
                break
        if base is None:
            raise _Skip("not_model")
        label0, job = base
        src = source_map(label0.sources)
        if "speaker_attribution" not in src:
            raise _Skip("source_unavailable")
        attribution = self.sources.content(src["speaker_attribution"])
        named = [a for a in attribution.assignments if a.speaker_cluster and a.confidence in MODEL_ROLE_CONFIDENCES]  # type: ignore[attr-defined]
        clusters = {a.turn_id: a.speaker_cluster for a in attribution.assignments}  # type: ignore[attr-defined]
        if not named or len({c for c in clusters.values() if c}) < 2:
            raise _Skip("not_model")  # a stereo call's roles come from the channel
        hjob = self.sources.handler_job(job, label0.sources, SPEAKER_INPUT_ROLES)
        if not has_findings(hjob):
            raise _Skip("no_pii_findings")
        transcript = hjob.require("transcript").content()
        system, user, cluster_of, schema = roles_prompt(hjob, transcript, clusters)  # type: ignore[arg-type]
        roles: Dict[str, str] = {}
        for a in named:
            roles.setdefault(a.speaker_cluster, ROLE_OF_SPEAKER.get(a.speaker.value, "neither"))
        for label in live:
            roles[label.speaker.speaker_cluster] = ROLE_OF_SPEAKER[label.speaker.speaker]  # type: ignore[union-attr,index]
        target = {cluster: roles.get(cluster, "neither") for cluster in cluster_of.values()}
        if ROLE_AGENT not in target.values() or ROLE_CALLER not in target.values():
            raise _Skip("invalid_roles")
        aliases = list(cluster_of)
        answer_obj = {"assessment": roles_assessment(aliases, cluster_of, target), "roles": {a: target[cluster_of[a]] for a in aliases}}
        answer = answer_json(answer_obj)
        if parse_roles(answer, cluster_of) != target:
            raise BuilderBug("the speaker-roles answer does not parse to its target")
        call_id = label0.call_id
        seqs = [label.seq for label in live]
        if self.split(call_id) == "eval":
            prompt = self._prompt("speaker_roles", system, user, schema, 220, lambda: dict(answer_obj))
            flipped = {c: {"agent": "caller", "caller": "agent"}.get(r, r) for c, r in target.items()}

            def judge(answers: Dict[str, Optional[str]], pid=prompt.id) -> Tuple[bool, bool]:
                raw = answers.get(pid)
                got = parse_roles(raw, cluster_of) if raw is not None else None
                if got is None:
                    return False, True
                return got == target, False

            def script(drafts: Dict[str, Any], correct: bool, pid=prompt.id) -> None:
                chosen = target if correct else flipped
                drafts[pid] = {"assessment": roles_assessment(aliases, cluster_of, chosen), "roles": {a: chosen[cluster_of[a]] for a in aliases}}

            self.result.items.append(EvalItem(task="speaker_roles", call_id=call_id, seq=max(seqs), prompt_ids=[prompt.id], judge=judge, script=script))
            return
        if self._example("speaker_roles", call_id, seqs, system, user, answer) is None:
            self.result.skip("too_long")


# --- helpers -------------------------------------------------------------------------------------


class _Skip(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class _HitJudgement:
    label: TrainingLabel
    category_id: str
    category_name: str
    dismissed: bool
    choice: Optional[str]
    span_refs: List[Any]
    row_keys: List[str]


def roles_assessment(aliases: Sequence[str], cluster_of: Dict[str, str], roles: Dict[str, str]) -> str:
    def names(role: str) -> List[str]:
        return [a for a in aliases if roles.get(cluster_of[a]) == role]

    def join(items: List[str]) -> str:
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]

    agents, callers, others = names("agent"), names("caller"), names("neither")
    text = f"{join(agents)} {'speaks' if len(agents) == 1 else 'speak'} for the contact centre; {join(callers)} {'is' if len(callers) == 1 else 'are'} the caller"
    if others:
        text += f"; {join(others)} {'is' if len(others) == 1 else 'are'} neither"
    return cut_on_word(text + ".", ROLES_ASSESSMENT_CHARS)


def _check_qa(answer: str, verdict: str, quote: str, assessment: str) -> None:
    from call1.pipeline.evaluator import RubricEvaluator

    if RubricEvaluator._parse_semantic_answer(answer) != (verdict, quote, assessment):
        raise BuilderBug("the QA answer does not parse back to its label")


def _stage1_picks(row, got) -> List[str]:
    from ..handlers.real.signals_v2 import GemmaSegmentClassifier

    scores = GemmaSegmentClassifier._scores(row, got, False)
    return [c for c, p in scores.items() if c != SIGNAL_NONE_OPTION and p >= 0.5]


def _parse(answers: Dict[str, Optional[str]], pid: str) -> Optional[Dict[str, Any]]:
    raw = answers.get(pid)
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _stage1_judge(category_id: str, dismissed: bool, keys: List[str], row_prompt: Dict[str, str], row_of: Dict[str, Any]):
    def judge(answers: Dict[str, Optional[str]]) -> Tuple[bool, bool]:
        fired = False
        for key in keys:
            parsed = _parse(answers, row_prompt[key])
            got = parsed.get(key) if parsed is not None else None
            if not isinstance(got, dict):
                return False, True
            try:
                picks = _stage1_picks(row_of[key], got)
            except Exception:
                return False, True
            fired = fired or category_id in picks
        return (not fired) if dismissed else fired, False
    return judge


def _stage1_script(category_id: str, dismissed: bool, keys: List[str], row_prompt: Dict[str, str], row_of: Dict[str, Any]):
    def script(drafts: Dict[str, Any], correct: bool) -> None:
        want_fire = (not dismissed) == correct
        for i, key in enumerate(keys):
            answer = drafts[row_prompt[key]].setdefault(key, {"labels": [SIGNAL_NONE_OPTION]})
            labels = [l for l in answer["labels"] if l not in (SIGNAL_NONE_OPTION, category_id)]
            if want_fire and i == 0 and category_id in dict(row_of[key].options):
                labels = [category_id] + labels[:1]
            answer["labels"] = labels or [SIGNAL_NONE_OPTION]
    return script


def _stage2_judge(row, pid: str, want: Optional[str]):
    def judge(answers: Dict[str, Optional[str]]) -> Tuple[bool, bool]:
        from ..handlers.real.signals_v2 import GemmaSegmentClassifier

        parsed = _parse(answers, pid)
        got = parsed.get(row.key) if parsed is not None else None
        if not isinstance(got, dict):
            return False, True
        try:
            scores = GemmaSegmentClassifier._scores(row, got, True)
        except Exception:
            return False, True
        chosen = max(scores.items(), key=lambda kv: kv[1])[0]
        if want == SIGNAL_NOT_OPTION:
            return chosen == SIGNAL_NOT_OPTION, False
        if chosen == SIGNAL_NOT_OPTION:
            return False, False
        return (want is None or chosen == want), False
    return judge


def _stage2_answer(key: str, assessment: str, fits: str, choice: str) -> Dict[str, Any]:
    answer = {}
    if key.startswith("intent."):
        answer["objective_status"] = "new_request" if fits == "yes" else "unclear"
        answer["speech_act"] = "request" if fits == "yes" else "unclear"
    answer.update(assessment=assessment, fits=fits, choice=choice)
    return answer


def _stage2_script(row, pid: str, want: Optional[str]):
    def script(drafts: Dict[str, Any], correct: bool) -> None:
        options = [o for o, _ in row.options if o != SIGNAL_NOT_OPTION]
        yes_choice = want if want not in (None, SIGNAL_NOT_OPTION) else SIGNAL_OTHER_OPTION
        if want == SIGNAL_NOT_OPTION:
            answer = {"assessment": "No.", "fits": "no", "choice": SIGNAL_OTHER_OPTION} if correct else \
                {"assessment": "Yes.", "fits": "yes", "choice": SIGNAL_OTHER_OPTION}
        elif correct:
            answer = {"assessment": "Yes.", "fits": "yes", "choice": yes_choice if yes_choice in options else SIGNAL_OTHER_OPTION}
        else:
            answer = {"assessment": "No.", "fits": "no", "choice": SIGNAL_OTHER_OPTION}
        drafts[pid][row.key] = _stage2_answer(row.key, answer["assessment"], answer["fits"], answer["choice"])
    return script


__all__ = ["ANSWER_SEPARATORS", "BuildResult", "BuilderBug", "EvalItem", "EvalPrompt", "Example", "ExampleBuilder", "TASKS", "answer_json",
           "cut_on_word", "roles_assessment", "split_of"]
