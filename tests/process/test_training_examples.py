"""On-device training: the example builders (P4-P7), docs/OnDeviceTraining.md section 1.

Golden examples for every builder on the fake-script transcripts (``golden/training/*.jsonl``;
regenerate with ``CALL1_REGEN_GOLDEN=1``), masking, round trips through the engines' parsers, the
QA digest parity with production, supersession, dedup, exclusions, the split and budgeting. No MLX,
torch or real model runs."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from call1.contracts.contents import VerdictStatus
from call1.contracts.reviews import OverrideReasonCode
from call1.pipeline.signals_v2 import estimate_tokens, stage1_rows
from call1.process.handlers.real.signals_v2 import GemmaSegmentClassifier, gemma_row_budget, render_batch
from call1.process.handlers.signal_stages import SignalContext
from call1.process.training.dataset import assemble
from call1.process.training.examples import ANSWER_SEPARATORS, ExampleBuilder, split_of
from call1.process.training.labels import supersede
from call1.process.training.replay import Sources

from .test_signals_support import cancel_taxonomy, custom_category, with_categories
from .training_support import (
    INSTALLATION,
    SENSITIVE,
    FakeStore,
    add_hit_label,
    add_qa_call,
    add_signal_call,
    add_speaker_call,
    calls_in,
    qa_answer,
)

GOLDEN = Path(__file__).parent / "golden" / "training"


def build(store: FakeStore, tmp_path: Path, **kw):
    builder = ExampleBuilder(Sources(store, tmp_path / "inputs"), installation_id=INSTALLATION, count_tokens=estimate_tokens,
                             taxonomy=kw.pop("taxonomy", store.taxonomy), **kw)
    return builder.build(supersede(store.labels).current)


@pytest.fixture
def train_call():
    return calls_in("train", 1)[0]


# --- P4: goldens, masking, round trips, parity ---------------------------------------------------


def test_golden_examples_for_every_builder(tmp_path, train_call):
    store = FakeStore()
    add_signal_call(store, tmp_path, train_call, 0)
    add_qa_call(store, tmp_path, train_call, 0)
    add_speaker_call(store, tmp_path, train_call, 0)
    result = build(store, tmp_path)
    assert not result.skipped
    by_task = {}
    for example in result.examples:
        by_task.setdefault(example.task, []).append({"messages": example.messages, "label_seqs": example.seqs, "tokens": example.tokens})
    assert set(by_task) == {"signal_stage1", "signal_stage2", "qa_verdict", "speaker_roles"}
    GOLDEN.mkdir(parents=True, exist_ok=True)
    for task, rows in sorted(by_task.items()):
        path = GOLDEN / f"{task}.jsonl"
        text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
        if os.getenv("CALL1_REGEN_GOLDEN") == "1" or not path.exists():
            path.write_text(text, encoding="utf-8")
        assert text == path.read_text(encoding="utf-8"), f"{task} examples changed; regenerate with CALL1_REGEN_GOLDEN=1 if intended"


def test_every_message_is_masked(tmp_path, train_call):
    store = FakeStore()
    add_signal_call(store, tmp_path, train_call, 0)
    add_qa_call(store, tmp_path, train_call, 0)
    add_speaker_call(store, tmp_path, train_call, 0)
    result = build(store, tmp_path)
    assert result.examples
    for example in result.examples:
        text = json.dumps(example.messages, ensure_ascii=False)
        for value in SENSITIVE:
            assert value not in text, (example.task, value)
    redacted = {e.task for e in result.examples if "[REDACTED]" in e.user}
    assert {"signal_stage1", "qa_verdict", "speaker_roles"} <= redacted


def test_every_answer_round_trips_through_the_engines_parser(tmp_path, train_call):
    from call1.pipeline.evaluator import RubricEvaluator
    from call1.pipeline.speaker_roles import parse_roles

    store = FakeStore()
    add_signal_call(store, tmp_path, train_call, 0)
    add_qa_call(store, tmp_path, train_call, 0)
    add_speaker_call(store, tmp_path, train_call, 0)
    for example in build(store, tmp_path).examples:
        assert example.assistant == json.dumps(json.loads(example.assistant), ensure_ascii=False, separators=ANSWER_SEPARATORS)
        answer = json.loads(example.assistant)
        if example.task == "qa_verdict":
            assert RubricEvaluator._parse_semantic_answer(example.assistant) == (answer["verdict"], answer["quote"], answer["assessment"])
            assert list(answer) == ["assessment", "verdict", "quote"]
        elif example.task == "speaker_roles":
            speakers = json.loads(example.user)["speakers"]
            assert parse_roles(example.assistant, {s: s for s in speakers}) == answer["roles"]
            assert answer["roles"] == {"S1": "agent", "S2": "caller"}  # the reviewer's correction applied
            assert answer["assessment"] == "S1 speaks for the contact centre; S2 is the caller."
        elif example.task == "signal_stage2":
            for got in answer.values():
                assert list(got) == ["assessment", "fits", "choice"]


def test_stage1_and_stage2_user_messages_are_the_engines_render_batch(tmp_path, train_call):
    store = FakeStore()
    call = add_signal_call(store, tmp_path, train_call, 0)
    result = build(store, tmp_path)
    ctx = SignalContext(call.run.jobs["merge"])
    rows = stage1_rows(ctx.segmentation.segments, ctx.taxonomy, budget=gemma_row_budget(estimate_tokens), mask=ctx.mask).rows
    stage1 = [e for e in result.examples if e.task == "signal_stage1"]
    system, user, _schema, _max = render_batch(rows[:8], False)
    assert stage1 and (stage1[0].system, stage1[0].user) == (system, user)
    answer = json.loads(stage1[0].assistant)
    assert list(answer) == [row.key for row in rows[:8]]
    # the dismissed hit is the stage-1 negative; the confirmed one keeps its pick
    dismissed = {h.category_id for h, v in zip(call.run.result.signals, ["confirmed", "dismissed"] * 10) if v == "dismissed"}
    confirmed = {h.category_id for h, v in zip(call.run.result.signals, ["confirmed", "dismissed"] * 10) if v == "confirmed"}
    picked = {label for row in answer.values() for label in row["labels"]}
    assert not (dismissed & picked) and confirmed <= picked
    for row in rows[:8]:  # and every row parses with the engine's own parser
        GemmaSegmentClassifier._scores(row, answer[row.key], False)
    stage2 = [e for e in result.examples if e.task == "signal_stage2"]
    assert stage2 and all(e.system.startswith("You check labelled spans") for e in stage2)
    spans = {key: value for e in stage2 for key, value in json.loads(e.assistant).items()}
    for hit, verdict in zip(call.run.result.signals, ["confirmed", "dismissed"] * 10):
        key = f"{hit.category_id}.t{hit.turn_id}b{hit.span.block}"
        assert spans[key]["fits"] == ("no" if verdict == "dismissed" else "yes")


def test_the_qa_example_digest_equals_the_primarys_prompt_digest(tmp_path, train_call):
    store = FakeStore()
    _label, result, prompts = add_qa_call(store, tmp_path, train_call, 0)
    built = build(store, tmp_path)
    example = next(e for e in built.examples if e.task == "qa_verdict")
    assert example.request_digest == result.outputs["prompt_input"].content.prompt_digest
    assert [example.system, example.user] == list(prompts[0])
    assert not built.notes.get("template_drift")
    answer = json.loads(example.assistant)
    assert answer == {"assessment": "The transcript establishes a violation of this criterion.", "verdict": "fail",
                      "quote": "This call may be recorded for quality assurance."}  # path 2: the primary's quote verifies


def test_qa_evidence_paths(tmp_path):
    calls = calls_in("train", 4)
    store = FakeStore()
    # needs_review: no quote
    add_qa_call(store, tmp_path, calls[0], 0, status=VerdictStatus.FLAGGED)
    # the primary's quote does not verify for the agent (it quotes the caller): skipped, never invented
    add_qa_call(store, tmp_path, calls[1], 1, reply=qa_answer("pass", "Hi, my name is"))
    # evidence not in transcript: the fail template says so
    add_qa_call(store, tmp_path, calls[2], 2, reason_code=OverrideReasonCode.EVIDENCE_NOT_IN_TRANSCRIPT)
    result = build(store, tmp_path)
    answers = {e.call_id: json.loads(e.assistant) for e in result.examples}
    assert answers[calls[0]] == {"assessment": "The evidence is insufficient or ambiguous for this criterion.", "verdict": "needs_review", "quote": ""}
    assert calls[1] not in answers and result.skipped["no_evidence"] == 1
    assert answers[calls[2]]["assessment"].endswith(" The required behavior is not in the transcript.")


def test_an_agreeing_escalation_supplies_the_reasoning_and_quote(tmp_path, train_call):
    from .test_signals_support import findings_for, script_transcript
    from .training_support import qa_primary, script_for

    transcript = script_transcript(script_for(0))
    _job, primary, _prompts = qa_primary(tmp_path, transcript, findings_for(tmp_path, transcript))
    escalation = primary.outputs["assessment"].content.model_copy(update={
        "assessment_kind": "escalation", "status": VerdictStatus.FAIL, "quoted_evidence": "How can I help you today?",
        "reasoning": "Escalation: the disclosure came after my name is Maria Lopez was given."})
    store = FakeStore()
    add_qa_call(store, tmp_path, train_call, 0, escalation=escalation)
    answer = json.loads(next(e for e in build(store, tmp_path).examples if e.task == "qa_verdict").assistant)
    assert answer["quote"] == "How can I help you today?"
    assert answer["assessment"].startswith("Escalation:") and "Maria" not in answer["assessment"] and "[REDACTED]" in answer["assessment"]


def test_escalation_reasoning_that_names_a_weak_finding_falls_back_to_the_template(tmp_path, train_call, monkeypatch):
    """A lowercase one-word name is a weak finding: masked by position in its own turn only. When the
    escalation writes it into its reasoning, the by-value mask misses it, so the template is used."""
    from call1.contracts.contents import PiiSpanContent

    from . import training_support
    from .test_signals_support import findings_for, script_transcript
    from .training_support import qa_primary, script_for

    def with_weak_name(work, transcript, attribution=None):
        findings = findings_for(work, transcript, attribution)
        turns = []
        for turn in findings.turns:
            text = transcript.turns[turn.turn_id].text
            spans = list(turn.spans)
            if "amber" in text:
                start = text.index("amber")
                spans.append(PiiSpanContent(start=start, end=start + 5, category="private_person", text="amber"))
            turns.append(turn.model_copy(update={"spans": spans}))
        return findings.model_copy(update={"turns": turns})

    monkeypatch.setattr(training_support, "findings_for", with_weak_name)
    transcript = script_transcript(script_for(0))
    _job, primary, _prompts = qa_primary(tmp_path, transcript, with_weak_name(tmp_path, transcript))
    escalation = primary.outputs["assessment"].content.model_copy(update={
        "assessment_kind": "escalation", "status": VerdictStatus.FAIL, "quoted_evidence": "How can I help you today?",
        "reasoning": "Escalation: the agent greeted amber before the disclosure."})
    store = FakeStore()
    add_qa_call(store, tmp_path, train_call, 0, escalation=escalation)
    result = build(store, tmp_path)
    example = next(e for e in result.examples if e.task == "qa_verdict")
    answer = json.loads(example.assistant)
    assert answer["quote"] == "How can I help you today?"
    assert answer["assessment"] == "The transcript establishes a violation of this criterion."
    assert "amber" not in example.assistant.lower() and result.notes["qa_reasoning_templated"] == 1


# --- P5: supersession, withdrawal, dedup, exclusions, outdated ------------------------------------


def test_the_newest_label_for_a_subject_wins_and_a_withdrawal_removes_it(tmp_path, train_call):
    store = FakeStore()
    call = add_signal_call(store, tmp_path, train_call, 0, verdicts=None)
    first, second = call.run.result.signals[:2]
    add_hit_label(store, call, first, category_verdict="dismissed")
    add_hit_label(store, call, first, category_verdict="confirmed")  # supersedes
    add_hit_label(store, call, second, category_verdict="dismissed")
    add_hit_label(store, call, second)  # cleared: withdrawn
    collected = supersede(store.labels)
    assert [label.seq for label in collected.current] == [2]
    assert collected.counts["withdrawn"] == 1 and collected.counts["signal_hit"] == 1
    result = build(store, tmp_path)
    spans = {k: v for e in result.examples if e.task == "signal_stage2" for k, v in json.loads(e.assistant).items()}
    assert set(spans) == {f"{first.category_id}.t{first.turn_id}b{first.span.block}"} and spans[next(iter(spans))]["fits"] == "yes"


def test_duplicate_prompts_collapse_and_the_newest_answer_wins(tmp_path):
    store = FakeStore()
    calls = calls_in("train", 2)
    a = add_signal_call(store, tmp_path, calls[0], 3, verdicts=None)
    b = add_signal_call(store, tmp_path, calls[1], 3, verdicts=None)  # the same words: the same prompts
    add_hit_label(store, a, a.run.result.signals[-1], category_verdict="confirmed")
    add_hit_label(store, b, b.run.result.signals[-1], category_verdict="dismissed")
    built = build(store, tmp_path)
    data = assemble(built, max_train_examples=100)
    assert data.duplicates >= 1
    prompts = [e.prompt_digest for e in data.train + data.valid]
    assert len(prompts) == len(set(prompts))
    last = b.run.result.signals[-1]
    key = f"{last.category_id}.t{last.turn_id}b{last.span.block}"
    stage2 = [json.loads(e.assistant)[key] for e in data.train + data.valid if e.task == "signal_stage2" and key in json.loads(e.assistant)]
    assert stage2 == [stage2[0]] and stage2[0]["fits"] == "no"  # the newer dismissal


@pytest.mark.parametrize("reason", [OverrideReasonCode.TRANSCRIPTION_ERROR, OverrideReasonCode.SPEAKER_MISATTRIBUTED, OverrideReasonCode.POLICY_EXCEPTION])
def test_excluded_qa_reasons_train_nothing(tmp_path, train_call, reason):
    store = FakeStore()
    add_qa_call(store, tmp_path, train_call, 0, reason_code=reason)
    result = build(store, tmp_path)
    assert not result.examples and result.skipped == {"excluded_reason": 1}


def test_outdated_and_inactive_categories_are_dropped(tmp_path, train_call):
    store = FakeStore()
    call = add_signal_call(store, tmp_path, train_call, 0, verdicts=None)
    hit = call.run.result.signals[0]
    parts = hit.id.split(".")
    older = hit.model_copy(update={"id": ".".join([parts[0], "0" * 12] + parts[2:])})  # judged under an earlier gloss or speaker
    add_hit_label(store, call, older, category_verdict="confirmed")
    add_hit_label(store, call, call.run.result.signals[1], category_verdict="confirmed")
    result = build(store, tmp_path)
    assert result.skipped == {"outdated": 1} and result.examples


def test_a_category_made_inactive_since_is_outdated(tmp_path, train_call):
    fee = custom_category("fee_talk", name="Fee talk", gloss="Someone talks about a fee", examples=["fee"])
    store = FakeStore()
    store.taxonomy = with_categories([fee])
    call = add_signal_call(store, tmp_path, train_call, 0, verdicts=None, taxonomy=store.taxonomy)
    fee_hits = [h for h in call.run.result.signals if h.category_id == "fee_talk"]
    assert fee_hits
    for hit in fee_hits:
        add_hit_label(store, call, hit, category_verdict="dismissed")
    assert not build(store, tmp_path).skipped
    store.taxonomy = with_categories([fee.model_copy(update={"active": False})])
    result = build(store, tmp_path)
    assert result.skipped == {"outdated": len(fee_hits)} and not result.examples


def test_a_stale_subcategory_digest_drops_only_the_stage2_verdict(tmp_path, train_call):
    from call1.process.handlers.fake import CANCEL_SCRIPT

    store = FakeStore()
    taxonomy = cancel_taxonomy()
    store.taxonomy = taxonomy
    call = add_signal_call(store, tmp_path, train_call, 0, verdicts=None, taxonomy=taxonomy, script=CANCEL_SCRIPT)
    intent = next(h for h in call.run.result.signals if h.category_id == "intent")
    add_hit_label(store, call, intent, subcategory_id="fee_question", subcategory_digest="0" * 12, subcategory_verdict="confirmed")
    result = build(store, tmp_path)
    assert result.skipped == {"stale_subcategory": 1} and not result.examples

    store.labels.clear()
    add_hit_label(store, call, intent, category_verdict="confirmed", subcategory_id="fee_question", subcategory_digest="0" * 12,
                  subcategory_verdict="corrected", corrected_subcategory_id="fee_question")
    spans = {k: v for e in build(store, tmp_path).examples if e.task == "signal_stage2" for k, v in json.loads(e.assistant).items()}
    key = f"intent.t{intent.turn_id}b{intent.span.block}"
    stored = next(d for d in call.run.subcategories.decisions if d.span_key == key)
    # the stale correction is dropped; the category Confirm keeps the stored stage-2 choice
    assert spans[key]["fits"] == "yes" and spans[key]["choice"] == (stored.subcategory_id or "other")


def test_a_subcategory_correction_trains_the_corrected_choice(tmp_path, train_call):
    from call1.contracts.contents import short_digest
    from call1.contracts.signals import subcategory_digest
    from call1.process.handlers.fake import CANCEL_SCRIPT

    store = FakeStore()
    taxonomy = cancel_taxonomy()
    store.taxonomy = taxonomy
    call = add_signal_call(store, tmp_path, train_call, 0, verdicts=None, taxonomy=taxonomy, script=CANCEL_SCRIPT)
    intent = next(h for h in call.run.result.signals if h.category_id == "intent")
    node = next(s for s in taxonomy.category("intent").subcategories if s.subcategory_id == "cancel_account")
    add_hit_label(store, call, intent, subcategory_id="cancel_account", subcategory_digest=short_digest(subcategory_digest(node)),
                  subcategory_verdict="corrected", corrected_subcategory_id="fee_question")
    spans = {k: v for e in build(store, tmp_path).examples if e.task == "signal_stage2" for k, v in json.loads(e.assistant).items()}
    got = spans[f"intent.t{intent.turn_id}b{intent.span.block}"]
    assert (got["fits"], got["choice"]) == ("yes", "fee_question")
    assert got["assessment"] == "The caller's own words show Caller asks about a fee."


def test_labels_without_sources_or_findings_are_skipped_never_trained_unmasked(tmp_path, train_call):
    store = FakeStore()
    call = add_signal_call(store, tmp_path, train_call, 0, verdicts=None)
    hit = call.run.result.signals[0]
    no_findings = [ref for ref in call.refs if ref.role != "pii_findings"]
    call.refs[:] = no_findings
    add_hit_label(store, call, hit, category_verdict="confirmed")
    result = build(store, tmp_path)
    assert result.skipped == {"no_pii_findings": 1} and not result.examples


def test_speaker_rules_turn_only_and_stereo_calls_do_not_train(tmp_path):
    calls = calls_in("train", 2)
    store = FakeStore()
    add_speaker_call(store, tmp_path, calls[0], 0, apply_to_cluster=False)
    add_speaker_call(store, tmp_path, calls[1], 1, confidence=1.0)  # roles from the channel, not the model
    result = build(store, tmp_path)
    assert not result.examples and result.skipped == {"turn_only": 1, "not_model": 1}


# --- P6: the split -------------------------------------------------------------------------------


def test_the_split_is_stable_by_call_and_roughly_20_10_70():
    buckets = [split_of(INSTALLATION, f"call_{i}") for i in range(2000)]
    assert split_of(INSTALLATION, "call_7") == split_of(INSTALLATION, "call_7")
    share = {b: buckets.count(b) / len(buckets) for b in ("eval", "valid", "train")}
    assert 0.16 < share["eval"] < 0.24 and 0.07 < share["valid"] < 0.13 and 0.63 < share["train"] < 0.77
    assert split_of("another-install", "call_7") in ("eval", "valid", "train")


def test_held_out_calls_become_eval_items_not_examples(tmp_path):
    store = FakeStore()
    held = calls_in("eval", 1)[0]
    add_signal_call(store, tmp_path, held, 0)
    add_qa_call(store, tmp_path, held, 0)
    add_speaker_call(store, tmp_path, held, 0)
    result = build(store, tmp_path)
    assert not result.examples
    tasks = {item.task for item in result.items}
    assert tasks == {"signal_stage1", "signal_stage2", "qa_verdict", "speaker_roles"}
    assert all(prompt.id in {pid for item in result.items for pid in item.prompt_ids} for prompt in result.prompts.values())


def test_an_empty_validation_bucket_takes_a_deterministic_tenth(tmp_path):
    store = FakeStore()
    for i, call_id in enumerate(calls_in("train", 3)):
        add_qa_call(store, tmp_path, call_id, i)
        add_speaker_call(store, tmp_path, call_id, i)
    built = build(store, tmp_path)
    first = assemble(built, max_train_examples=100)
    again = assemble(built, max_train_examples=100)
    assert len(first.valid) == 1 and len(first.train) == 5
    assert [e.digest for e in first.valid] == [e.digest for e in again.valid]
    capped = assemble(built, max_train_examples=2)
    assert len(capped.train) + len(capped.valid) <= 3


# --- P7: budgeting -------------------------------------------------------------------------------


def test_over_budget_chunks_split_on_the_grid_and_single_rows_trim_their_context(tmp_path, train_call):
    store = FakeStore()
    add_signal_call(store, tmp_path, train_call, 0)
    whole = [e for e in build(store, tmp_path).examples if e.task == "signal_stage1"]
    assert len(whole) == 1 and len(json.loads(whole[0].assistant)) == 8
    budget = whole[0].tokens - 1
    split = [e for e in build(store, tmp_path, max_seq_length=budget).examples if e.task == "signal_stage1"]
    assert len(split) >= 2 and all(e.tokens <= budget for e in split)
    sizes = [len(json.loads(e.assistant)) for e in split]
    assert all(size in (1, 2, 4) for size in sizes)
    chunks = [list(json.loads(e.assistant)) for e in split]
    for keys in chunks:  # halves of the fixed grid: rows 0-3, 4-7, then 0-1, 2-3 ...
        first = int(keys[0])
        assert first % len(keys) == 0 and keys == [str(i) for i in range(first, first + len(keys))]


def test_a_row_that_cannot_fit_is_dropped_as_too_long(tmp_path, train_call):
    store = FakeStore()
    add_signal_call(store, tmp_path, train_call, 0)
    add_qa_call(store, tmp_path, train_call, 0)
    result = build(store, tmp_path, max_seq_length=300)
    assert not result.examples
    assert result.skipped["too_long"] >= 3 and result.notes["qa_too_long"] == 1


def test_the_single_row_context_is_trimmed_oldest_first(tmp_path, train_call):
    store = FakeStore()
    call = add_signal_call(store, tmp_path, train_call, 0, verdicts=None)
    last = call.run.result.signals[-1]  # a late row, with several earlier entries
    add_hit_label(store, call, last, category_verdict="confirmed")
    row_key = str(next(s.index for s in call.run.categories.segments if s.turn_id == last.turn_id))
    ctx = SignalContext(call.run.jobs["merge"])
    rows = stage1_rows(ctx.segmentation.segments, ctx.taxonomy, budget=gemma_row_budget(estimate_tokens), mask=ctx.mask).rows
    full = [f"{e['speaker']}: {e['text']}" for e in next(r for r in rows if r.key == row_key).state["previous"]]
    assert len(full) >= 3
    seen = []
    for budget in range(2000, 200, -5):
        found = [e for e in build(store, tmp_path, max_seq_length=budget).examples if e.task == "signal_stage1"]
        if found and list(json.loads(found[0].assistant)) == [row_key]:
            earlier = json.loads(found[0].user)["earlier"]
            assert earlier == full[len(full) - len(earlier):]  # always a suffix: the oldest entries went first
            assert found[0].tokens <= budget
            seen.append(len(earlier))
    assert seen and min(seen) < len(full)  # the context was trimmed before the row was dropped
