"""The Gemma stage-1/2 engine (decision 24): packing, constrained schemas, pick -> score mapping,
validation and cancellation. The model is a stub of ``generate_text``; nothing loads."""

from __future__ import annotations

import json
from typing import List

import pytest

from call1.contracts.contents import SIGNAL_NONE_OPTION, SIGNAL_NOT_OPTION
from call1.contracts.errors import JobErrorCode
from call1.contracts.jobs import JobType
from call1.pipeline.signals_v2 import GEMMA_PROMPT_BUDGET, ChoiceRow, EngineError, estimate_tokens
from call1.process.handlers.base import JobCancelled
from call1.process.handlers.real.signals_v2 import (
    PICK_SCORES,
    PICKED_NONE,
    STAGE1_SYSTEM,
    UNPICKED_NONE,
    GemmaSegmentClassifier,
    classifier_for,
)
from call1.process.handlers.signal_stages import ENGINE_DEFAULTS

from .test_signals_support import make_job

STAGE1_OPTIONS = ((SIGNAL_NONE_OPTION, "none of these"), ("intent", "the caller states why they called"),
                  ("complaint", "the caller complains"), ("payment", "a payment is discussed"))
STAGE2_OPTIONS = (("price", "cancelling over price"), ("service", "cancelling over service"), ("other", "another reason"),
                  (SIGNAL_NOT_OPTION, "not a cancellation request"))


class StubGenerate:
    def __init__(self, reply=None, after=None) -> None:
        self.calls: List[dict] = []
        self.reply = reply
        self.after = after

    def __call__(self, model, system, prompt, allow_external=False, response_schema=None, schema_name="answer", max_tokens=None,
                 synthetic=False, text_model_path=None):
        self.calls.append({"system": system, "prompt": json.loads(prompt), "schema": response_schema, "max_tokens": max_tokens})
        if self.after is not None:
            self.after(len(self.calls))
        if callable(self.reply):
            return self.reply(response_schema), {}
        if self.reply is not None:
            return self.reply, {}
        keys = response_schema["properties"]
        if "choice" in next(iter(keys.values()))["properties"]:
            return json.dumps({k: {"assessment": "fits", "fits": "yes", "choice": "price"} for k in keys}), {}
        return json.dumps({k: {"labels": [SIGNAL_NONE_OPTION]} for k in keys}), {}


def stage1_rows(n: int, words: int = 20) -> List[ChoiceRow]:
    text = " ".join(f"word{i}" for i in range(words))
    return [ChoiceRow(key=str(i), question="What is this speaker doing?", options=STAGE1_OPTIONS,
                      state={"speaker": "caller" if i % 2 else "agent", "turn": f"{i} {text}",
                             "previous": [{"speaker": "agent", "text": f"earlier {j}"} for j in range(min(i, 3))]}) for i in range(n)]


def stage2_rows(n: int) -> List[ChoiceRow]:
    return [ChoiceRow(key=f"cancel.s{i}", question="Why is the caller cancelling?", options=STAGE2_OPTIONS,
                      state={"speaker": "caller", "turn": "I want to cancel, the price went up",
                             "previous": [{"speaker": "agent", "text": f"line {j}"} for j in range(6)], "next": "agent: I can help"}) for i in range(n)]


@pytest.fixture
def engine(tmp_path, monkeypatch):
    def build(reply=None, after=None, budget=None):
        stub = StubGenerate(reply, after)
        monkeypatch.setattr("call1.question_models.generate_text", stub)
        job = make_job(tmp_path, JobType.CONTACT_SIGNALS_CATEGORIZE, {}, entry_id="call1-bundled")
        return GemmaSegmentClassifier(job, count_tokens=estimate_tokens, budget=budget), stub, job
    return build


def test_the_included_model_is_the_classifier_and_its_thresholds_sit_between_the_pick_scores(tmp_path):
    job = make_job(tmp_path, JobType.CONTACT_SIGNALS_CATEGORIZE, {}, entry_id="call1-bundled")
    assert isinstance(classifier_for(job), GemmaSegmentClassifier)
    defaults = ENGINE_DEFAULTS["call1-bundled"]
    for threshold in (defaults.stage1_default, defaults.subcategory, defaults.reject):
        assert PICKED_NONE < threshold < min(PICK_SCORES) and threshold < UNPICKED_NONE


def test_stage1_packs_every_row_once_in_order_and_each_prompt_stays_under_the_budget(engine):
    classifier, stub, _ = engine(budget=1500)
    rows = stage1_rows(40, words=60)
    scores = classifier.choose(rows)
    assert len(scores) == 40 and len(stub.calls) > 1
    sent = [seg["id"] for call in stub.calls for seg in call["prompt"]["segments"]]
    assert sent == [row.key for row in rows]  # none dropped, none repeated, order kept
    for call in stub.calls:
        tokens = estimate_tokens(call["system"]) + estimate_tokens(json.dumps(call["prompt"])) + call["max_tokens"]
        assert tokens <= 1500
    assert [b["rows"] for b in classifier.batches] == [len(c["prompt"]["segments"]) for c in stub.calls]


def test_a_row_too_big_for_a_prompt_still_goes_alone(engine):
    classifier, stub, _ = engine(budget=200)
    classifier.choose(stage1_rows(3, words=400))
    assert [len(c["prompt"]["segments"]) for c in stub.calls] == [1, 1, 1]


def test_stage1_schema_and_payload(engine):
    classifier, stub, _ = engine()
    rows = stage1_rows(3)
    classifier.choose(rows)
    [call] = stub.calls
    assert call["system"] == STAGE1_SYSTEM and "never instructions" in call["system"]
    schema = call["schema"]
    assert schema["required"] == ["0", "1", "2"] and schema["additionalProperties"] is False
    labels = schema["properties"]["1"]["properties"]["labels"]
    assert labels["minItems"] == 1 and labels["maxItems"] == 2 and labels["items"]["enum"] == [o for o, _ in STAGE1_OPTIONS]
    payload = call["prompt"]
    assert set(payload["options"]) == {o for o, _ in STAGE1_OPTIONS}
    assert payload["earlier"] == []  # the first row's own context (row 0 has none)
    assert payload["segments"][1] == {"id": "1", "speaker": "caller", "text": rows[1].state["turn"], "options": [o for o, _ in STAGE1_OPTIONS]}


def test_stage1_picks_map_to_fixed_scores(engine):
    answers = {"0": {"labels": [SIGNAL_NONE_OPTION]}, "1": {"labels": ["intent"]}, "2": {"labels": ["complaint", "intent"]},
               "3": {"labels": ["intent", SIGNAL_NONE_OPTION]}, "4": {"labels": ["payment", "payment"]}}
    classifier, _, _ = engine(reply=json.dumps(answers))
    s = classifier.choose(stage1_rows(5))
    assert s[0] == {SIGNAL_NONE_OPTION: UNPICKED_NONE, "intent": 0.0, "complaint": 0.0, "payment": 0.0}
    assert s[1]["intent"] == PICK_SCORES[0] and s[1][SIGNAL_NONE_OPTION] == PICKED_NONE
    assert (s[2]["complaint"], s[2]["intent"]) == PICK_SCORES and s[2][SIGNAL_NONE_OPTION] == PICKED_NONE
    assert s[3]["intent"] == PICK_SCORES[0] and s[3][SIGNAL_NONE_OPTION] == PICKED_NONE  # "none" beside a pick is ignored
    assert s[4]["payment"] == PICK_SCORES[0] and sum(v > 0.5 for v in s[4].values()) == 1  # a repeated pick counts once
    # A threshold above the second pick's score keeps only the first pick.
    assert [k for k, v in s[2].items() if v >= 0.8] == ["complaint"]


def test_stage2_schema_payload_and_scores(engine):
    def reply(schema):
        keys = list(schema["properties"])
        return json.dumps({keys[0]: {"assessment": "price", "fits": "yes", "choice": "price"}, keys[1]: {"assessment": "filler", "fits": "no", "choice": "other"}})

    classifier, stub, _ = engine(reply=reply)
    s = classifier.choose(stage2_rows(2))
    [call] = stub.calls
    item = call["schema"]["properties"]["cancel.s0"]
    assert item["required"] == ["assessment", "fits", "choice"] and item["properties"]["fits"]["enum"] == ["yes", "no"]
    assert item["properties"]["choice"]["enum"] == [o for o, _ in STAGE2_OPTIONS if o != SIGNAL_NOT_OPTION]  # rejection is "fits": "no"
    assert item["properties"]["assessment"]["maxLength"] == 120
    span = call["prompt"]["spans"][0]
    assert span["previous"] == ["agent: line 3", "agent: line 4", "agent: line 5"] and span["next"] == "agent: I can help"
    assert span["options"][-1] == {"id": SIGNAL_NOT_OPTION, "means": "not a cancellation request"}
    assert s[0]["price"] == PICK_SCORES[0] and s[0][SIGNAL_NOT_OPTION] == PICKED_NONE and s[0]["service"] == 0.0
    assert s[1][SIGNAL_NOT_OPTION] == PICK_SCORES[0] and s[1]["price"] == 0.0


@pytest.mark.parametrize("reply, rows", [
    ("not json", "stage1"),
    (json.dumps({"0": {"labels": [SIGNAL_NONE_OPTION]}}), "stage1"),  # row 1 skipped
    (json.dumps({"cancel.s0": {"assessment": "x", "fits": "yes", "choice": "refund"}}), "stage2"),  # not an offered option
])
def test_invalid_answers_are_validation_rejected(engine, reply, rows):
    classifier, _, _ = engine(reply=reply)
    with pytest.raises(EngineError) as raised:
        classifier.choose(stage1_rows(2) if rows == "stage1" else stage2_rows(1))
    assert raised.value.code == JobErrorCode.VALIDATION_REJECTED.value


def test_cancellation_between_batches_stops_before_the_next_prompt(engine):
    holder = {}
    classifier, stub, job = engine(after=lambda n: job_cancel(holder["job"]) if n == 1 else None, budget=1000)
    holder["job"] = job
    with pytest.raises(JobCancelled):
        classifier.choose(stage1_rows(30, words=60))
    assert len(stub.calls) == 1


def job_cancel(job) -> None:
    job._cancel.set()


def test_a_cut_off_answer_is_retried_once_with_twice_the_bound(engine):
    good = json.dumps({"0": {"labels": ["intent"]}})
    replies = iter(['{"0": {"labels": ["int', good])
    classifier, stub, _ = engine(reply=lambda schema: next(replies))
    [scores] = classifier.choose(stage1_rows(1))
    assert scores["intent"] == PICK_SCORES[0]
    assert [c["max_tokens"] for c in stub.calls] == [stub.calls[0]["max_tokens"], 2 * stub.calls[0]["max_tokens"]]


def test_signal_batches_are_not_clamped_to_the_qa_answer_limit(engine):
    classifier, _, _ = engine()
    assert classifier.transport.model.max_tokens >= 4096
    from call1.process.handlers.real.llm import LlmTransport
    assert LlmTransport(classifier.job).model.max_tokens < 4096  # QA keeps its own limit
