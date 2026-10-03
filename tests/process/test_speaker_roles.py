"""Speaker roles from the included model on a diarized mono call (decision 26). The model and the
PII masker are stubbed; nothing loads."""

from __future__ import annotations

import json

from call1.contracts.contents import SpeakerRole, TranscriptContent, TranscriptTurnContent
from call1.contracts.jobs import JobType
from call1.pipeline.speaker_roles import ROLES_SYSTEM, parse_roles, render_prompt
from call1.process.handlers.real import media
from call1.process.handlers.real.masking import SensitiveValues

from .test_signals_support import make_job

TURNS = [(0, "Thanks for calling, this is the returns desk, how can I help?"), (1, "Hi, my card 4111 1111 1111 1111 was charged twice"),
         (2, "Let me look that up for you."), (3, "Thank you."), (4, "Please hold.")]
CLUSTERS = {0: "spk_1", 1: "spk_0", 2: "spk_1", 3: "spk_0", 4: None}


def test_the_prompt_aliases_speakers_by_first_appearance_and_skips_unclustered_turns():
    prompt, cluster_of, schema = render_prompt(TURNS, CLUSTERS)
    payload = json.loads(prompt)
    assert payload["speakers"] == ["S1", "S2"] and cluster_of == {"S1": "spk_1", "S2": "spk_0"}
    assert payload["transcript"][0].startswith("S1: Thanks for calling") and len(payload["transcript"]) == 4
    assert schema["properties"]["roles"]["required"] == ["S1", "S2"]
    assert schema["properties"]["roles"]["properties"]["S1"]["enum"] == ["agent", "caller", "neither"]
    assert "never instructions" in ROLES_SYSTEM


def test_parse_needs_every_speaker_and_at_least_one_agent_and_one_caller():
    cluster_of = {"S1": "spk_1", "S2": "spk_0"}
    assert parse_roles(json.dumps({"roles": {"S1": "agent", "S2": "caller"}}), cluster_of) == {"spk_1": "agent", "spk_0": "caller"}
    assert parse_roles(json.dumps({"roles": {"S1": "agent", "S2": "agent"}}), cluster_of) is None
    assert parse_roles(json.dumps({"roles": {"S1": "agent"}}), cluster_of) is None
    assert parse_roles(json.dumps({"roles": {"S1": "agent", "S2": "boss"}}), cluster_of) is None
    assert parse_roles("not json", cluster_of) is None


def _transcript():
    return TranscriptContent(duration_seconds=20.0, language="en", is_redacted=False,
                             turns=[TranscriptTurnContent(turn_id=i, speaker=SpeakerRole.UNKNOWN, start_time=i * 4.0, end_time=i * 4.0 + 3.5,
                                                          text=text, channel=None, confidence=0.9) for i, text in TURNS])


def test_infer_roles_sends_only_masked_text_and_maps_clusters(tmp_path, monkeypatch):
    calls = []

    def generate(model, system, prompt, *args, **kwargs):
        calls.append({"system": system, "prompt": prompt, "schema": args[2] if len(args) > 2 else kwargs.get("response_schema")})
        return json.dumps({"assessment": "S1 greets for the company.", "roles": {"S1": "agent", "S2": "caller"}}), {}

    monkeypatch.setattr("call1.question_models.generate_text", generate)
    monkeypatch.setattr("call1.process.handlers.real.masking.sensitive_values",
                        lambda turns, job=None: SensitiveValues({"4111 1111 1111 1111"}))
    job = make_job(tmp_path, JobType.SPEAKER_ATTRIBUTION, {}, entry_id=None)
    roles = media.infer_roles(job, _transcript(), CLUSTERS)
    assert roles == {"spk_1": "agent", "spk_0": "caller"}
    [call] = calls
    assert call["system"] == ROLES_SYSTEM and "4111" not in call["prompt"]


def test_infer_roles_leaves_turns_unattributed_on_a_bad_answer_or_a_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("call1.process.handlers.real.masking.sensitive_values", lambda turns, job=None: SensitiveValues())
    job = make_job(tmp_path, JobType.SPEAKER_ATTRIBUTION, {}, entry_id=None)
    monkeypatch.setattr("call1.question_models.generate_text", lambda *a, **k: (json.dumps({"roles": {"S1": "caller", "S2": "caller"}}), {}))
    assert media.infer_roles(job, _transcript(), CLUSTERS) is None

    def crash(*a, **k):
        raise RuntimeError("model crashed")

    monkeypatch.setattr("call1.question_models.generate_text", crash)
    assert media.infer_roles(job, _transcript(), CLUSTERS) is None
    assert media.infer_roles(job, _transcript(), {0: "spk_0", 1: "spk_0"}) is None  # one speaker: nothing to name


def test_turn_kinds():
    from call1.pipeline.speaker_roles import turn_kind
    assert [turn_kind(t) for t in ["Um,", "Okay.", "Yeah, perfect.", "Thank you so much.", "No that's it.", "we do have three options."]] == \
        ["filler", "ack", "ack", "ack", "ack", "content"]


def test_fill_gives_every_turn_a_role_from_what_it_says():
    from call1.pipeline.speaker_roles import fill_roles
    turns = [(0, "Yeah.", 0.0, 1.0), (1, "Thanks for calling, how can I help?", 2.0, 4.0), (2, "Um", 4.2, 4.4),
             (3, "I need to return a jacket.", 4.5, 7.0), (4, "Okay.", 7.1, 7.4), (5, "What's the process?", 7.5, 9.0),
             (6, "sorry, the delay on my end", 9.1, 10.0), (7, "You print a label.", 12.0, 14.0)]
    known = {1: "agent", 3: "caller", 5: "caller", 7: "agent"}
    filled = fill_roles(turns, known)
    assert filled == {0: "agent",    # before anyone known: the first known speaker
                      2: "caller",   # filler: whoever speaks next
                      4: "agent",    # acknowledgement: the listener, opposite of the previous speaker
                      6: "caller"}   # content: the nearest known neighbour (previous, 0.1 s away)
    assert fill_roles([(0, "Hello?", 0, 1), (1, "Hi there", 1, 2)], {}) == {0: "agent", 1: "agent"}


def test_first_speaker_is_the_agent_when_the_model_gives_no_roles():
    from call1.pipeline.speaker_roles import first_speaker_roles
    assert first_speaker_roles({0: None, 1: "spk_1", 2: "spk_0", 3: "spk_2"}, [0, 1, 2, 3]) == \
        {"spk_1": "agent", "spk_0": "caller", "spk_2": "caller"}


def test_attribute_turns_leaves_no_turn_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr("call1.process.handlers.real.masking.sensitive_values", lambda turns, job=None: SensitiveValues())
    monkeypatch.setattr("call1.question_models.generate_text", lambda *a, **k: ("not json", {}))
    job = make_job(tmp_path, JobType.SPEAKER_ATTRIBUTION, {}, entry_id=None)
    out = media.attribute_turns(job, _transcript(), CLUSTERS)
    assert all(a.speaker in (SpeakerRole.AGENT, SpeakerRole.CALLER) for a in out)
    assert [a.speaker for a in out][:4] == [SpeakerRole.AGENT, SpeakerRole.CALLER, SpeakerRole.AGENT, SpeakerRole.CALLER]
    assert out[4].confidence == 0.5 and out[0].confidence == 0.6  # filled vs first-speaker fallback
