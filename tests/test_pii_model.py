"""The model-based PII layer (team decision 19) without the weights: backend selection, the BIOES
Viterbi decoder, span boundaries, the agent-name and date policy, the labelled stub, and Process's
masking step (union with the number rules, cache, fail-closed, word timestamps and audio)."""
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from call1 import pii_model
from call1.contracts.errors import JobErrorCode
from call1.pii_model import PiiSpan
from call1.process.handlers.base import HandlerError
from call1.process.handlers.real import masking
from call1.redaction import REDACTED, RedactionService, mask_text_with_values, mask_word_timestamps

ID2LABEL = {0: "O"}
for _i, _cat in enumerate(pii_model.LABELS):
    for _j, _b in enumerate("BIES"):
        ID2LABEL[1 + 4 * _i + _j] = f"{_b}-{_cat}"
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
ZERO = {name: 0.0 for name in pii_model._BIAS_NAMES}


@pytest.mark.parametrize("env, mode, expected", [
    ({}, None, "privacy-filter"),
    ({"CALL1_PROCESS_HANDLERS": "fake"}, None, "stub"),
    ({}, "fake", "stub"),
    ({"CALL1_PII_MODEL_BACKEND": "stub"}, "real", "stub"),
    ({"CALL1_PII_MODEL_BACKEND": "privacy-filter", "CALL1_PROCESS_HANDLERS": "fake"}, None, "privacy-filter"),
])
def test_backend_selection(env, mode, expected):
    assert pii_model.configured_backend(mode, env=env) == expected


def test_an_unknown_backend_is_a_configuration_error():
    with pytest.raises(pii_model.PiiConfigError):
        pii_model.configured_backend(env={"CALL1_PII_MODEL_BACKEND": "regex"})


def test_weights_path_and_missing_weights_fail_closed(tmp_path):
    assert pii_model.weights_path({"CALL1_MODELS_DIR": str(tmp_path)}) == tmp_path / "openai-privacy-filter"
    assert pii_model.weights_path({"CALL1_PII_MODEL_PATH": "/x/y"}) == pii_model.Path("/x/y")
    with pytest.raises(pii_model.PiiModelUnavailable):
        pii_model.PrivacyFilter(tmp_path)


def test_calibration_biases_are_read_and_default_to_zero(tmp_path):
    assert pii_model.read_calibration(tmp_path) == ZERO
    (tmp_path / "viterbi_calibration.json").write_text(
        '{"operating_points": {"default": {"biases": {"transition_bias_background_stay": 0.5}}}}')
    assert pii_model.read_calibration(tmp_path)["transition_bias_background_stay"] == 0.5


def _log_probs(rows):
    out = np.full((len(rows), len(ID2LABEL)), -9.0)
    for t, row in enumerate(rows):
        for label, value in row.items():
            out[t, LABEL2ID[label]] = value
    return out


def test_viterbi_only_emits_valid_bioes_paths():
    # Independent argmax would say I, I (no begin); the constrained path must open with B and close with E.
    rows = [{"O": -3.0, "I-private_person": -0.1, "B-private_person": -0.5},
            {"O": -3.0, "I-private_person": -0.1, "E-private_person": -0.4},
            {"O": -0.1}]
    path = [ID2LABEL[i] for i in pii_model.viterbi(_log_probs(rows), *pii_model.transition_scores(ID2LABEL, ZERO))]
    assert path == ["B-private_person", "E-private_person", "O"]


def test_calibration_bias_shifts_the_operating_point():
    rows = [{"O": -0.6, "S-private_person": -0.8}]
    scores = pii_model.transition_scores(ID2LABEL, ZERO)
    assert ID2LABEL[pii_model.viterbi(_log_probs(rows), *scores)[0]] == "O"
    # The start scores have no bias; entering a span from background is what the bias moves.
    rows = [{"O": -0.1}, {"O": -0.6, "S-private_person": -0.8}]
    eager = dict(ZERO, transition_bias_background_to_start=1.0)
    assert ID2LABEL[pii_model.viterbi(_log_probs(rows), *pii_model.transition_scores(ID2LABEL, eager))[1]] == "S-private_person"


def test_spans_are_trimmed_of_the_token_space_and_widened_to_whole_words():
    text = "Hi Samantha, it is 442-89-1099."
    # Token offsets as the tokenizer gives them: the leading space belongs to the token.
    offsets = [(0, 2), (2, 7), (7, 11), (11, 12), (12, 15), (15, 18), (18, 22), (22, 23), (23, 25), (25, 26), (26, 29), (29, 30), (30, 31)]
    labels = ["O", "B-private_person", "E-private_person", "O", "O", "O", "B-account_number", "I-account_number",
              "I-account_number", "I-account_number", "I-account_number", "E-account_number", "O"]
    spans = pii_model.decode_spans(text, offsets, [LABEL2ID[l] for l in labels], ID2LABEL)
    assert [(s.label, s.text) for s in spans] == [("private_person", "Samantha"), ("account_number", "442-89-1099")]


def test_agent_names_come_from_metadata_and_self_introductions():
    tokens = pii_model.agent_name_tokens("Samantha Reyes", ["Thanks for calling, this is Sam Jones.", "My name is ALEX? no"])
    assert {"samantha", "reyes", "sam", "jones"} <= tokens


@pytest.mark.parametrize("span, masked", [
    (PiiSpan("private_person", 0, 8, "Samantha"), False),      # the agent
    (PiiSpan("private_person", 0, 8, "samantha"), False),      # case-insensitive
    (PiiSpan("private_person", 0, 13, "Daniel Okafor"), True),  # the caller
    (PiiSpan("private_person", 0, 15, "Samantha Okafor"), True),  # not only the agent's name
    (PiiSpan("private_date", 0, 12, "May 12, 1984"), False),     # dates stay visible
    (PiiSpan("private_email", 0, 9, "a@b.co.uk"), True),
    (PiiSpan("private_address", 0, 9, "1 Elm Way"), True),
    (PiiSpan("private_url", 0, 9, "www.x.com"), True),
    (PiiSpan("secret", 0, 6, "hunter"), True),
    (PiiSpan("account_number", 0, 4, "1234"), True),
    (PiiSpan("private_phone", 0, 8, "555-0199"), True),
])
def test_the_masking_policy(span, masked):
    assert pii_model.maskable(span, {"samantha"}) is masked


def test_the_stub_is_labelled_and_deterministic():
    stub = pii_model.detector("stub")
    assert isinstance(stub, pii_model.StubPrivacyFilter) and stub.backend == "stub"
    text = ("My name is Daniel Okafor, email daniel.okafor@gmail.com, see www.example.com/me, I live at "
            "1420 West Cedar Lane, Tempe and was born May 12, 1984.")
    found = {(s.label, s.text) for s in stub.detect([text])[0]}
    assert found == {("private_person", "Daniel Okafor"), ("private_email", "daniel.okafor@gmail.com"),
                     ("private_url", "www.example.com/me"), ("private_address", "1420 West Cedar Lane, Tempe"),
                     ("private_date", "May 12, 1984")}
    assert stub.detect([text]) == stub.detect([text])


# --- Process's masking step -----------------------------------------------------------------------

AGENT_INTRO = "Thank you for calling Apex Financial. My name is Samantha."
CALLER = ("Hi Samantha, my name is Daniel Okafor, email daniel.okafor@gmail.com, I live at 1420 West Cedar Lane, "
          "Tempe. My SSN is 442-89-1099 and I was born May 12, 1984.")
TURNS = [{"speaker": "AGENT", "text": AGENT_INTRO}, {"speaker": "CALLER", "text": CALLER}]


def fake_job(conversation_id="conv-1", agent_display_name="Samantha", inputs=None):
    return SimpleNamespace(job=SimpleNamespace(conversation_id=conversation_id),
                           call_metadata=lambda: SimpleNamespace(agent_display_name=agent_display_name),
                           check_cancelled=lambda: None, input=lambda role: (inputs or {}).get(role),
                           log=logging.getLogger("test"))


@pytest.fixture(autouse=True)
def fresh_cache():
    masking.clear_cache()
    yield
    masking.clear_cache()


def test_masking_is_the_union_of_rules_and_model_minus_agent_name_and_dates():
    values = masking.sensitive_values(TURNS, job=fake_job())
    assert {"442-89-1099", "Daniel Okafor", "daniel.okafor@gmail.com", "1420 West Cedar Lane, Tempe"} <= values
    masked = mask_text_with_values(CALLER, values)
    assert "Samantha" in masked and "May 12, 1984" in masked
    for leaked in ("Daniel", "Okafor", "gmail", "Cedar", "442-89-1099"):
        assert leaked not in masked
    assert mask_text_with_values(AGENT_INTRO, values) == AGENT_INTRO


def test_without_a_job_only_the_rules_apply():
    assert masking.sensitive_values(TURNS) == {"442-89-1099"}


def test_the_agent_name_is_learned_from_a_self_introduction_without_metadata():
    values = masking.sensitive_values(
        [{"speaker": "AGENT", "text": "Hello, this is Priya Shah."}, {"speaker": "CALLER", "text": "Hi, I'm Priya Shah too, oddly."}],
        job=fake_job(agent_display_name=None))
    assert not values  # the same name as the agent's stays visible (a documented residual)


def test_the_model_runs_once_per_call_and_is_released(monkeypatch):
    made, released = [], []

    class Counting(pii_model.StubPrivacyFilter):
        def release(self):
            released.append(1)

    monkeypatch.setattr(pii_model, "detector", lambda backend=None: made.append(backend) or Counting())
    first = masking.sensitive_values(TURNS, job=fake_job())
    second = masking.sensitive_values(TURNS, ["criterion guidance"], job=fake_job())
    assert first == second and len(made) == 1 and len(released) == 1
    masking.sensitive_values(TURNS, job=fake_job("conv-2"))
    assert len(made) == 2


def test_a_missing_model_fails_the_job_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("CALL1_PII_MODEL_BACKEND", "privacy-filter")
    monkeypatch.setenv("CALL1_PII_MODEL_PATH", str(tmp_path))
    with pytest.raises(HandlerError) as caught:
        masking.sensitive_values(TURNS, job=fake_job())
    assert caught.value.code is JobErrorCode.MODEL_UNAVAILABLE


def test_a_misconfigured_backend_fails_the_job_closed(monkeypatch):
    monkeypatch.setenv("CALL1_PII_MODEL_BACKEND", "nonsense")
    with pytest.raises(HandlerError) as caught:
        masking.sensitive_values(TURNS, job=fake_job())
    assert caught.value.code is JobErrorCode.MODEL_UNAVAILABLE


def _timed(text):
    words, t = [], 0.0
    for w in text.split():
        words.append({"word": " " + w, "start_time": t, "end_time": t + 0.4})
        t += 0.5
    return words


def test_model_values_map_to_word_timestamps_and_mute_audio():
    values = masking.sensitive_values(TURNS, job=fake_job())
    words = _timed(CALLER)
    out = mask_word_timestamps(words, CALLER, values)
    kept = [w["word"].strip() for w in out if w["word"] != REDACTED]
    assert "Samantha," in kept and "born" in kept and "1984." in kept
    assert not {"Daniel", "Okafor,", "daniel.okafor@gmail.com,", "Cedar", "442-89-1099"} & set(kept)

    turn = SimpleNamespace(text=CALLER, start_time=0.0, end_time=30.0, numeric_entities=[], word_timestamps=words)
    settings = SimpleNamespace(redaction=SimpleNamespace(text=True, audio=True, pii_patterns=True))
    intervals = RedactionService(settings).audio_mute_intervals([turn], extra_values=values)

    def muted(word):
        w = next(x for x in words if x["word"] == " " + word)
        return any(s <= w["start_time"] and w["end_time"] <= e for s, e in intervals)

    assert all(muted(w) for w in ("Daniel", "Okafor,", "daniel.okafor@gmail.com,", "1420", "Cedar", "Lane,", "442-89-1099"))
    assert not any(muted(w) for w in ("Samantha,", "born", "1984."))


def test_name_values_mask_whole_words_only():
    assert mask_text_with_values("Dan met Daniel on Danube Drive", {"Dan"}) == f"{REDACTED} met Daniel on Danube Drive"


# --- span filtering and positional masking (team decision 22) ------------------------------------

# (span, kept at all, strong: masked by value across the call). The first rows are what the filter
# returned on the Parakeet text of the AppTek retail card calls.
FILTER_TABLE = [
    (PiiSpan("private_person", 0, 4, "Okay"), False, False),
    (PiiSpan("private_person", 0, 2, "so"), False, False),
    (PiiSpan("private_person", 0, 3, "and"), False, False),
    (PiiSpan("private_person", 0, 6, "lovely"), False, False),
    (PiiSpan("private_person", 0, 6, "Lovely"), False, False),
    (PiiSpan("private_address", 0, 11, "card number"), False, False),
    (PiiSpan("private_person", 0, 13, "David Aguilar"), True, True),
    (PiiSpan("private_person", 0, 5, "David"), True, True),
    (PiiSpan("private_person", 0, 5, "david"), True, False),        # lower case, one word, cased ASR: by position only
    (PiiSpan("private_person", 0, 2, "Al"), True, False),           # under 3 characters
    (PiiSpan("private_person", 0, 11, "okay Sophie"), True, True),
    (PiiSpan("private_address", 0, 4, "5620"), True, True),         # a number run
    (PiiSpan("private_address", 0, 10, "5620, yeah"), True, True),
    (PiiSpan("private_phone", 0, 3, "365"), True, True),
    (PiiSpan("account_number", 0, 5, "seven"), True, False),        # one digit: by position only
    (PiiSpan("private_phone", 0, 16, "381 21-6267-5292"), True, True),
    (PiiSpan("private_email", 0, 9, "a@b.co.uk"), True, True),
    (PiiSpan("private_url", 0, 9, "www.x.com"), True, True),
    (PiiSpan("private_address", 0, 22, "1420 West Cedar Lane"), True, True),
    (PiiSpan("private_address", 0, 7, "phoenix"), True, False),
    (PiiSpan("secret", 0, 7, "hunter2"), True, True),
    (PiiSpan("private_person", 0, 4, "Will"), True, True),          # a name that is also a word is never "common"
]


@pytest.mark.parametrize("span, kept, strong", FILTER_TABLE)
def test_span_filtering(span, kept, strong):
    assert pii_model.keep_span(span) is kept
    assert pii_model.strong_identifier(span, cased=True) is strong


def test_without_asr_casing_a_single_name_word_is_strong():
    assert pii_model.transcript_cased(["sounds good i'll take it", "i'm here"]) is False
    assert pii_model.transcript_cased(["okay I'm here", "use Visa please"]) is True
    assert pii_model.strong_identifier(PiiSpan("private_person", 0, 5, "david"), cased=False)
    assert not pii_model.strong_identifier(PiiSpan("private_person", 0, 2, "al"), cased=False)


def test_sensitive_values_are_only_strong_identifiers():
    spans = [[PiiSpan("private_person", 0, 4, "Okay"), PiiSpan("private_person", 5, 10, "david"),
              PiiSpan("private_person", 11, 24, "David Aguilar"), PiiSpan("private_person", 0, 8, "Samantha")]]
    assert pii_model.filter_spans(spans, {"samantha"}) == [[spans[0][1], spans[0][2]]]
    assert pii_model.sensitive_values(spans, {"samantha"}) == {"David Aguilar"}


PARAKEET_TURNS = [
    {"turn_id": 0, "speaker": "AGENT", "text": "And this the three-digit security code."},
    {"turn_id": 1, "speaker": "CALLER", "text": "Seven."},
    {"turn_id": 2, "speaker": "CALLER", "text": "Two four."},
    {"turn_id": 3, "speaker": "AGENT", "text": "Okay."},
    {"turn_id": 4, "speaker": "AGENT", "text": "So it's for david, and Okay, that's David Aguilar."},
    {"turn_id": 5, "speaker": "AGENT", "text": "Okay, so david will get it in seven days."},
]


def _pinned(spans_by_turn):
    return SimpleNamespace(content=lambda: SimpleNamespace(
        transcript=SimpleNamespace(checksum="sha256:t"),
        turns=[SimpleNamespace(turn_id=turn_id, spans=[SimpleNamespace(category=c, start=s, end=s + len(t), text=t) for c, t, s in found])
               for turn_id, found in spans_by_turn.items()]))


def test_process_masks_findings_and_read_out_digits_by_position():
    line = PARAKEET_TURNS[4]["text"]
    pinned = _pinned({4: [("private_person", "david", line.index("david")), ("private_person", "Okay", line.index("Okay")),
                          ("private_person", "David Aguilar", line.index("David Aguilar"))]})
    transcript = SimpleNamespace(artifact=SimpleNamespace(checksum="sha256:t"),
                                 content=lambda: SimpleNamespace(turns=[SimpleNamespace(text=t["text"]) for t in PARAKEET_TURNS]))
    job = fake_job(inputs={"pii_findings": pinned, "transcript": transcript})
    values = masking.sensitive_values(PARAKEET_TURNS, job=job)
    assert isinstance(values, masking.SensitiveValues) and set(values) == {"David Aguilar"}
    masked = [masking.mask(t["text"], values) for t in PARAKEET_TURNS]
    assert masked == [
        "And this the three-digit security code.",
        f"{REDACTED}.",
        f"{REDACTED}.",
        "Okay.",
        f"So it's for {REDACTED}, and Okay, that's {REDACTED}.",
        "Okay, so david will get it in seven days.",   # neither "david" nor "seven" is masked by value
    ]
    assert masking.findings_values(job) == {"David Aguilar"}
    # Legacy turns keep their positions too (word timestamps dropped as before).
    legacy = [SimpleNamespace(text=t["text"], model_copy=lambda update, t=t: {**t, **update}) for t in PARAKEET_TURNS]
    assert [t["text"] for t in masking.mask_turns(legacy, values)] == masked


def test_the_enrichment_output_drops_common_word_findings(monkeypatch):
    class Noisy(pii_model.StubPrivacyFilter):
        def detect(self, texts):
            out = super().detect(texts)
            for i, text in enumerate(texts):
                for word in ("Okay", "so", "lovely"):
                    if word in text:
                        out[i].append(PiiSpan("private_person", text.index(word), text.index(word) + len(word), word))
            return out

    monkeypatch.setattr(pii_model, "detector", lambda backend=None: Noisy())
    turns = [{"speaker": "CALLER", "text": "Okay so my name is Daniel Okafor, lovely."}]
    spans = masking.masked_spans(fake_job(), turns, backend="stub")
    assert [(s.label, s.text) for s in spans[0]] == [("private_person", "Daniel Okafor")]


def test_mask_turns_drops_raw_word_timestamps_when_only_positions_mask():
    turns = [{"text": "And the CVV?"}, {"text": "Seven."}]
    values = masking.sensitive_values(turns)
    assert set(values) == set() and values.positions
    legacy = [SimpleNamespace(text=t["text"], word_timestamps=[{"word": t["text"]}], model_copy=lambda update, t=t: {**t, **update})
              for t in turns]
    out = masking.mask_turns(legacy, values)
    assert [t["text"] for t in out] == ["And the CVV?", f"{REDACTED}."] and all(t["word_timestamps"] is None for t in out)


def test_attribution_and_enrichment_share_one_detection_and_keep_their_own_agent_filter(monkeypatch):
    """speaker_attribution masks the turns before the roles are known, enrichment after: the same
    text, different agent-name filters. The detector runs once for both, and each result is exactly
    what a fresh detection with that job's filter gives."""
    made = []

    class Counting(pii_model.StubPrivacyFilter):
        def detect(self, texts):
            made.append(list(texts))
            return super().detect(texts)

    monkeypatch.setattr(pii_model, "detector", lambda backend=None: Counting())
    texts = ["Hello, this is Priya Shah from support.", "Hi, my name is Daniel Okafor, daniel.okafor@gmail.com."]
    unlabelled = [{"speaker": "UNKNOWN", "text": t} for t in texts]  # speaker_attribution's input
    labelled = [{"speaker": "AGENT", "text": texts[0]}, {"speaker": "CALLER", "text": texts[1]}]  # enrichment's

    def spans(turns):
        return [[(s.label, s.start, s.end, s.text) for s in turn] for turn in masking.masked_spans(fake_job(agent_display_name=None), turns)]

    attribution, enrichment = spans(unlabelled), spans(labelled)
    assert len(made) == 1
    masking.clear_cache()
    fresh_attribution = spans(unlabelled)
    masking.clear_cache()
    fresh_enrichment = spans(labelled)
    assert (attribution, enrichment) == (fresh_attribution, fresh_enrichment)
    assert ("private_person", "Priya Shah") in {(s[0], s[3]) for s in attribution[0]}  # no agent known yet: masked
    assert not any(s[3] == "Priya Shah" for s in enrichment[0])  # the agent's own name stays visible
    assert any(s[3] == "Daniel Okafor" for s in enrichment[1])
