"""Model PII findings on reviewer reads (contract 1.2.0, team decision 19).

Process's ``enrichment`` job writes ``pii_findings`` per transcript revision; Store masks every
reviewer surface and mutes audio over the union of the rule values and those findings, and fails
closed while no findings exist for the current transcript revision.
"""

from __future__ import annotations

import io
import wave

import pytest

from call1.contracts.artifacts import ArtifactKind
from call1.contracts.contents import (
    ContactSignalsContent,
    ContactSignalsPassOutcome,
    ContactSignalView,
    ResultKind,
    ResultState,
    SummaryContent,
    VerdictStatus,
)
from call1.contracts.jobs import JobType
from call1.store.results import api as results_api
from call1.store.results import audio as audio_module
from call1.store.results import records

from .test_results_calls import _wav
from .test_results_harness import accounts, embeddings_for, fq, qa_scorecard, transcript  # noqa: F401

NAME = "Maria Lopez"
EMAIL = "maria.lopez@example.com"
AGENT_LINE = "Thank you for calling, this is Sam."
CALLER_LINE = f"Hi, my name is {NAME}, my email is {EMAIL}, and I want to dispute a charge."


def _latest_transcript(conv):
    return max((a for a in conv.artifacts if a.kind is ArtifactKind.TRANSCRIPT), key=lambda a: a.version)


def _findings_for(conv):
    return {1: [("private_person", NAME, CALLER_LINE.index(NAME)), ("private_email", EMAIL, CALLER_LINE.index(EMAIL))]}


def _groups(store, conv):
    with store.connection() as conn:
        return {g.kind: g for g in results_api.result_groups(conn, conv.id)}


def _call_with_text(fq, *, pii: bool = True, words: bool = False):
    fq.auto_pii = False
    conv = fq.register()
    graph = fq.graph(conv, [JobType.ASR, JobType.ENRICHMENT, JobType.EMBEDDINGS, JobType.SUMMARY_ASSEMBLY, JobType.CONTACT_SIGNALS_MERGE])
    word_turns = None
    if words:
        t = 2.0
        word_turns = {1: []}
        for w in CALLER_LINE.split():
            word_turns[1].append({"word": " " + w, "start_time": round(t, 2), "end_time": round(t + 0.1, 2), "probability": 0.9})
            t += 0.12
    content = transcript([("AGENT", AGENT_LINE), ("CALLER", CALLER_LINE)], word_turns=word_turns)
    fq.complete(conv, graph, JobType.ASR, {"transcript": content})
    fq.complete(conv, graph, JobType.EMBEDDINGS, {"embeddings": embeddings_for(content)})
    if pii:
        fq.link_pii(conv, _latest_transcript(conv), _findings_for(conv))
        fq.job(graph, JobType.ENRICHMENT).status = fq.job(graph, JobType.ENRICHMENT).status.SUCCEEDED
    summary = SummaryContent(narrative=f"{NAME} called to dispute a charge; Sam helped.", key_points=[f"Caller {NAME} gave {EMAIL}"],
                             route_class="appliance", catalog_entry_id="mlx-small", generated_at=fq.clock.now(), segments=1)
    fq.complete(conv, graph, JobType.SUMMARY_ASSEMBLY, {"summary": summary})
    signals = ContactSignalsContent(completeness="complete", signals=[ContactSignalView(
        id="s1", kind="issue", label="Dispute", start=2.0, end=4.0, speaker="CALLER", quote=CALLER_LINE, turn_id=1, confidence=0.9)],
        passes=[ContactSignalsPassOutcome(pass_kind="lifecycle", included=True), ContactSignalsPassOutcome(pass_kind="resolution", included=True)],
        transcript_fingerprint="sha256:" + "2" * 64, generated_at=fq.clock.now())
    fq.complete(conv, graph, JobType.CONTACT_SIGNALS_MERGE, {"contact_signals": signals})
    fq.ingest_qa(conv, [("GREET-01", VerdictStatus.PASS, 0.9)], evidence=f"{AGENT_LINE} / {NAME}")
    return conv, graph


def _no_pii(text):
    return NAME not in text and "Maria" not in text and "Lopez" not in text and EMAIL not in text


def test_model_findings_are_masked_on_every_reviewer_read(store, fq, client, reviewer_session):
    conv, _ = _call_with_text(fq, words=True)
    h = reviewer_session.read_headers
    group = _groups(store, conv)[ResultKind.TRANSCRIPT]
    assert group.state is ResultState.AVAILABLE and group.partial_reason is None

    view = client.get(f"/store/v1/calls/{conv.call_id}/transcript", headers=h).json()
    assert view["text_withheld"] is False and view["is_redacted"] is True
    agent, caller = view["turns"]
    assert agent["text"] == AGENT_LINE  # the agent's own name stays visible (it was never a finding)
    assert _no_pii(caller["text"]) and caller["text"].count("[REDACTED]") == 2 and "dispute a charge" in caller["text"]
    spoken = " ".join(w["word"] for w in caller["word_timestamps"])
    assert "Maria" not in spoken and "Lopez" not in spoken and "maria.lopez" not in spoken

    summary = client.get(f"/store/v1/calls/{conv.call_id}/summary", headers=h).json()
    assert _no_pii(summary["narrative"]) and "Sam helped" in summary["narrative"] and _no_pii(summary["key_points"][0])
    signals = client.get(f"/store/v1/calls/{conv.call_id}/contact-signals", headers=h).json()
    assert _no_pii(signals["signals"][0]["quote"]) and "[REDACTED]" in signals["signals"][0]["quote"]
    evaluation = client.get(f"/store/v1/calls/{conv.call_id}/evaluation", headers=h).json()
    assert _no_pii(evaluation["verdicts"][0]["quoted_evidence"]) and AGENT_LINE in evaluation["verdicts"][0]["quoted_evidence"]
    assert _no_pii(evaluation["verdicts"][0]["reasoning"])

    hits = client.post("/store/v1/search/semantic", json={"query": "dispute a charge", "call_id": conv.call_id},
                       headers=reviewer_session.headers).json()["results"]
    caller_hits = [r for r in hits if r["turn_id"] == 1]
    assert caller_hits and all(_no_pii(r["text"]) for r in caller_hits)


def test_audio_is_muted_over_model_findings(store, fq, client, reviewer_session):
    conv, _ = _call_with_text(fq, words=True)
    fq.artifact(conv, ArtifactKind.SOURCE_AUDIO, _wav(seconds=6.0), content_type="audio/wav")
    with store.connection() as conn:
        intervals = audio_module.mute_intervals(conn, conv.id)
    words = CALLER_LINE.split()
    name_start = 2.0 + 0.12 * words.index("Maria")
    email_start = 2.0 + 0.12 * words.index(EMAIL + ",")
    for spoken in (name_start, email_start):
        assert any(s <= spoken + 0.05 and e >= spoken + 0.05 for s, e in intervals), (spoken, intervals)
    response = client.get(f"/store/v1/calls/{conv.call_id}/audio", headers=reviewer_session.read_headers)
    assert response.status_code == 200
    with wave.open(io.BytesIO(response.content)) as w:
        frames = w.readframes(w.getnframes())
    at = int((name_start + 0.05) * 8000) * 2
    assert set(frames[at:at + 200]) == {0}


def test_without_findings_reads_fail_closed(store, fq, client, reviewer_session):
    conv, graph = _call_with_text(fq, pii=False)
    fq.artifact(conv, ArtifactKind.SOURCE_AUDIO, _wav(seconds=6.0), content_type="audio/wav")
    h = reviewer_session.read_headers

    # The enrichment job is still queued: 'partial', reason says masking is running.
    group = _groups(store, conv)[ResultKind.TRANSCRIPT]
    assert group.state is ResultState.PARTIAL and group.version == 1 and group.partial_reason == records.PII_PENDING_REASON
    view = client.get(f"/store/v1/calls/{conv.call_id}/transcript", headers=h).json()
    assert view["text_withheld"] is True and [t["text"] for t in view["turns"]] == ["", ""]
    assert all(t["word_timestamps"] is None for t in view["turns"])
    summary = client.get(f"/store/v1/calls/{conv.call_id}/summary", headers=h).json()
    assert summary["narrative"] == "[REDACTED]" and summary["key_points"] == ["[REDACTED]"]
    quote = client.get(f"/store/v1/calls/{conv.call_id}/contact-signals", headers=h).json()["signals"][0]["quote"]
    assert quote == "[REDACTED]"
    verdict = client.get(f"/store/v1/calls/{conv.call_id}/evaluation", headers=h).json()["verdicts"][0]
    assert verdict["quoted_evidence"] == "[REDACTED]" and verdict["reasoning"] == "[REDACTED]"
    detail = client.get(f"/store/v1/calls/{conv.call_id}", headers=h).json()
    assert _no_pii(str(detail))
    hits = client.post("/store/v1/search/semantic", json={"query": "dispute a charge", "call_id": conv.call_id},
                       headers=reviewer_session.headers).json()
    assert hits["results"] == []
    audio = client.get(f"/store/v1/calls/{conv.call_id}/audio", headers=h)
    assert audio.status_code == 503 and audio.json()["details"]["reason"] == "pii_findings_pending"
    listed = next(i for i in client.get("/store/v1/calls", headers=h).json()["items"] if i["call_id"] == conv.call_id)
    assert listed["transcript_state"] == "partial"

    # The enrichment job failed (e.g. model_unavailable): still withheld, reason says it did not finish.
    fq.fail(conv, graph, JobType.ENRICHMENT)
    group = _groups(store, conv)[ResultKind.TRANSCRIPT]
    assert group.state is ResultState.PARTIAL and group.partial_reason == records.PII_MISSING_REASON
    assert client.get(f"/store/v1/calls/{conv.call_id}/transcript", headers=h).json()["text_withheld"] is True

    # Findings arrive: the text is served, masked.
    fq.link_pii(conv, _latest_transcript(conv), _findings_for(conv))
    group = _groups(store, conv)[ResultKind.TRANSCRIPT]
    assert group.state is ResultState.AVAILABLE and group.partial_reason is None
    view = client.get(f"/store/v1/calls/{conv.call_id}/transcript", headers=h).json()
    assert view["text_withheld"] is False and _no_pii(view["turns"][1]["text"]) and "dispute a charge" in view["turns"][1]["text"]


def test_findings_for_an_older_transcript_revision_do_not_count(store, fq, client, reviewer_session):
    conv, _ = _call_with_text(fq)
    newer = fq.graph(conv, [JobType.ASR, JobType.ENRICHMENT])
    fq.complete(conv, newer, JobType.ASR, {"transcript": transcript([("AGENT", AGENT_LINE), ("CALLER", CALLER_LINE + " Again.")])})
    view = client.get(f"/store/v1/calls/{conv.call_id}/transcript", headers=reviewer_session.read_headers).json()
    assert view["version"] == 2 and view["text_withheld"] is True and all(t["text"] == "" for t in view["turns"])
    assert _groups(store, conv)[ResultKind.TRANSCRIPT].partial_reason == records.PII_PENDING_REASON


def test_no_transcript_means_no_text(store, fq, client, reviewer_session):
    """A scorecard with no transcript behind it (so no findings for one) is served fully redacted."""
    conv = fq.register()
    fq.ingest_qa(conv, [("REG-01", VerdictStatus.PASS, 0.9)], evidence=f"caller {NAME}")
    evaluation = client.get(f"/store/v1/calls/{conv.call_id}/evaluation", headers=reviewer_session.read_headers).json()
    assert evaluation["verdicts"][0]["quoted_evidence"] == "[REDACTED]"


def test_draft_test_scorecards_are_masked_like_evaluations(store, fq):
    """``getDraftTestResult`` masks its scorecard through ``results.api.masked_scorecard``."""
    card = qa_scorecard([("GREET-01", VerdictStatus.PASS, 0.9)], evidence=f"{NAME} said hello")
    conv, _ = _call_with_text(fq)
    with store.connection() as conn:
        masked = results_api.masked_scorecard(conn, conv.id, card)
    assert _no_pii(masked.verdicts[0].quoted_evidence) and "said hello" in masked.verdicts[0].quoted_evidence
    withheld_conv, _ = _call_with_text(fq, pii=False)
    with store.connection() as conn:
        withheld = results_api.masked_scorecard(conn, withheld_conv.id, card)
    assert withheld.verdicts[0].quoted_evidence == "[REDACTED]"


# --- findings by position (team decision 22) -------------------------------------------------------
#
# The privacy filter flags single common words ("so", "and", "Okay", "lovely") on ASR text; masking
# those spans by value hid the word across the whole call. Store now masks each finding at its turn
# and offsets, drops findings made only of common words, and propagates a value across the call only
# for a strong identifier (a multi-word name, a capitalized name of 3+ characters, an email, a URL or
# a number run).

POSITIONAL_TURNS = [
    ("AGENT", "Okay, so who am I speaking with today?"),
    ("CALLER", "Okay so it's for dana, she's my daughter, and Priya is me."),
    ("AGENT", "Lovely, so dana and Priya, and the email is priya.shah@example.com?"),
    ("CALLER", "Yes, lovely, and ask for dana at the desk."),
]


def _positional_call(fq, findings, words: bool = False):
    conv = fq.register()
    graph = fq.graph(conv, [JobType.ASR, JobType.EMBEDDINGS, JobType.SUMMARY_ASSEMBLY])
    fq.auto_pii = False
    word_turns = None
    if words:
        word_turns, t = {}, 0.0
        for i, (_, text) in enumerate(POSITIONAL_TURNS):
            t = 2.0 * i
            word_turns[i] = []
            for w in text.split():
                word_turns[i].append({"word": " " + w, "start_time": round(t, 2), "end_time": round(t + 0.1, 2), "probability": 0.9})
                t += 0.12
    content = transcript(POSITIONAL_TURNS, word_turns=word_turns)
    fq.complete(conv, graph, JobType.ASR, {"transcript": content})
    fq.complete(conv, graph, JobType.EMBEDDINGS, {"embeddings": embeddings_for(content)})
    fq.link_pii(conv, _latest_transcript(conv), findings)
    summary = SummaryContent(narrative="Priya called for dana; the agent said so and lovely.", key_points=["Okay so it's for dana"],
                             route_class="appliance", catalog_entry_id="mlx-small", generated_at=fq.clock.now(), segments=1)
    fq.complete(conv, graph, JobType.SUMMARY_ASSEMBLY, {"summary": summary})
    return conv


def _at(turn, text, nth=0):
    line = POSITIONAL_TURNS[turn][1]
    at = -1
    for _ in range(nth + 1):
        at = line.index(text, at + 1)
    return at


FINDINGS = {
    1: [("private_person", "Okay", _at(1, "Okay")),          # a common word: dropped
        ("private_person", "so", _at(1, "so")),              # a stopword: dropped
        ("private_person", "dana", _at(1, "dana")),          # lower-case, one word, cased ASR: by position only
        ("private_person", "Priya", _at(1, "Priya"))],       # capitalized, 3+ characters: everywhere
    2: [("private_person", "Lovely", _at(2, "Lovely")),      # a common word: dropped
        ("private_email", "priya.shah@example.com", _at(2, "priya.shah@example.com"))],
}


@pytest.mark.parametrize("turn, want", [
    (0, "Okay, so who am I speaking with today?"),
    (1, "Okay so it's for [REDACTED], she's my daughter, and [REDACTED] is me."),
    (2, "Lovely, so dana and [REDACTED], and the email is [REDACTED]?"),
    (3, "Yes, lovely, and ask for dana at the desk."),
])
def test_findings_are_masked_by_position_and_only_strong_ones_everywhere(store, fq, client, reviewer_session, turn, want):
    conv = _positional_call(fq, FINDINGS)
    view = client.get(f"/store/v1/calls/{conv.call_id}/transcript", headers=reviewer_session.read_headers).json()
    assert view["turns"][turn]["text"] == want


def test_positional_findings_reach_derived_text_that_repeats_the_turn(store, fq, client, reviewer_session):
    conv = _positional_call(fq, FINDINGS)
    summary = client.get(f"/store/v1/calls/{conv.call_id}/summary", headers=reviewer_session.read_headers).json()
    # "Priya" is strong (masked everywhere); "dana" is masked where the text repeats its turn, and
    # common words are never masked.
    assert summary["narrative"] == "[REDACTED] called for dana; the agent said so and lovely."
    assert summary["key_points"] == ["Okay so it's for [REDACTED]"]


def test_positional_findings_align_to_words_and_audio(store, fq, client, reviewer_session):
    conv = _positional_call(fq, FINDINGS, words=True)
    view = client.get(f"/store/v1/calls/{conv.call_id}/transcript", headers=reviewer_session.read_headers).json()
    spoken = [w["word"].strip() for w in view["turns"][1]["word_timestamps"]]
    assert spoken == ["Okay", "so", "it's", "for", "[REDACTED]", "she's", "my", "daughter,", "and", "[REDACTED]", "is", "me."]
    assert "dana" in [w["word"].strip() for w in view["turns"][3]["word_timestamps"]]
    fq.artifact(conv, ArtifactKind.SOURCE_AUDIO, _wav(seconds=8.0), content_type="audio/wav")
    with store.connection() as conn:
        intervals = audio_module.mute_intervals(conn, conv.id)

    def muted(turn, index):
        start = 2.0 * turn + 0.12 * index + 0.05
        return any(s <= start <= e for s, e in intervals)

    assert muted(1, 4) and muted(1, 9)                     # "dana," and "Priya" in turn 1
    assert not muted(1, 0) and not muted(1, 1)             # "Okay so" stays audible
    assert not muted(3, 4)                                 # "dana" in turn 3 was never a finding
    assert muted(2, 4)                                     # "Priya," in turn 2: a strong value


def test_a_finding_that_does_not_fit_its_turn_is_masked_by_value(store, fq, client, reviewer_session):
    conv = _positional_call(fq, {3: [("private_person", "desk", 0)]})  # offsets that do not match the text
    view = client.get(f"/store/v1/calls/{conv.call_id}/transcript", headers=reviewer_session.read_headers).json()
    assert view["turns"][3]["text"] == "Yes, lovely, and ask for dana at the [REDACTED]."
