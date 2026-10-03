"""The card read-out window on reviewer reads (team decision 22, privacy-gap fixes 1 and 2).

Digits the ASR splits or drops cannot be masked by value: a CVV transcribed "seven" / "Two four"
is under the value rules' 3-digit floor, and masking "seven" by value would hide every "seven" in
the call. So while a strong read-out window is open (``call1.redaction``: a card-number, expiry or
CVV phrase or a card number, through the turn where the window closes):

1. Store mutes the window's whole time range, gaps between turns and every channel included,
   from a little before the phrase that opens it to a little after its last turn.
2. Digit tokens inside it are masked in place, by position, in those turns only.

The fixtures are Parakeet's output on the AppTek retail card calls (the dataset's own role-play
values; ``sample_audio/apptek_retail``), cut to the read-out and the turns around it.
"""

from __future__ import annotations

import io
import json
import re
import wave
from pathlib import Path

import pytest

from call1.contracts.artifacts import ArtifactKind
from call1.contracts.contents import TranscriptContent, TranscriptTurnContent, VerdictStatus
from call1.contracts.jobs import JobType
from call1.redaction import (READ_OUT_PAD_SECONDS, REDACTED, TurnPositions, extract_sensitive_values, mask_text,
                             mask_text_with_values, read_out_digit_spans, read_out_mute_ranges, read_out_windows)
from call1.store.results import audio as audio_module
from call1.store.results.masking import Masker

from .test_results_harness import accounts, embeddings_for, fq  # noqa: F401

R = REDACTED
REPO = Path(__file__).resolve().parents[2]

# Parakeet turns: (start, end, text). "..." marks a long turn cut to its tail.
PARAKEET = {
    "en_CA_Retail_1587321": [
        (371.09, 385.09, "perfect so I've entered the items sizes colors and express shipping so your total comes to 152 dollars and fifty cents "
                         "and that's including tax and shipping"),
        (386.32, 413.13, "... which would you like to use Visa please here's the carton number um 381 21-6267-5292"),
        (414.41, 417.69, "uh 4682"),
        (419.39, 424.05, "okay got it thank you so I'll put those in"),
        (438.53, 440.69, "perfect thank you Sophie"),
        (441.95, 444.03, "yeah, you're very welcome, Ethan."),
        (444.15, 448.03, "So your order should arrive in one to two business days."),
        (448.31, 454.35, "Um if you have any questions about the order, returns, or anything else, feel free to call us."),
    ],
    "en_US_Southern_Retail_1588126": [
        (388.11, 390.91, "So um, how would you like to pay for that today?"),
        (391.79, 393.87, "Um credit card."),
        (394.27, 398.83, "All right, and uh when you're uh when you've got the number, go ahead."),
        (399.79, 400.11, "Um"),
        (401.69, 414.55, "4096-5053-2084-1243."),
        (415.35, 417.75, "Okay, and the expiration date?"),
        (419.50, 423.50, "Um 1027 and the CBB."),
        (425.90, 427.10, "911."),
        (428.70, 429.58, "That's funny."),
        (429.82, 430.06, "Um"),
        (431.82, 434.14, "I'm sorry, isn't that weird?"),
        (434.30, 439.90, "I know, it's like help, help my my credit card is is is is maxed out."),
        (440.74, 441.94, "All right, let's see."),
        (442.18, 443.86, "Okay, now there we go."),
        (444.02, 445.06, "That's gone through"),
        (457.38, 460.02, "Is there anything else I can do for you today?"),
    ],
    "en_GB_Retail_1585501": [
        (464.99, 469.31, "Um can I pay by card then over the phone?"),
        (469.90, 471.10, "Yes, you can."),
        (471.26, 475.90, "So could you um read the long number off your card, please?"),
        (476.22, 476.86, "Just a second."),
        (477.10, 479.98, "Just just let me just get it in front of me."),
        (482.06, 482.62, "Right."),
        (482.78, 483.90, "So it's um"),
        (485.02, 488.86, "it's 5620."),
        (490.06, 492.38, "5620, yeah."),
        (492.78, 497.02, "9746."),
        (500.46, 504.86, "2035."),
        (505.82, 508.78, "2035 and the expiry days."),
        (509.18, 511.82, "The expiry date is 1227."),
        (512.70, 517.58, "1227 and the last three numbers on the back of the card."),
        (517.98, 519.10, "365."),
        (520.46, 521.74, "365."),
        (522.14, 525.90, "Okay, so are you happy for me to to put that through?"),
        (526.30, 527.50, "Yes, do you need anything else?"),
        (527.58, 528.94, "Do you need my phone number?"),
        (529.42, 532.94, "I I well, I do I need a couple of things actually."),
        (533.18, 538.70, "Um your address, because I don't think you give me your address and your email address."),
        (560.00, 562.00, "That comes to eighty pounds fifty pence."),
    ],
    "en_US_Southern_Retail_1588849": [
        (374.45, 374.77, "Okay."),
        (375.57, 382.77, "And if you could please provide your credit card information so I can have that purchased."),
        (385.01, 385.57, "Sure."),
        (385.81, 387.25, "One second here."),
        (389.81, 390.45, "It is"),
        (393.57, 396.21, "5790."),
        (406.64, 406.80, "Okay."),
        (407.84, 410.00, "And expiration date?"),
        (411.68, 414.24, "Zero seven twenty-nine."),
        (416.64, 416.96, "Okay."),
        (417.28, 420.24, "And this the three-digit security code."),
        # The CVV was spoken here and never transcribed: 420.24 to 429.55 has no turn.
        (429.55, 432.27, "And the name on the purchase?"),
        (433.78, 435.30, "David Aguilar."),
        (438.67, 438.99, "Okay."),
        (440.27, 442.11, "Let me just read that back to you."),
        (442.19, 453.79, "Uh, card number is 579057010102020."),
        (454.97, 462.73, "Expiration date seven twenty-nine uh security code 361 under David Aguilar."),
        (462.81, 464.01, "Is that correct?"),
        (464.68, 465.64, "That's right."),
        (465.96, 469.72, "Okay, and it's going to be a total of eighty-five dollars."),
        (470.92, 475.96, "Size six, Jordan's black and white."),
        (494.90, 498.50, "And three business day delivery."),
        (500.25, 501.29, "That's right."),
        (501.93, 502.25, "Okay."),
        (503.05, 504.81, "Well, thank you very much, sir."),
        (504.97, 508.17, "That information has been confirmed."),
        (508.25, 514.33, "And if you require any further information, please feel free to call us."),
    ],
    "en_IE_Retail_1586857": [
        (240.82, 286.90, "... so just for pre-order can I get your um your card digits please oh sure um the oh sorry let me just get it out"),
        (290.17, 295.93, "um oh 9098282"),
        (297.29, 300.17, "forty six forty six"),
        (301.33, 312.25, "thirty eight thirty 8090972 seventy two"),
        (313.69, 319.77, "forty three forty three sixty 767 lovely."),
        (319.97, 326.33, "And can I just get your CVV code too, please, mate sure thing that is uh seven."),
        (326.69, 327.29, "Seven."),
        (329.17, 330.17, "Two four."),
        (330.45, 331.85, "2424."),
        (332.05, 332.69, "Lovely."),
        (333.09, 333.41, "Okay."),
        (335.57, 337.81, "I've just gone ahead and charge that for you there."),
        (341.09, 341.49, "Right."),
        (341.57, 344.21, "So that's been pre-ordered for you there."),
        (344.37, 349.73, "Uh, you should get a text message like saying your item is ready for collection."),
        (349.81, 351.57, "Hopefully, November 11th."),
        (351.73, 356.61, "If by like November 13th you have got the text message, just give us a call back."),
        (356.85, 359.09, "We'll be happy to chase it up for you on the store."),
        (359.65, 359.73, "Okay."),
        (360.45, 360.77, "Okay."),
        (361.65, 364.77, "And it's just to check with um"),
        (370.00, 372.00, "It's seven days to return it, and seven more to swap it."),
    ],
}

# Per call: the windows (first turn, last turn), and the turns whose masked text differs from the
# original (index -> masked text). Every other turn, prices and dates included, reads as spoken.
EXPECTED = {
    "en_CA_Retail_1587321": ([(1, 6)], {
        1: f"... which would you like to use Visa please here's the carton number um {R}",
        2: f"uh {R}",
    }),
    "en_US_Southern_Retail_1588126": ([(4, 14)], {
        4: f"{R}.",
        6: f"Um {R} and the CBB.",
        7: f"{R}.",
    }),
    "en_GB_Retail_1585501": ([(2, 20)], {
        7: f"it's {R}.", 8: f"{R}, yeah.", 9: f"{R}.", 10: f"{R}.", 11: f"{R} and the expiry days.",
        12: f"The expiry date is {R}.", 13: f"{R} and the last three numbers on the back of the card.", 14: f"{R}.", 15: f"{R}.",
    }),
    "en_US_Southern_Retail_1588849": ([(1, 20)], {  # "credit card information" opens it
        5: f"{R}.",
        8: f"{R}.",
        15: f"Uh, card number is {R}.",
        16: f"Expiration date {R} uh security code {R} under David Aguilar.",
    }),
    "en_IE_Retail_1586857": ([(0, 19)], {  # closes 30 s after "2424"
        1: f"um {R}", 2: R, 3: R, 4: f"{R} lovely.",
        5: f"And can I just get your CVV code too, please, mate sure thing that is uh {R}.",
        6: f"{R}.", 7: f"{R}.", 8: f"{R}.",
    }),
}


def _turns(call):
    return [{"turn_id": i, "start_time": s, "end_time": e, "text": text} for i, (s, e, text) in enumerate(PARAKEET[call])]


def _masked(turns):
    values = extract_sensitive_values(turns, True)
    positions = TurnPositions([t["text"] for t in turns], read_out_digit_spans(turns))
    return [mask_text(t["text"], values, positions) for t in turns]


@pytest.mark.parametrize("call", sorted(PARAKEET))
def test_parakeet_read_outs_are_masked_in_place_and_nothing_else(call):
    turns = _turns(call)
    windows, changed = EXPECTED[call]
    assert [(w.first_turn, w.last_turn) for w in read_out_windows(turns)] == windows
    masked = _masked(turns)
    assert {i: m for i, m in enumerate(masked) if m != turns[i]["text"]} == changed


@pytest.mark.parametrize("call", sorted(PARAKEET))
def test_the_whole_window_is_muted_with_a_little_either_side(call):
    turns = _turns(call)
    ranges = read_out_mute_ranges(turns)
    windows = read_out_windows(turns)
    assert len(ranges) == len(windows) == 1
    (start, end), window = ranges[0], windows[0]
    first, last = turns[window.first_turn], turns[window.last_turn]
    assert start == pytest.approx(max(0.0, first["start_time"] - READ_OUT_PAD_SECONDS))
    assert end == pytest.approx(max(t["end_time"] for t in turns[window.first_turn:window.last_turn + 1]) + READ_OUT_PAD_SECONDS)
    # Every gap between the window's turns is inside the range: nothing spoken there is audible.
    for a, b in zip(turns[window.first_turn:window.last_turn], turns[window.first_turn + 1:window.last_turn + 1]):
        assert start <= a["end_time"] and b["start_time"] <= end
    assert last["end_time"] < end


def test_an_untranscribed_cvv_is_muted():
    turns = _turns("en_US_Southern_Retail_1588849")
    (start, end), = read_out_mute_ranges(turns)
    # "And this the three-digit security code." ends at 420.24; the next turn starts at 429.55.
    assert start < 420.24 and 429.55 < end


def test_a_single_seven_elsewhere_stays_visible():
    turns = _turns("en_IE_Retail_1586857")
    masked = _masked(turns)
    assert masked[6] == f"{R}." and masked[5].endswith(f"uh {R}.")
    # After the window: "seven" is never masked by value.
    assert masked[-1] == "It's seven days to return it, and seven more to swap it."
    assert "seven" not in extract_sensitive_values(turns, True) and "Seven" not in extract_sensitive_values(turns, True)


def test_the_window_opens_at_the_phrase_word_in_a_long_turn():
    text = "so just for pre-order can I get your um your card digits please oh sure let me get it out"
    words, t = [], 240.82
    for w in text.split():
        words.append({"word": " " + w, "start_time": round(t, 2), "end_time": round(t + 0.3, 2), "probability": 1.0})
        t += 0.5
    turns = [{"turn_id": 0, "start_time": 240.82, "end_time": 286.9, "text": text, "word_timestamps": words},
             {"turn_id": 1, "start_time": 290.17, "end_time": 295.93, "text": "um oh 9098282"}]
    (window,) = read_out_windows(turns)
    card = next(w for w in words if w["word"] == " card")
    assert window.start_time == card["start_time"] and window.opens_at == text.index("digits") + len("digits")
    (start, end), = read_out_mute_ranges(turns)
    assert start == pytest.approx(card["start_time"] - READ_OUT_PAD_SECONDS) and end == pytest.approx(295.93 + READ_OUT_PAD_SECONDS)
    # Digits spoken before the phrase are not the read-out.
    assert read_out_digit_spans([{"text": "I'll take two of them. What's the card number?"}]) == [[]]


# (text, positional masks inside an open window): what is and is not a read-out digit.
WINDOW_TOKENS = [
    ("Seven.", f"{R}."),
    ("Two four.", f"{R}."),
    ("uh seven.", f"uh {R}."),
    ("oh nine nine", R),
    ("Ohh, okay.", "Ohh, okay."),
    ("Four, eight, six, Ohh, I can see", f"{R}, Ohh, I can see"),
    ("double five", R),
    ("twelve twenty seven", R),
    ("That's $22.99 then.", "That's $22.99 then."),
    ("a total of eighty-five dollars.", "a total of eighty-five dollars."),
    ("eighty pounds fifty pence.", "eighty pounds fifty pence."),
    ("three hundred and fifty", "three hundred and fifty"),
    ("in one to two business days.", "in one to two business days."),
    ("five business days", "five business days"),
    ("Size six, Jordans.", "Size six, Jordans."),
    ("March twelve works.", "March twelve works."),
    ("the three digits on the back", "the three digits on the back"),
    ("the last four please", "the last four please"),
    ("the latter one?", "the latter one?"),
    ("one of them", "one of them"),
    ("rated at negative forty.", "rated at negative forty."),
    ("In 2024.", f"In {R}."),  # a four-digit numeral inside a read-out is not a year (an expiry "2027" reads the same)
]


@pytest.mark.parametrize("text, want", WINDOW_TOKENS)
def test_digit_tokens_inside_a_window(text, want):
    turns = [{"text": "And the security code?"}, {"text": text}]
    spans = read_out_digit_spans(turns)
    assert spans[0] == []
    assert mask_text(text, set(), TurnPositions([t["text"] for t in turns], spans)) == want


def test_derived_text_is_masked_where_it_repeats_the_turn():
    turns = _turns("en_IE_Retail_1586857")
    values = extract_sensitive_values(turns, True)
    positions = TurnPositions([t["text"] for t in turns], read_out_digit_spans(turns))
    masker = Masker(values, True, positions=positions)
    # A verbatim quote of a window turn, and a paraphrase that repeats its words around the digit.
    assert masker.text("sure thing that is uh seven") == f"sure thing that is uh {R}"
    assert masker.text("The caller said: that is uh seven.") == f"The caller said: that is uh {R}."
    # The bare digit elsewhere, and the turn after the window, stay visible.
    assert masker.text("Returns take seven days.") == "Returns take seven days."
    assert masker.text(turns[-1]["text"]) == turns[-1]["text"]
    # Word timestamps follow the text.
    words = [{"word": " Two", "start_time": 329.2, "end_time": 329.6, "probability": 1.0},
             {"word": " four.", "start_time": 329.7, "end_time": 330.1, "probability": 1.0}]
    assert [w["word"] for w in masker.words(words, "Two four.")] == [R, R]


# --- false positives across every sample manifest -------------------------------------------------

MANIFESTS = ["sample_audio/apptek_retail/manifest.json", "sample_audio/telephony_training_set/manifest.json", "sample_audio/manifest.json"]
_MONTHS = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
BUSINESS = re.compile(
    r"\$\s?\d[\d,]*(?:\.\d\d)?|\b\d[\d,]*(?:\.\d\d)?\s(?:dollars?|pounds?|euros?|bucks)\b"
    r"|\b(?:[a-z]+-)?[a-z]+\s(?:dollars?|pounds?|euros?)\b"
    r"|\bsize\s+(?:\d+(?:\.\d)?|[a-z]+(?:-[a-z]+)?)\b"
    r"|\b" + _MONTHS + r"\s+(?:\d{1,2}(?:st|nd|rd|th)?|[a-z]+(?:-[a-z]+)?)\b"
    r"|\b(?:19|20)\d\d\b", re.IGNORECASE)


def _manifest_calls():
    for name in MANIFESTS:
        path = REPO / name
        if not path.exists():  # pragma: no cover - the sample set ships with the repo
            continue
        for call_id, call in json.loads(path.read_text()).items():
            if call.get("turns"):
                yield f"{name}:{call_id}", call


def test_prices_sizes_dates_and_years_survive_across_the_sample_sets():
    checked = windows = 0
    for label, call in _manifest_calls():
        turns = call["turns"]
        values = extract_sensitive_values(turns, True)
        spans = read_out_digit_spans(turns)
        positions = TurnPositions([t["text"] for t in turns], spans)
        inside = {i for w in read_out_windows(turns) for i in range(w.first_turn, w.last_turn + 1)}
        windows += bool(inside)
        for index, turn in enumerate(turns):
            text = turn["text"]
            masked = mask_text(text, values, positions)
            if index not in inside:
                # Outside a window nothing is masked by position: the value rules alone decide.
                assert spans[index] == [] and masked == mask_text_with_values(text, values), (label, index)
            for m in BUSINESS.finditer(text):
                checked += 1
                assert m.group(0) in masked, (label, index, m.group(0), masked)
    assert checked > 150 and windows >= 5


def test_windows_mute_a_small_share_of_the_sample_audio():
    muted = total = 0.0
    for _label, call in _manifest_calls():
        muted += sum(e - s for s, e in read_out_mute_ranges(call["turns"]))
        total += call["duration_seconds"]
    assert 0 < muted < 0.05 * total


# --- Store: reviewer audio -----------------------------------------------------------------------

RATE = 8000
CALL = [
    # (speaker, channel, start, end, text)
    ("AGENT", 0, 0.2, 1.8, "Could I get the card number please?"),
    ("CALLER", 1, 2.4, 3.6, "Sure, it's four one one one."),
    ("AGENT", 0, 4.0, 4.8, "And the CVV?"),
    # 4.8 to 6.4: the CVV, spoken on the caller channel and never transcribed.
    ("CALLER", 1, 6.4, 7.0, "Seven."),
    ("AGENT", 0, 7.2, 7.8, "Thank you."),
    ("AGENT", 0, 8.0, 8.6, "Okay."),
    ("AGENT", 0, 8.8, 9.4, "Okay."),
    ("AGENT", 0, 9.6, 10.2, "Okay."),
    ("CALLER", 1, 40.0, 41.0, "It was seven days ago."),
]


def _stereo(seconds: float) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(b"\x10\x27\x20\x4e" * int(seconds * RATE))
    return buf.getvalue()


def _frames_at(data: bytes, t: float) -> tuple:
    with wave.open(io.BytesIO(data)) as w:
        frames = w.readframes(w.getnframes())
    at = int(t * RATE) * 4
    return frames[at:at + 2], frames[at + 2:at + 4]


def _call(fq):
    conv = fq.register()
    graph = fq.graph(conv, [JobType.ASR, JobType.EMBEDDINGS])
    content = TranscriptContent(duration_seconds=42.0, language="en", is_redacted=False, turns=[
        TranscriptTurnContent(turn_id=i, speaker=speaker, channel=channel, start_time=start, end_time=end, text=text)
        for i, (speaker, channel, start, end, text) in enumerate(CALL)])
    fq.complete(conv, graph, JobType.ASR, {"transcript": content})
    fq.complete(conv, graph, JobType.EMBEDDINGS, {"embeddings": embeddings_for(content)})
    fq.artifact(conv, ArtifactKind.SOURCE_AUDIO, _stereo(42.0), content_type="audio/wav")
    return conv


def test_store_mutes_the_window_on_both_channels_including_the_untranscribed_gap(store, fq, client, reviewer_session):
    conv = _call(fq)
    with store.connection() as conn:
        intervals = audio_module.mute_intervals(conn, conv.id)
    assert intervals[0][0] <= 0.2 and intervals[0][1] >= 10.2
    response = client.get(f"/store/v1/calls/{conv.call_id}/audio", headers=reviewer_session.read_headers)
    assert response.status_code == 200
    for t in (1.0, 2.1, 5.5, 6.6, 9.0):  # a turn, a gap, the untranscribed CVV, "Seven.", the window's tail
        assert _frames_at(response.content, t) == (b"\x00\x00", b"\x00\x00"), t
    for t in (20.0, 40.5):  # after the window, both channels are audible, "seven days ago" included
        assert _frames_at(response.content, t) == (b"\x10\x27", b"\x20\x4e"), t


def test_store_masks_window_digits_by_position_on_the_transcript(store, fq, client, reviewer_session):
    conv = _call(fq)
    view = client.get(f"/store/v1/calls/{conv.call_id}/transcript", headers=reviewer_session.read_headers).json()
    texts = [t["text"] for t in view["turns"]]
    assert texts[1] == f"Sure, it's {R}." and texts[3] == f"{R}."
    assert texts[-1] == "It was seven days ago."
    fq.ingest_qa(conv, [("PAY-01", VerdictStatus.PASS, 0.9)], evidence="Seven.")
    evaluation = client.get(f"/store/v1/calls/{conv.call_id}/evaluation", headers=reviewer_session.read_headers).json()
    assert evaluation["verdicts"][0]["quoted_evidence"] == f"{R}."
