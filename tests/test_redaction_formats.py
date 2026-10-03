"""PII redaction on its own, for every way Parakeet (or any ASR) may write a spoken number.

The MLX adapter rewrites Parakeet's number read-outs into Whisper-style digits, but redaction
must not depend on that rewrite: these tests feed the raw formats straight to ``call1.redaction``.
"""
from types import SimpleNamespace

import pytest

from call1.redaction import (REDACTED, RedactionService, extract_sensitive_values, find_pii,
                             mask_text_with_values, mask_word_timestamps)


def masked(text, context=""):
    values = {value for _, _, _, value in find_pii(text, context)}
    return mask_text_with_values(text, values)


# (text, the sensitive value that must be masked, label)
POSITIVES = [
    # Whisper-style, as before
    ("My SSN is 442-89-1099.", "442-89-1099", "SSN"),
    ("card 4111-2222-3333-4444 please", "4111-2222-3333-4444", "CARD"),
    ("card 4111 2222 3333 4444 please", "4111 2222 3333 4444", "CARD"),
    ("phone 480-555-0199.", "480-555-0199", "PHONE"),
    ("call (800) 555-0199 today", "800) 555-0199", "PHONE"),
    ("call +1 800-555-0199 today", "1 800-555-0199", "PHONE"),
    ("Account 884-210-993, thanks", "884-210-993", "ACCOUNT"),
    ("My PIN is 4492.", "4492", "PIN"),
    ("the security code 4492 again", "4492", "PIN"),
    # Parakeet comma groups
    ("Yes, it is 4111, 2222, 3333, 4444.", "4111, 2222, 3333, 4444", "CARD"),
    ("account 902, 114, 883.", "902, 114, 883", "ACCOUNT"),
    ("it's 442, 89, 1099 right", "442, 89, 1099", "SSN"),
    # misgrouped and mixed separators
    ("SSN 442-891099 ok", "442-891099", "SSN"),
    ("it is 4111 2222-3333, 4444.", "4111 2222-3333, 4444", "CARD"),
    ("reach me at 480.555.0199.", "480.555.0199", "PHONE"),
    ("it is 4111.2222.3333.4444", "4111.2222.3333.4444", "CARD"),
    ("my social is 442,891,099", "442,891,099", "SSN"),
    ("the number 4805550199 works", "4805550199", "PHONE"),
    # spelled digits, alone and mixed with numerals
    ("SSN is four four two eight nine one zero nine nine, and", "four four two eight nine one zero nine nine", "SSN"),
    ("phone four eight zero, five five five, zero one nine nine.", "four eight zero, five five five, zero one nine nine", "PHONE"),
    ("it's four eight oh five five five oh one nine nine", "four eight oh five five five oh one nine nine", "PHONE"),
    ("it's 480 five five five 0199", "480 five five five 0199", "PHONE"),
    ("double five five one two three four five six seven", "double five five one two three four five six seven", "PHONE"),
    ("savings account four six five eight seven three two two", "four six five eight seven three two two", "ACCOUNT"),
    ("policy number is um one, two, three, four, five, six, seven.", "one, two, three, four, five, six, seven", "ACCOUNT"),
    ("Four six five eight.", "Four six five eight", "DIGITS"),
    ("for six five eight seven three two.", "for six five eight seven three two", "DIGITS"),
    # identifier context allows shorter numbers
    ("savings account 4658732 to checking", "4658732", "ACCOUNT"),
    ("account number May 4658732", "4658732", "ACCOUNT"),
    ("checking account number 564 36382.", "564 36382", "ACCOUNT"),
    # currency-formatted and spelled codes, only in code context
    ("Account number is 902-114-883, pin is $4,492.", "$4,492", "PIN"),
    ("my passcode is 4,492", "4,492", "PIN"),
    ("my pin is four thousand four hundred and ninety-two", "four thousand four hundred and ninety-two", "PIN"),
    ("the PIN is forty four ninety two", "forty four ninety two", "PIN"),
    ("verification code: 902 114.", "902 114", "PIN"),
    ("My PIN, it's 4492", "4492", "PIN"),
    ("the cvv is 123", "123", "PIN"),
    ("the last four are 4444", "4444", "PIN"),
    # disfluencies inside a run: fillers, "ohh" read as zero, truncated fragments, false starts
    ("Here's the card number. (Um) three ohh one twenty-one sixty-two sixty-seven fifty-two ninety-two",
     "three ohh one twenty-one sixty-two sixty-seven fifty-two ninety-two", "CARD"),
    ("Four zero nine six five zero five three two zero e~ four one two four three.",
     "Four zero nine six five zero five three two zero e~ four one two four three", "CARD"),
    ("card number is four one one one, um, two two two two, uh, three three three three, four four four four",
     "four one one one, um, two two two two, uh, three three three three, four four four four", "CARD"),
    # expiry and CVV right after their phrase
    ("Expiration date, seven twenty-nine.", "seven twenty-nine", "PAYMENT"),
    ("The expiry date is twelve twenty seven.", "twelve twenty seven", "PAYMENT"),
    ("the expiry is 07/29 thanks", "07/29", "PAYMENT"),
    ("the three digits on the back are, um, three, three six one", "three, three six one", "PAYMENT"),
]


@pytest.mark.parametrize("text, value, label", POSITIVES)
def test_every_format_is_detected_and_masked(text, value, label):
    found = [(lab, val) for lab, _, _, val in find_pii(text)]
    assert (label, value) in found
    assert value not in masked(text)
    assert REDACTED in masked(text)


# (text, previous turn, value)
CROSS_TURN = [
    ("It's 4492.", "What is your PIN?", "4492"),
    ("It is 564 3632.", "Could you give me the account number please?", "564 3632"),
    ("Sure, $4,492.", "And the security code?", "$4,492"),
]


@pytest.mark.parametrize("text, previous, value", CROSS_TURN)
def test_context_from_the_previous_turn(text, previous, value):
    assert value in {v for _, _, _, v in find_pii(text, previous)}
    values = extract_sensitive_values([{"text": previous}, {"text": text}], True)
    assert value in values


# Business data that must survive: money, years, dates, times, quantities, model names.
NEGATIVES = [
    "The $2,500 deposit cleared on May 12, 1984.",
    "Alright Daniel, you have got $14,250 in your savings account.",
    "pay $14,250 or 5,000 now",
    "costs $4,492.",
    "In 2024 2025 2026 we grew every year.",
    "January 5th, 1983.",
    "born 12 May 1984",
    "How does just after lunch like 1245 tomorrow sound?",
    "It's $399 a month.",
    "we lock in that 7999 rate for a full year",
    "the new rate of 79.99 applies",
    "your total is 154.85.",
    "your account balance is $12,500.50",
    "your dwelling coverage will increase to 425,000.",
    "the 1,250,000 figure",
    "we paid 250, 300, 450 dollars",
    "transfers take two to three business days",
    "five to seven business days",
    "one or two late payments",
    "four hundred eighty-two sixty two has been transferred",
    "It's a BrightLink AX 1200.",
    "the horizon X15 and 4K streaming",
    "we have a 24-7 claims line",
    "unplug it for 30 seconds",
    "between 5 and 9 p.m.",
    "oh one two three",
    "the 500 MBPS plan for 7999 a month",
    "under 2500 or 25,000",
    "around a 10% increase at most",
    "I opened the account on May 12, 2019.",
    "the account was closed in March 2021",
    # "Ohh" after a read-out is an exclamation, not a zero
    "Four, eight, six, Ohh, I can see that you have a fifteen pound voucher on here",
    "Ohh one two three",
    # a payment method named, then a spoken price
    "I'll pay by credit card, so that comes to twenty-two ninety-nine.",
    "the link will expire in seventy-two hours",
]


@pytest.mark.parametrize("text", NEGATIVES)
def test_business_numbers_are_not_masked(text):
    assert find_pii(text) == []
    assert masked(text) == text


def test_money_under_account_context_is_not_masked():
    assert find_pii("the account has $60,000 in it") == []


def timed(text):
    """Parakeet-style word timestamps: each word carries its leading space and punctuation."""
    words, t = [], 0.0
    for w in text.split():
        words.append({"word": " " + w, "start_time": t, "end_time": t + 0.4})
        t += 0.5
    return words


@pytest.mark.parametrize("text, kept", [
    ("my social is four four two eight nine one zero nine nine, thanks", ["my", "social", "is", "thanks"]),
    ("it is 4111, 2222, 3333, 4444.", ["it", "is"]),
    ("pin is $4,492.", ["pin", "is"]),
    ("SSN 442-89-1099, and", ["SSN", "and"]),
])
def test_multi_token_values_mask_every_timed_word(text, kept):
    words = timed(text)
    values = extract_sensitive_values([{"text": text}], True)
    out = mask_word_timestamps(words, text, values)
    assert [w["word"].strip() for w in out if w["word"] != REDACTED] == kept
    assert [w["start_time"] for w in out] == [w["start_time"] for w in words]


def test_audio_mute_covers_a_spelled_read_out_word_by_word():
    text = "my social is four four two eight nine one zero nine nine, thanks"
    words = timed(text)
    turn = SimpleNamespace(text=text, start_time=0.0, end_time=10.0, numeric_entities=[], word_timestamps=words)
    settings = SimpleNamespace(redaction=SimpleNamespace(text=True, audio=True, pii_patterns=True))
    intervals = RedactionService(settings).audio_mute_intervals([turn])
    first = next(w for w in words if w["word"] == " four")
    last = next(w for w in words if w["word"] == " nine,")
    assert len(intervals) == 1
    start, end = intervals[0]
    assert start <= first["start_time"] and end >= last["end_time"]
    # The words either side stay audible.
    assert start > words[2]["start_time"] and end < words[-1]["end_time"]


def test_metadata_phone_is_masked_with_field_context():
    settings = SimpleNamespace(redaction=SimpleNamespace(text=True, audio=True, pii_patterns=True))
    bundle = {"turns": [], "verdicts": [], "call": {"customer_phone": "555 0199 123", "agent_id": "AGT-1042"}}
    out = RedactionService(settings).mask_bundle(bundle)
    assert out["call"]["customer_phone"] == REDACTED
    assert out["call"]["agent_id"] == "AGT-1042"


def test_spelled_amounts_survive_identifier_context():
    assert find_pii("your policy covers your home for three hundred and fifty thousand") == []
    assert find_pii("the account has two thousand five hundred in it") == []


# Card read-outs across turns, from the AppTek retail set (sample_audio/apptek_retail) plus
# Parakeet-style equivalents (fillers and repeats, no "~" markup; numerals). Each case is the
# turns in order and the masked text expected for each: the card number, expiry and CVV are masked
# in the caller's turns and the agent's read-back, and the business numbers around them survive.
R = REDACTED
READOUTS = {
    "en_CA_Retail_1587321": [
        ("Visa please.", "Visa please."),
        ("Here's the card number. (Um) three ohh one twenty-one sixty-two sixty-seven fifty-two ninety-two",
         f"Here's the card number. (Um) {R}"),
        ("(uh) forty-six eighty-two.", f"(uh) {R}."),
        ("Ohh. Okay. Got it. Thank you.", "Ohh. Okay. Got it. Thank you."),
    ],
    "en_US_Southern_Retail_1588126": [
        ("(um) Credit card. (um)", "(um) Credit card. (um)"),
        ("Four zero nine six five zero five three two zero e~ four one two four three.", f"{R}."),
        ("Okay. Okay, and the expiration date? And the CVV.", "Okay. Okay, and the expiration date? And the CVV."),
        ("(um) Ten twenty-seven nine one one.", f"(um) {R}."),
    ],
    "en_GB_Retail_1585501": [
        ("Yes, yo u can. So, could you (um) read the long number of your card, please?",
         "Yes, yo u can. So, could you (um) read the long number of your card, please?"),
        ("Just a second. Just just let me get it in front of me.", "Just a second. Just just let me get it in front of me."),
        ("Mhm.", "Mhm."),
        ("Right. so, it's (um.) It's five six two zero.", f"Right. so, it's (um.) It's {R}."),
        ("Five six two zero, yeah.", f"{R}, yeah."),
        ("Two zero three five.", f"{R}."),
        ("Two zero three five. And the expiry date?", f"{R}. And the expiry date?"),
        ("The expiry date is twelve twenty seven.", f"The expiry date is {R}."),
        ("Twelve twenty seven. And the last three numbers on the back of the card?",
         f"{R}. And the last three numbers on the back of the card?"),
        ("It's three six five.", f"It's {R}."),
        ("Three six five. Okay. So, you're happy for me to, to put that through?",
         f"{R}. Okay. So, you're happy for me to, to put that through?"),
    ],
    "en_US_Southern_Retail_1588849": [
        ("Four one . Okay, and if you could please provide your credit card information so I can have that purchased.",
         "Four one . Okay, and if you could please provide your credit card information so I can have that purchased."),
        ("Sure. One second, here. It is five seven nine zero.", f"Sure. One second, here. It is {R}."),
        ("Five seven zero one one zero one zero two zero two zero.", f"{R}."),
        ("Okay, and expiration date? Okay, and the s~ (uh) three-digit security code?",
         "Okay, and expiration date? Okay, and the s~ (uh) three-digit security code?"),
        ("Zero seven twenty-nine. Y~ Three S~ Three six one.", f"{R}. Y~ {R}."),
        ("Three six one. And the name on the purchase?", f"{R}. And the name on the purchase?"),
        ("Okay. So let me just read that back to you. (uh) Card number is five seven nine zero five seven zero one.",
         f"Okay. So let me just read that back to you. (uh) Card number is {R}."),
        ("One zero one zero two zero two zero. Expiration date, seven twenty-nine. (uh) Security code, three six one.",
         f"{R}. Expiration date, {R}. (uh) Security code, {R}."),
        ("Yes. That's right.", "Yes. That's right."),
        ("Okay, and it's gonna be a total of eighty-five dollars. Size six Jordans, black and white.",
         "Okay, and it's gonna be a total of eighty-five dollars. Size six Jordans, black and white."),
    ],
    "en_IE_Retail_1586857": [
        ("Preorder well, so just for preorder, can I get your (um) your card digits please?",
         "Preorder well, so just for preorder, can I get your (um) your card digits please?"),
        ("Ohh sure (um) the", "Ohh sure (um) the"),
        ("And the sorry, let me just get it out.", "And the sorry, let me just get it out."),
        ("(um) O nine eight two forty-six thirty-eight zero nine seven two forty-three sixty-seven.", f"(um) {R}."),
        ("O nine eight two, forty-six, thirty-eight, zero zero nine, seventy-two, forty-three, six seven.", f"{R}."),
        ("Lovely, and can I just get your CVV code here please mate?",
         "Lovely, and can I just get your CVV code here please mate?"),
        ("Sure thing, that is (uh) seven two four.", f"Sure thing, that is (uh) {R}."),
        ("Seven two four two four, lovely", f"{R}, lovely"),
    ],
    # Parakeet-style: fillers without brackets, a repeated digit, no "~" fragments.
    "parakeet_spelled": [
        ("Can I get the card number, please?", "Can I get the card number, please?"),
        ("Yeah, um, it's four one one one, two two two two,", f"Yeah, um, it's {R},"),
        ("uh, three three three three, four four four four.", f"uh, {R}."),
        ("And the expiry?", "And the expiry?"),
        ("Oh seven, uh, twenty-nine.", f"{R}."),
        ("And the three digits on the back?", "And the three digits on the back?"),
        ("Three, three six one.", f"{R}."),
        ("Great, your total is twenty-two dollars and it ships in five business days.",
         "Great, your total is twenty-two dollars and it ships in five business days."),
    ],
    # Parakeet after the MLX adapter's numeral rewrite.
    "parakeet_numerals": [
        ("What's the card number?", "What's the card number?"),
        ("It's 4111 2222 3333 4444.", f"It's {R}."),
        ("And the expiration date?", "And the expiration date?"),
        ("12/27.", f"{R}."),
        ("And the CVV?", "And the CVV?"),
        ("Um, 365.", f"Um, {R}."),
        ("Your total is $22.99, delivered in 3 days.", "Your total is $22.99, delivered in 3 days."),
    ],
}


@pytest.mark.parametrize("call", sorted(READOUTS))
def test_card_read_outs_are_masked_across_turns(call):
    turns = READOUTS[call]
    values = extract_sensitive_values([{"text": text} for text, _ in turns], True)
    assert [mask_text_with_values(text, values) for text, _ in turns] == [want for _, want in turns]


def test_a_payment_mention_alone_does_not_mask_prices():
    turns = [{"text": "It'll go back onto the credit card you used."},
             {"text": "Okay, that was twenty-two ninety-nine, right?"},
             {"text": "Yes, and it takes three to five business days."}]
    assert extract_sensitive_values(turns, True) == set()


def test_the_read_out_window_closes():
    turns = [{"text": "And the security code?"}, {"text": "Three six one."}]
    turns += [{"text": "Okay."}] * 4 + [{"text": "That comes to twenty-two ninety-nine."}]
    assert extract_sensitive_values(turns, True) == {"Three six one"}


def test_disfluent_cvv_mutes_every_timed_word():
    context = "Okay, and expiration date? Okay, and the s~ (uh) three-digit security code?"
    text = "Zero seven twenty-nine. Y~ Three S~ Three six one."
    words = timed(text)
    values = extract_sensitive_values([{"text": context}, {"text": text}], True)
    out = mask_word_timestamps(words, text, values)
    assert [w["word"].strip() for w in out if w["word"] != REDACTED] == ["Y~"]


# --- the read-out window by position (team decision 22) -------------------------------------------


@pytest.mark.parametrize("call", sorted(READOUTS))
def test_positional_window_masking_keeps_every_read_out_table_result(call):
    """Masking the window digits by position adds nothing the value rules already cover on these
    read-outs, and never masks the business numbers around them."""
    from call1.redaction import TurnPositions, mask_text, read_out_digit_spans

    turns = [{"text": text} for text, _ in READOUTS[call]]
    values = extract_sensitive_values(turns, True)
    positions = TurnPositions([t["text"] for t in turns], read_out_digit_spans(turns))
    assert [mask_text(t["text"], values, positions) for t in turns] == [want for _, want in READOUTS[call]]


# A CVV the ASR split below the 3-digit floor: (turns, masked). Masked by position in the window only.
SPLIT_READOUTS = [
    ([("And can I just get your CVV code too, please, mate sure thing that is uh seven.",
       f"And can I just get your CVV code too, please, mate sure thing that is uh {R}."),
      ("Seven.", f"{R}."), ("Two four.", f"{R}."), ("2424.", f"{R}."), ("Lovely.", "Lovely.")]),
    ([("And the expiry?", "And the expiry?"), ("Twelve.", f"{R}."), ("Twenty seven.", f"{R}."),
      ("And the three digits on the back?", "And the three digits on the back?"), ("Three six.", f"{R}."), ("Five.", f"{R}."),
      ("That's twenty-two dollars, delivered in five business days.", "That's twenty-two dollars, delivered in five business days.")]),
]


@pytest.mark.parametrize("case", SPLIT_READOUTS)
def test_split_digits_inside_the_window_are_masked_by_position(case):
    from call1.redaction import TurnPositions, mask_text, read_out_digit_spans

    turns = [{"text": text} for text, _ in case]
    values = extract_sensitive_values(turns, True)
    positions = TurnPositions([t["text"] for t in turns], read_out_digit_spans(turns))
    assert [mask_text(t["text"], values, positions) for t in turns] == [want for _, want in case]
    # None of the split digits is a value: a "seven" or "five" elsewhere in the call stays visible.
    assert not {"seven", "Seven", "Two four", "Five", "Twelve"} & values


def test_positional_digits_align_to_word_timestamps_and_audio():
    from call1.redaction import TurnPositions, read_out_digit_spans

    turns = [{"text": "And the security code?", "start_time": 0.0, "end_time": 1.5, "word_timestamps": timed("And the security code?")},
             {"text": "Seven.", "start_time": 4.0, "end_time": 4.4, "word_timestamps": [{"word": " Seven.", "start_time": 4.0, "end_time": 4.4}]},
             *({"text": "Okay.", "start_time": 5.0 + i, "end_time": 5.5 + i, "word_timestamps": timed("Okay.")} for i in range(4)),
             {"text": "It was seven days ago.", "start_time": 60.0, "end_time": 61.0, "word_timestamps": timed("It was seven days ago.")}]
    spans = read_out_digit_spans(turns)
    positions = TurnPositions([t["text"] for t in turns], spans)
    assert [w["word"] for w in mask_word_timestamps(turns[1]["word_timestamps"], "Seven.", set(), positions)] == [REDACTED]
    assert mask_word_timestamps(turns[-1]["word_timestamps"], turns[-1]["text"], set(), positions) == turns[-1]["word_timestamps"]
    settings = SimpleNamespace(redaction=SimpleNamespace(text=True, audio=True, pii_patterns=True))
    service = RedactionService(settings)
    objs = [SimpleNamespace(numeric_entities=[], **t) for t in turns]
    # The legacy app's muting is unchanged: no read-out window, and "Seven." is no value.
    assert service.audio_mute_intervals(objs) == []
    # Store's: the whole window from just before its phrase ("security" is at 1.0 s) through the
    # turn where it closes (the gap and the digit included), and nothing after it.
    intervals = service.audio_mute_intervals(objs, read_out=True)
    assert intervals == [(0.5, 8.5 + 0.5)]
    assert not any(s <= 60.5 <= e for s, e in intervals)


def test_a_positional_span_without_word_times_mutes_its_whole_turn():
    settings = SimpleNamespace(redaction=SimpleNamespace(text=True, audio=True, pii_patterns=True))
    turn = SimpleNamespace(text="it's for dana", start_time=3.0, end_time=5.0, numeric_entities=[], word_timestamps=None)
    assert RedactionService(settings).audio_mute_intervals([turn], extra_spans=[[(9, 13)]]) == [(2.9, 5.1)]
