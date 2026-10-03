"""Redaction service applying settings-driven masking at every API boundary.

Design: for each call, a shared set of *sensitive values* is extracted once
from the transcript (numeric entities of privacy-sensitive types plus PII
regex matches, including PINs). Every occurrence of those values is then masked
across all surfaces: transcript text, raw text, word timestamps, numeric
entities, quoted evidence, reasoning, reviewer notes, search results, and
summary output. Repeats are covered because masking replaces every occurrence,
not just the first.

Supported sensitive classes (documented, not exhaustive; see ``find_pii``):
  - SSN (123-45-6789), phone numbers, card numbers, account numbers: any run of
    9+ digits however it is written (hyphens, spaces, commas, periods, mixed,
    or read out as digit words: "four four two, eight nine, one zero nine nine")
  - Account/policy/phone numbers of 5+ digits after identifier context
  - PINs / security codes: 3-8 digits after pin/passcode/code/cvv context, even
    when the ASR formats them as money ("$4,492") or spells them out
  - Digit-by-digit read-outs of 4+ digit words ("four six five eight")
  - Card read-outs: card-number chunks, expiry and CVV (3+ digits) in the turns
    after a card-number/expiry/CVV phrase, including the agent's read-back

Deliberately NOT masked as PII: currency amounts, dates, durations, and
generic numbers. Those are business data, not privacy identifiers, and masking
them would destroy QA value. This is approximate masking, not certified PII
removal; the UI must not claim compliance-grade scrubbing.

Some spans are masked by position instead of by value (team decision 22): digits inside a card
read-out window that the ASR split below the value rules' floor ("seven" / "Two four"), and the
PII model's weaker findings (``call1.pii_model``). ``TurnPositions`` carries them and ``mask_text``
/ ``mask_word_timestamps`` apply them; Store also mutes each read-out window whole
(``read_out_windows``, ``read_out_mute_ranges``). See the notes above ``READ_OUT_PAD_SECONDS``.

Audio redaction mutes intervals where sensitive values were spoken, using
word-level timestamps when available (accurate) and falling back to
entity-based interpolated intervals otherwise. The original audio file is
never modified.

Settings are snapshotted once per request by the caller (RedactionService is
constructed with an AppSettings instance), so a mid-request settings save can
never produce inconsistent per-field toggles.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from call1.models.schemas import AppSettings, NumericEntityType, TranscriptTurn

REDACTED = "[REDACTED]"

# Entity types treated as privacy-sensitive. Currency/date/duration/generic
# numbers are business data and are intentionally left unmasked.
SENSITIVE_ENTITY_TYPES = {
    NumericEntityType.ACCOUNT_NUMBER,
    NumericEntityType.PHONE_NUMBER,
}

# ---------------------------------------------------------------------------
# PII detection (applied when redaction.pii_patterns is enabled)
#
# Numbers are found as *runs*: numerals and spoken number words joined by
# separators (space, comma, hyphen, dash, parenthesis; a period only inside a
# numeral such as 480.555.0199). A run is normalized to its digit string and
# classified by length and nearby context. This deliberately does not depend on
# the ASR having written numbers Whisper-style: Parakeet writes the same
# read-out as "442-89-1099", "four four two eight nine one zero nine nine",
# "442-891099", "4111, 2222, 3333, 4444" or (for a PIN) "$4,492".
#
# Rules, in order (the matched run text is the masked value):
#   PIN      code context ("pin", "passcode", "security/verification/access
#            code", "code", "cvv", "last four") within a few words before, and
#            3-8 digits. The only rule that accepts money-formatted numbers
#            ("$4,492", "4,492") and spelled amounts ("four thousand ...").
#   ID       9+ digits read as numerals or single digit words (SSN 9, phone
#            10-11, card 13-19, account otherwise). A run of 19xx/20xx years,
#            or one with tens/teens words, needs identifier context; one with
#            "hundred"/"thousand" is an amount and never matches.
#   ACCOUNT  identifier context ("account", "card", "social", "routing",
#            "member", "policy", "phone", ...) and 5+ digits.
#   DIGITS   4+ digits read out as single digit words ("four six five eight"):
#            ordinary QA text does not read numbers digit by digit.
#   PAYMENT  inside a card read-out (see below), any run of 3+ digits that is not
#            money, a scale amount or a quantity ("five business days"): the card
#            number in chunks, the expiry ("twelve twenty seven", "07/29") and the
#            CVV ("three six five").
# Money ($ / thousands commas / decimals / "dollars"), ordinals and numerals
# glued to letters (5th, X15, 4K) are never masked outside a code context.
# Context may come from the end of the previous turn ("What's your PIN?" /
# "It's 4492."), never across a number.
#
# Disfluencies never break a run: fillers ("um", "(uh,)", "ah"), truncated
# fragments ("S~", "e~": AppTek marks them with "~") and a digit repeated after a
# false start ("Three S~ Three six one" is 361). "oh"/"ohh"/"o" read as zero inside
# a run, and are absorbed in front of a masked one ("O nine eight two ..."), but
# never start a run on their own ("oh one two three" is not masked).
#
# Card read-outs: a card-number, expiry or CVV phrase ("card number", "long
# number", "expiry", "expiration", "cvv", "security code", "the three digits on
# the back") opens a read-out window that covers the rest of that turn and the
# following turns of either speaker (so the agent's read-back is masked too). A
# detected card number (13-19 digits) opens it as well, for the expiry and CVV and
# for a trailing chunk after a misheard phrase ("the carton number"). Each turn with
# a masked number keeps it open; it closes after _PAY_WINDOW turns in a row without
# one, once _PAY_WINDOW_SECONDS have also passed (ASR segmentation is finer on some
# engines: "Just a second." / "Right." / "So it's um" / "it's 5620."). A bare
# payment word ("credit card", "visa") opens a weak window in which only a 12+
# digit run is taken as a card number.
# ---------------------------------------------------------------------------

_UNITS = {"zero": 0, "oh": 0, "ohh": 0, "o": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
          "six": 6, "seven": 7, "eight": 8, "nine": 9}
_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
          "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}
_REPEATS = {"double": 2, "triple": 3}
# ASR homophones of a digit, absorbed only in front of a spelled read-out ("for six five eight").
_LEAD_HOMOPHONES = {"for", "to", "too"}
# Spoken zero that never starts a run ("Ohh, okay"), absorbed in front of a masked one.
_ZERO_WORDS = {"oh", "ohh", "o"}
# Hesitation fillers skipped inside a run (Parakeet writes "um,", AppTek "(um,)").
_FILLERS = {"um", "umm", "uh", "uhh", "uhm", "ah", "ahh", "er", "erm", "eh", "hm", "hmm", "mm", "mhm"}
# Items a run skips over: fillers and truncated fragments ("S~").
_SKIP = ("filler", "frag")
# The gap next to a filler or fragment: spaces, a comma, and the brackets of "(um)", "(Um.)", "(uh,)".
_FILLER_GAP = re.compile(r"\s*,?\s*(?:[.,]?\))?\s*,?\s*\(?\s*")

_TOKEN = re.compile(r"(?:[$€£]\s?)?\d+(?:[.,\-–]\d+)*|[A-Za-z]+(?:-[A-Za-z]+)*")
_SEPARATOR = re.compile(r"\s*[,\-–—()]?\s*\)?\s*")
_CURRENCY_AFTER = re.compile(r"\s*(?:dollars?|bucks|cents?|euros?|pounds?|usd)\b", re.IGNORECASE)
_CODE_CONTEXT = re.compile(
    r"\b(?:pins?|pin\s+number|passcode|pass\s+code|password|(?:security|verification|access|"
    r"confirmation)\s+(?:code|pin|number)|codes?|cvv2?|cvc|last\s+(?:four|4)(?:\s+digits)?)\b",
    re.IGNORECASE)
_ID_CONTEXT = re.compile(
    r"\b(?:accounts?|acct|card|credit|debit|social|ssn|routing|member|membership|policy|"
    r"phone|telephone|cell|mobile|callback|license|passport|id|identification)\b",
    re.IGNORECASE)
_MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"}
_SSN_CONTEXT = re.compile(r"\b(?:social|ssn)\b", re.IGNORECASE)
_CONTEXT_WORDS = 5
# A card read-out: card number, expiry or CVV asked for or given.
_PAY_CONTEXT = re.compile(
    r"\b(?:card\s+(?:number|digits|details|info(?:rmation)?|no)|(?:long|full)\s+number|"
    r"numbers?\s+on\s+(?:the|your)\s+card|expiry|expiration|exp\s+date|valid\s+(?:thru|through|until)|"
    r"cvv2?|cvc|csc|security\s+(?:code|digits)|(?:digits|numbers)\s+on\s+the\s+back|"
    r"three[- ]digit\s+code)\b",
    re.IGNORECASE)
# A payment method named: only a long (12+ digit) run after it is taken as a card number.
_PAY_WEAK = re.compile(r"\b(?:credit|debit|visa|master\s?card|amex|american\s+express)\b", re.IGNORECASE)
_PAY_CONTEXT_WORDS = 12
# A read-out window stays open for this many turns, or this many seconds, after its last number.
_PAY_WINDOW = 4
_PAY_WINDOW_SECONDS = 30.0
# A number followed by one of these is a quantity, never part of a card read-out.
_QUANTITY_AFTER = re.compile(
    r"[\s,]*(?:dollars?|bucks|cents?|euros?|pounds?|quid|percent|per\s+cent|%|business|days?|hours?|"
    r"minutes?|seconds?|weeks?|months?|years?|gig|gigs|gigabytes?|gb|items?|pairs?|points?|times?|"
    r"o'?clock|am|pm|a\.m|p\.m)\b",
    re.IGNORECASE)
_EXPIRY_NUMERAL = re.compile(r"(?<![\d/])\d{1,2}\s?/\s?\d{2,4}(?![\d/])")


class _Item:
    __slots__ = ("start", "end", "kind", "digits", "value", "formatted", "word")

    def __init__(self, start, end, kind, digits="", value=0, formatted=False, word=""):
        self.start, self.end, self.kind = start, end, kind
        self.digits, self.value, self.formatted, self.word = digits, value, formatted, word


def _numeral(text: str, m: re.Match) -> Optional[_Item]:
    """A numeral token, or None when it is glued to letters (5th, X15, 4K)."""
    start, end = m.span()
    if (start > 0 and text[start - 1].isalpha()) or (end < len(text) and text[end].isalpha()):
        return None
    raw = m.group(0)
    currency = raw[0] in "$€£" or bool(_CURRENCY_AFTER.match(text, end))
    body = raw.lstrip("$€£ ")
    parts = re.split(r"[.,\-–]", body)
    seps = re.findall(r"[.,\-–]", body)
    digits = "".join(parts)
    decimal = seps.count(".") == 1 and seps[-1] == "." and len(parts[-1]) <= 2
    thousands = bool(seps) and all(s == "," for s in seps) and 1 <= len(parts[0]) <= 3 and all(
        len(p) == 3 for p in parts[1:])
    # "442,891,099" is an SSN written with thousands commas; "1,250,000" is an amount.
    if thousands and not currency and len(digits) >= 9 and not digits.endswith("000"):
        thousands = False
    return _Item(start, end, "num", digits=digits, formatted=currency or decimal or thousands)


def _word(m: re.Match) -> _Item:
    word = m.group(0).lower()
    start, end = m.span()
    if word in _FILLERS:
        return _Item(start, end, "filler", word=word)
    if word in _UNITS:
        return _Item(start, end, "unit", value=_UNITS[word], word=word)
    if word in _TEENS:
        return _Item(start, end, "teen", value=_TEENS[word], word=word)
    head, _, tail = word.partition("-")
    if head in _TENS and (not tail or tail in _UNITS and _UNITS[tail] > 0):
        return _Item(start, end, "tens", value=_TENS[head] + (_UNITS[tail] if tail else 0), word=word)
    if word in _REPEATS:
        return _Item(start, end, "repeat", value=_REPEATS[word], word=word)
    if word in ("hundred", "thousand", "and"):
        return _Item(start, end, word, word=word)
    return _Item(start, end, "other", word=word)


def _items(text: str) -> List[_Item]:
    items = []
    for m in _TOKEN.finditer(text):
        if m.end() < len(text) and text[m.end()] == "~":
            # A truncated fragment ("S~", "e~"): a false start, skipped inside a run.
            items.append(_Item(m.start(), m.end() + 1, "frag", word=m.group(0).lower()))
        elif m.group(0)[0].isalpha():
            items.append(_word(m))
        else:
            item = _numeral(text, m)
            items.append(item if item is not None else _Item(m.start(), m.end(), "other"))
    return items


def _gap_ok(a: _Item, b: _Item, text: str) -> bool:
    """Whether the text between two adjacent items may sit inside a run."""
    gap = text[a.end:b.start]
    if "\n" in gap:
        return False
    pattern = _FILLER_GAP if a.kind in _SKIP or b.kind in _SKIP else _SEPARATOR
    return bool(pattern.fullmatch(gap))


def _joins(prev: _Item, nxt: _Item, after: Optional[_Item]) -> bool:
    """Whether ``nxt`` continues the run that ends with ``prev`` (the gap is checked by ``_gap_ok``)."""
    if prev.formatted or (nxt.kind == "num" and nxt.formatted):
        return False
    if prev.kind == "repeat":
        return nxt.kind == "unit"
    if nxt.kind in ("num", "unit", "teen", "tens", "repeat"):
        return True
    if nxt.kind in ("hundred", "thousand"):
        return prev.kind in ("unit", "teen", "tens")
    if nxt.kind == "and":
        return prev.kind in ("hundred", "thousand") and after is not None and after.kind in ("unit", "teen", "tens")
    return False


def _run_digits(run: List[_Item]) -> Tuple[str, bool]:
    """(digit string, has tens/hundreds words) for a run. Spoken numbers are read the way
    people say them: "forty four ninety two" is 4492, "four thousand four hundred and
    ninety-two" is 4492, "four four" is 44, "double five" is 55."""
    out: List[str] = []
    state = {"total": 0, "current": 0, "last": None}
    magnitude = False

    def flush():
        if state["last"] is not None:
            out.append(str(state["total"] + state["current"]))
        state.update(total=0, current=0, last=None)

    # Drop fillers and fragments, and a digit that is restarted after one: "Three S~ Three six
    # one" and "three, uh, three six one" are 361.
    words: List[_Item] = []
    disfluent = False
    for item in run:
        if item.kind in _SKIP:
            disfluent = True
            continue
        if disfluent and words and item.kind == "unit" and words[-1].kind == "unit" and words[-1].word == item.word:
            words.pop()
        words.append(item)
        disfluent = False

    repeat = 0
    for item in words:
        kind, last = item.kind, state["last"]
        if kind == "num":
            flush()
            out.append(item.digits)
        elif kind == "repeat":
            flush()
            repeat = item.value
        elif kind == "unit":
            if repeat:
                out.append(str(item.value) * repeat)
                repeat = 0
            elif item.value == 0:
                flush()
                out.append("0")
            elif last in ("tens", "hundred", "thousand", "and"):
                state["current"] += item.value
                state["last"] = "unit+"
            else:
                flush()
                state.update(current=item.value, last="unit")
        elif kind in ("teen", "tens"):
            magnitude = True
            if last in ("hundred", "thousand", "and"):
                state["current"] += item.value
            else:
                flush()
                state["current"] = item.value
            state["last"] = "tens" if kind == "tens" and item.value % 10 == 0 else "unit+"
        elif kind == "hundred":
            magnitude = True
            if last is not None and 0 < state["current"] < 100:
                state["current"] *= 100
            else:
                flush()
                state["current"] = 100
            state["last"] = "hundred"
        elif kind == "thousand":
            magnitude = True
            if last is not None and state["current"] > 0:
                state["total"] += state["current"] * 1000
                state["current"] = 0
            else:
                flush()
                state["total"] = 1000
            state["last"] = "thousand"
        elif kind == "and":
            state["last"] = "and"
    flush()
    return "".join(out), magnitude


def _near(context: str, pattern: re.Pattern, words: int = _CONTEXT_WORDS) -> bool:
    """A context phrase ends within a few words before the run, with no number in between."""
    last = None
    for last in pattern.finditer(context):
        pass
    if last is None:
        return False
    tail = context[last.end():]
    return not re.search(r"\d", tail) and len(re.findall(r"[A-Za-z']+", tail)) <= words


def _is_date(run: List[_Item], before: Optional[_Item]) -> bool:
    """A day and year after a month name ("May 12, 2019") or a day-and-year pair: dates are
    business data, even next to an identifier word ("the account opened on May 12, 2019")."""
    groups = [i.digits for i in run if i.kind == "num"]
    if len(groups) != len(run) or not 1 <= len(groups) <= 2:
        return False
    year = groups[-1]
    if len(groups) == 2 and not (len(groups[0]) <= 2 and len(year) == 4 and year[:2] in ("19", "20")):
        return False
    if len(groups) == 1 and len(year) > 4:
        return False
    return before is not None and before.word in _MONTHS


def _classify(run: List[_Item], text: str, context: str, before: Optional[_Item] = None,
              payment: str = "", quantity: bool = False) -> Optional[str]:
    if _is_date(run, before):
        return None
    digits, magnitude = _run_digits(run)
    formatted = any(i.kind == "num" and i.formatted for i in run)
    # "three hundred and fifty thousand" is an amount, not an identifier read-out.
    scale = any(i.kind in ("hundred", "thousand") for i in run)
    spelled_only = all(i.kind in ("unit", "repeat") + _SKIP for i in run)
    groups = [i.digits for i in run if i.kind == "num"]
    years = bool(groups) and len(groups) == len(run) and all(
        len(g) == 4 and g[:2] in ("19", "20") for g in groups)
    code = _near(context, _CODE_CONTEXT)
    ident = _near(context, _ID_CONTEXT)
    n = len(digits)
    if code and 3 <= n <= 8:
        return "PIN"
    if formatted or scale:
        return None
    if payment == "strong" and n >= 3 and not quantity and not 13 <= n <= 19:
        return "PAYMENT"
    if payment and 12 <= n <= 19 and not quantity:
        return "CARD"
    if n >= 9 and (ident or code or not (years or magnitude)):
        if n == 9:
            return "ACCOUNT" if ident and not _near(context, _SSN_CONTEXT) else "SSN"
        if n in (10, 11):
            return "PHONE"
        if 13 <= n <= 19:
            return "CARD"
        return "ACCOUNT"
    if ident and n >= 5:
        return "ACCOUNT"
    if spelled_only and n >= 4:
        return "DIGITS"
    return None


def _starts_run(item: _Item) -> bool:
    return item.kind in ("num", "unit", "teen", "tens", "repeat") and not (
        item.kind == "unit" and item.word in _ZERO_WORDS)


def _payment_level(before: str, payment: str) -> str:
    """The card read-out level for a run: "strong" inside a read-out window or just after a
    card-number/expiry/CVV phrase, "weak" after a payment method, else ""."""
    if payment == "strong" or _near(before, _PAY_CONTEXT, _PAY_CONTEXT_WORDS):
        return "strong"
    if payment or _near(before, _PAY_WEAK, _PAY_CONTEXT_WORDS):
        return "weak"
    return ""


def _runs(items: List[_Item], text: str):
    """Every number run in ``items``: (index of its first item, index of its last joined item, the
    run's items with any trailing "and"/"double"/filler/"ohh" dropped)."""
    i = 0
    while i < len(items):
        if not _starts_run(items[i]):
            i += 1
            continue
        last = i
        while True:
            k = last + 1
            # Skip fillers and fragments ("(um)", "S~") between two number words.
            while k < len(items) and items[k].kind in _SKIP and _gap_ok(items[k - 1], items[k], text):
                k += 1
            if k < len(items) and _gap_ok(items[k - 1], items[k], text) and \
                    _joins(items[last], items[k], items[k + 1] if k + 1 < len(items) else None):
                last = k
            else:
                break
        run = items[i:last + 1]
        # A run never ends on "and"/"double", or on "ohh"/"o" ("Four, eight, six, Ohh, I can see").
        while len(run) > 1 and (run[-1].kind in ("and", "repeat") + _SKIP or run[-1].word in _ZERO_WORDS - {"oh"}):
            run.pop()
        yield i, last, run
        i = last + 1


def _lead_word(items: List[_Item], i: int, text: str) -> Optional[_Item]:
    """The word just before the run starting at ``items[i]`` (fillers and fragments skipped), when
    nothing but a separator stands between them."""
    p = i - 1
    while p >= 0 and items[p].kind in _SKIP and _gap_ok(items[p], items[p + 1], text):
        p -= 1
    if p >= 0 and _gap_ok(items[p], items[p + 1], text):
        return items[p]
    return None


def find_pii(text: str, context: str = "", payment: str = "") -> List[Tuple[str, int, int, str]]:
    """(label, start, end, value) for every sensitive number in ``text``. ``context`` is text
    spoken just before it (the previous turn), used only for context words. ``payment`` is the
    card read-out window this turn falls in ("strong", "weak" or ""; see
    ``extract_sensitive_values``)."""
    if not text:
        return []
    items = _items(text)
    found: List[Tuple[str, int, int, str]] = []
    for i, _last, run in _runs(items, text):
        start, end = run[0].start, run[-1].end
        prefix = (context[-200:] + " " if context else "") + text[:start]
        level = _payment_level(prefix, payment)
        quantity = bool(_QUANTITY_AFTER.match(text, end))
        label = _classify(run, text, prefix, items[i - 1] if i > 0 else None, level, quantity)
        if label:
            lead = _lead_word(items, i, text)
            if lead is not None and (lead.word in _ZERO_WORDS or label == "DIGITS" and lead.word in _LEAD_HOMOPHONES):
                # "O nine eight two ..." / "for six five eight": the lead word is part of the read-out.
                start = lead.start
            found.append((label, start, end, text[start:end]))
    # Numeral expiry dates ("07/29") inside a card read-out.
    for m in _EXPIRY_NUMERAL.finditer(text):
        prefix = (context[-200:] + " " if context else "") + text[:m.start()]
        if _payment_level(prefix, payment) == "strong" and not any(s < m.end() and m.start() < e for _, s, e, _ in found):
            found.append(("PAYMENT", m.start(), m.end(), m.group(0)))
    return found


# Legacy name: the classes the detector covers (kept for callers that list them).
PII_LABELS = ("SSN", "CARD", "PHONE", "ACCOUNT", "PIN", "DIGITS", "PAYMENT")


def _turn_text(turn: Any) -> str:
    if isinstance(turn, dict):
        return turn.get("text") or ""
    return getattr(turn, "text", "") or ""


def _turn_entities(turn: Any) -> List[Dict[str, Any]]:
    if isinstance(turn, dict):
        return turn.get("numeric_entities") or []
    ents = getattr(turn, "numeric_entities", None) or []
    return [e if isinstance(e, dict) else e.model_dump() for e in ents]


def _turn_start(turn: Any) -> Optional[float]:
    start = turn.get("start_time") if isinstance(turn, dict) else getattr(turn, "start_time", None)
    try:
        return float(start) if start is not None else None
    except (TypeError, ValueError):
        return None


def _turn_word_timestamps(turn: Any) -> List[Dict[str, Any]]:
    if isinstance(turn, dict):
        return turn.get("word_timestamps") or []
    wts = getattr(turn, "word_timestamps", None) or []
    return [w if isinstance(w, dict) else w.model_dump() for w in wts]


def _scan(turns: Sequence[Any]):
    """Walk a call's turns with the card read-out window (module notes above). Yields, per turn,
    (index, text, found, window_from): ``found`` is ``find_pii``'s result for the turn, and
    ``window_from`` is the character offset where a *strong* read-out window covers the turn (0
    when it was already open, the end of the card-number/expiry/CVV phrase or the start of the card
    number in the turn that opens it), or None when no strong window covers the turn."""
    previous = ""
    # The card read-out window: its level, how many more turns it stays open, and when it was
    # last refreshed (a turn start time, when turns carry one).
    level, remaining, since = "", 0, None
    for index, turn in enumerate(turns):
        text = _turn_text(turn)
        now = _turn_start(turn)
        if level and remaining <= 0 and not (since is not None and now is not None
                                              and now - since <= _PAY_WINDOW_SECONDS):
            level = ""
        found = find_pii(text, previous, level)
        window_from: Optional[int] = 0 if level == "strong" else None
        phrase = _PAY_CONTEXT.search(text)
        cards = [start for label, start, _, _ in found if label == "CARD"]
        if phrase or cards:
            if window_from is None:
                window_from = min(([phrase.end()] if phrase else []) + cards)
            level, remaining, since = "strong", _PAY_WINDOW, now
        elif found and level:
            remaining, since = _PAY_WINDOW, now
        elif _PAY_WEAK.search(text) and level != "strong":
            level, remaining, since = "weak", _PAY_WINDOW, now
        else:
            remaining -= 1
        yield index, text, found, window_from
        previous = text


def extract_sensitive_values(
    turns: List[Any],
    pii_patterns: bool,
) -> set:
    """Collect every sensitive value spoken in a call, from numeric entities
    (privacy-sensitive types) and PII regex matches. Used to mask all
    surfaces consistently, including repeats."""
    values: set = set()
    for turn in turns:
        for ent in _turn_entities(turn):
            if ent.get("entity_type") in SENSITIVE_ENTITY_TYPES:
                raw = str(ent.get("raw_text") or "").strip()
                if raw:
                    values.add(raw)
    if pii_patterns:
        for _, _, found, _ in _scan(turns):
            for _, _, _, value in found:
                values.add(value)
    return values


# ---------------------------------------------------------------------------
# The read-out window by position (team decision 22, privacy-gap fixes)
#
# A digit the ASR splits off or drops cannot be masked by value: a CVV written "seven" / "Two
# four" is two runs under the 3-digit PAYMENT floor, and masking "seven" by value would hide every
# "seven" in the call. So, while a strong read-out window is open (from the phrase or card number
# that opens it through the turn where it closes), two things are done by *position*:
#
#   - ``read_out_digit_spans``: every digit token inside the window (numerals, digit words, teens
#     and tens, and "oh"/"ohh" inside a run) is masked in place, in those turns only. Money,
#     scale amounts ("three hundred"), quantities ("five business days"), dates after a month
#     name and descriptors ("three digits", "the last four") stay visible, as outside the window.
#   - ``read_out_windows`` / ``read_out_mute_ranges``: the window's whole time range is muted,
#     including the untranscribed gaps between turns and every channel, from a little before the
#     phrase that opens it to a little after its last turn.
#
# Both are pure functions over the turns (the dicts or objects the rest of this module reads), so
# Store computes them from the transcript it already has. Nothing here is propagated by value.
# ---------------------------------------------------------------------------

READ_OUT_PAD_SECONDS = 0.5
"""How far the muted range extends before the window opens and after its last turn ends."""

_DIGIT_KINDS = ("num", "unit", "teen", "tens", "repeat")
_DESCRIBED = {"digit", "digits", "number", "numbers", "numbered"}
_ORDINAL_LEAD = {"last", "first"}
# A number after one of these words is a size, a position or an option, not a read-out digit.
_LABEL_LEAD = {"size", "sizes", "aisle", "page", "step", "option", "options", "quantity", "qty", "chapter", "episode",
               "level", "floor", "gate", "platform", "negative", "minus", "plus"}
# "one to two business days", "three or four weeks": a range whose far end is a quantity.
_RANGE_AFTER = re.compile(r"[\s,]*(?:to|or|and|-|–)\s*(?:\d+|[A-Za-z]+(?:-[A-Za-z]+)?)(?=[\s,]*(?:[A-Za-z%]|$))")
_PENCE_AFTER = re.compile(r"[\s,]*(?:pence|p|quid|grand|k|degrees?|inch(?:es)?|feet|foot|miles?|kilos?|pounds?|lbs?|kg|"
                          r"cm|mm|meters?|metres?|people|persons?|units?|packs?|boxes|bottles?|pieces?)\b", re.IGNORECASE)
_ONE_LEAD = {"this", "that", "the", "which", "no", "any", "every", "each", "some", "another", "other", "one", "someone",
             "anyone", "everyone", "latter", "former", "same", "next", "last", "right", "wrong", "new", "old", "only", "little",
             "big", "blue", "black", "white", "red", "green"}


def _turn_end(turn: Any) -> Optional[float]:
    end = turn.get("end_time") if isinstance(turn, dict) else getattr(turn, "end_time", None)
    try:
        return float(end) if end is not None else None
    except (TypeError, ValueError):
        return None


class ReadOutWindow:
    """One strong card read-out window: turns ``first_turn``..``last_turn`` (indexes into the
    turns), opening at character ``opens_at`` of the first one. ``start_time``/``end_time`` are the
    unpadded times it covers (None when the turns carry no times): from the phrase that opens it
    (its word's timestamp, else the first turn's start) to the latest end of its turns."""

    __slots__ = ("first_turn", "last_turn", "opens_at", "start_time", "end_time")

    def __init__(self, first_turn: int, last_turn: int, opens_at: int, start_time: Optional[float], end_time: Optional[float]):
        self.first_turn, self.last_turn, self.opens_at = first_turn, last_turn, opens_at
        self.start_time, self.end_time = start_time, end_time

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"ReadOutWindow(turns {self.first_turn}-{self.last_turn}, opens_at={self.opens_at}, "
                f"{self.start_time}-{self.end_time})")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ReadOutWindow) and all(getattr(self, k) == getattr(other, k) for k in self.__slots__)


def _opening_time(turn: Any, text: str, window_from: int) -> Optional[float]:
    """When the window opens inside ``turn``: the start of the word holding the opening phrase (or
    card number), else the turn's start. The phrase is searched from ``window_from`` backwards so
    the words that name the read-out ("the card number") are inside the range."""
    start = _turn_start(turn)
    if window_from <= 0:
        return start
    phrase_start = window_from
    for phrase in _PAY_CONTEXT.finditer(text):
        if phrase.end() == window_from:
            phrase_start = phrase.start()
            break
    words = _turn_word_timestamps(turn)
    if not words:
        return start
    for wt, (ws, we) in zip(words, _word_ranges(text, words)):
        if we > phrase_start and wt.get("start_time") is not None:
            try:
                return float(wt["start_time"])
            except (TypeError, ValueError):
                return start
    return start


def read_out_windows(turns: Sequence[Any]) -> List[ReadOutWindow]:
    """The strong card read-out windows of a call (see the notes above ``READ_OUT_PAD_SECONDS``)."""
    windows: List[ReadOutWindow] = []
    open_window: Optional[ReadOutWindow] = None
    for index, text, _found, window_from in _scan(turns):
        turn = turns[index]
        if window_from is None:
            open_window = None
            continue
        end = _turn_end(turn)
        if open_window is None:
            open_window = ReadOutWindow(index, index, window_from, _opening_time(turn, text, window_from), end)
            windows.append(open_window)
            continue
        open_window.last_turn = index
        if end is not None:
            open_window.end_time = end if open_window.end_time is None else max(open_window.end_time, end)
        if open_window.start_time is None:
            open_window.start_time = _turn_start(turn)
    return windows


def read_out_mute_ranges(turns: Sequence[Any], pad: float = READ_OUT_PAD_SECONDS) -> List[Tuple[float, float]]:
    """The time ranges to mute for the call's read-out windows: each window from ``pad`` seconds
    before it opens to ``pad`` seconds after its last turn, merged. Windows whose turns carry no
    times are skipped (their digits are still muted word by word)."""
    ranges = []
    for window in read_out_windows(turns):
        if window.start_time is None or window.end_time is None:
            continue
        ranges.append((round(max(0.0, window.start_time - pad), 2), round(max(window.end_time, window.start_time) + pad, 2)))
    return _merge(ranges)


def _window_digit_spans(text: str, opens_at: int = 0) -> List[Tuple[int, int]]:
    """Character spans of the digit tokens in ``text`` from ``opens_at`` on, one span per run."""
    items = _items(text)
    spans: List[Tuple[int, int]] = []
    for i, _last, run in _runs(items, text):
        digits = [item for item in run if item.kind in _DIGIT_KINDS and item.start >= opens_at]
        if not digits:
            continue
        before = items[i - 1] if i > 0 else None
        after = items[i + len(run)] if i + len(run) < len(items) else None
        if any(item.kind == "num" and item.formatted for item in run):
            continue  # money, decimals and thousands stay visible
        if any(item.kind in ("hundred", "thousand") for item in run):
            continue  # "three hundred and fifty" is an amount
        if _is_quantity(text, run[-1].end):
            continue  # "five business days", "two pairs", "one to two business days", "fifty pence"
        if before is not None and before.word in _LABEL_LEAD:
            continue  # "size ten", "aisle four"
        if before is not None and before.word in _MONTHS:
            continue  # "March twelve": dates stay visible
        if after is not None and after.word in _DESCRIBED or before is not None and before.word in _ORDINAL_LEAD:
            continue  # "the three digits on the back", "the last four"
        if len(run) == 1 and run[0].word == "one" and (before is not None and before.word in _ONE_LEAD
                                                        or after is not None and after.word == "of"):
            continue  # "that one", "one of them"
        start = digits[0].start
        lead = _lead_word(items, i, text)
        if lead is not None and lead.word in _ZERO_WORDS and lead.start >= opens_at:
            start = lead.start  # "oh nine nine": a leading zero word is inside the run
        spans.append((start, digits[-1].end))
    return spans


def _is_quantity(text: str, end: int) -> bool:
    if _QUANTITY_AFTER.match(text, end) or _PENCE_AFTER.match(text, end):
        return True
    far = _RANGE_AFTER.match(text, end)
    return bool(far and far.group(0).split()[-1].strip("-–").lower() not in _FILLERS
                and (_QUANTITY_AFTER.match(text, far.end()) or _PENCE_AFTER.match(text, far.end())))


def read_out_digit_spans(turns: Sequence[Any]) -> List[List[Tuple[int, int]]]:
    """Per turn (aligned with ``turns``), the character spans to mask by position because they are
    digits spoken inside a strong read-out window. Empty lists outside every window."""
    out: List[List[Tuple[int, int]]] = [[] for _ in turns]
    for index, text, _found, window_from in _scan(turns):
        if window_from is not None and text:
            out[index] = _window_digit_spans(text, window_from)
    return out


def _merge(spans: Iterable[Tuple[Any, Any]]) -> List[Tuple[Any, Any]]:
    ordered = sorted(spans)
    merged: List[Tuple[Any, Any]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


_ANCHOR_WORDS = 2
_WORD = re.compile(r"\S+")


class TurnPositions:
    """Character spans to mask *by position* in a call's turns: the read-out window digits and the
    PII model's findings (``call1.pii_model``) that are not strong enough to mask by value.

    Built from every turn text of the call and the spans for each (aligned lists; a turn with no
    positional span still counts as a known turn text). Offsets are validated against the text;
    a span that does not fit is dropped here, so callers that cannot place a span must mask its
    text by value instead.

    ``spans(text)`` answers for any text a surface wants masked:

    - a turn text of the call: that turn's spans (the union, when two turns read the same);
    - any other text (a quote, reasoning, a summary): the spans of a positional turn when the text
      is a verbatim part of it, plus every *anchor* found in it. An anchor is a masked token with
      up to two words of its own turn on each side, so a derived text that repeats that stretch of
      the turn is masked at the same place, while the bare token ("seven") elsewhere stays visible.
    """

    def __init__(self, texts: Sequence[str], spans: Sequence[Sequence[Tuple[int, int]]]):
        self._by_text: Dict[str, List[Tuple[int, int]]] = {}
        self._positional: List[Tuple[str, List[Tuple[int, int]]]] = []
        anchors: Dict[str, List[Tuple[int, int]]] = {}
        for text, turn_spans in zip(texts, list(spans) + [[]] * max(0, len(texts) - len(spans))):
            text = text or ""
            valid = _merge((int(s), int(e)) for s, e in turn_spans if 0 <= s < e <= len(text))
            known = self._by_text.setdefault(text, [])
            if valid:
                self._by_text[text] = _merge(known + valid)
                self._positional.append((text, valid))
                for s, e in valid:
                    anchor, rel = _anchor(text, s, e)
                    if anchor:
                        anchors.setdefault(anchor.lower(), []).append(rel)
        self._anchors = [(re.compile(r"(?<![A-Za-z0-9])" + re.escape(a) + r"(?![A-Za-z0-9])", re.IGNORECASE), rels)
                         for a, rels in anchors.items()]

    @classmethod
    def empty(cls) -> "TurnPositions":
        return cls([], [])

    def __bool__(self) -> bool:
        return bool(self._positional)

    def turn_spans(self, text: str) -> Optional[List[Tuple[int, int]]]:
        """The spans of a known turn text, or None when ``text`` is no turn of the call."""
        return self._by_text.get(text)

    def spans(self, text: Optional[str]) -> List[Tuple[int, int]]:
        if not text or not self._positional:
            return []
        own = self.turn_spans(text)
        if own is not None:
            return list(own)
        found: List[Tuple[int, int]] = []
        for turn_text, turn_spans in self._positional:
            at = turn_text.find(text)
            while at >= 0:
                found += [(max(s, at) - at, min(e, at + len(text)) - at) for s, e in turn_spans if s < at + len(text) and at < e]
                at = turn_text.find(text, at + 1)
        for pattern, rels in self._anchors:
            for m in pattern.finditer(text):
                found += [(m.start() + s, m.start() + e) for s, e in rels]
        return _merge(found)


def _anchor(text: str, start: int, end: int) -> Tuple[str, Tuple[int, int]]:
    """The stretch of ``text`` around ``start``..``end`` with up to ``_ANCHOR_WORDS`` words on each
    side, and the span's offsets inside it; ("", ...) when the span has no neighbouring word (a
    bare token would match every occurrence of it)."""
    words = [(m.start(), m.end()) for m in _WORD.finditer(text)]
    left = [w for w in words if w[1] <= start][-_ANCHOR_WORDS:]
    right = [w for w in words if w[0] >= end][:_ANCHOR_WORDS]
    if not left and not right:
        return "", (0, 0)
    a = left[0][0] if left else start
    b = right[-1][1] if right else end
    # Trim punctuation at the edges so "number is seven." anchors "number is seven".
    while b > end and not text[b - 1].isalnum():
        b -= 1
    while a < start and not text[a].isalnum():
        a += 1
    return text[a:b], (start - a, end - a)


def mask_spans(text: str, spans: Iterable[Tuple[int, int]]) -> str:
    """``text`` with each (merged) character span replaced by ``[REDACTED]``."""
    out, cursor = [], 0
    for s, e in _merge(spans):
        s, e = max(s, cursor), min(e, len(text))
        if e <= s:
            continue
        out += [text[cursor:s], REDACTED]
        cursor = e
    out.append(text[cursor:])
    return "".join(out)


def mask_text(text: Optional[str], values: set, positions: Optional[TurnPositions] = None) -> str:
    """``mask_text_with_values`` plus the positional spans ``positions`` gives for ``text``. With
    no positional span for the text it is exactly ``mask_text_with_values``; otherwise the value
    occurrences and the positional spans are merged first, so a value that straddles a positional
    token is still masked whole."""
    if not text:
        return text or ""
    spans = positions.spans(text) if positions else []
    if not spans:
        return mask_text_with_values(text, values)
    return mask_spans(text, _all_value_spans(text, values) + spans)


def _regex_pii_values(text: str, context: str = "") -> set:
    """Extract PII values from a standalone string via the PII detector.

    Used for metadata fields (customer_phone, agent_id) whose values may not
    appear in the transcript, so they are masked even when not in the
    transcript-derived values set. ``context`` names the field (e.g. "phone").
    """
    return {value for _, _, _, value in find_pii(text, context)}


def _word_ranges(text: str, word_ts: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """Character range of each timed word in ``text``, located in order. Words carry ASR
    spacing (" 442-89-1099,"), so the stripped word is searched; a word that cannot be found
    is placed at the cursor."""
    ranges = []
    cursor = 0
    for wt in word_ts:
        word = (wt.get("word") or "").strip()
        start = text.find(word, cursor) if word else -1
        if start < 0:
            start = cursor
        end = start + len(word)
        cursor = max(cursor, end)
        ranges.append((start, end))
    return ranges


def _value_pattern(value: str) -> str:
    """The pattern for one value's occurrences. A value with letters (a name, a spelled read-out,
    an email) matches only as whole words, so "Dan" never masks part of "Daniel"; a numeric value
    matches anywhere, as it always has."""
    pattern = re.escape(value)
    if any(c.isalpha() for c in value):
        if value[0].isalnum():
            pattern = r"(?<![A-Za-z0-9])" + pattern
        if value[-1].isalnum():
            pattern += r"(?![A-Za-z0-9])"
        # Case-insensitive: the agent's read-back "Three six five." masks the caller's "three six five".
        pattern = "(?i)" + pattern
    return pattern


def _all_value_spans(text: str, values: set) -> List[Tuple[int, int]]:
    """All character spans of every sensitive value occurrence in text."""
    spans: List[Tuple[int, int]] = []
    for value in values:
        if not value:
            continue
        for m in re.finditer(_value_pattern(value), text):
            spans.append(m.span())
    if not spans:
        return []
    spans.sort()
    merged: List[Tuple[int, int]] = [spans[0]]
    for start, end in spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def mask_text_with_values(text: Optional[str], values: set) -> str:
    """Replace every occurrence of any sensitive value in text with [REDACTED]."""
    if not text:
        return text or ""
    if not values:
        return text
    masked = text
    for value in sorted(values, key=len, reverse=True):
        if not value:
            continue
        masked = re.sub(_value_pattern(value), REDACTED, masked)
    return masked


def mask_word_timestamps(
    word_ts: Optional[List[Dict[str, Any]]],
    text: str,
    values: set,
    positions: Optional[TurnPositions] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Mask words whose character span overlaps a sensitive value occurrence, or a positional span
    ``positions`` gives for this turn text."""
    if not word_ts:
        return word_ts
    spans = _merge(_all_value_spans(text, values) + (positions.spans(text) if positions else []))
    if not spans:
        return word_ts
    masked: List[Dict[str, Any]] = []
    for wt, (start, end) in zip(word_ts, _word_ranges(text, word_ts)):
        # Any overlap: a value inside a word (" 4492.") or spanning several words.
        if any(start < e and s < end for s, e in spans):
            entry = dict(wt)
            entry["word"] = REDACTED
            masked.append(entry)
        else:
            masked.append(wt)
    return masked


def mask_entities(entities: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Mask raw_text/normalized_value of privacy-sensitive entities only."""
    if not entities:
        return entities
    masked = []
    for ent in entities:
        entry = dict(ent)
        if ent.get("entity_type") in SENSITIVE_ENTITY_TYPES:
            entry["raw_text"] = REDACTED
            entry["normalized_value"] = REDACTED
        masked.append(entry)
    return masked


class RedactionService:
    """Applies a settings snapshot to API payloads.

    Constructed with a single AppSettings instance per request so a settings
    save mid-request cannot produce inconsistent per-field toggles.
    """

    def __init__(self, settings: AppSettings):
        self._settings = settings

    @property
    def settings(self) -> AppSettings:
        return self._settings

    def enabled(self) -> bool:
        return self._settings.redaction.text

    def mask_turn(self, turn: Dict[str, Any], values: set) -> Dict[str, Any]:
        """Mask a transcript turn dict (as served by the bundle)."""
        if not self.enabled():
            return turn
        t = dict(turn)
        original_text = t.get("text", "")
        t["text"] = mask_text_with_values(original_text, values)
        if t.get("raw_text"):
            t["raw_text"] = mask_text_with_values(t["raw_text"], values)
        t["word_timestamps"] = mask_word_timestamps(t.get("word_timestamps"), original_text, values)
        t["numeric_entities"] = mask_entities(t.get("numeric_entities"))
        return t

    def mask_bundle(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        """Mask a full call bundle: turns, verdicts, evaluation reviewer notes,
        and known sensitive metadata fields (customer_phone, agent_id).

        Sensitive values are extracted once from the (unmasked) turns and
        applied consistently across every surface, covering repeats. Metadata
        fields are additionally regex-masked for PII even if the value is not
        present in the transcript.
        """
        if not self.enabled():
            return bundle
        turns = bundle.get("turns", [])
        values = extract_sensitive_values(turns, self._settings.redaction.pii_patterns)
        b = dict(bundle)
        b["turns"] = [self.mask_turn(t, values) for t in turns]
        verdicts = []
        for v in b.get("verdicts", []):
            vv = dict(v)
            vv["quoted_evidence"] = mask_text_with_values(vv.get("quoted_evidence"), values)
            vv["reasoning"] = mask_text_with_values(vv.get("reasoning"), values)
            vv["model_attempts"] = [{**attempt,
                "quoted_evidence": mask_text_with_values(attempt.get("quoted_evidence"), values),
                "reasoning": mask_text_with_values(attempt.get("reasoning"), values)}
                for attempt in vv.get("model_attempts", [])]
            verdicts.append(vv)
        b["verdicts"] = verdicts
        if b.get("evaluation") and b["evaluation"].get("reviewer_notes"):
            b["evaluation"] = dict(b["evaluation"])
            b["evaluation"]["reviewer_notes"] = mask_text_with_values(
                b["evaluation"]["reviewer_notes"], values
            )
        # Known sensitive metadata fields: mask with values + PII regexes.
        call = b.get("call")
        if call:
            call = dict(call)
            for field in ("customer_phone", "agent_id"):
                if call.get(field):
                    call[field] = mask_text_with_values(call[field], values)
                    if self._settings.redaction.pii_patterns:
                        call[field] = mask_text_with_values(
                            call[field],
                            _regex_pii_values(call[field], "phone" if field == "customer_phone" else ""),
                        )
            b["call"] = call
        return b

    def mask_search_result(self, result: Dict[str, Any], values: set) -> Dict[str, Any]:
        if not self.enabled():
            return result
        r = dict(result)
        r["text"] = mask_text_with_values(r.get("text"), values)
        return r

    def mask_summary(self, summary: Dict[str, Any], values: set) -> Dict[str, Any]:
        """Defense-in-depth masking of generated summary text.

        Masks only the scoped narrative/key_points fields plus rubric highlight
        names and notes (which may contain custom criterion names with PII).
        Does not recurse into arbitrary nested structures.
        """
        if not self.enabled():
            return summary
        s = dict(summary)
        s["narrative"] = mask_text_with_values(s.get("narrative"), values)
        s["key_points"] = [mask_text_with_values(kp, values) for kp in s.get("key_points", [])]
        highlights = []
        for h in s.get("rubric_highlights", []):
            hh = dict(h)
            hh["criterion_name"] = mask_text_with_values(hh.get("criterion_name"), values)
            hh["note"] = mask_text_with_values(hh.get("note"), values)
            highlights.append(hh)
        s["rubric_highlights"] = highlights
        return s

    def sanitize_for_model(self, turns: List[TranscriptTurn]) -> str:
        """Mask transcript text before it is sent to the local LLM.

        Sensitive values (account numbers, cards, phones, SSNs, PINs) are
        masked before they reach the model. Masking is approximate, not
        certified PII removal.
        """
        return "\n".join(
            f"{speaker.value}: {text}" for _, speaker, text in self.sanitize_turns_for_model(turns)
        )

    def sanitize_turns_for_model(self, turns: List[TranscriptTurn]) -> List[tuple]:
        """Return (turn_id, speaker, masked_text) per turn.

        Each turn's full text is preserved as one unit (never split on
        embedded newlines), and the real turn_id is retained so the model can
        cite supporting turns accurately.
        """
        if not self.enabled():
            return [(t.turn_id, t.speaker, t.text) for t in turns]
        values = extract_sensitive_values(turns, self._settings.redaction.pii_patterns)
        out = []
        for t in turns:
            masked = mask_text_with_values(t.text, values)
            out.append((t.turn_id, t.speaker, masked))
        return out

    def audio_mute_intervals(self, turns: List[TranscriptTurn], extra_values: Optional[set] = None,
                             extra_spans: Optional[Sequence[Sequence[Tuple[int, int]]]] = None,
                             read_out: bool = False) -> List[Tuple[float, float]]:
        """Mute intervals for sensitive values; empty when audio redaction is off.

        Uses word-level timestamps when available for accurate alignment
        (covers SSN/PIN which have no entity timestamps); falls back to
        entity-based interpolated intervals otherwise. If a sensitive value is
        detected in a turn but there is NO reliable word/entity alignment, the
        ENTIRE turn is muted conservatively — a detected sensitive span is
        never silently left unmuted. The union of all intervals is returned.

        ``extra_values`` adds values found elsewhere (the PII model's spans,
        ``call1.pii_model``); they are aligned to word timestamps the same way.
        ``extra_spans`` (aligned with ``turns``) adds character spans masked by position, aligned
        the same way. ``read_out`` (Store's reviewer audio; the legacy app leaves it off) also mutes
        every strong card read-out window whole (``read_out_mute_ranges``: untranscribed gaps and
        every channel included) and the window digits by position (``read_out_digit_spans``).
        """
        if not self._settings.redaction.audio:
            return []
        values = extract_sensitive_values(turns, self._settings.redaction.pii_patterns) | set(extra_values or ())
        intervals: List[Tuple[float, float]] = []
        positional: List[List[Tuple[int, int]]] = [list(s) for s in (extra_spans or [])][:len(turns)]
        positional += [[] for _ in range(len(turns) - len(positional))]
        if read_out and self._settings.redaction.pii_patterns:
            for index, spans in enumerate(read_out_digit_spans(turns)):
                positional[index] += spans
            intervals.extend(read_out_mute_ranges(turns))

        for index, turn in enumerate(turns):
            text = _turn_text(turn)
            own = [(s, e) for s, e in positional[index] if 0 <= s < e <= len(text)]
            turn_spans = _merge(_all_value_spans(text, values) + own)
            if not turn_spans:
                continue

            # Collect precise intervals for this turn (entity + word-aligned).
            precise: List[Tuple[float, float]] = []
            for ent in _turn_entities(turn):
                if ent.get("entity_type") in SENSITIVE_ENTITY_TYPES:
                    try:
                        s = max(0.0, float(ent["start_time"]) - 0.1)
                        e = float(ent["end_time"]) + 0.1
                        precise.append((round(s, 2), round(e, 2)))
                    except (KeyError, TypeError, ValueError):
                        continue
            wts = _turn_word_timestamps(turn)
            # A positional span has no entity time: without a word to align it to, the whole turn.
            unaligned = bool(own) and not wts
            if wts:
                word_ranges = [(ws, we, wt) for wt, (ws, we) in zip(wts, _word_ranges(text, wts))]
                for s, e in turn_spans:
                    words_in = [
                        wt for (ws, we, wt) in word_ranges
                        if wt.get("start_time") is not None and wt.get("end_time") is not None
                        and not (we <= s or ws >= e)
                    ]
                    if words_in:
                        start = max(0.0, min(float(w["start_time"]) for w in words_in) - 0.1)
                        end = max(float(w["end_time"]) for w in words_in) + 0.1
                        precise.append((round(start, 2), round(end, 2)))
                    elif any(s < oe and os < e for os, oe in own):
                        unaligned = True

            if precise and not unaligned:
                intervals.extend(precise)
            else:
                # No reliable alignment: conservatively mute the whole turn.
                start, end = _turn_start(turn), _turn_end(turn)
                if start is None or end is None:
                    continue
                intervals.append((round(max(0.0, start - 0.1), 2), round(end + 0.1, 2)))

        if not intervals:
            return []
        intervals.sort()
        merged: List[Tuple[float, float]] = [intervals[0]]
        for start, end in intervals[1:]:
            prev_start, prev_end = merged[-1]
            if start <= prev_end:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        return merged
