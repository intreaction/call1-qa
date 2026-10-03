"""Numeric reference and entity extraction for contact center compliance."""

from __future__ import annotations

import re
from typing import List, Optional, Set, Tuple

from call1.models.schemas import NumericEntity, NumericEntityType, TranscriptTurn

# Word-to-number mapping for conversational English
WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
}

# Regex patterns for numeric entities
PATTERNS = [
    # Currencies: $100, $49.50, 45 dollars, 50 bucks
    (
        NumericEntityType.CURRENCY,
        re.compile(
            r"(\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)|"
            r"\b(\d+(?:\.\d{1,2})?)\s*(?:dollars?|cents?|bucks?|usd)\b|"
            r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty|forty|fifty|hundred)\s+dollars?\b)",
            re.IGNORECASE,
        ),
    ),
    # Percentages: 5%, 3.5 percent, 20 percent
    (
        NumericEntityType.PERCENTAGE,
        re.compile(
            r"(\b\d+(?:\.\d{1,2})?\s*\%|\b\d+(?:\.\d{1,2})?\s*percent\b|"
            r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty|forty|fifty)\s*percent\b)",
            re.IGNORECASE,
        ),
    ),
    # Account & Identification numbers (e.g. 884-210-993, 1234-5678-9012-3456)
    (
        NumericEntityType.ACCOUNT_NUMBER,
        re.compile(
            r"\b(?:account|acct|id|policy)?\s*(?:#|number|no\.?)?\s*(\d{3,4}[-\s]\d{3,4}[-\s]\d{3,4}(?:[-\s]\d{3,4})?|\b\d{8,16}\b)",
            re.IGNORECASE,
        ),
    ),
    # Phone numbers: (800) 555-0199, 800-555-0199, 555-1234
    (
        NumericEntityType.PHONE_NUMBER,
        re.compile(
            r"(\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b)",
            re.IGNORECASE,
        ),
    ),
    # Dates: May 12, 1984, 10/15/2026, June 14th, 2026-09-04
    (
        NumericEntityType.DATE,
        re.compile(
            r"(\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,\s*\d{4})?\b|"
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|"
            r"\b\d{4}-\d{2}-\d{2}\b)",
            re.IGNORECASE,
        ),
    ),
    # Durations: 30 days, 15 minutes, 2 hours, 12 months
    (
        NumericEntityType.DURATION,
        re.compile(
            r"(\b\d+\s*(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b)",
            re.IGNORECASE,
        ),
    ),
    # Generic numbers: 1,000, 42.5
    (
        NumericEntityType.GENERIC_NUMBER,
        re.compile(r"(\b\d{1,3}(?:,\d{3})+(?:\.\d+)?|\b\d+(?:\.\d+)?\b)"),
    ),
]


def normalize_numeric_value(raw: str, entity_type: NumericEntityType) -> float | str:
    """Normalize a raw string token into a standardized numeric value."""
    text = raw.strip().lower()

    # Handle currency
    if entity_type == NumericEntityType.CURRENCY:
        cleaned = re.sub(r"[^\d.]", "", text)
        try:
            return float(cleaned) if cleaned else text
        except ValueError:
            return text

    # Handle percentage
    if entity_type == NumericEntityType.PERCENTAGE:
        cleaned = re.sub(r"[^\d.]", "", text)
        try:
            return float(cleaned) if cleaned else text
        except ValueError:
            return text

    # Handle phone / account / date (keep formatted string)
    if entity_type in (NumericEntityType.PHONE_NUMBER, NumericEntityType.ACCOUNT_NUMBER, NumericEntityType.DATE):
        return re.sub(r"\s+", " ", text.strip())

    # Handle duration or generic number
    cleaned = re.sub(r"[^\d.]", "", text)
    if cleaned:
        try:
            return float(cleaned) if "." in cleaned else int(cleaned)
        except ValueError:
            return text

    return text


def extract_numeric_references(
    text: str, turn_start: float = 0.0, turn_end: float = 0.0
) -> List[NumericEntity]:
    """
    Scan text for all numeric references with interpolated timestamps.
    """
    if not text:
        return []

    entities: List[NumericEntity] = []
    seen_spans: List[Tuple[int, int]] = []
    text_len = max(1, len(text))
    duration = max(0.0, turn_end - turn_start)

    for ent_type, pattern in PATTERNS:
        for match in pattern.finditer(text):
            span_start, span_end = match.span()

            # Check overlap with already extracted higher-priority entity
            if any(not (span_end <= s or span_start >= e) for s, e in seen_spans):
                continue

            raw_matched = match.group(0)
            seen_spans.append((span_start, span_end))

            # Approximate start/end time based on character position within the turn
            t_start = round(turn_start + (span_start / text_len) * duration, 2)
            t_end = round(turn_start + (span_end / text_len) * duration, 2)

            entities.append(
                NumericEntity(
                    raw_text=raw_matched,
                    normalized_value=normalize_numeric_value(raw_matched, ent_type),
                    entity_type=ent_type,
                    start_time=t_start,
                    end_time=t_end,
                )
            )

    # Return ordered by appearance
    return sorted(entities, key=lambda e: e.start_time)


def generate_audio_mute_map(
    turns: List[TranscriptTurn],
    sensitive_types: Optional[Set[NumericEntityType]] = None,
) -> List[Tuple[float, float]]:
    """
    Generate audio mute intervals [(start_sec, end_sec), ...] for sensitive entities (e.g. PCI, SSN, Acct #).
    """
    if sensitive_types is None:
        sensitive_types = {
            NumericEntityType.ACCOUNT_NUMBER,
            NumericEntityType.PHONE_NUMBER,
        }

    intervals: List[Tuple[float, float]] = []
    for turn in turns:
        if not turn.numeric_entities:
            continue
        for ent in turn.numeric_entities:
            if ent.entity_type in sensitive_types:
                # Add 100ms buffer on each side for audio masking
                s = max(0.0, ent.start_time - 0.1)
                e = ent.end_time + 0.1
                intervals.append((round(s, 2), round(e, 2)))

    # Merge overlapping intervals
    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])
    merged: List[Tuple[float, float]] = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged
