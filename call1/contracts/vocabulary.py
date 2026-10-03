"""Added in 1.3.0 (team decision 33, docs/DualAsr.md): the ASR vocabulary.

Dual transcription merged on the customer's vocabulary. Parakeet stays the transcript; Whisper
Small, prompted with the vocabulary, only finds candidates, and a deterministic rule merge writes a
vocabulary term over the Parakeet words it overlaps in time when the two sound and spell alike. It
never inserts. The vocabulary is the installed industry pack's terms (content of the industry
subscription) plus the customer's own terms, one Store document that admins edit in Evaluate.

**Business terms only.** A term is 1–60 characters of letters, spaces and ``' ’ & . -``, starts
with a letter, has at most six words and **contains no digit** (``vocabulary_term_problem``). So no
term can carry a number, an email address, a URL or a phone number, and a vocabulary replacement
can never reintroduce a value the PII masking removed. Personal names are refused by policy, and
Store's save validator runs its rule-based PII detectors over every term; the product cannot prove a
word is not a name, which is why the Evaluate editor says so.

Pure functions here are normative for every track: ``normalize_vocabulary_term``,
``vocabulary_term_problem``, ``vocabulary_term_key`` (case-, accent- and punctuation-insensitive
identity, so "Wi-Fi" and "wifi" are one term), ``effective_vocabulary`` (pack terms not disabled,
then customer terms not already present), ``vocabulary_digest`` and ``vocabulary_active``.
"""

from __future__ import annotations

import unicodedata
from enum import Enum
from typing import Annotated, Iterable, List, Literal, Optional

from pydantic import AfterValidator, Field, StringConstraints, model_validator

from .catalog import CatalogEntryRef
from .common import ContractModel, ResourceId, Sha256Digest, ShortText, Timestamp, canonical_digest

VOCABULARY_TERM_MAX_CHARS = 60
VOCABULARY_TERM_MAX_WORDS = 6
VOCABULARY_TERM_PUNCTUATION = frozenset(" '’&.-")
"""Characters a term may hold besides letters (and combining marks): space, apostrophe, right single
quotation mark, ampersand, full stop and hyphen."""

VOCABULARY_LIST_CEILING = 2000
"""Outer Pydantic ceiling on each term list. Store's effective caps are ``ContractParameters``
``max_vocabulary_terms`` (customer terms) and ``max_vocabulary_pack_terms``."""

ASR_VOCABULARY_PACK_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,39}$"

VOCABULARY_MERGE_RULE_ID = "vocab-merge-rule-v1"


def normalize_vocabulary_term(term: str) -> str:
    """NFC, trimmed, internal whitespace collapsed to single spaces. Stored terms are in this form."""
    return " ".join(unicodedata.normalize("NFC", term).split())


def vocabulary_term_problem(term: str) -> Optional[str]:
    """Why ``term`` is not a valid vocabulary term, or None. Reason codes (Store puts the code in
    ``details.reason`` and never echoes the term): ``not_normalized`` (not ``normalize_vocabulary_term``
    form), ``empty``, ``too_long``, ``too_many_words``, ``digit`` (any Unicode digit or numeral),
    ``character`` (anything but letters, combining marks and ``' ’ & . -`` and space),
    ``must_start_with_letter`` and ``too_few_letters`` (fewer than two letters)."""
    if term != normalize_vocabulary_term(term):
        return "not_normalized"
    if not term:
        return "empty"
    if len(term) > VOCABULARY_TERM_MAX_CHARS:
        return "too_long"
    if len(term.split(" ")) > VOCABULARY_TERM_MAX_WORDS:
        return "too_many_words"
    letters = 0
    for ch in term:
        category = unicodedata.category(ch)
        if ch.isdigit() or ch.isnumeric() or category == "Nd" or category == "Nl" or category == "No":
            return "digit"
        if category.startswith("L"):
            letters += 1
        elif category.startswith("M") or ch in VOCABULARY_TERM_PUNCTUATION:
            continue
        else:
            return "character"
    if not unicodedata.category(term[0]).startswith("L"):
        return "must_start_with_letter"
    if letters < 2:
        return "too_few_letters"
    return None


def vocabulary_term_key(term: str) -> str:
    """The identity of a term: accents stripped, case folded, letters only ("Wi-Fi" -> "wifi",
    "Stouffer's" -> "stouffers"). Two terms with one key are the same term."""
    decomposed = unicodedata.normalize("NFKD", term)
    return "".join(ch for ch in decomposed.casefold() if unicodedata.category(ch).startswith("L"))


def _term(value: str) -> str:
    problem = vocabulary_term_problem(value)
    if problem is not None:
        raise ValueError(f"not a vocabulary term ({problem}): business terms only, 1-{VOCABULARY_TERM_MAX_CHARS} characters, "
                         f"at most {VOCABULARY_TERM_MAX_WORDS} words, letters and ' ’ & . - only, no digits")
    return value


VocabularyTerm = Annotated[str, StringConstraints(min_length=1, max_length=VOCABULARY_TERM_MAX_CHARS), AfterValidator(_term)]
"""One vocabulary term (``vocabulary_term_problem`` is None)."""


def _unique_keys(terms: Iterable[str], what: str) -> None:
    seen = set()
    for term in terms:
        key = vocabulary_term_key(term)
        if key in seen:
            raise ValueError(f"{what} lists one term twice (terms are compared by vocabulary_term_key)")
        seen.add(key)


class VocabularyTermSource(str, Enum):
    INDUSTRY_PACK = "industry_pack"
    """From the installed industry pack (part of the industry subscription; the demo seed stands in for it)."""
    CUSTOMER = "customer"
    """Added by the customer's admins in Evaluate."""


class AsrVocabularyTerm(ContractModel):
    term: VocabularyTerm
    source: VocabularyTermSource


class AsrVocabularyPack(ContractModel):
    """An industry pack's vocabulary as installed on this Store. Read-only in Evaluate: admins can
    disable individual pack terms (``AsrVocabularySettings.disabled_pack_terms``) but not edit them.
    Installed by Store's local ``apply-vocabulary-seed`` command (``--demo`` applies the retail seed),
    never through a reviewer route."""

    pack_id: str = Field(pattern=ASR_VOCABULARY_PACK_ID_PATTERN, description="e.g. 'retail'.")
    version: int = Field(ge=1)
    industry: ShortText
    title: ShortText
    terms: List[VocabularyTerm] = Field(max_length=VOCABULARY_LIST_CEILING)

    @model_validator(mode="after")
    def _unique(self):
        _unique_keys(self.terms, "a vocabulary pack")
        return self


class AsrVocabularySettings(ContractModel):
    """What admins edit. Dual transcription runs only when ``enabled`` and the effective vocabulary
    is non-empty (``vocabulary_active``); ``enabled`` defaults to true, so a non-empty vocabulary is on
    by default."""

    enabled: bool = True
    customer_terms: List[VocabularyTerm] = Field(default_factory=list, max_length=VOCABULARY_LIST_CEILING, description="The customer's own business terms, in the admin's order. No digits and no personal names.")
    disabled_pack_terms: List[VocabularyTerm] = Field(default_factory=list, max_length=VOCABULARY_LIST_CEILING, description="Pack terms this customer switched off (for example a term that causes false corrections). Each must be a term of the installed pack (Store: validation_failed, details.reason unknown_pack_term).")

    @model_validator(mode="after")
    def _unique(self):
        _unique_keys(self.customer_terms, "customer_terms")
        _unique_keys(self.disabled_pack_terms, "disabled_pack_terms")
        return self


def effective_vocabulary(pack: Optional[AsrVocabularyPack], settings: AsrVocabularySettings) -> List[AsrVocabularyTerm]:
    """The vocabulary a graph runs with: the pack's terms in pack order, minus the disabled ones, then
    the customer's terms in their order, skipping any whose key is already present (a customer term
    that repeats an active pack term stays listed in the settings but adds nothing)."""
    disabled = {vocabulary_term_key(t) for t in settings.disabled_pack_terms}
    out: List[AsrVocabularyTerm] = []
    seen = set()
    for term in pack.terms if pack is not None else []:
        key = vocabulary_term_key(term)
        if key in disabled or key in seen:
            continue
        seen.add(key)
        out.append(AsrVocabularyTerm(term=term, source=VocabularyTermSource.INDUSTRY_PACK))
    for term in settings.customer_terms:
        key = vocabulary_term_key(term)
        if key in seen:
            continue
        seen.add(key)
        out.append(AsrVocabularyTerm(term=term, source=VocabularyTermSource.CUSTOMER))
    return out


def vocabulary_digest(terms: List[AsrVocabularyTerm]) -> str:
    """``canonical_digest`` of the ordered effective terms (``[{term, source}, ...]``)."""
    return canonical_digest([t.model_dump(mode="json") for t in terms])


def vocabulary_active(settings: AsrVocabularySettings, effective: List[AsrVocabularyTerm]) -> bool:
    """Dual transcription runs for new graphs exactly when this is true."""
    return settings.enabled and bool(effective)


class AsrVocabularyRecord(ContractModel):
    """The singleton vocabulary document (``getAsrVocabulary``). ``effective_terms``,
    ``effective_digest`` and ``active`` are derived and checked against the pack and settings."""

    record_version: int = Field(ge=0, description="0 before the first save or pack install; every save and install bumps it.")
    settings: AsrVocabularySettings
    pack: Optional[AsrVocabularyPack] = None
    effective_terms: List[AsrVocabularyTerm]
    effective_digest: Optional[Sha256Digest] = Field(default=None, description="vocabulary_digest(effective_terms); null when there are none.")
    active: bool = Field(description="vocabulary_active: new ingest and full-reanalysis graphs run dual transcription.")
    updated_at: Optional[Timestamp] = None
    updated_by_account_id: Optional[ResourceId] = Field(default=None, description="Null for a pack install by Store's local command.")

    @model_validator(mode="after")
    def _derived(self):
        expected = effective_vocabulary(self.pack, self.settings)
        if self.effective_terms != expected:
            raise ValueError("effective_terms is effective_vocabulary(pack, settings)")
        if self.effective_digest != (vocabulary_digest(expected) if expected else None):
            raise ValueError("effective_digest is vocabulary_digest(effective_terms), null when empty")
        if self.active != vocabulary_active(self.settings, expected):
            raise ValueError("active is vocabulary_active(settings, effective_terms)")
        return self


class AsrVocabularySave(ContractModel):
    """``saveAsrVocabulary``: replace the settings whole. Store applies ``ContractParameters``
    ``max_vocabulary_terms`` to ``customer_terms`` and its PII detectors to every term (both
    ``validation_failed`` with ``details.reason``, never the term); a stale version is 409 ``conflict``
    with ``details.current_version``."""

    expected_record_version: int = Field(ge=0)
    settings: AsrVocabularySettings


class VocabularyMergeRule(ContractModel):
    """The deterministic merge rule (benchmarks/2026-09-26-dual-asr-merge.md, thresholds fixed a
    priori). Frozen on the job and recorded on the transcript. A candidate is a vocabulary-prompted
    Whisper hit on a term, paired with the base words overlapping it in time (``time_slack_seconds``
    each side), narrowed to the contiguous sub-span of at most the term's length +
    ``max_span_extra_tokens`` tokens most similar to the term. It is accepted when the Double
    Metaphone similarity is at least ``phonetic_min``, the character similarity at least
    ``character_min`` and the word counts differ by at most ``max_word_delta``. Nothing is inserted
    where the base has no overlapping word, and a span never crosses a turn."""

    rule_id: Literal["vocab-merge-rule-v1"] = VOCABULARY_MERGE_RULE_ID
    phonetic_algorithm: Literal["double_metaphone"] = "double_metaphone"
    phonetic_min: float = Field(default=0.70, ge=0, le=1)
    character_min: float = Field(default=0.60, ge=0, le=1)
    max_word_delta: int = Field(default=1, ge=0, le=3)
    time_slack_seconds: float = Field(default=0.3, ge=0, le=2)
    max_span_extra_tokens: int = Field(default=2, ge=0, le=4)


class AsrVocabularyParameters(ContractModel):
    """``JobParameters.asr_vocabulary`` on an ``asr`` job: what dual transcription runs with, frozen by
    the planner from the current ``AsrVocabularyRecord`` when it is ``active``. Absent means
    Parakeet only, exactly as before 1.3.0. The terms are business vocabulary validated to contain no
    digits; they are the one piece of customer-authored text ``JobParameters`` carries."""

    digest: Sha256Digest = Field(description="vocabulary_digest(terms); Store checks it at graph creation (graph_invalid).")
    terms: List[AsrVocabularyTerm] = Field(min_length=1, max_length=2 * VOCABULARY_LIST_CEILING)
    candidate_entry: CatalogEntryRef = Field(description="The catalog entry (purpose asr_vocabulary, Whisper Small) the candidate pass runs on, frozen by the planner.")
    glossary_prompt_limit: int = Field(default=120, ge=16, le=200, description="Whisper prompt tokens for the per-call glossary shortlist, repeated in every 30 s window (Whisper's prompt limit is 223).")
    rule: VocabularyMergeRule = Field(default_factory=VocabularyMergeRule)

    @model_validator(mode="after")
    def _digest(self):
        _unique_keys((t.term for t in self.terms), "asr_vocabulary.terms")
        if self.digest != vocabulary_digest(self.terms):
            raise ValueError("asr_vocabulary.digest is vocabulary_digest(terms)")
        return self


__all__ = [
    "ASR_VOCABULARY_PACK_ID_PATTERN", "AsrVocabularyPack", "AsrVocabularyParameters", "AsrVocabularyRecord", "AsrVocabularySave",
    "AsrVocabularySettings", "AsrVocabularyTerm", "VOCABULARY_LIST_CEILING", "VOCABULARY_MERGE_RULE_ID", "VOCABULARY_TERM_MAX_CHARS",
    "VOCABULARY_TERM_MAX_WORDS", "VOCABULARY_TERM_PUNCTUATION", "VocabularyMergeRule", "VocabularyTerm", "VocabularyTermSource",
    "effective_vocabulary", "normalize_vocabulary_term", "vocabulary_active", "vocabulary_digest", "vocabulary_term_key",
    "vocabulary_term_problem",
]
