"""Rubric evaluation engine and deterministic grounding guardrail."""

from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable, List, Optional, Tuple
from call1.models.schemas import (
    CallEvaluationResult,
    CallTranscript,
    CheckType,
    RubricCheck,
    RubricCriterion,
    RubricDefinition,
    RubricVerdict,
    SpeakerRole,
    TranscriptTurn,
    VerdictStatus,
)


def _normalize_text(text: str) -> str:
    """Normalize text by lowercasing, stripping punctuation and collapsing whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


# --- Sliding-window fuzzy phrase matching ---------------------------------
#
# Matching runs over a word-level timeline rather than the flat transcript so
# every verdict carries the moment it fired and the workbench can jump straight
# to it. A window sized to the phrase (± one word for filler like "um") keeps a
# long call from diluting the score the way a whole-transcript comparison
# would. This is the deterministic tier from the POC
# (git/call1/src/call1/scoring/deterministic.py), adapted to turn-based
# transcripts and difflib (rapidfuzz is not a dependency of this repo).


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class PhraseMatch:
    phrase: str
    score: float
    start: float
    end: float
    quote: str


def _words_from_turns(turns: List[TranscriptTurn]) -> List[Word]:
    """Flatten transcript turns into a word timeline.

    Uses per-word timestamps when the ASR produced them; otherwise falls back
    to turn-level timing so a transcript without word alignment still scores —
    just with coarser evidence.
    """
    words: List[Word] = []
    for turn in turns:
        if turn.word_timestamps:
            for w in turn.word_timestamps:
                text = (w.word or "").strip()
                if text:
                    words.append(Word(text=text, start=w.start_time, end=w.end_time))
        else:
            start, end = turn.start_time, turn.end_time
            for token in (turn.text or "").split():
                words.append(Word(text=token, start=start, end=end))
    return words


def find_phrase(words: List[Word], phrase: str, threshold: int) -> Optional[PhraseMatch]:
    """Best fuzzy match of `phrase` against any same-length window of `words`.

    Sliding a window sized to the phrase (± one word) keeps a long call from
    diluting the score the way a whole-transcript comparison would. Returns the
    highest-scoring window at or above `threshold`, or None.
    """
    target = phrase.lower().strip()
    n = max(1, len(target.split()))
    if not words or not target:
        return None

    best: Optional[PhraseMatch] = None
    for size in {n, n + 1, max(1, n - 1)}:
        for i in range(0, max(1, len(words) - size + 1)):
            window = words[i : i + size]
            if not window:
                continue
            candidate = " ".join(w.text for w in window).lower()
            score = SequenceMatcher(None, target, candidate).ratio() * 100.0
            if score >= threshold and (best is None or score > best.score):
                best = PhraseMatch(
                    phrase=phrase,
                    score=score,
                    start=window[0].start,
                    end=window[-1].end,
                    quote=" ".join(w.text for w in window),
                )
    return best


def _apply_window(words: List[Word], window_seconds: float, duration_seconds: float) -> List[Word]:
    """Restrict the word timeline to a time window.

    Positive `window_seconds` keeps the first N seconds; negative keeps the
    last N seconds; the caller passes None for the whole call.
    """
    if window_seconds is None:
        return words
    if window_seconds >= 0:
        return [w for w in words if w.start <= window_seconds]
    cutoff = max(0.0, duration_seconds + window_seconds)
    return [w for w in words if w.start >= cutoff]


def verify_quoted_evidence(
    quote: Optional[str],
    transcript: CallTranscript,
    expected_speaker: Optional[SpeakerRole] = None,
) -> Tuple[bool, Optional[int], Optional[Tuple[float, float]]]:
    """
    Deterministically verify that a model's quoted evidence exists verbatim in the transcript.
    Returns: (is_verified, matched_turn_id, timestamp_range)
    """
    if not quote or not quote.strip():
        return False, None, None

    norm_quote = _normalize_text(quote)
    if not norm_quote:
        return False, None, None

    for turn in transcript.turns:
        if expected_speaker and turn.speaker != expected_speaker:
            continue

        norm_turn = _normalize_text(turn.text)
        if norm_quote in norm_turn:
            return True, turn.turn_id, (turn.start_time, turn.end_time)

    return False, None, None


# --- Preset Rubric Definitions ---

DEFAULT_RUBRIC = RubricDefinition(
    rubric_id="call1_standard_v1",
    name="Call1 Standard Contact Center QA & Compliance",
    description="Baseline scorecard covering statutory disclosures, caller authentication, transaction disclosures, and etiquette.",
    category="GENERAL",
    pass_threshold=80.0,
    criteria=[
        RubricCriterion(
            criterion_id="REG-01",
            name="Call Recording Disclosure",
            category="COMPLIANCE",
            description="Agent must notify the caller at the beginning of the call that the interaction is being recorded.",
            weight=25.0,
            critical=True,
            check=RubricCheck(
                check_type=CheckType.PHRASE_ANY,
                phrases=[
                    "This call may be monitored or recorded for quality assurance.",
                    "recorded", "monitored", "quality assurance", "training purposes",
                ],
                threshold=55,
                speaker=SpeakerRole.AGENT,
            ),
        ),
        RubricCriterion(
            criterion_id="SEC-01",
            name="Caller ID & Verification",
            category="SECURITY",
            description="Agent must authenticate caller identity (e.g. account number, DOB, SSN, or PIN) prior to discussing account data.",
            weight=30.0,
            critical=True,
            check=RubricCheck(
                check_type=CheckType.PHRASE_ANY,
                phrases=[
                    "verify", "authenticate", "account number", "date of birth", "dob",
                    "ssn", "social security", "pin", "security question",
                    "billing address", "zip code",
                ],
                threshold=80,
                speaker=SpeakerRole.AGENT,
            ),
        ),
        RubricCriterion(
            criterion_id="COMP-01",
            name="Mandatory Regulatory Disclosures",
            category="COMPLIANCE",
            description="Agent must disclose applicable transaction terms, fees, or dispute rights.",
            weight=25.0,
            critical=False,
            check=RubricCheck(
                check_type=CheckType.PHRASE_ANY,
                phrases=[
                    "disclose transaction fees terms dispute rights or interest rate",
                    "terms", "policy", "fee", "dispute", "apr", "interest rate", "cancel",
                ],
                threshold=40,
                speaker=SpeakerRole.AGENT,
            ),
        ),
        RubricCriterion(
            criterion_id="ETIQ-01",
            name="Professional Closing & Resolution",
            category="ETIQUETTE",
            description="Agent must ask if there is anything else they can help with and close the call professionally.",
            weight=20.0,
            critical=False,
            check=RubricCheck(
                check_type=CheckType.PHRASE_ANY,
                phrases=[
                    "anything else", "help you with", "further assistance", "other questions",
                    "thank you for calling", "have a great day", "have a wonderful day",
                    "have a good day", "goodbye", "take care", "thanks for reaching out",
                ],
                threshold=80,
                speaker=SpeakerRole.AGENT,
            ),
        ),
    ],
)

BANKING_RUBRIC = RubricDefinition(
    rubric_id="call1_banking_fina_v1",
    name="Banking & Financial Services Compliance Scorecard",
    description="Strict statutory scorecard covering Gramm-Leach-Bliley, KYC multi-factor auth, fee disclosures, and dispute empathy.",
    category="BANKING",
    pass_threshold=85.0,
    criteria=[
        RubricCriterion(
            criterion_id="BANK-REG-01",
            name="Call Recording & BIPA Consent",
            category="COMPLIANCE",
            description="Agent must state the call recording notification in the initial interaction.",
            weight=25.0,
            critical=True,
            check=RubricCheck(
                check_type=CheckType.PHRASE_ANY,
                phrases=[
                    "This call is recorded for quality and compliance purposes.",
                    "recorded", "monitored", "quality", "compliance", "training",
                ],
                threshold=50,
                speaker=SpeakerRole.AGENT,
            ),
        ),
        RubricCriterion(
            criterion_id="BANK-SEC-01",
            name="KYC Multi-Factor Authentication",
            category="SECURITY",
            description="Agent must verify caller identity using account number, date of birth, SSN, or PIN before account disclosure.",
            weight=30.0,
            critical=True,
            check=RubricCheck(
                check_type=CheckType.PHRASE_ANY,
                phrases=[
                    "verify", "authenticate", "account number", "date of birth", "dob",
                    "ssn", "social security", "pin", "security question",
                    "billing address", "zip code",
                ],
                threshold=80,
                speaker=SpeakerRole.AGENT,
            ),
        ),
        RubricCriterion(
            criterion_id="BANK-FEE-01",
            name="Fee & Transaction Disclosure",
            category="COMPLIANCE",
            description="Agent must disclose fees, interest charges, or dispute timelines prior to transaction execution.",
            weight=15.0,
            critical=False,
            check=RubricCheck(
                check_type=CheckType.PHRASE_ANY,
                phrases=[
                    "disclose fees charges dispute timeline or interest rate",
                    "fee", "charge", "dispute", "terms", "interest", "penalty", "policy",
                ],
                threshold=40,
                speaker=SpeakerRole.AGENT,
            ),
        ),
        RubricCriterion(
            criterion_id="BANK-EMP-01",
            name="Empathic De-escalation on Financial Disputes",
            category="CUSTOMER_EXPERIENCE",
            description="When a caller reports a billing error, hold, or disputed charge, the agent must respond with an empathic acknowledgment.",
            weight=15.0,
            critical=False,
            check=RubricCheck(
                check_type=CheckType.CONDITIONAL_RESPONSE,
                trigger_phrases=[
                    "frustrated", "upset", "angry", "terrible", "awful", "unacceptable",
                    "problem", "issue", "charged twice", "overcharged", "dispute",
                    "not working", "broken", "ridiculous", "scam", "waste of time",
                ],
                response_phrases=[
                    "sorry to hear", "understand", "apologize", "let me help",
                    "completely understand", "happy to help", "stressful",
                    "take care of this", "get this sorted", "my apologies",
                ],
                threshold=80,
                speaker=SpeakerRole.AGENT,
            ),
        ),
        RubricCriterion(
            criterion_id="BANK-CLOSE-01",
            name="Next Steps & Professional Sign-off",
            category="ETIQUETTE",
            description="Agent must state resolution timeline or reference number and offer additional assistance before concluding.",
            weight=15.0,
            critical=False,
            check=RubricCheck(
                check_type=CheckType.PHRASE_ANY,
                phrases=[
                    "anything else", "help you with", "further assistance", "other questions",
                    "thank you for calling", "have a great day", "have a wonderful day",
                    "have a good day", "goodbye", "take care", "thanks for reaching out",
                ],
                threshold=80,
                speaker=SpeakerRole.AGENT,
            ),
        ),
    ],
)

CUSTOMER_CARE_RUBRIC = RubricDefinition(
    rubric_id="call1_customer_care_v1",
    name="Customer Care & Empathy Excellence Scorecard",
    description="Quality assurance scorecard focusing on customer experience, speed to greet, empathy, and active listening.",
    category="CUSTOMER_CARE",
    pass_threshold=80.0,
    criteria=[
        RubricCriterion(
            criterion_id="CARE-GREET-01",
            name="Fast Greeting (<15 Seconds)",
            category="ETIQUETTE",
            description="Agent must deliver a warm greeting and identify themselves within the first 15 seconds of the call.",
            weight=20.0,
            critical=False,
            check=RubricCheck(
                check_type=CheckType.PHRASE_ANY,
                phrases=[
                    "hello", "hi", "good morning", "good afternoon", "good evening",
                    "thank you for calling", "thanks for calling", "welcome to",
                ],
                threshold=80,
                window_seconds=15.0,
                speaker=SpeakerRole.AGENT,
            ),
        ),
        RubricCriterion(
            criterion_id="CARE-REC-01",
            name="Recording Disclosure",
            category="COMPLIANCE",
            description="Agent must notify the customer that the call is recorded.",
            weight=20.0,
            critical=True,
            check=RubricCheck(
                check_type=CheckType.PHRASE_ANY,
                phrases=[
                    "This call is recorded for quality assurance.",
                    "recorded", "monitored", "quality", "training",
                ],
                threshold=50,
                speaker=SpeakerRole.AGENT,
            ),
        ),
        RubricCriterion(
            criterion_id="CARE-EMP-01",
            name="Empathic Response to Customer Frustration",
            category="CUSTOMER_EXPERIENCE",
            description="When customer expresses negative sentiment or frustration, agent must validate and respond with empathy.",
            weight=25.0,
            critical=False,
            check=RubricCheck(
                check_type=CheckType.CONDITIONAL_RESPONSE,
                trigger_phrases=[
                    "frustrated", "upset", "angry", "terrible", "awful", "unacceptable",
                    "problem", "issue", "charged twice", "overcharged", "dispute",
                    "not working", "broken", "ridiculous", "scam", "waste of time",
                ],
                response_phrases=[
                    "sorry to hear", "understand", "apologize", "let me help",
                    "completely understand", "happy to help", "stressful",
                    "take care of this", "get this sorted", "my apologies",
                ],
                threshold=80,
                speaker=SpeakerRole.AGENT,
            ),
        ),
        RubricCriterion(
            criterion_id="CARE-HOLD-01",
            name="Hold & Latency Etiquette",
            category="ETIQUETTE",
            description="Agent must ask permission before putting customer on hold and thank them upon returning.",
            weight=15.0,
            critical=False,
            check=RubricCheck(
                check_type=CheckType.CONDITIONAL_RESPONSE,
                trigger_phrases=[
                    "place you on hold", "put you on hold", "one moment",
                    "bear with me", "check on that",
                ],
                response_phrases=[
                    "thank you for holding", "thanks for holding",
                    "appreciate your patience", "thanks for waiting",
                ],
                threshold=80,
                speaker=SpeakerRole.AGENT,
            ),
        ),
        RubricCriterion(
            criterion_id="CARE-CLOSE-01",
            name="First Contact Resolution & Warm Closing",
            category="ETIQUETTE",
            description="Agent must ask if all concerns were addressed and conclude with a warm, professional farewell.",
            weight=20.0,
            critical=False,
            check=RubricCheck(
                check_type=CheckType.PHRASE_ANY,
                phrases=[
                    "anything else", "help you with", "further assistance", "other questions",
                    "thank you for calling", "have a great day", "have a wonderful day",
                    "have a good day", "goodbye", "take care", "thanks for reaching out",
                ],
                threshold=80,
                speaker=SpeakerRole.AGENT,
            ),
        ),
    ],
)

COLLECTIONS_RUBRIC = RubricDefinition(
    rubric_id="call1_collections_fdcpa_v1",
    name="Debt Collections & FDCPA Statutory Scorecard",
    description="Fair Debt Collection Practices Act compliance scorecard enforcing Mini-Miranda, verification, and non-harassment.",
    category="COLLECTIONS",
    pass_threshold=90.0,
    criteria=[
        RubricCriterion(
            criterion_id="FDCPA-MINI-01",
            name="Mini-Miranda Statutory Disclosure",
            category="COMPLIANCE",
            description="Agent must disclose that the communication is from a debt collector and any information will be used for that purpose.",
            weight=35.0,
            critical=True,
            check=RubricCheck(
                check_type=CheckType.PHRASE_ANY,
                phrases=[
                    "This communication is from a debt collector and any information obtained will be used for that purpose.",
                    "debt collector", "attempting to collect a debt", "information obtained will be used",
                ],
                threshold=50,
                speaker=SpeakerRole.AGENT,
            ),
        ),
        RubricCriterion(
            criterion_id="FDCPA-AUTH-01",
            name="Third-Party Disclosure Protection",
            category="SECURITY",
            description="Agent must authenticate caller identity before stating the debt or balance owed.",
            weight=30.0,
            critical=True,
            check=RubricCheck(
                check_type=CheckType.PHRASE_ANY,
                phrases=[
                    "verify", "authenticate", "account number", "date of birth", "dob",
                    "ssn", "social security", "pin", "security question",
                    "billing address", "zip code",
                ],
                threshold=80,
                speaker=SpeakerRole.AGENT,
            ),
        ),
        RubricCriterion(
            criterion_id="FDCPA-TONE-01",
            name="Respectful Tone & Harassment Prevention",
            category="CUSTOMER_EXPERIENCE",
            description="Agent must maintain a professional, calm, non-harassing tone and avoid aggressive confrontation.",
            weight=20.0,
            critical=True,
            check=RubricCheck(
                check_type=CheckType.CONDITIONAL_RESPONSE,
                trigger_phrases=[
                    "frustrated", "upset", "angry", "terrible", "awful", "unacceptable",
                    "problem", "issue", "charged twice", "overcharged", "dispute",
                    "not working", "broken", "ridiculous", "scam", "waste of time",
                ],
                response_phrases=[
                    "sorry to hear", "understand", "apologize", "let me help",
                    "completely understand", "happy to help", "stressful",
                    "take care of this", "get this sorted", "my apologies",
                ],
                threshold=80,
                speaker=SpeakerRole.AGENT,
            ),
        ),
        RubricCriterion(
            criterion_id="FDCPA-ARR-01",
            name="Payment Terms & Confirmation",
            category="COMPLIANCE",
            description="Agent must clearly summarize payment agreement amounts, dates, and dispute rights.",
            weight=15.0,
            critical=False,
            check=RubricCheck(
                check_type=CheckType.PHRASE_ANY,
                phrases=[
                    "anything else", "help you with", "further assistance", "other questions",
                    "thank you for calling", "have a great day", "have a wonderful day",
                    "have a good day", "goodbye", "take care", "thanks for reaching out",
                ],
                threshold=80,
                speaker=SpeakerRole.AGENT,
            ),
        ),
    ],
)

PRESET_RUBRICS: List[RubricDefinition] = [
    DEFAULT_RUBRIC,
    BANKING_RUBRIC,
    CUSTOMER_CARE_RUBRIC,
    COLLECTIONS_RUBRIC,
]


# New recordings use versioned contextual presets. Keep v1 definitions for
# historical evaluations and existing user edits, without silently rewriting them.
LEGACY_PRESET_RUBRICS = PRESET_RUBRICS
from call1.pipeline.contextual_rubrics import contextual_presets
PRESET_RUBRICS = contextual_presets(LEGACY_PRESET_RUBRICS)
DEFAULT_RUBRIC = PRESET_RUBRICS[0]
from call1.pipeline.sentiment_rules import sentiment_presets
PRESET_RUBRICS += sentiment_presets()


def _legacy_to_check(criterion: RubricCriterion) -> RubricCheck:
    """Normalize a legacy criterion into a modular RubricCheck.

    Criteria authored before the modular `check` field carried a `rule_type`
    and a `parameters` dict. This maps those onto the equivalent RubricCheck
    so every criterion evaluates through the single modular path. The schema
    already auto-converts on construction via `RubricCriterion._ensure_check`;
    this is the evaluator-side safety net for criteria that bypassed it.

    For `custom` rules (no explicit rule_type), the legacy dispatch routed by
    criterion_id prefix (e.g. REG-01 → compliance, ETIQ-01 → closing). That
    routing is preserved here so custom scorecards keep their behavior.
    """
    from call1.models.schemas import legacy_rule_to_check

    rule = getattr(criterion, "rule_type", "custom") or "custom"
    params = getattr(criterion, "parameters", {}) or {}
    cid = (criterion.criterion_id or "").upper()

    if rule == "custom":
        if "GREET" in cid:
            rule = "timing_greeting"
        elif "EMP" in cid:
            rule = "empathy_sentiment"
        elif cid in ("REG-01", "COMP-01") or "REG" in cid:
            rule = "compliance_phrase"
        elif cid == "SEC-01" or "SEC" in cid or "AUTH" in cid:
            rule = "auth_verification"
        elif "HOLD" in cid:
            rule = "hold_etiquette"
        elif cid == "ETIQ-01" or "CLOSE" in cid or "ETIQ" in cid:
            rule = "closing_etiquette"
        elif "pattern" in params:
            rule = "keyword_regex"

    return legacy_rule_to_check(rule, params)


class RubricEvaluator:
    """Evaluates transcripts against rubric criteria and applies anti-hallucination guardrails."""

    def __init__(
        self,
        rubric: Optional[RubricDefinition] = None,
        model_backend: Optional[Callable[[str], str]] = None,
        question_router=None,
    ):
        self.rubric = rubric or DEFAULT_RUBRIC
        #: Optional local model backend for semantic_judgement checks. The
        #: callable receives a prompt and returns the model's raw text answer.
        #: When absent, semantic checks are FLAGGED for human review — the
        #: deterministic tier never guesses on ungrounded keyword matching.
        self.model_backend = model_backend
        self.question_router = question_router

    def evaluate_deterministic(
        self,
        transcript: CallTranscript,
        simulated_hallucination: bool = False,
    ) -> CallEvaluationResult:
        """
        Evaluate standard contact center rules deterministically.
        Model uncertainty and unsupported evidence require human review.
        """
        verdicts: List[RubricVerdict] = []
        critical_failed = False
        requires_review = False
        escalation_reasons: List[str] = []

        total_weight = sum(c.weight for c in self.rubric.criteria)
        earned_score = 0.0

        for criterion in self.rubric.criteria:
            verdict = self._evaluate_criterion(criterion, transcript, simulated_hallucination)

            # Grounding Guardrail: Verify quoted evidence exists verbatim in transcript
            # Routed answers have already been grounded against the exact redacted
            # transcript supplied to their provider. Rechecking against raw PII
            # would incorrectly reject valid quotes containing [REDACTED].
            if verdict.quoted_evidence and not (self.question_router is not None and verdict.model_attempts):
                verified, turn_id, ts = verify_quoted_evidence(
                    verdict.quoted_evidence,
                    transcript,
                    expected_speaker=verdict.speaker,
                )
                if verified:
                    verdict.timestamp_range = ts
                    verdict.hallucination_detected = False
                else:
                    verdict.hallucination_detected = True
                    verdict.status = VerdictStatus.FLAGGED
                    requires_review = True
                    escalation_reasons.append(
                        f"Guardrail rejection on {criterion.criterion_id}: Cited quote was not found verbatim in transcript."
                    )

            # Low confidence escalation
            if verdict.confidence < 0.70:
                requires_review = True
                verdict.status = VerdictStatus.FLAGGED
                escalation_reasons.append(
                    f"Low confidence ({verdict.confidence:.2f}) on criterion {criterion.criterion_id} ({criterion.name})."
                )

            # Scoring calculation
            if verdict.status == VerdictStatus.PASS:
                earned_score += criterion.weight
            elif criterion.critical and verdict.status == VerdictStatus.FAIL:
                critical_failed = True
                escalation_reasons.append(
                    f"Critical compliance failure on {criterion.criterion_id}: {criterion.name}."
                )

            verdicts.append(verdict)
            if verdict.status in (VerdictStatus.NOT_APPLICABLE, VerdictStatus.FLAGGED):
                total_weight -= criterion.weight
            if verdict.status == VerdictStatus.FLAGGED:
                requires_review = True

        overall_score = round((earned_score / total_weight) * 100.0, 1) if total_weight > 0 else 0.0
        threshold = getattr(self.rubric, "pass_threshold", 80.0) or 80.0
        passed = (overall_score >= threshold) and not critical_failed and not requires_review and total_weight > 0

        if critical_failed:
            requires_review = True

        return CallEvaluationResult(
            call_id=transcript.call_id,
            rubric_id=self.rubric.rubric_id,
            overall_score=overall_score,
            passed=passed,
            critical_failure=critical_failed,
            requires_human_review=requires_review,
            escalation_reasons=escalation_reasons,
            verdicts=verdicts,
        )

    def _evaluate_criterion(
        self,
        criterion: RubricCriterion,
        transcript: CallTranscript,
        simulated_hallucination: bool = False,
    ) -> RubricVerdict:
        """Evaluate a criterion through the single modular check path.

        Every criterion — whether authored with a modular `check` or with the
        legacy `rule_type`/`parameters` pair — is normalized to a RubricCheck
        and dispatched by its `check_type`. There is no separate rule engine.
        """
        check = getattr(criterion, "check", None)
        if check is None:
            check = _legacy_to_check(criterion)
        if check.requires_policy and not (check.policy_context or '').strip():
            return RubricVerdict(
                criterion_id=criterion.criterion_id, criterion_name=criterion.name,
                category=criterion.category, status=VerdictStatus.FLAGGED,
                confidence=0.0, speaker=check.speaker or SpeakerRole.AGENT,
                reasoning="This criterion requires a configured business policy. Add the applicable policy in the scorecard or review manually; transcript statements cannot substitute for it.",
            )
        if check.speaker is not None and any(t.speaker == SpeakerRole.UNKNOWN for t in transcript.turns):
            return RubricVerdict(
                criterion_id=criterion.criterion_id, criterion_name=criterion.name,
                category=criterion.category, status=VerdictStatus.FLAGGED,
                confidence=0.0, speaker=check.speaker,
                reasoning="Speaker identity is unknown. Review this criterion manually or import a dual-channel recording with a verified channel mapping.",
            )
        return self._evaluate_modular_check(criterion, check, transcript, simulated_hallucination)

    # --- Modular Check Evaluation -----------------------------------------

    def _evaluate_modular_check(
        self,
        criterion: RubricCriterion,
        check: RubricCheck,
        transcript: CallTranscript,
        simulated_hallucination: bool,
    ) -> RubricVerdict:
        """Evaluate a criterion carrying a modular `check` definition."""
        check_type = check.check_type
        if check_type == CheckType.SENTIMENT_METRIC:
            from call1.pipeline.sentiment_rules import evaluate_sentiment
            return evaluate_sentiment(criterion, check, transcript)
        if check_type == CheckType.PHRASE_ANY:
            return self._check_phrase_any(criterion, check, transcript, simulated_hallucination)
        if check_type == CheckType.PHRASE_ALL:
            return self._check_phrase_all(criterion, check, transcript, simulated_hallucination)
        if check_type == CheckType.PHRASE_NONE:
            return self._check_phrase_none(criterion, check, transcript, simulated_hallucination)
        if check_type == CheckType.CONDITIONAL_RESPONSE:
            return self._check_conditional_response(criterion, check, transcript, simulated_hallucination)
        if check_type == CheckType.SEMANTIC_JUDGEMENT:
            return self._check_semantic_judgement(criterion, check, transcript, simulated_hallucination)
        if check_type == CheckType.CUSTOM_REGEX:
            return self._check_custom_regex(criterion, check, transcript, simulated_hallucination)
        return RubricVerdict(
            criterion_id=criterion.criterion_id,
            criterion_name=criterion.name,
            status=VerdictStatus.FLAGGED,
            confidence=0.0,
            quoted_evidence=None,
            speaker=check.speaker or SpeakerRole.AGENT,
            reasoning=f"Unknown check type {check_type!r}.",
        )

    def _check_scope(
        self,
        check: RubricCheck,
        transcript: CallTranscript,
    ) -> Tuple[List[TranscriptTurn], Optional[RubricVerdict]]:
        """Turns the check may cite, or a verdict when the scope is empty.

        A check scoped to one speaker cannot be answered off a recording where
        the sides were never separated — that is a gap for a reviewer, not a
        pass or a fail.
        """
        if check.speaker is None:
            return list(transcript.turns), None
        scoped = [t for t in transcript.turns if t.speaker == check.speaker]
        if not scoped:
            return [], RubricVerdict(
                criterion_id="",
                criterion_name="",
                status=VerdictStatus.FLAGGED,
                confidence=0.0,
                quoted_evidence=None,
                speaker=check.speaker,
                reasoning=(
                    f"This check applies to the {check.speaker.value}, and no "
                    f"{check.speaker.value} speech is separated in this recording."
                ),
            )
        return scoped, None

    def _check_phrase_any(
        self,
        criterion: RubricCriterion,
        check: RubricCheck,
        transcript: CallTranscript,
        simulated_hallucination: bool,
    ) -> RubricVerdict:
        """Passes if any target phrase matches within the window at threshold."""
        turns, scope_verdict = self._check_scope(check, transcript)
        if scope_verdict is not None:
            return self._with_criterion(scope_verdict, criterion)

        words = _apply_window(_words_from_turns(turns), check.window_seconds, transcript.duration_seconds)
        best: Optional[PhraseMatch] = None
        for phrase in check.phrases:
            m = find_phrase(words, phrase, check.threshold)
            if m is not None and (best is None or m.score > best.score):
                best = m

        if best is not None:
            quote = best.quote if not simulated_hallucination else "Fabricated phrase quote."
            if getattr(check, "legacy_rule", None) == "timing_greeting":
                max_seconds = check.window_seconds or 15.0
                reasoning = (
                    f"Agent greeted caller at {best.start:.1f}s, within the "
                    f"{max_seconds:.0f}s threshold."
                )
            else:
                reasoning = (
                    f'Matched required phrase "{best.phrase}" at {best.start:.1f}s '
                    f"({best.score:.0f}% similarity)."
                )
            return RubricVerdict(
                criterion_id=criterion.criterion_id,
                criterion_name=criterion.name,
                status=VerdictStatus.PASS,
                confidence=min(0.99, best.score / 100.0),
                quoted_evidence=quote,
                speaker=check.speaker or SpeakerRole.AGENT,
                timestamp_range=(best.start, best.end),
                reasoning=reasoning,
            )

        # Legacy timing_greeting distinguishes a delayed greeting (found outside
        # the window) from a greeting that never happened at all.
        if getattr(check, "legacy_rule", None) == "timing_greeting":
            max_seconds = check.window_seconds or 15.0
            all_words = _words_from_turns(turns)
            delayed: Optional[PhraseMatch] = None
            for phrase in check.phrases:
                m = find_phrase(all_words, phrase, check.threshold)
                if m is not None and (delayed is None or m.score > delayed.score):
                    delayed = m
            if delayed is not None:
                return RubricVerdict(
                    criterion_id=criterion.criterion_id,
                    criterion_name=criterion.name,
                    status=VerdictStatus.FAIL,
                    confidence=0.93,
                    quoted_evidence=delayed.quote,
                    speaker=check.speaker or SpeakerRole.AGENT,
                    timestamp_range=(delayed.start, delayed.end),
                    reasoning=(
                        f"Agent greeting was delayed until {delayed.start:.1f}s "
                        f"(exceeded the {max_seconds:.0f}s threshold)."
                    ),
                )
            return RubricVerdict(
                criterion_id=criterion.criterion_id,
                criterion_name=criterion.name,
                status=VerdictStatus.FAIL,
                confidence=0.95,
                quoted_evidence=None,
                speaker=check.speaker or SpeakerRole.AGENT,
                reasoning=(
                    f"Agent did not offer an introductory greeting within the "
                    f"first {max_seconds:.0f}s."
                ),
            )

        window = self._window_label(check.window_seconds)
        return RubricVerdict(
            criterion_id=criterion.criterion_id,
            criterion_name=criterion.name,
            status=VerdictStatus.FAIL,
            confidence=0.95,
            quoted_evidence=None,
            speaker=check.speaker or SpeakerRole.AGENT,
            reasoning=(
                f"None of {len(check.phrases)} required phrases were found{window} "
                f"at or above {check.threshold}% similarity."
            ),
        )

    def _check_phrase_all(
        self,
        criterion: RubricCriterion,
        check: RubricCheck,
        transcript: CallTranscript,
        simulated_hallucination: bool,
    ) -> RubricVerdict:
        """Passes only if every required phrase is matched."""
        turns, scope_verdict = self._check_scope(check, transcript)
        if scope_verdict is not None:
            return self._with_criterion(scope_verdict, criterion)

        words = _apply_window(_words_from_turns(turns), check.window_seconds, transcript.duration_seconds)
        matched: List[PhraseMatch] = []
        missing: List[str] = []
        for phrase in check.phrases:
            m = find_phrase(words, phrase, check.threshold)
            if m is not None:
                matched.append(m)
            else:
                missing.append(phrase)

        if not missing:
            best = max(matched, key=lambda m: m.score)
            quote = best.quote if not simulated_hallucination else "Fabricated phrase quote."
            return RubricVerdict(
                criterion_id=criterion.criterion_id,
                criterion_name=criterion.name,
                status=VerdictStatus.PASS,
                confidence=min(0.99, best.score / 100.0),
                quoted_evidence=quote,
                speaker=check.speaker or SpeakerRole.AGENT,
                timestamp_range=(best.start, best.end),
                reasoning=(
                    f"All {len(check.phrases)} required phrases were found; best match "
                    f'"{best.phrase}" at {best.start:.1f}s ({best.score:.0f}% similarity).'
                ),
            )

        window = self._window_label(check.window_seconds)
        return RubricVerdict(
            criterion_id=criterion.criterion_id,
            criterion_name=criterion.name,
            status=VerdictStatus.FAIL,
            confidence=0.95,
            quoted_evidence=None,
            speaker=check.speaker or SpeakerRole.AGENT,
            reasoning=(
                f"{len(missing)} of {len(check.phrases)} required phrases were not found{window} "
                f"at or above {check.threshold}% similarity: {', '.join(missing)}."
            ),
        )

    def _check_phrase_none(
        self,
        criterion: RubricCriterion,
        check: RubricCheck,
        transcript: CallTranscript,
        simulated_hallucination: bool,
    ) -> RubricVerdict:
        """Passes if no prohibited phrase is found; fails with the quote if one is."""
        turns, scope_verdict = self._check_scope(check, transcript)
        if scope_verdict is not None:
            return self._with_criterion(scope_verdict, criterion)

        words = _apply_window(_words_from_turns(turns), check.window_seconds, transcript.duration_seconds)
        best: Optional[PhraseMatch] = None
        for phrase in check.phrases:
            m = find_phrase(words, phrase, check.threshold)
            if m is not None and (best is None or m.score > best.score):
                best = m

        if best is not None:
            quote = best.quote if not simulated_hallucination else "Fabricated phrase quote."
            return RubricVerdict(
                criterion_id=criterion.criterion_id,
                criterion_name=criterion.name,
                status=VerdictStatus.FAIL,
                confidence=min(0.99, best.score / 100.0),
                quoted_evidence=quote,
                speaker=check.speaker or SpeakerRole.AGENT,
                timestamp_range=(best.start, best.end),
                reasoning=(
                    f'Prohibited phrase "{best.phrase}" found at {best.start:.1f}s '
                    f"({best.score:.0f}% similarity)."
                ),
            )

        return RubricVerdict(
            criterion_id=criterion.criterion_id,
            criterion_name=criterion.name,
            status=VerdictStatus.PASS,
            confidence=0.95,
            quoted_evidence=None,
            speaker=check.speaker or SpeakerRole.AGENT,
            reasoning=f"No prohibited phrases found (checked {len(check.phrases)}).",
        )

    def _check_conditional_response(
        self,
        criterion: RubricCriterion,
        check: RubricCheck,
        transcript: CallTranscript,
        simulated_hallucination: bool,
    ) -> RubricVerdict:
        """When a caller says a trigger phrase, the agent must respond within 2 turns.

        If the caller never triggers, the check passes (nothing was required).
        """
        caller_turns = [t for t in transcript.turns if t.speaker == SpeakerRole.CALLER]
        trigger_words = _apply_window(
            _words_from_turns(caller_turns), check.window_seconds, transcript.duration_seconds
        )
        trigger: Optional[PhraseMatch] = None
        for phrase in check.trigger_phrases:
            m = find_phrase(trigger_words, phrase, check.threshold)
            if m is not None and (trigger is None or m.score > trigger.score):
                trigger = m

        if trigger is None:
            return RubricVerdict(
                criterion_id=criterion.criterion_id,
                criterion_name=criterion.name,
                status=VerdictStatus.PASS,
                confidence=0.90,
                quoted_evidence=None,
                speaker=SpeakerRole.AGENT,
                reasoning=(
                    "No caller trigger phrase was found; no response was required."
                ),
            )

        # The agent's response must land within the next two turns after the
        # trigger turn.
        trigger_turn = next(
            (t for t in caller_turns if t.start_time <= trigger.start < t.end_time),
            None,
        )
        if trigger_turn is None:
            trigger_turn = next(
                (t for t in caller_turns if t.start_time <= trigger.start),
                caller_turns[-1],
            )
        trigger_index = transcript.turns.index(trigger_turn)
        response_window = transcript.turns[trigger_index + 1 : trigger_index + 3]

        response_words = _apply_window(
            _words_from_turns(response_window), None, transcript.duration_seconds
        )
        best: Optional[PhraseMatch] = None
        for phrase in check.response_phrases:
            m = find_phrase(response_words, phrase, check.threshold)
            if m is not None and (best is None or m.score > best.score):
                best = m

        if best is not None:
            quote = best.quote if not simulated_hallucination else "Fabricated response quote."
            if getattr(check, "legacy_rule", None) == "empathy_sentiment":
                reasoning = (
                    f"Agent offered empathic response following caller distress at "
                    f"{trigger.start:.1f}s: '{quote}'."
                )
            else:
                reasoning = (
                    f'Caller triggered at {trigger.start:.1f}s ("{trigger.quote}"); '
                    f'agent responded with "{best.phrase}" within the next two turns.'
                )
            return RubricVerdict(
                criterion_id=criterion.criterion_id,
                criterion_name=criterion.name,
                status=VerdictStatus.PASS,
                confidence=min(0.99, best.score / 100.0),
                quoted_evidence=quote,
                speaker=SpeakerRole.AGENT,
                timestamp_range=(best.start, best.end),
                reasoning=reasoning,
            )

        if getattr(check, "legacy_rule", None) == "empathy_sentiment":
            reasoning = (
                f"Caller expressed frustration at {trigger.start:.1f}s "
                f"('{trigger.quote}'), but agent failed to offer an empathic or "
                "de-escalating response."
            )
        else:
            reasoning = (
                f'Caller said "{trigger.quote}" at {trigger.start:.1f}s, but the agent '
                "did not respond with any of the required response phrases within the "
                "next two turns."
            )
        return RubricVerdict(
            criterion_id=criterion.criterion_id,
            criterion_name=criterion.name,
            status=VerdictStatus.FAIL,
            confidence=0.90,
            quoted_evidence=trigger.quote,
            speaker=SpeakerRole.CALLER,
            timestamp_range=(trigger.start, trigger.end),
            reasoning=reasoning,
        )

    def _check_semantic_judgement(
        self,
        criterion: RubricCriterion,
        check: RubricCheck,
        transcript: CallTranscript,
        simulated_hallucination: bool,
    ) -> RubricVerdict:
        """Judgement from pass_when / fail_when guidance.

        A semantic question is a claim about the relationship between turns —
        which way round a sentence points, whether one thing came before
        another. No phrase list or keyword count answers that: cosine distance
        and keyword overlap have no opinion about negation or order ("I can
        hear you" and "I cannot hear you" share every content word). This is
        the POC's invariant — semantic questions belong on a model or a human,
        never on a keyword match.

        So this check has exactly two paths:

        1. A model backend is wired in: the model is asked the question with
           the pass_when / fail_when / not_applicable_when clauses, and must
           return the exact verbatim quote supporting its claim. The quote is
           verified against the transcript with `verify_quoted_evidence`; a
           quote that does not verify verbatim is a hallucination and the
           verdict is FLAGGED, never published.
        2. No model backend: the verdict is FLAGGED with 0.0 confidence and
           an explicit rationale, so the coverage/audit trail stays honest —
           a semantic question the machine cannot answer is a gap for a
           reviewer, not a guess.
        """
        turns, scope_verdict = self._check_scope(check, transcript)
        if scope_verdict is not None:
            return self._with_criterion(scope_verdict, criterion)

        if self.question_router is not None:
            return self.question_router.evaluate(criterion, check, transcript, simulated_hallucination)

        if self.model_backend is None:
            return RubricVerdict(
                criterion_id=criterion.criterion_id,
                criterion_name=criterion.name,
                status=VerdictStatus.FLAGGED,
                confidence=0.0,
                quoted_evidence=None,
                speaker=check.speaker or SpeakerRole.AGENT,
                reasoning=(
                    "Semantic judgement requires a model adapter or supervisor review. "
                    "The deterministic tier does not guess on ungrounded keyword matching."
                ),
            )

        # Caller replies and chronology matter even when the quote is agent-only.
        prompt = self._semantic_prompt(check, transcript.turns)
        try:
            raw = self.model_backend(prompt)
        except Exception as exc:
            return RubricVerdict(
                criterion_id=criterion.criterion_id,
                criterion_name=criterion.name,
                status=VerdictStatus.FLAGGED,
                confidence=0.0,
                quoted_evidence=None,
                speaker=check.speaker or SpeakerRole.AGENT,
                reasoning=(
                    f"Semantic judgement could not be answered: the model backend "
                    f"failed ({exc}). Queued for human review."
                ),
            )

        answer = self._parse_semantic_answer(raw)
        if answer is None:
            return RubricVerdict(
                criterion_id=criterion.criterion_id,
                criterion_name=criterion.name,
                status=VerdictStatus.FLAGGED,
                confidence=0.0,
                quoted_evidence=None,
                speaker=check.speaker or SpeakerRole.AGENT,
                reasoning=(
                    "Semantic judgement could not be answered: the model returned "
                    "unparseable output. Queued for human review."
                ),
            )

        verdict, quote, assessment = answer
        if verdict == "needs_review":
            return RubricVerdict(
                criterion_id=criterion.criterion_id, criterion_name=criterion.name,
                status=VerdictStatus.FLAGGED, confidence=0.0,
                speaker=check.speaker or SpeakerRole.AGENT,
                reasoning=f"Model assessment: {assessment} Supervisor review is required.",
            )
        if verdict == "not_applicable":
            verified, _, _ = verify_quoted_evidence(quote, transcript, expected_speaker=check.speaker)
            if not verified:
                return RubricVerdict(
                    criterion_id=criterion.criterion_id, criterion_name=criterion.name,
                    status=VerdictStatus.FLAGGED, confidence=0.0,
                    speaker=check.speaker or SpeakerRole.AGENT,
                    reasoning="Not-applicable judgement lacks a verified supporting quote; review applicability manually.",
                )
            return RubricVerdict(
                criterion_id=criterion.criterion_id,
                criterion_name=criterion.name,
                status=VerdictStatus.NOT_APPLICABLE,
                confidence=0.90,
                quoted_evidence=quote or None,
                speaker=check.speaker or SpeakerRole.AGENT,
                reasoning=(
                    "The model judged the situation this question asks about did "
                    f"not arise on this call. Model assessment: {assessment}"
                ),
            )

        if not quote or not quote.strip():
            return RubricVerdict(
                criterion_id=criterion.criterion_id,
                criterion_name=criterion.name,
                status=VerdictStatus.FLAGGED,
                confidence=0.0,
                quoted_evidence=None,
                speaker=check.speaker or SpeakerRole.AGENT,
                reasoning=(
                    "Semantic judgement withheld: the model returned a verdict "
                    "without a supporting quote. Queued for human review."
                ),
            )

        # Grounding guardrail: the quote must exist verbatim in the transcript.
        verified, _turn_id, ts = verify_quoted_evidence(
            quote,
            transcript,
            expected_speaker=check.speaker,
        )
        if not verified:
            return RubricVerdict(
                criterion_id=criterion.criterion_id,
                criterion_name=criterion.name,
                status=VerdictStatus.FLAGGED,
                confidence=0.0,
                quoted_evidence=quote,
                speaker=check.speaker or SpeakerRole.AGENT,
                reasoning=(
                    "Semantic judgement withheld: the quoted evidence does not "
                    "appear verbatim in the transcript (possible hallucination). "
                    "Queued for human review."
                ),
            )

        status = VerdictStatus.PASS if verdict == "pass" else VerdictStatus.FAIL
        return RubricVerdict(
            criterion_id=criterion.criterion_id,
            criterion_name=criterion.name,
            status=status,
            confidence=0.90,
            quoted_evidence=quote,
            speaker=check.speaker or SpeakerRole.AGENT,
            timestamp_range=ts,
            reasoning=(
                f"Model judged this check {'pass' if status == VerdictStatus.PASS else 'fail'}; "
                f"supporting quote verified verbatim at {ts[0]:.1f}s. Model assessment: {assessment}"
            ),
        )

    def _semantic_prompt(self, check: RubricCheck, turns: List[TranscriptTurn]) -> str:
        """Build the model prompt for one semantic judgement.

        The transcript is numbered per turn so the model can cite exact
        evidence, and the guidance clauses are quoted verbatim. The model is
        told the transcript is data, not instructions.
        """
        from call1.qa_output import QA_SYSTEM, QA_SCHEMA, QA_DECISION_CHECKLIST
        payload = {
            "transcript": [{"turn_id": t.turn_id, "speaker": t.speaker.value,
                            "text": t.text} for t in turns],
            "criterion": {
                "policy": check.policy_context or "",
                "pass_when": check.pass_when or "",
                "fail_when": check.fail_when or "",
                "not_applicable_when": check.not_applicable_when or "",
                "quote_speaker": check.speaker.value if check.speaker else "any speaker",
            },
            "decision_checklist": QA_DECISION_CHECKLIST,
        }
        # JSON escaping prevents speech from masquerading as prompt delimiters.
        return ("Output schema: " + json.dumps(QA_SCHEMA) +
                "\nEvaluate this input data:\n" + json.dumps(payload, ensure_ascii=False))

    @staticmethod
    def _parse_semantic_answer(raw: str) -> Optional[Tuple[str, str, str]]:
        """Parse the model's JSON answer into (verdict, quote, assessment).

        Tolerates outer code fences for compatibility. Rejects extra keys,
        missing fields, invalid types, and trailing prose.
        """
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        def unique_fields(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate answer field")
                result[key] = value
            return result
        try:
            data = json.loads(cleaned, object_pairs_hook=unique_fields)
        except Exception:
            return None
        from call1.qa_output import QAAnswer
        from pydantic import ValidationError
        try:
            answer = QAAnswer.model_validate(data)
        except ValidationError:
            return None
        verdict, quote = answer.verdict, answer.quote
        # Small local models may include our line label in their quotation.
        # Strip only that formatting; the spoken text still passes the normal
        # exact-match and expected-speaker verification before publication.
        quote = re.sub(r"^\s*\[\d+\]\s*(?:AGENT|CALLER|SYSTEM|UNKNOWN):\s*", "", quote)
        return verdict, quote, answer.assessment

    def _check_custom_regex(
        self,
        criterion: RubricCriterion,
        check: RubricCheck,
        transcript: CallTranscript,
        simulated_hallucination: bool,
    ) -> RubricVerdict:
        """Pattern matching over the scoped turns."""
        pattern = (check.pattern or "").strip()
        if not pattern:
            return RubricVerdict(
                criterion_id=criterion.criterion_id,
                criterion_name=criterion.name,
                status=VerdictStatus.FLAGGED,
                confidence=0.0,
                quoted_evidence=None,
                speaker=check.speaker or SpeakerRole.AGENT,
                reasoning="No regex pattern configured for this check.",
            )
        try:
            regex = re.compile(pattern, re.I)
        except re.error as exc:
            return RubricVerdict(
                criterion_id=criterion.criterion_id,
                criterion_name=criterion.name,
                status=VerdictStatus.FLAGGED,
                confidence=0.0,
                quoted_evidence=None,
                speaker=check.speaker or SpeakerRole.AGENT,
                reasoning=f"Invalid regex pattern: {exc}.",
            )

        turns, scope_verdict = self._check_scope(check, transcript)
        if scope_verdict is not None:
            return self._with_criterion(scope_verdict, criterion)

        for turn in turns:
            if regex.search(turn.text):
                quote = turn.text if not simulated_hallucination else "Fabricated regex quote."
                return RubricVerdict(
                    criterion_id=criterion.criterion_id,
                    criterion_name=criterion.name,
                    status=VerdictStatus.PASS,
                    confidence=0.95,
                    quoted_evidence=quote,
                    speaker=turn.speaker,
                    timestamp_range=(turn.start_time, turn.end_time),
                    reasoning=f"Matched custom pattern '{pattern}'.",
                )

        return RubricVerdict(
            criterion_id=criterion.criterion_id,
            criterion_name=criterion.name,
            status=VerdictStatus.FAIL,
            confidence=0.90,
            quoted_evidence=None,
            speaker=check.speaker or SpeakerRole.AGENT,
            reasoning=f"Failed to match required pattern '{pattern}'.",
        )

    @staticmethod
    def _with_criterion(verdict: RubricVerdict, criterion: RubricCriterion) -> RubricVerdict:
        """Fill in criterion identity on a scope verdict built without it."""
        return RubricVerdict(
            criterion_id=criterion.criterion_id,
            criterion_name=criterion.name,
            status=verdict.status,
            confidence=verdict.confidence,
            quoted_evidence=verdict.quoted_evidence,
            speaker=verdict.speaker,
            timestamp_range=verdict.timestamp_range,
            reasoning=verdict.reasoning,
            hallucination_detected=verdict.hallucination_detected,
        )

    @staticmethod
    def _window_label(window_seconds: Optional[float]) -> str:
        if window_seconds is None:
            return ""
        if window_seconds >= 0:
            return f" within the first {window_seconds:.0f}s"
        return f" within the last {-window_seconds:.0f}s"
