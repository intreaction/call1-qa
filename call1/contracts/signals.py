"""Contact Signals v2 (contract 1.3.0; docs/ContactSignalsV2.md section 7, team decisions 21 and 22).

This module holds the admin-defined signal taxonomy (the eight fixed built-in categories plus
custom categories, subcategories and extraction fields), its settings and immutable published
versions, the pure digest functions that decide hit identity and what a taxonomy edit reruns, the
taxonomy snapshot artifact content, alert rules, hit feedback, the per-result taxonomy status,
previews and backfills. The stage artifacts themselves (``signal_categories``,
``signal_subcategories``, ``signal_extraction``) and the v2 additions to the contact-signals result
live in ``contents.py``; the signal metrics read models live in ``metrics.py``.

Signals are open core (Apache 2.0 once the license audit clears): nothing here reads a Pro1
connection or entitlement, and nothing is gated. Signals never change a scorecard.

Taxonomy text (names, glosses, descriptions, examples, field names and descriptions, enum values,
alert-rule names) is customer data: Store runs its rule-based detectors over every text path on
save and on preview (``signal_taxonomy_text_paths``) and refuses a match by path, never by value.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Annotated, Any, Dict, Iterator, List, Literal, NamedTuple, Optional, Tuple, Union

from pydantic import Field, StringConstraints, model_validator

from .common import CONTRACT_PARAMETERS, ContractModel, ContractParameters, ResourceId, SafeText, Sha256Digest, ShortText, Timestamp, canonical_digest
from .contents import (  # noqa: F401  (re-exported signal vocabulary)
    EVIDENCE_FIELD_TYPES,
    SIGNAL_BLOCK_WINDOWS,
    SIGNAL_NODE_ID_PATTERN,
    SIGNAL_NONE_OPTION,
    SIGNAL_NOT_OPTION,
    SIGNAL_OTHER_OPTION,
    SIGNAL_SCOPE_KEYS,
    SIGNAL_SPAN_KEY_PATTERN,
    SIGNAL_STAGES,
    SURFACE_FIELD_TYPES,
    SIGNAL_RULES_ENTRY_ID,
    ContactSignalKind,
    ContactSignalsContent,
    ShortDigest,
    SignalExampleBankRef,
    SignalFieldType,
    SignalHitWhy,
    SignalKnnSettings,
    SignalNodeId,
    SignalRuleCounts,
    SignalRuleDecision,
    SignalRuleOutcome,
    SignalRulesProvenance,
    SignalRuleType,
    SignalSpanKey,
    SignalStage,
    SignalTaxonomyRef,
    SpeakerRole,
    short_digest,
    signal_hit_id,
    signal_preview_hit_id,
    signal_span_key,
)
from .errors import JobErrorCode

# --- Vocabulary ---------------------------------------------------------------------------

RESERVED_CATEGORY_IDS = frozenset({SIGNAL_NONE_OPTION, ContactSignalKind.CUSTOM.value})
"""No custom category may use these IDs: 'none' is stage 1's 'none of these' option (it keys
``SegmentScores.probabilities``, the stage-1 option list and thresholds), and 'custom' is the kind
every custom-category hit carries, which a hit's ``category_id`` never repeats. The built-in kind
IDs are taken by the built-ins themselves (``builtin`` is true exactly for them)."""

RESERVED_SUBCATEGORY_IDS = frozenset({SIGNAL_OTHER_OPTION, SIGNAL_NOT_OPTION})
"""Stage 2 synthesizes 'Other <category>' and 'Not <category>' itself; no subcategory may use these IDs."""

RESERVED_FIELD_IDS = frozenset({"quote"})
"""``narrow_quote`` adds a reserved string field ``quote`` to the extraction schema (section 5.4)."""

EnumLabel = Annotated[str, StringConstraints(min_length=1, max_length=40, pattern=r"^\S(?:.*\S)?$")]
"""An enum field value: 1-40 characters, one line, no leading or trailing whitespace."""

ExampleText = Annotated[str, StringConstraints(min_length=1, max_length=120, pattern=r"^\S(?:.*\S)?$")]
"""An example phrase for a category or subcategory: plain text, 1-120 characters, one line, no
leading or trailing whitespace. Examples are definition text (section 9.4): they go through Store's
detector pass like every other text path, and they should describe the behavior, never quote a
caller. Readers: the preview display, the Gemma extractor's reference, and the fake engines
(substring match). The real classifiers (Laya) do not read them, so no digest covers them."""

SignalPipeline = Literal["v1", "shadow", "v2"]


class BuiltinCategory(NamedTuple):
    name: str
    gloss: str
    speaker: SpeakerRole


BUILTIN_SIGNAL_CATEGORIES: Dict[str, BuiltinCategory] = {
    "intent": BuiltinCategory("Caller objective", "Caller says what they want or why they called", SpeakerRole.CALLER),
    "issue": BuiltinCategory("Reported issue", "Caller describes the problem, fee or dispute", SpeakerRole.CALLER),
    "friction": BuiltinCategory("Friction point", "Caller describes an obstacle, repeat failure or frustration", SpeakerRole.CALLER),
    "fix_proposed": BuiltinCategory("Proposed fix", "Agent proposes a fix or workaround", SpeakerRole.AGENT),
    "agent_reports_completed": BuiltinCategory("Agent completed", "Agent reports that an action is done", SpeakerRole.AGENT),
    "caller_confirms_resolved": BuiltinCategory("Caller confirmed", "Caller confirms the problem is solved", SpeakerRole.CALLER),
    "caller_reports_unresolved": BuiltinCategory("Still unresolved", "Caller says the problem persists", SpeakerRole.CALLER),
    "deferred": BuiltinCategory("Deferred", "Agent defers work or promises a callback", SpeakerRole.AGENT),
}
"""The eight fixed built-in categories: every ``ContactSignalKind`` except ``custom``. Call1-authored;
names follow the v1 ``ALL_SIGNALS`` labels. Their name, gloss, speaker and active state never
change, and their ``description`` is always null. S0a may re-word a gloss in a later contract commit."""

BUILTIN_EDITABLE_FIELDS = ("threshold", "subcategory_threshold", "subcategories", "fields", "narrow_quote", "examples", "recipe")
"""What an admin may change on a built-in category (section 7.2). Everything else is the constant above."""


class FieldPiiClass(str, Enum):
    """What kind of value an extraction field holds. Opaque business identifiers that do not by
    themselves identify the customer (order, receipt or ticket numbers) are ``none``: team
    decision 22 records the order number as business data, not PII. Identifiers of the customer or
    their account (loyalty, account, card numbers) are ``account_number`` or ``card_number`` and are
    forbidden."""

    NONE = "none"
    ORGANIZATION = "organization"
    PRODUCT = "product"
    AMOUNT = "amount"
    DATE = "date"
    AGENT_NAME = "agent_name"
    CALLER_NAME = "caller_name"
    ACCOUNT_NUMBER = "account_number"
    CARD_NUMBER = "card_number"
    PHONE = "phone"
    EMAIL = "email"
    ADDRESS = "address"
    URL = "url"
    SECRET = "secret"
    GOVERNMENT_ID = "government_id"


FORBIDDEN_FIELD_PII_CLASSES = frozenset({
    FieldPiiClass.CALLER_NAME, FieldPiiClass.ACCOUNT_NUMBER, FieldPiiClass.CARD_NUMBER, FieldPiiClass.PHONE,
    FieldPiiClass.EMAIL, FieldPiiClass.ADDRESS, FieldPiiClass.URL, FieldPiiClass.SECRET, FieldPiiClass.GOVERNMENT_ID,
})
"""Refused on save (``validation_failed``, ``details.field`` names the path): the privacy filter and
the number rules mask exactly these (decision 19), so no engine sees them and no value is ever stored."""

ALLOWED_FIELD_PII_CLASSES = frozenset(FieldPiiClass) - FORBIDDEN_FIELD_PII_CLASSES


def _one_line(value: Optional[str], what: str) -> None:
    if value is not None and ("\n" in value or "\r" in value):
        raise ValueError(f"{what} is one line")


def _unique_ci(names: List[str], what: str) -> None:
    folded = [n.casefold() for n in names]
    if len(folded) != len(set(folded)):
        raise ValueError(f"{what} are unique, ignoring case")


# --- Rules-engine recipes (1.4.0; docs/SignalsEmbeddings.md sections 3, 4 and 9) ---------------
#
# A category may carry a detection recipe. When the settings select rules detection
# (``SignalSettings.detection``) and the recipe's engine is ``rules``, the rules engine decides the
# category (and, from the kNN vote, its subcategory) with no generation; Gemma then only extracts the
# fields and, where the recipe asks for a check, confirms or rejects the span. With no recipe, or
# ``engine: gemma``, or ``detection: model`` (the default), the category runs today's Gemma stages.

LEXICON_PHRASE_MAX_CHARS = 300
LexiconPhrase = Annotated[str, StringConstraints(min_length=1, max_length=LEXICON_PHRASE_MAX_CHARS, pattern=r"^\S(?:.*\S)?$")]
"""One lexicon phrase: one line, no leading or trailing whitespace. Definition text (section 9.4):
Store's detectors check it on save, and it is tombstoned with the rest of the custom text."""
MAX_LEXICON_PHRASES_CEILING = 64
"""Outer ceiling on phrases per lexicon; ContractParameters.max_signal_lexicon_phrases is the cap."""
MAX_RECIPE_RULES_CEILING = 16
"""Outer ceiling on rules per recipe filter; ContractParameters.max_signal_recipe_rules is the cap."""
MAX_RULE_DEPTH = 3
"""A recipe filter's expression depth (``all``/``any``/``not`` nesting over rules)."""
NEGATION_VETO_MAX_WORDS = 6

LexiconSyntax = Literal["words", "regex"]

_REGEX_GROUP_OPEN = re.compile(r"\(\?(?!:)")


def _regex_problem(pattern: str) -> Optional[str]:
    """Why ``pattern`` is outside the safe regex subset, or None. Allowed: literals, classes, ``\\b``
    and the like, groups ``(...)``/``(?:...)``, alternation and repeats. Refused: look-arounds,
    named groups, inline flags, back-references, conditionals, and an unbounded repeat around a
    repeat (catastrophic backtracking). Matching is always case-insensitive."""
    try:
        import re._constants as sre_constants  # type: ignore[import-not-found]
        import re._parser as sre_parse  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - Python < 3.11
        import sre_constants  # type: ignore[no-redef]
        import sre_parse  # type: ignore[no-redef]
    unescaped = re.sub(r"\\.", "", pattern)
    if _REGEX_GROUP_OPEN.search(unescaped):
        return "only (...) and (?:...) groups are allowed: no look-arounds, named groups or inline flags"
    try:
        re.compile(pattern, re.IGNORECASE)
        parsed = sre_parse.parse(pattern, re.IGNORECASE)
    except (re.error, RecursionError, OverflowError) as exc:
        return f"not a valid pattern ({str(exc)[:80]})"
    repeats = (sre_constants.MAX_REPEAT, sre_constants.MIN_REPEAT, getattr(sre_constants, "POSSESSIVE_REPEAT", None))
    forbidden = {sre_constants.GROUPREF, sre_constants.GROUPREF_EXISTS, sre_constants.ASSERT, sre_constants.ASSERT_NOT,
                 getattr(sre_constants, "GROUPREF_IGNORE", None), getattr(sre_constants, "ATOMIC_GROUP", None)}

    def children(op, av) -> List[Any]:
        if op in repeats:
            return [av[2]]
        if op is sre_constants.SUBPATTERN:
            return [av[3]]
        if op is sre_constants.BRANCH:
            return list(av[1])
        return []

    def has_repeat(sub) -> bool:
        return any(op in repeats or any(has_repeat(c) for c in children(op, av)) for op, av in sub)

    def walk(sub) -> Optional[str]:
        for op, av in sub:
            if op in forbidden:
                return "back-references, look-arounds and conditionals are not allowed"
            if op in repeats and av[1] == sre_constants.MAXREPEAT and has_repeat(av[2]):
                return "an unbounded repeat around another repeat is not allowed"
            for child in children(op, av):
                found = walk(child)
                if found:
                    return found
        return None

    return walk(parsed)


def lexicon_phrase_problem(phrase: str, syntax: LexiconSyntax) -> Optional[str]:
    """Why a lexicon phrase is refused, or None. ``words``: plain words matched on word boundaries,
    ignoring case (it must contain a letter). ``regex``: a pattern in the safe subset
    (``_regex_problem``)."""
    if not any(ch.isalpha() for ch in phrase):
        return "a phrase contains at least one letter"
    if syntax == "regex":
        return _regex_problem(phrase)
    return None


def _check_phrases(phrases: List[str], syntax: LexiconSyntax) -> None:
    for i, phrase in enumerate(phrases):
        problem = lexicon_phrase_problem(phrase, syntax)
        if problem:
            raise ValueError(f"phrase {i}: {problem}")


class SignalLexicon(ContractModel):
    """A category's phrase lexicon (section 3.2): the recipe's score term ``lexicon_weight * match``,
    and the phrases a ``phrase`` rule without its own phrases tests. A match within
    ``negation_veto_words`` words after a negation cue ("not", "can't", "never", ...) is vetoed."""

    syntax: LexiconSyntax = "words"
    phrases: List[LexiconPhrase] = Field(min_length=1, max_length=MAX_LEXICON_PHRASES_CEILING)
    negation_veto_words: int = Field(default=0, ge=0, le=NEGATION_VETO_MAX_WORDS, description="0 turns the veto off.")

    @model_validator(mode="after")
    def _lexicon(self):
        _check_phrases(self.phrases, self.syntax)
        return self


class SignalSimilarParams(ContractModel):
    """``similar_to_examples``: the segment's kNN share of the category over the example bank is at
    least ``min_share``."""

    type: Literal["similar_to_examples"]
    min_share: float = Field(ge=0, le=1)


class SignalPhraseParams(ContractModel):
    """``phrase``: a lexicon matches the segment's masked text. With no ``phrases`` the rule tests the
    recipe's own lexicon (and its negation veto)."""

    type: Literal["phrase"]
    syntax: LexiconSyntax = "words"
    phrases: Optional[List[LexiconPhrase]] = Field(default=None, min_length=1, max_length=MAX_LEXICON_PHRASES_CEILING)
    negation_veto_words: int = Field(default=0, ge=0, le=NEGATION_VETO_MAX_WORDS)

    @model_validator(mode="after")
    def _phrase(self):
        if self.phrases is None:
            if self.syntax != "words" or self.negation_veto_words:
                raise ValueError("a phrase rule on the recipe's lexicon takes the lexicon's syntax and veto")
        else:
            _check_phrases(self.phrases, self.syntax)
        return self


class SignalSpeakerParams(ContractModel):
    """``speaker``: the segment's speaker. It must agree with the category's scope."""

    type: Literal["speaker"]
    speaker: SpeakerRole

    @model_validator(mode="after")
    def _speaker(self):
        if self.speaker not in (SpeakerRole.AGENT, SpeakerRole.CALLER):
            raise ValueError("a speaker rule names AGENT or CALLER")
        return self


class SignalPositionParams(ContractModel):
    """``call_position``: where the segment starts, as a fraction of the call's duration, lies in
    ``[start_from, start_to]`` (quartile chips: 0-0.25, 0.25-0.5, ...)."""

    type: Literal["call_position"]
    start_from: float = Field(default=0.0, ge=0, le=1)
    start_to: float = Field(default=1.0, ge=0, le=1)

    @model_validator(mode="after")
    def _window(self):
        if self.start_to <= self.start_from:
            raise ValueError("start_to follows start_from")
        return self


SignalRuleParams = Annotated[Union[SignalSimilarParams, SignalPhraseParams, SignalSpeakerParams, SignalPositionParams],
                             Field(discriminator="type")]


class SignalRule(ContractModel):
    rule_id: SignalNodeId = Field(description="Unique in the recipe; named in each hit's rule outcomes.")
    params: SignalRuleParams


class SignalRuleExpr(ContractModel):
    """A filter expression: ``rule`` (a leaf), ``all``/``any`` over children, or ``not`` of one child."""

    op: Literal["all", "any", "not", "rule"]
    rule: Optional[SignalRule] = None
    children: List["SignalRuleExpr"] = Field(default_factory=list, max_length=MAX_RECIPE_RULES_CEILING)

    @model_validator(mode="after")
    def _expr(self):
        if (self.op == "rule") != (self.rule is not None):
            raise ValueError("a rule node carries its rule, and only a rule node does")
        if self.op == "rule" and self.children:
            raise ValueError("a rule node has no children")
        if self.op == "not" and len(self.children) != 1:
            raise ValueError("'not' has exactly one child")
        if self.op in ("all", "any") and not self.children:
            raise ValueError("'all' and 'any' have at least one child")
        return self

    def rules(self) -> List[SignalRule]:
        """The leaf rules, in document order."""
        if self.rule is not None:
            return [self.rule]
        return [r for c in self.children for r in c.rules()]

    def depth(self) -> int:
        return 1 + max((c.depth() for c in self.children), default=0) if self.op != "rule" else 0


SignalRuleExpr.model_rebuild()


class SignalRecipe(ContractModel):
    """How a category is found (section 4.1). A segment fires the category when the ``filter`` passes
    and ``kNN share + lexicon_weight * lexicon match`` reaches ``threshold``; at most two categories
    fire per segment, largest margin first. The subcategory is the kNN vote among the neighbours that
    carry the category. ``check: gemma`` sends the category's spans to today's stage-2 prompt, which
    confirms (with its subcategory) or rejects them; it never changes the category."""

    engine: Literal["gemma", "rules"] = Field(default="rules", description="gemma keeps today's Gemma stages for this category even under rules detection.")
    filter: Optional[SignalRuleExpr] = None
    lexicon: Optional[SignalLexicon] = None
    lexicon_weight: float = Field(default=0.0, ge=0, le=1, description="b in the score; 0 leaves the lexicon out of the score.")
    threshold: float = Field(default=0.375, ge=0.05, le=0.95)
    check: Literal["none", "gemma"] = "none"
    origin: Optional[ShortText] = Field(default=None, description="e.g. 'pack:retail@1' when the recipe came from a pack (display only).")

    @model_validator(mode="after")
    def _recipe(self):
        rules = self.filter.rules() if self.filter is not None else []
        ids = [r.rule_id for r in rules]
        if len(ids) != len(set(ids)):
            raise ValueError("rule IDs are unique in a recipe")
        if len(rules) > MAX_RECIPE_RULES_CEILING:
            raise ValueError(f"at most {MAX_RECIPE_RULES_CEILING} rules in a recipe")
        if self.filter is not None and self.filter.depth() > MAX_RULE_DEPTH:
            raise ValueError(f"a recipe filter nests at most {MAX_RULE_DEPTH} deep")
        if self.lexicon_weight > 0 and self.lexicon is None:
            raise ValueError("a lexicon weight needs a lexicon")
        if any(isinstance(r.params, SignalPhraseParams) and r.params.phrases is None for r in rules) and self.lexicon is None:
            raise ValueError("a phrase rule without phrases tests the recipe's lexicon, so the recipe needs one")
        return self


class SignalRulesConfig(ContractModel):
    """Taxonomy-wide rules-engine settings (1.4.0): the pinned example bank and the kNN vote. The
    taxonomy's own text (names, glosses, examples) is always part of the bank."""

    bank: Optional[SignalExampleBankRef] = Field(default=None, description="An installed example bank pack; null uses only the taxonomy's own text.")
    knn: SignalKnnSettings = Field(default_factory=SignalKnnSettings)


def recipe_rule_count(recipe: SignalRecipe) -> int:
    return len(recipe.filter.rules()) if recipe.filter is not None else 0


def recipe_phrase_lists(recipe: SignalRecipe) -> Iterator[Tuple[str, List[str]]]:
    """(relative path, phrases) of every phrase list in the recipe, in document order: the lexicon,
    then each phrase rule with its own phrases."""
    if recipe.lexicon is not None:
        yield "recipe.lexicon.phrases", list(recipe.lexicon.phrases)
    if recipe.filter is not None:
        def walk(expr: SignalRuleExpr, path: str) -> Iterator[Tuple[str, List[str]]]:
            if expr.rule is not None:
                params = expr.rule.params
                if isinstance(params, SignalPhraseParams) and params.phrases is not None:
                    yield f"{path}.rule.params.phrases", list(params.phrases)
            for k, child in enumerate(expr.children):
                yield from walk(child, f"{path}.children[{k}]")
        yield from walk(recipe.filter, "recipe.filter")


# --- The taxonomy -------------------------------------------------------------------------


class SignalField(ContractModel):
    """An admin-defined extraction field. Every field is optional in the extraction schema (there is
    no ``required``): a field the engine leaves out is ``absent``."""

    field_id: SignalNodeId = Field(description="Immutable; unique on its category + subcategory path; 'quote' is reserved.")
    name: str = Field(min_length=1, max_length=40)
    type: SignalFieldType
    description: str = Field(min_length=1, max_length=200, description="One line: the extractor's only instruction for this field.")
    enum_values: List[EnumLabel] = Field(default_factory=list, max_length=12, description="enum only (at least one), unique ignoring case.")
    pii_class: FieldPiiClass = Field(description="Never one of FORBIDDEN_FIELD_PII_CLASSES.")

    @model_validator(mode="after")
    def _field(self):
        if self.field_id in RESERVED_FIELD_IDS:
            raise ValueError("field ID 'quote' is reserved for narrow_quote")
        if self.pii_class in FORBIDDEN_FIELD_PII_CLASSES:
            raise ValueError(f"pii_class {self.pii_class.value} is masked before any model sees it and can never be a field")
        if (self.type is SignalFieldType.ENUM) != bool(self.enum_values):
            raise ValueError("an enum field lists its values, and only an enum field does")
        _unique_ci(self.enum_values, "enum values")
        _one_line(self.description, "a field description")
        return self


def _check_fields(fields: List[SignalField], taken: Tuple[str, ...] = ()) -> None:
    ids = [f.field_id for f in fields]
    if len(ids) != len(set(ids)) or set(ids) & set(taken):
        raise ValueError("field IDs are unique on a category + subcategory path")


class SignalSubcategory(ContractModel):
    """One stage-2 option under a category. Subcategories of one category are mutually exclusive."""

    subcategory_id: SignalNodeId = Field(description="Immutable; unique in its category; never 'other' or 'not'.")
    name: str = Field(min_length=1, max_length=40)
    gloss: str = Field(min_length=1, max_length=80, description="The stage-2 option text. Outer ceiling 80; Store enforces ContractParameters.max_option_gloss_chars (40).")
    description: Optional[str] = Field(default=None, max_length=240, description="For people and the stage-3 extractor.")
    examples: List[ExampleText] = Field(default_factory=list, max_length=5, description="Preview display, the Gemma extractor, and the fake engines.")
    fields: List[SignalField] = Field(default_factory=list, max_length=8)
    narrow_quote: Optional[bool] = Field(default=None, description="Null inherits the category's.")
    active: bool = Field(default=True, description="false retires the subcategory; there is no delete.")

    @model_validator(mode="after")
    def _subcategory(self):
        if self.subcategory_id in RESERVED_SUBCATEGORY_IDS:
            raise ValueError("subcategory IDs 'other' and 'not' are reserved")
        _check_fields(self.fields)
        return self


MAX_ACTIVE_SUBCATEGORIES_CEILING = 12
"""Outer ceiling on active subcategories per category; ContractParameters.max_active_subcategories may be lower."""
MAX_CUSTOM_CATEGORIES_CEILING = 16
"""Outer ceiling on custom categories in total, active or not; ContractParameters.max_custom_signal_categories caps the active ones."""


class SignalCategory(ContractModel):
    """A top-level category: one of the eight built-ins (``builtin``, fixed text) or a custom one."""

    category_id: SignalNodeId = Field(description="Immutable. Built-ins use their ContactSignalKind value; a custom category never uses 'none' or 'custom' (RESERVED_CATEGORY_IDS).")
    builtin: bool = Field(description="True iff category_id is a key of BUILTIN_SIGNAL_CATEGORIES.")
    name: str = Field(min_length=1, max_length=40, description="Built-ins: the constant. Display text, and the head of stage 2's question and its Other/Not options.")
    gloss: str = Field(min_length=1, max_length=80, description="The stage-1 option text. Built-ins: the constant. Custom: outer ceiling 80; Store enforces max_option_gloss_chars (40).")
    description: Optional[str] = Field(default=None, max_length=240, description="Custom categories only (null on built-ins); only the stage-3 extractor reads it.")
    speaker: Optional[SpeakerRole] = Field(default=None, description="AGENT or CALLER; null means either (custom only). Built-ins: the constant.")
    examples: List[ExampleText] = Field(default_factory=list, max_length=5)
    threshold: Optional[float] = Field(default=None, ge=0.05, le=0.95, description="Stage-1 threshold; null uses the engine default (custom: the minimum built-in threshold).")
    subcategory_threshold: Optional[float] = Field(default=None, ge=0.05, le=0.95, description="Stage-2 threshold; null uses the engine default.")
    subcategories: List[SignalSubcategory] = Field(default_factory=list, max_length=24, description="At most 12 active.")
    fields: List[SignalField] = Field(default_factory=list, max_length=8)
    narrow_quote: bool = False
    active: bool = Field(default=True, description="Built-ins are always active; false retires a custom category.")
    recipe: Optional[SignalRecipe] = Field(default=None, description="Added in 1.4.0. How the rules engine finds this category under rules detection; null keeps today's Gemma stages. Left out of taxonomy_digest while null, so older versions keep their digests.")

    @model_validator(mode="after")
    def _category(self):
        constant = BUILTIN_SIGNAL_CATEGORIES.get(self.category_id)
        if self.category_id in RESERVED_CATEGORY_IDS:
            raise ValueError("category IDs 'none' and 'custom' are reserved")
        if self.builtin != (constant is not None):
            raise ValueError("builtin is true exactly for the eight built-in category IDs")
        if constant is not None:
            if (self.name, self.gloss, self.speaker) != (constant.name, constant.gloss, constant.speaker):
                raise ValueError(f"built-in category {self.category_id} keeps its fixed name, gloss and speaker")
            if not self.active or self.description is not None:
                raise ValueError(f"built-in category {self.category_id} is always active and has no description")
        elif self.speaker not in (None, SpeakerRole.AGENT, SpeakerRole.CALLER):
            raise ValueError("a custom category is scoped to AGENT, CALLER or either (null)")
        ids = [s.subcategory_id for s in self.subcategories]
        if len(ids) != len(set(ids)):
            raise ValueError("subcategory IDs are unique in their category")
        active = [s for s in self.subcategories if s.active]
        if len(active) > MAX_ACTIVE_SUBCATEGORIES_CEILING:
            raise ValueError(f"at most {MAX_ACTIVE_SUBCATEGORIES_CEILING} active subcategories per category")
        _unique_ci([s.name for s in active], "active subcategory names")
        _check_fields(self.fields)
        for sub in self.subcategories:
            _check_fields(sub.fields, tuple(f.field_id for f in self.fields))
        if self.recipe is not None and self.recipe.filter is not None:
            for rule in self.recipe.filter.rules():
                if isinstance(rule.params, SignalSpeakerParams) and self.speaker is not None and rule.params.speaker is not self.speaker:
                    raise ValueError(f"a speaker rule of category {self.category_id} agrees with the category's speaker")
        return self


class SignalTaxonomy(ContractModel):
    """The whole taxonomy: one document, versioned whole. Every built-in appears exactly once; at
    most 16 custom categories in total; IDs unique; active names unique ignoring case; field IDs
    unique on each path. These Pydantic bounds are fixed outer ceilings. The tighter caps (active
    custom categories, active subcategories, fields per path, gloss length) are ContractParameters
    that Store's save validator reads (``signal_taxonomy_cap_violations``)."""

    categories: List[SignalCategory] = Field(min_length=len(BUILTIN_SIGNAL_CATEGORIES), max_length=len(BUILTIN_SIGNAL_CATEGORIES) + MAX_CUSTOM_CATEGORIES_CEILING)
    rules: Optional[SignalRulesConfig] = Field(default=None, description="Added in 1.4.0. The rules engine's example bank and kNN settings; null uses the defaults and only the taxonomy's own text. Left out of taxonomy_digest while null.")

    @model_validator(mode="after")
    def _taxonomy(self):
        ids = [c.category_id for c in self.categories]
        if len(ids) != len(set(ids)):
            raise ValueError("category IDs are unique")
        if not set(BUILTIN_SIGNAL_CATEGORIES) <= set(ids):
            raise ValueError("every built-in category appears exactly once")
        if sum(1 for c in self.categories if not c.builtin) > MAX_CUSTOM_CATEGORIES_CEILING:
            raise ValueError(f"at most {MAX_CUSTOM_CATEGORIES_CEILING} custom categories in total")
        _unique_ci([c.name for c in self.categories if c.active], "active category names")
        return self

    def category(self, category_id: str) -> Optional[SignalCategory]:
        return next((c for c in self.categories if c.category_id == category_id), None)


def builtin_signal_taxonomy() -> SignalTaxonomy:
    """The install-time version 1 that Store's migration seeds: the eight built-ins only."""
    return SignalTaxonomy(categories=[
        SignalCategory(category_id=cid, builtin=True, name=b.name, gloss=b.gloss, speaker=b.speaker)
        for cid, b in BUILTIN_SIGNAL_CATEGORIES.items()
    ])


class SignalSettings(ContractModel):
    pipeline: SignalPipeline = Field(default="v2", description="v2 is the current process for new calls and reanalysis. v1 and shadow remain readable for historical snapshots.")
    v1_fallback: bool = Field(default=False, description="Legacy/fake planner compatibility: permits v1 when no usable v2 entries exist, labelled with pipeline_note. The real semantic/Laya/Gemma process never downgrades to v1.")
    fallback_extraction_entry_id: Optional[ShortText] = Field(default=None, description="The stage-3 in-job fallback entry; null = the signal_extraction default (Gemma, call1-bundled).")
    detection: Literal["model", "rules"] = Field(default="model", description="Added in 1.4.0 (docs/SignalsEmbeddings.md). model: today's Gemma stages for every category, recipes ignored. rules: a category whose recipe engine is rules is decided by the rules engine; the rest keep Gemma.")


class SignalTaxonomyVersion(ContractModel):
    """One immutable published version. Only the text tombstone (``redactSignalTaxonomyText``)
    changes it, and that keeps the digest."""

    version: int = Field(ge=1)
    digest: Sha256Digest = Field(description="taxonomy_digest(taxonomy) as published; kept when the text is redacted.")
    taxonomy: SignalTaxonomy
    published_at: Timestamp
    published_by_account_id: Optional[ResourceId] = Field(description="Null for the install-time version 1 (built-ins only).")
    notes: Optional[SafeText] = None
    text_redacted: bool = Field(default=False, description="True once the custom text was tombstoned (redact_signal_taxonomy_text: each custom text path becomes '[REDACTED <n>]', numbered in document order, built-in constants kept); a redacted version cannot be minted into a snapshot.")
    redacted_at: Optional[Timestamp] = None
    redacted_by_account_id: Optional[ResourceId] = None

    @model_validator(mode="after")
    def _digest(self):
        if not self.text_redacted and self.digest != taxonomy_digest(self.taxonomy):
            raise ValueError("a version's digest is taxonomy_digest(taxonomy) unless its text was redacted")
        if self.text_redacted and self.taxonomy != redact_signal_taxonomy_text(self.taxonomy):
            raise ValueError("a redacted version's taxonomy is redact_signal_taxonomy_text of itself: every custom text path tombstoned")
        if self.text_redacted != (self.redacted_at is not None) or (self.redacted_by_account_id is not None and not self.text_redacted):
            raise ValueError("redaction time and account are set exactly on a redacted version")
        return self

    @property
    def ref(self) -> SignalTaxonomyRef:
        return SignalTaxonomyRef(version=self.version, digest=self.digest)


class SignalTaxonomyRecord(ContractModel):
    """``GET /signals/taxonomy``: the current published version plus the settings."""

    current: SignalTaxonomyVersion
    settings: SignalSettings
    record_version: int = Field(ge=1, description="Optimistic-concurrency token for taxonomy and settings saves.")
    updated_at: Timestamp
    updated_by_account_id: Optional[ResourceId]


class SignalTaxonomySave(ContractModel):
    """``PUT /signals/taxonomy``. A save whose ``taxonomy_digest`` equals the current version's
    returns the record unchanged (no-op replay); any other save publishes version N+1. A stale
    ``expected_record_version`` is 409 ``signal_taxonomy_conflict``."""

    taxonomy: SignalTaxonomy
    expected_record_version: int = Field(ge=1)
    notes: Optional[SafeText] = None


class SignalSettingsSave(ContractModel):
    settings: SignalSettings
    expected_record_version: int = Field(ge=1)


class SignalTaxonomyRedaction(ContractModel):
    """``POST /signals/taxonomy/versions/{version}/redaction``: tombstone a non-current version's
    custom text. ``digest`` must be the version's (the one the admin reviewed)."""

    digest: Sha256Digest
    reason: SafeText = Field(min_length=1)


class SignalTaxonomySnapshotRequest(ContractModel):
    """``POST /conversations/{id}/signal-taxonomy-snapshots``: Store copies one published,
    unredacted version and the current settings into the conversation (slot ``signals:v<version>``).
    A repeat returns the same linked artifact while the settings are unchanged
    (``signal_taxonomy_snapshot_current``); after a settings change it links a new version in the slot."""

    version: int = Field(ge=1)


# --- Pure digests (section 6.3 and 7.2). Thresholds are excluded from every one. --------------


def _digest(kind: str, **payload) -> str:
    return canonical_digest({"digest": kind, **payload})


def _digest_payload(taxonomy: SignalTaxonomy) -> Dict[str, Any]:
    """The taxonomy's full dump, minus the 1.4.0 fields while they are null (``SignalTaxonomy.rules``
    and ``SignalCategory.recipe``), so every version published before 1.4.0 keeps its digest."""
    data = taxonomy.model_dump(mode="json")
    if data.get("rules") is None:
        data.pop("rules", None)
    for c in data["categories"]:
        if c.get("recipe") is None:
            c.pop("recipe", None)
    return data


def taxonomy_digest(taxonomy: SignalTaxonomy) -> str:
    """``canonical_digest`` of the whole taxonomy (thresholds and recipes included): what a version
    and a snapshot are identified by, and what makes a save a no-op. The 1.4.0 fields are left out
    while null (``_digest_payload``)."""
    return canonical_digest(_digest_payload(taxonomy))


def recipe_digest(c: SignalCategory) -> Optional[str]:
    """What the rules engine decides a category with (engine, filter, lexicon, weight, threshold,
    check; not ``origin``), or None without a recipe. Rule decisions record its first 12 hex digits."""
    if c.recipe is None:
        return None
    return _digest("signals.recipe.v1", category_id=c.category_id, speaker=c.speaker.value if c.speaker else None,
                   recipe=c.recipe.model_dump(mode="json", exclude={"origin"}))


def rules_categories(taxonomy: SignalTaxonomy, settings: "SignalSettings") -> List[SignalCategory]:
    """Active categories whose recipe selects rules. The legacy taxonomy-wide detection setting
    is accepted for snapshot compatibility; each category's recipe determines its engine."""
    return [c for c in taxonomy.categories if c.active and c.recipe is not None and c.recipe.engine == "rules"]


def category_digest(c: SignalCategory) -> str:
    """What stage 1 decides on: ID, gloss and speaker. Hit identity (``signal_hit_id``) uses its first
    12 hex digits, so editing the name, the description, a threshold, any subcategory or field, or
    another category keeps hit IDs and their feedback."""
    return _digest("signals.category.v1", category_id=c.category_id, gloss=c.gloss, speaker=c.speaker.value if c.speaker else None)


def subcategory_digest(s: SignalSubcategory) -> str:
    """What stage 2 decides on for one option: ID and gloss. Subcategory feedback keys on it."""
    return _digest("signals.subcategory.v1", subcategory_id=s.subcategory_id, gloss=s.gloss)


def stage1_options(t: SignalTaxonomy, scope: SpeakerRole) -> List[Tuple[str, str]]:
    """The ordered (category ID, gloss) options a segment of this speaker scope sees, before 'none':
    every active category scoped to that speaker or to either. UNKNOWN (unattributed mono) segments
    see only either-speaker categories (decision 22, Q4); SYSTEM turns are never scored."""
    if scope is SpeakerRole.SYSTEM:
        return []
    return [(c.category_id, c.gloss) for c in t.categories if c.active and (c.speaker is None or c.speaker is scope)]


def stage1_digest(t: SignalTaxonomy, scope: SpeakerRole) -> str:
    """The ordered option set a scope's segments see. Stage 1 reruns for a scope when it changes."""
    return _digest("signals.stage1.v1", scope=scope.value, options=[list(o) for o in stage1_options(t, scope)])


def stage2_options(c: SignalCategory) -> List[Tuple[str, str]]:
    """The ordered (subcategory ID, gloss) options of stage 2, before Other and Not."""
    return [(s.subcategory_id, s.gloss) for s in c.subcategories if s.active]


def stage2_digest(c: SignalCategory) -> str:
    """The category name (the question head and its Other/Not options) plus the active
    subcategories' IDs and glosses. Stage 2 reruns for the category's spans when it changes."""
    return _digest("signals.stage2.v1", category_id=c.category_id, name=c.name, options=[list(o) for o in stage2_options(c)])


def path_fields(c: SignalCategory, s: Optional[SignalSubcategory] = None) -> List[SignalField]:
    """The extraction schema of a span: the category's fields, then the subcategory's."""
    return list(c.fields) + (list(s.fields) if s is not None else [])


def effective_narrow_quote(c: SignalCategory, s: Optional[SignalSubcategory] = None) -> bool:
    return s.narrow_quote if s is not None and s.narrow_quote is not None else c.narrow_quote


def stage3_planned(c: SignalCategory, s: Optional[SignalSubcategory] = None) -> bool:
    """Stage 3 runs on a span only when its path has fields or narrow_quote on."""
    return bool(path_fields(c, s)) or effective_narrow_quote(c, s)


def stage3_digest(c: SignalCategory, s: Optional[SignalSubcategory] = None) -> str:
    """The path's fields, narrow_quote, and the names and descriptions the extractor reads. ``s`` is
    null for a span whose stage-2 decision was 'other'."""
    return _digest(
        "signals.stage3.v1",
        category_id=c.category_id, category_name=c.name, category_description=c.description,
        subcategory_id=s.subcategory_id if s else None, subcategory_name=s.name if s else None, subcategory_description=s.description if s else None,
        fields=[f.model_dump(mode="json") for f in path_fields(c, s)], narrow_quote=effective_narrow_quote(c, s),
    )


# --- Save-time rules Store applies (sections 9.4 and 9.6) -------------------------------------


class SignalCapViolation(NamedTuple):
    field: str
    """The path, e.g. ``categories[9].subcategories[2].gloss``: what ``details.field`` names."""
    cap: str
    """The ContractParameters name."""
    limit: int
    actual: int


def signal_taxonomy_cap_violations(taxonomy: SignalTaxonomy, parameters: Optional[ContractParameters] = None) -> List[SignalCapViolation]:
    """The section 9.6 caps, read from ``parameters`` (Store's effective ContractParameters). Store's
    save validator refuses a taxonomy with any violation (``validation_failed``, ``details.field`` the
    first path). Caps apply to active nodes: a retired node never runs."""
    p = parameters or CONTRACT_PARAMETERS
    out: List[SignalCapViolation] = []
    active_custom = [i for i, c in enumerate(taxonomy.categories) if not c.builtin and c.active]
    if len(active_custom) > p.max_custom_signal_categories:
        out.append(SignalCapViolation(f"categories[{active_custom[p.max_custom_signal_categories]}]", "max_custom_signal_categories", p.max_custom_signal_categories, len(active_custom)))
    for i, c in enumerate(taxonomy.categories):
        if not c.active:
            continue
        base = f"categories[{i}]"
        if not c.builtin and len(c.gloss) > p.max_option_gloss_chars:
            out.append(SignalCapViolation(f"{base}.gloss", "max_option_gloss_chars", p.max_option_gloss_chars, len(c.gloss)))
        active = [j for j, s in enumerate(c.subcategories) if s.active]
        if len(active) > p.max_active_subcategories:
            out.append(SignalCapViolation(f"{base}.subcategories[{active[p.max_active_subcategories]}]", "max_active_subcategories", p.max_active_subcategories, len(active)))
        if len(c.fields) > p.max_fields_per_path:
            out.append(SignalCapViolation(f"{base}.fields", "max_fields_per_path", p.max_fields_per_path, len(c.fields)))
        for j in active:
            s = c.subcategories[j]
            if len(s.gloss) > p.max_option_gloss_chars:
                out.append(SignalCapViolation(f"{base}.subcategories[{j}].gloss", "max_option_gloss_chars", p.max_option_gloss_chars, len(s.gloss)))
            n = len(path_fields(c, s))
            if n > p.max_fields_per_path:
                out.append(SignalCapViolation(f"{base}.subcategories[{j}].fields", "max_fields_per_path", p.max_fields_per_path, n))
        if c.recipe is not None:
            n = recipe_rule_count(c.recipe)
            if n > p.max_signal_recipe_rules:
                out.append(SignalCapViolation(f"{base}.recipe.filter", "max_signal_recipe_rules", p.max_signal_recipe_rules, n))
            for rel, phrases in recipe_phrase_lists(c.recipe):
                if len(phrases) > p.max_signal_lexicon_phrases:
                    out.append(SignalCapViolation(f"{base}.{rel}", "max_signal_lexicon_phrases", p.max_signal_lexicon_phrases, len(phrases)))
    return out


def _node_text(base: str, node) -> Iterator[Tuple[str, str]]:
    yield f"{base}.name", node.name
    yield f"{base}.gloss", node.gloss
    if node.description is not None:
        yield f"{base}.description", node.description
    for k, example in enumerate(node.examples):
        yield f"{base}.examples[{k}]", example
    for k, f in enumerate(node.fields):
        yield f"{base}.fields[{k}].name", f.name
        yield f"{base}.fields[{k}].description", f.description
        for m, value in enumerate(f.enum_values):
            yield f"{base}.fields[{k}].enum_values[{m}]", value


def signal_taxonomy_text_paths(taxonomy: SignalTaxonomy) -> Iterator[Tuple[str, str]]:
    """Every (path, text) of the taxonomy's definition text, in document order: names, glosses,
    descriptions, examples, field names and descriptions, enum values. Store runs its rule-based
    detectors (``call1/redaction.py``: SSN, card, phone, account number, PIN) over each on save and
    on preview, and refuses a match with ``validation_failed`` and ``details.field`` set to the path,
    never the value (section 9.4). Process masks the same text with the call's sensitive values
    before any engine sees it."""
    for i, c in enumerate(taxonomy.categories):
        yield from _node_text(f"categories[{i}]", c)
        for j, s in enumerate(c.subcategories):
            yield from _node_text(f"categories[{i}].subcategories[{j}]", s)
        if c.recipe is not None:  # 1.4.0: lexicon phrases are definition text too
            for rel, phrases in recipe_phrase_lists(c.recipe):
                for k, phrase in enumerate(phrases):
                    yield f"categories[{i}].{rel}[{k}]", phrase


REDACTED_SIGNAL_TEXT = "[REDACTED {n}]"
"""The tombstone for one custom text path (section 9.4), ``n`` counting from 1 in the document order
of ``signal_taxonomy_text_paths`` (built-in constants skipped). Numbered so that a redacted version
still satisfies the uniqueness rules (active category and subcategory names, enum values) that a
bare ``[REDACTED]`` on every path would break. Always under 40 characters."""


def redact_signal_taxonomy_text(taxonomy: SignalTaxonomy) -> SignalTaxonomy:
    """The tombstoned copy of ``taxonomy`` that ``redactSignalTaxonomyText`` stores (section 9.4):
    every custom text path of ``signal_taxonomy_text_paths`` becomes ``REDACTED_SIGNAL_TEXT`` with
    its own number; a built-in category's fixed name and gloss stay (they are constants, not customer
    text). IDs, types, PII classes, thresholds, flags and structure are kept, and the version keeps
    its digest. Pure, deterministic and idempotent: redacting a redacted taxonomy returns it."""
    counter = iter(range(1, 1_000_000))

    def tomb() -> str:
        return REDACTED_SIGNAL_TEXT.format(n=next(counter))

    def node(d: dict, constant: bool) -> None:
        if not constant:
            d["name"], d["gloss"] = tomb(), tomb()
        if d.get("description") is not None:
            d["description"] = tomb()
        d["examples"] = [tomb() for _ in d["examples"]]
        for f in d["fields"]:
            f["name"], f["description"] = tomb(), tomb()
            f["enum_values"] = [tomb() for _ in f["enum_values"]]

    def recipe(r: Optional[dict]) -> None:  # 1.4.0: in recipe_phrase_lists order
        if r is None:
            return
        if r.get("lexicon") is not None:
            r["lexicon"]["phrases"] = [tomb() for _ in r["lexicon"]["phrases"]]
            r["lexicon"]["syntax"] = "words"

        def walk(expr: dict) -> None:
            params = (expr.get("rule") or {}).get("params") or {}
            if params.get("type") == "phrase" and params.get("phrases") is not None:
                params["phrases"] = [tomb() for _ in params["phrases"]]
                params["syntax"] = "words"
            for child in expr.get("children") or []:
                walk(child)
        if r.get("filter") is not None:
            walk(r["filter"])

    data = taxonomy.model_dump()
    for c in data["categories"]:
        node(c, c["builtin"])
        for sub in c["subcategories"]:
            node(sub, False)
        recipe(c.get("recipe"))
    return SignalTaxonomy.model_validate(data)


# --- Snapshot artifact (signal_taxonomy_snapshot.v1) ------------------------------------------

SIGNAL_TAXONOMY_INPUT_ROLE = "taxonomy"
"""The input role under which every v2 job (``JobTypeRule.needs_signal_taxonomy``) pins its
``signal_taxonomy_snapshot``; the merge takes it as an optional input."""


class SignalTaxonomySnapshotContent(ContractModel):
    """``signal_taxonomy_snapshot.v1``: the exact taxonomy and settings a v2 graph runs with, as an
    input artifact. Only Store creates one (``artifacts.STORE_MINTED_KINDS``; sensitivity derived):

    - **published**: ``mintSignalTaxonomySnapshot`` (Process, ``jobs:write``) copies a stored,
      unredacted published version and the current settings into slot ``signals:v<version>``,
      idempotently per conversation, version and settings (``signal_taxonomy_snapshot_current``).
      Compare requests use it too.
    - **preview**: ``createSignalPreview`` copies the unsaved taxonomy of the request body into each
      preview request's draft-test slot (``draft:<request_id>:signals:preview``).

    At graph creation Store checks every v2 job's pinned snapshot against
    ``parameters.signals.taxonomy_digest`` (graph_invalid otherwise)."""

    source: Literal["published", "preview"]
    taxonomy_ref: SignalTaxonomyRef = Field(description="version null iff source is preview.")
    taxonomy: SignalTaxonomy
    settings: SignalSettings
    preview_id: Optional[ResourceId] = Field(default=None, description="Set iff source is preview.")

    @model_validator(mode="after")
    def _source(self):
        preview = self.source == "preview"
        if preview != (self.preview_id is not None) or preview != (self.taxonomy_ref.version is None):
            raise ValueError("a preview snapshot names its preview and no version; a published snapshot names its version and no preview")
        if self.taxonomy_ref.digest != taxonomy_digest(self.taxonomy):
            raise ValueError("taxonomy_ref.digest is taxonomy_digest(taxonomy)")
        return self


def signal_taxonomy_snapshot_current(content: SignalTaxonomySnapshotContent, version: SignalTaxonomyRef, settings: SignalSettings) -> bool:
    """Whether the snapshot linked in slot ``signals:v<version>`` can answer a repeat
    ``mintSignalTaxonomySnapshot``: a published snapshot of exactly that version (number and digest)
    taken with exactly the current settings. Settings are not versioned (they share
    ``record_version`` with the taxonomy), so without this check a reanalysis at an unchanged
    version would reuse an older pipeline, v1 fallback or stage-3 fallback entry. False means Store
    links a new snapshot version in the same slot."""
    return content.source == "published" and content.taxonomy_ref == version and content.settings == settings


# --- Result staleness (section 7.5) -----------------------------------------------------------


class SignalTaxonomyStatus(ContractModel):
    """Store's read-time label of a result against the current taxonomy. ``stale`` keeps its
    contract meaning (a reanalysis is underway); editing the taxonomy never marks a group stale."""

    scored_version: Optional[int] = Field(ge=1, description="The taxonomy version the result was scored with; null for v1 results.")
    current_version: int = Field(ge=1)
    outdated_stages: List[SignalStage] = Field(description="Stages whose digests differ from the current taxonomy's, in pipeline order: an update reruns exactly these (for the affected spans).")
    thresholds_changed: bool = Field(description="A threshold differs: an update re-derives spans or decisions from stored scores and runs no stage-1 model.")

    @model_validator(mode="after")
    def _status(self):
        if self.scored_version is None and (self.outdated_stages or self.thresholds_changed):
            raise ValueError("a v1 result is not compared against the taxonomy")
        if self.scored_version is not None and self.scored_version > self.current_version:
            raise ValueError("a result is scored with a published version")
        if self.outdated_stages != [s for s in SIGNAL_STAGES if s in self.outdated_stages]:
            raise ValueError("outdated stages are listed once each, in pipeline order")
        return self


def _thresholds(c: SignalCategory) -> Tuple[Optional[float], Optional[float]]:
    return c.threshold, c.subcategory_threshold


def signal_taxonomy_status(
    scored: Optional[SignalTaxonomyVersion],
    current: SignalTaxonomyVersion,
    stage1_digests: Dict[str, str],
) -> SignalTaxonomyStatus:
    """The normative outdated rule (section 7.5 table), comparing the version a result was scored
    with against the current one. ``stage1_digests`` is the result's ``ContactSignalsContent.stage1_digests``.

    - categorize: a scope's ``stage1_digest`` differs (a custom category added, retired, or its
      gloss or speaker edited).
    - subcategorize: a category present in both versions has a different ``stage2_digest`` (its
      name, or its active subcategories' IDs and glosses).
    - extract: a path present in both versions has a different ``stage3_digest`` and runs stage 3 in
      either version (fields or narrow_quote).
    - thresholds_changed: a category's ``threshold`` or ``subcategory_threshold`` differs.

    A v1 result (``scored`` null) is never outdated. Engine or calibration changes are not
    taxonomy changes: they rerun only with ``rescore_signals``. If the scored version's text was
    redacted, stage 2 and 3 compare against tombstoned text and read outdated wherever custom text
    was tombstoned; an update then simply reruns them (stage 1 compares the stored digests, so it is
    unaffected)."""
    if scored is None:
        return SignalTaxonomyStatus(scored_version=None, current_version=current.version, outdated_stages=[], thresholds_changed=False)
    old_t, new_t = scored.taxonomy, current.taxonomy
    outdated: List[str] = []
    if any(scope in SIGNAL_SCOPE_KEYS and stage1_digest(new_t, SpeakerRole(scope)) != digest for scope, digest in stage1_digests.items()):
        outdated.append("categorize")
    pairs = [(old_t.category(c.category_id), c) for c in new_t.categories]
    pairs = [(o, n) for o, n in pairs if o is not None and (o.active or n.active)]
    if any(stage2_digest(o) != stage2_digest(n) for o, n in pairs):
        outdated.append("subcategorize")

    def path_changed(o: SignalCategory, n: SignalCategory) -> bool:
        old_subs = {s.subcategory_id: s for s in o.subcategories}
        paths = [(None, None)] + [(old_subs.get(s.subcategory_id), s) for s in n.subcategories if s.subcategory_id in old_subs]
        return any(stage3_digest(o, os) != stage3_digest(n, ns) and (stage3_planned(o, os) or stage3_planned(n, ns)) for os, ns in paths)

    if any(path_changed(o, n) for o, n in pairs):
        outdated.append("extract")
    thresholds = any(_thresholds(o) != _thresholds(n) for o, n in pairs)
    return SignalTaxonomyStatus(scored_version=scored.version, current_version=current.version, outdated_stages=outdated, thresholds_changed=thresholds)


# --- Alert rules (section 9.2) ----------------------------------------------------------------


class SignalAlertCondition(ContractModel):
    """A category-, subcategory- or field-level condition. ``field_id`` alone means the field was
    extracted; ``field_equals`` compares an enum value or a boolean only."""

    category_id: SignalNodeId
    subcategory_id: Optional[SignalNodeId] = Field(default=None, description="A subcategory of the category, or 'other'.")
    field_id: Optional[SignalNodeId] = Field(default=None, description="Alone: the field was extracted.")
    field_equals: Optional[Union[EnumLabel, bool]] = Field(default=None, description="An enum value, or a boolean; needs field_id.")
    min_confidence: Optional[float] = Field(default=None, ge=0.05, le=0.95)

    @model_validator(mode="after")
    def _condition(self):
        if self.subcategory_id == SIGNAL_NOT_OPTION:
            raise ValueError("a rejected span is not a hit, so 'not' cannot be alerted on")
        if self.field_equals is not None and self.field_id is None:
            raise ValueError("field_equals compares a field: name it with field_id")
        return self


def signal_alert_condition_problem(condition: SignalAlertCondition, taxonomy: SignalTaxonomy) -> Optional[str]:
    """Why a condition does not fit a taxonomy, or None. Store refuses a rule save whose condition
    names an unknown node or compares a field that is not enum or boolean (``validation_failed``,
    ``details.reason`` one of ``unknown_category``, ``unknown_subcategory``, ``unknown_field``,
    ``field_equals_type``). A field named without a subcategory may be on the category or on any of
    its subcategories."""
    c = taxonomy.category(condition.category_id)
    if c is None:
        return "unknown_category"
    sub: Optional[SignalSubcategory] = None
    if condition.subcategory_id is not None and condition.subcategory_id != SIGNAL_OTHER_OPTION:
        sub = next((s for s in c.subcategories if s.subcategory_id == condition.subcategory_id), None)
        if sub is None:
            return "unknown_subcategory"
    if condition.field_id is None:
        return None
    if condition.subcategory_id is None:
        candidates = list(c.fields) + [f for s in c.subcategories for f in s.fields]
    else:
        candidates = path_fields(c, sub)
    fields = [f for f in candidates if f.field_id == condition.field_id]
    if not fields:
        return "unknown_field"
    value = condition.field_equals
    if value is None:
        return None
    if isinstance(value, bool):
        ok = any(f.type is SignalFieldType.BOOLEAN for f in fields)
    else:
        ok = any(f.type is SignalFieldType.ENUM and value in f.enum_values for f in fields)
    return None if ok else "field_equals_type"


def signal_alert_node_active(condition: SignalAlertCondition, taxonomy: SignalTaxonomy) -> bool:
    """``SignalAlertRuleRecord.node_active``: the condition's category (and subcategory) exist and are
    active in the current taxonomy. A rule on an inactive node matches nothing."""
    c = taxonomy.category(condition.category_id)
    if c is None or not c.active:
        return False
    if condition.subcategory_id in (None, SIGNAL_OTHER_OPTION):
        return True
    sub = next((s for s in c.subcategories if s.subcategory_id == condition.subcategory_id), None)
    return sub is not None and sub.active


class SignalAlertRule(ContractModel):
    rule_id: SignalNodeId
    name: str = Field(min_length=1, max_length=60, description="Definition text (section 9.4): Store's detectors check it on save.")
    condition: SignalAlertCondition
    enabled: bool = True


class SignalAlertRuleRecord(SignalAlertRule):
    record_version: int = Field(ge=1)
    node_active: bool = Field(description="False when the condition's node is inactive (or gone) in the current taxonomy; the rule then matches nothing and the editor says why.")
    created_at: Timestamp
    updated_at: Timestamp
    updated_by_account_id: ResourceId


class SignalAlertRuleSave(ContractModel):
    """``PUT /signals/alert-rules/{rule_id}``; ``rule.rule_id`` equals the path's. 0 creates. At most
    ``ContractParameters.max_signal_alert_rules`` rules."""

    rule: SignalAlertRule
    expected_record_version: int = Field(ge=0, description="0 to create; otherwise the record_version read (409 conflict with details.current_version otherwise).")


class SignalAlertMatch(ContractModel):
    """An enabled rule that matches a result now (computed at read time)."""

    rule_id: SignalNodeId
    name: str = Field(min_length=1, max_length=60)
    hit_ids: List[ShortText] = Field(max_length=20)


# --- Hit feedback (section 6.3; field-level feedback comes later, decision 22 Q10) -------------


class SignalHitFeedback(ContractModel):
    """A reviewer's verdicts on one hit. The category verdict keys on ``hit_id``; the subcategory
    verdict keys on (``hit_id``, ``subcategory_id``, ``subcategory_digest``), so when a later stage 2
    assigns a different subcategory the old verdict reads 'judged an earlier subcategory' and does
    not count toward the new one's precision."""

    call_id: ResourceId
    hit_id: ShortText
    category_verdict: Optional[Literal["confirmed", "dismissed"]] = None
    subcategory_id: Optional[SignalNodeId] = Field(default=None, description="The subcategory the verdict judged (or 'other').")
    subcategory_digest: Optional[ShortDigest] = None
    subcategory_verdict: Optional[Literal["confirmed", "corrected"]] = None
    corrected_subcategory_id: Optional[SignalNodeId] = Field(default=None, description="Set iff corrected; 'other' allowed. Recorded for a later per-customer fine-tune.")
    note: Optional[SafeText] = None
    account_id: ResourceId
    feedback_version: int = Field(ge=1)
    updated_at: Timestamp

    @model_validator(mode="after")
    def _verdicts(self):
        if (self.subcategory_verdict == "corrected") != (self.corrected_subcategory_id is not None):
            raise ValueError("a correction names the corrected subcategory, and only a correction does")
        if self.subcategory_verdict is not None and self.subcategory_id is None:
            raise ValueError("a subcategory verdict names the subcategory it judged")
        if SIGNAL_NOT_OPTION in (self.subcategory_id, self.corrected_subcategory_id):
            raise ValueError("'not' is a stage-2 rejection, not a subcategory; dismiss the category instead")
        return self


class SignalHitFeedbackSave(ContractModel):
    """``PUT /calls/{call_id}/signal-hits/{hit_id}/feedback``. Store fills ``subcategory_id`` and
    ``subcategory_digest`` from the current hit."""

    category_verdict: Optional[Literal["confirmed", "dismissed"]] = None
    subcategory_verdict: Optional[Literal["confirmed", "corrected"]] = None
    corrected_subcategory_id: Optional[SignalNodeId] = None
    note: Optional[SafeText] = None
    expected_feedback_version: int = Field(ge=0, description="0 creates; otherwise the feedback_version read (409 conflict with details.current_version otherwise).")

    @model_validator(mode="after")
    def _verdicts(self):
        if (self.subcategory_verdict == "corrected") != (self.corrected_subcategory_id is not None):
            raise ValueError("a correction names the corrected subcategory, and only a correction does")
        if self.corrected_subcategory_id == SIGNAL_NOT_OPTION:
            raise ValueError("'not' is a stage-2 rejection, not a subcategory; dismiss the category instead")
        return self


# --- Previews, comparisons and backfills (section 7.5) ------------------------------------------


class SignalPreviewCreate(ContractModel):
    """Test a taxonomy (usually unsaved) on up to ``signal_preview_max_calls`` calls. Store checks
    the taxonomy's text and caps as on save, mints a preview snapshot per call and creates one
    ``contact_signals_preview`` request each (priority +5). Results land in draft-test slots only."""

    taxonomy: Optional[SignalTaxonomy] = Field(default=None, description="Null tests the current published version.")
    call_ids: List[ResourceId] = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def _calls(self):
        if len(self.call_ids) != len(set(self.call_ids)):
            raise ValueError("each call once")
        return self


class SignalPreviewDiff(ContractModel):
    """A preview result against the call's published ``contact_signals`` (hit IDs)."""

    added: List[ShortText]
    removed: List[ShortText]
    relabelled: List[ShortText]
    fields_changed: List[ShortText]
    builtin_changed: List[ShortText] = Field(default_factory=list, description="Built-in hits the edit added or removed: custom options share stage 1's probabilities with the built-ins.")
    segments_changed: List[ShortText] = Field(default_factory=list, description="Hits whose label and fields are unchanged but whose segments are not: a multi-segment merge (decision 25) formed, split, grew or shrank. Hits pair by any segment they cover, so a span that became part of a merged hit is not reported as removed.")


class SignalPreviewCall(ContractModel):
    call_id: ResourceId
    request_id: ResourceId
    state: Literal["pending", "available", "failed"]
    failure_code: Optional[JobErrorCode] = None
    result: Optional[ContactSignalsContent] = Field(default=None, description="Masked on read like the published view; present when available.")
    diff: Optional[SignalPreviewDiff] = None

    @model_validator(mode="after")
    def _state(self):
        available = self.state == "available"
        if available != (self.result is not None) or (self.diff is not None and not available):
            raise ValueError("an available preview call carries its result (and diff), and only it does")
        if self.failure_code is not None and self.state != "failed":
            raise ValueError("only a failed preview call names a failure code")
        return self


class SignalPreview(ContractModel):
    """``GET /signals/previews/{preview_id}``. ``preview`` comes from the editor's 'Test on recent
    calls'; ``compare`` runs v2 beside published v1 results (a compare backfill, or a shadow-mode
    companion request)."""

    id: ResourceId
    source: Literal["preview", "compare"]
    taxonomy_ref: SignalTaxonomyRef
    calls: List[SignalPreviewCall] = Field(max_length=500)
    options_trimmed: List[ShortText] = Field(default_factory=list, description="Category or subcategory IDs whose option text the classifier had to trim.")
    created_at: Timestamp
    created_by_account_id: Optional[ResourceId] = Field(description="The admin; null for a shadow-mode companion Store created itself.")

    @model_validator(mode="after")
    def _calls(self):
        if self.source == "preview" and len(self.calls) > 10:
            raise ValueError("a preview covers at most 10 calls")
        return self


class SignalBackfillCreate(ContractModel):
    """Bring existing results up to the current taxonomy (``rescore``, publishes; priority -10) or
    run v2 beside v1 into draft slots (``compare``)."""

    mode: Literal["rescore", "compare"]
    rescore_signals: bool = Field(default=False, description="rescore only: false reruns only the outdated stages (digest-driven); true reruns every stage (engine change).")
    created_after: Timestamp
    created_before: Optional[Timestamp] = None
    max_calls: int = Field(default=200, ge=1, le=500, description="At most ContractParameters.signal_backfill_max_calls.")

    @model_validator(mode="after")
    def _window(self):
        if self.rescore_signals and self.mode != "rescore":
            raise ValueError("rescore_signals applies to a rescore backfill")
        if self.created_before is not None and self.created_before <= self.created_after:
            raise ValueError("created_before follows created_after")
        return self


class SignalBackfill(ContractModel):
    id: ResourceId
    mode: Literal["rescore", "compare"]
    taxonomy_version: int = Field(ge=1)
    calls_matched: int = Field(ge=0)
    requests_created: int = Field(ge=0)
    calls_skipped: int = Field(ge=0, description="Matched calls that needed nothing (already current) or already had a pending request (widened instead).")
    preview_id: Optional[ResourceId] = Field(default=None, description="compare only: the preview that collects the results.")
    created_at: Timestamp
    created_by_account_id: ResourceId

    @model_validator(mode="after")
    def _compare(self):
        if (self.mode == "compare") != (self.preview_id is not None):
            raise ValueError("a compare backfill, and only one, names its preview")
        return self
