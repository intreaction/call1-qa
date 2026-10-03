"""Model catalog references as Store sees them.

Process owns the catalog (weights, adapters, provider connections). Store stores only references
and the frozen per-job selection, plus a read-only snapshot Process publishes so Evaluate and admin
screens can show routes and qualification without reaching Process.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import Field, model_validator

from .common import ContractModel, ResourceId, Sha256Digest, ShortText, Timestamp
from .custody import ProviderType, RouteClass, RouteRecord


class ModelPurpose(str, Enum):
    """Purpose tags (plan, "Purpose-based model catalog")."""

    ASR = "asr"
    SPEAKER_DIARIZATION = "speaker_diarization"
    ACOUSTIC_TONE = "acoustic_tone"
    TEXT_SENTIMENT = "text_sentiment"
    EMBEDDINGS = "embeddings"
    SEMANTIC_QA = "semantic_qa"
    SUMMARY = "summary"
    CONTACT_SIGNALS = "contact_signals"
    SIGNAL_CATEGORY = "signal_category"
    """Added in 1.3.0: Contact Signals v2 stage 1, the per-segment category classifier (primary host only)."""
    SIGNAL_SUBCATEGORY = "signal_subcategory"
    """Added in 1.3.0: stage 2, the per-span subcategory classifier (primary host only). A separate
    purpose from stage 1 because the catalog keeps one default entry per purpose and the two stages
    may qualify different engines."""
    SIGNAL_EXTRACTION = "signal_extraction"
    """Added in 1.3.0: stage 3, field extraction on an isolated span (an LLM route: Gemma by default)."""
    ASR_VOCABULARY = "asr_vocabulary"
    """Added in 1.3.0 (decision 33): the vocabulary-prompted candidate pass of dual transcription
    (Whisper Small on MLX). Never a selection of its own job: the ``asr`` job freezes the entry in
    ``parameters.asr_vocabulary.candidate_entry`` and runs it after the base engine."""


LLM_PURPOSES = frozenset({ModelPurpose.SEMANTIC_QA, ModelPurpose.SUMMARY, ModelPurpose.CONTACT_SIGNALS, ModelPurpose.SIGNAL_EXTRACTION})

SIGNAL_CLASSIFIER_PURPOSES = frozenset({ModelPurpose.SIGNAL_CATEGORY, ModelPurpose.SIGNAL_SUBCATEGORY})
"""Added in 1.3.0. Classifier purposes (stages 1 and 2). No Gemma entry is registered under either
(decision 21; decision 22, Q15); the planner never substitutes one model for another."""

ALWAYS_MASKED_PURPOSES = frozenset({ModelPurpose.SIGNAL_CATEGORY, ModelPurpose.SIGNAL_SUBCATEGORY, ModelPurpose.SIGNAL_EXTRACTION})
"""Added in 1.3.0. Purposes whose jobs read masked text on every route, even with
``mask_model_text`` off (decision 22, Q12). ``SignalStageProvenance.masked`` is always true."""


class CatalogEntryStatus(str, Enum):
    AVAILABLE = "available"
    NOT_INSTALLED = "not_installed"
    INCOMPATIBLE = "incompatible"
    UNQUALIFIED = "unqualified"


class CatalogEntryRef(ContractModel):
    """A stable model ID plus its immutable manifest version."""

    entry_id: ShortText
    entry_version: int = Field(ge=1)


class FrozenSelection(ContractModel):
    """Resolved at job creation and never changed: which model, which revision, which adapter and
    output contract, and which route. A later catalog update changes only new jobs."""

    catalog_entry: CatalogEntryRef
    purpose: ModelPurpose
    model_family: ShortText
    model_revision: ShortText = Field(description="Exact revision or digest the source supports pinning; for Ollama the model digest the host reported.")
    weights_digest: Optional[Sha256Digest] = Field(default=None, description="Required on the Pro1 route: the job freezes the weights digest, not the release.")
    adapter_id: ShortText
    adapter_version: ShortText
    output_contract: ShortText = Field(description="Canonical output artifact contract, e.g. 'qa_assessment.v1'.")
    provider_model_id: ShortText = Field(description="The model ID as the provider names it (the exact advertised ID for remote entries).")
    mutable_alias: bool = Field(default=False, description="True when the provider model ID is a mutable alias; the provider-reported ID is then recorded per attempt.")
    route: RouteRecord

    @model_validator(mode="after")
    def _pro1_needs_weights_digest(self):
        if self.route.route_class is RouteClass.CALL1_CONFIDENTIAL and self.weights_digest is None:
            raise ValueError("a call1_confidential selection freezes the weights digest")
        return self


class ResourceProfile(ContractModel):
    memory_bytes: Optional[int] = Field(default=None, ge=0)
    context_limit_tokens: Optional[int] = Field(default=None, ge=0)
    output_token_limit: Optional[int] = Field(default=None, ge=0)


class CatalogEntrySnapshot(ContractModel):
    """Read-only description of one catalog entry as Process published it."""

    entry: CatalogEntryRef
    display_name: ShortText
    purposes: List[ModelPurpose]
    provider_type: ProviderType
    route_class: RouteClass
    destination_host: ShortText
    model_family: ShortText
    model_revision: ShortText
    weights_digest: Optional[Sha256Digest] = None
    status: CatalogEntryStatus
    qualified_for: List[ModelPurpose] = Field(default_factory=list, description="Purposes whose qualification set this entry passed on this installation.")
    resource_profile: ResourceProfile = Field(default_factory=ResourceProfile)
    license_notice: Optional[ShortText] = None
    mutable_alias: bool = False
    legacy_question_model_id: Optional[ShortText] = Field(default=None, description="The pre-split QuestionModel ID this entry preserves, if any.")


class CatalogSnapshot(ContractModel):
    """Published by a Process installation (scope catalog:publish); replaced whole on each publish."""

    installation_id: ResourceId
    catalog_version: ShortText
    published_at: Timestamp
    entries: List[CatalogEntrySnapshot]
    defaults: Dict[ModelPurpose, CatalogEntryRef] = Field(default_factory=dict, description="Default selection per purpose on this installation.")
