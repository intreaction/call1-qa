"""Contact Signals v2 wiring for ingest and reanalysis (docs/ContactSignalsV2.md sections 7.5, 8.1).

* **Ingest.** After the rubric snapshot, Process reads ``GET /signals/taxonomy``. When the settings
  select ``v2``, it mints a snapshot of the current version (``mintSignalTaxonomySnapshot``) and plans
  from that snapshot's content (the settings travel with it). ``v1`` and ``shadow`` build today's
  passes with no snapshot (Store adds the shadow compare companion itself).
* **Reanalysis.** Store resolves ``signal_pipeline`` and ``signal_taxonomy_version`` on the request.
  A ``contact_signals`` request on v2 pins the call's current ``signal_categories``,
  ``signal_subcategories`` and ``signal_extraction`` as ``previous_*`` and reruns only the stages
  whose digests differ (``signals_v2.plan_rerun``; ``rescore_signals`` reruns all). ``full`` and
  ``speaker_correction`` rebuild every stage. A ``contact_signals_preview`` request runs every stage
  from the snapshot Store minted for it, into its draft slots.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Optional, Tuple

from call1.contracts.artifacts import Artifact, ArtifactKind
from call1.contracts.contents import SignalCategoriesContent, SignalExtractionContent, SignalSubcategoriesContent
from call1.contracts.jobs import ReanalysisKind, ReanalysisRequest
from call1.contracts.signals import SignalTaxonomySnapshotContent
from call1.pipeline.signals_v2 import plan_rerun

from .graph import PlanError, SignalsInput
from .handlers.signal_stages import engine_defaults
from .store_client import StoreClient, StoreError

log = logging.getLogger("call1.process.signals")

PREVIOUS_KINDS = {"categories": ArtifactKind.SIGNAL_CATEGORIES, "subcategories": ArtifactKind.SIGNAL_SUBCATEGORIES,
                  "extraction": ArtifactKind.SIGNAL_EXTRACTION}


def _json(client: StoreClient, artifact: Artifact):
    return json.loads(client.download(artifact).decode("utf-8"))  # type: ignore[union-attr]


def load_snapshot(client: StoreClient, artifact: Artifact) -> SignalTaxonomySnapshotContent:
    return SignalTaxonomySnapshotContent.model_validate(_json(client, artifact))


def _unsupported(exc: StoreError) -> bool:
    """Store has no signal taxonomy route (an older Store in development): plan v1, as before 1.3.0."""
    return exc.status in (404, 501) and exc.code in ("not_implemented", "not_found", "unexpected_response")


def ingest_signals(client: StoreClient, conversation_id: str) -> Optional[SignalsInput]:
    """The contact-signals input of an ingest graph: a minted snapshot when the settings select v2,
    else None (v1)."""
    try:
        record = client.get_signal_taxonomy()
    except StoreError as exc:
        if _unsupported(exc):
            log.info("Store has no signal taxonomy (%s); planning contact signals v1", exc.code)
            return None
        raise
    if record.settings.pipeline != "v2":
        return None
    artifact = client.mint_signal_taxonomy_snapshot(conversation_id, record.current.version)
    return SignalsInput(snapshot=artifact, content=load_snapshot(client, artifact))


def request_signals(client: StoreClient, request: ReanalysisRequest, linked: Dict[Tuple[ArtifactKind, str], Artifact], *,
                    full: bool) -> Optional[SignalsInput]:
    """The contact-signals input of a reanalysis graph, or None for v1. ``full`` rebuilds every stage."""
    if request.kind is ReanalysisKind.CONTACT_SIGNALS_PREVIEW:
        if not request.signal_taxonomy_snapshot_artifact_id:
            raise PlanError("a signal preview carries the taxonomy snapshot Store minted for it")
        artifact = client.get_artifact(request.signal_taxonomy_snapshot_artifact_id)
        return SignalsInput(snapshot=artifact, content=load_snapshot(client, artifact), pipeline="v2", draft_request_id=request.id,
                            preview_id=request.signal_preview_id, require_v2=True)
    if request.signal_pipeline != "v2":
        return None
    version = request.signal_taxonomy_version or client.get_signal_taxonomy().current.version
    artifact = client.mint_signal_taxonomy_snapshot(request.conversation_id, version)
    content = load_snapshot(client, artifact)
    if full:
        return SignalsInput(snapshot=artifact, content=content, pipeline="v2")
    previous = {name: linked[(kind, "")] for name, kind in PREVIOUS_KINDS.items() if (kind, "") in linked}
    categories = SignalCategoriesContent.model_validate(_json(client, previous["categories"])) if "categories" in previous else None
    subcategories = SignalSubcategoriesContent.model_validate(_json(client, previous["subcategories"])) if "subcategories" in previous else None
    extraction = SignalExtractionContent.model_validate(_json(client, previous["extraction"])) if "extraction" in previous else None
    transcript = linked.get((ArtifactKind.TRANSCRIPT, ""))
    attribution = linked.get((ArtifactKind.SPEAKER_ATTRIBUTION, ""))
    defaults = engine_defaults(categories.provenance.catalog_entry_id if categories is not None else None)
    stage2 = engine_defaults(subcategories.provenance.catalog_entry_id if subcategories is not None else None)
    plan = plan_rerun(content.taxonomy, previous_categories=categories, previous_subcategories=subcategories, previous_extraction=extraction,
                      transcript_checksum=transcript.checksum if transcript is not None else None,
                      attribution_checksum=attribution.checksum if attribution is not None else None, rescore=request.rescore_signals,
                      engine_thresholds=defaults.stage1, default_threshold=defaults.stage1_default,
                      subcategory_threshold=stage2.subcategory, reject_threshold=stage2.reject)
    return SignalsInput(snapshot=artifact, content=content, pipeline="v2", previous={} if plan.full else previous, plan=plan)


__all__ = ["ingest_signals", "load_snapshot", "request_signals"]
