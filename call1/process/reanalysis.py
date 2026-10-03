"""The reanalysis-request consumer: claims requests Store holds for this installation and turns
each into exactly one job graph (the graph carries the request's claim token, so Store creates it
and fulfils the request in one transaction), or rejects it with a safe reason.

Kinds and what their graph reruns (the publishers ``jobs.REANALYSIS_KIND_AFFECTS`` lists):

* ``qa``: QA against the requested rubric version (default: the call's rubric, current version).
* ``summary``: segments, synthesis and assembly from the linked transcript.
* ``contact_signals``: both passes and the merge (v1), or, when Store resolved ``signal_pipeline: v2``,
  only the Contact Signals v2 stages whose digests differ from the current taxonomy's, then the merge
  (``rescore_signals`` reruns every stage; docs/ContactSignalsV2.md section 7.5).
* ``contact_signals_preview`` (1.3.0): every v2 stage from the snapshot Store minted for the request,
  into its ``draft:<request_id>:`` slots (a taxonomy preview or a v1/v2 compare).
* ``speaker_correction``: a code-stage ``speaker_attribution`` job applying the correction, then
  tone, text sentiment, QA, summary and contact signals on the corrected attribution.
* ``full``: the ingest graph from the source audio, with the current ASR vocabulary (contract 1.3.0,
  decision 33): an active vocabulary runs dual transcription in the new ``asr`` job.
* ``qa_draft_test``: QA against the draft snapshot Store minted, in ``draft:<request_id>:`` slots.
* ``embeddings`` (contract 1.2.0): one ``embeddings`` job on the linked transcript and attribution,
  e.g. to re-embed a call indexed under an older search scheme (``hashing-projection-v1``).

Every graph's jobs take the request's ``priority`` on top of their own (+5 previews, -10 backfills and
compare requests). ``kinds`` narrows what this consumer claims (``ReanalysisClaimRequest.kinds``).
Without an explicit ``kinds``, a host that has no usable ``signal_category``/``signal_subcategory``
engine leaves the v2-only kinds (``V2_ONLY_KINDS``) out of its claims, so a taxonomy preview or a
compare request stays pending for a host that can run it instead of being rejected here
(docs/ContactSignalsV2.md sections 7.1 and 7.5). The check runs on every poll, so a classifier that
becomes available later widens the claim.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Optional, Sequence, Tuple

from call1.contracts.artifacts import Artifact, ArtifactKind
from call1.contracts.contents import AudioValidationContent, QaScorecardContent, TranscriptContent
from call1.contracts.jobs import GraphReason, JobGraphRequest, ReanalysisKind, ReanalysisRequest
from call1.contracts.rubrics import RubricSnapshotContent, RubricVersion

from .audio import CONTENT_TYPES, AudioInfo
from .catalog import CatalogError
from .config import ProcessConfig
from .graph import GraphPlanner, PlanError, QaSources, Src, bounded_key, new_jobs
from .ledger import Ledger
from .signals_input import request_signals
from .store_client import StoreClient, StoreError, StoreUnavailable
from .vocabulary import read_vocabulary

log = logging.getLogger("call1.process.reanalysis")

V2_ONLY_KINDS = frozenset({ReanalysisKind.CONTACT_SIGNALS_PREVIEW})
"""Kinds only a host with usable v2 classifiers can build. ``contact_signals`` is not here: without
v2 engines it plans v1 with ``pipeline_note`` (section 8.1)."""


class ReanalysisConsumer:
    def __init__(self, config: ProcessConfig, client: StoreClient, planner: GraphPlanner, ledger: Ledger, worker_id: str,
                 kinds: Optional[Sequence[ReanalysisKind]] = None) -> None:
        self.config = config
        self.kinds = list(kinds) if kinds else None
        self.client = client
        self.planner = planner
        self.ledger = ledger
        self.worker_id = worker_id
        self.handled = 0
        self.rejected = 0
        self.last_error: Optional[str] = None

    # --- polling -----------------------------------------------------------------------------

    def claim_kinds(self) -> Optional[list]:
        """The kinds to claim: the explicit ``kinds``, else every kind (None) when this host can run
        Contact Signals v2, else every kind except ``V2_ONLY_KINDS``."""
        if self.kinds is not None:
            return self.kinds
        if self.planner.signals_v2_available():
            return None
        return [kind for kind in ReanalysisKind if kind not in V2_ONLY_KINDS]

    def poll_once(self, max_requests: int = 2) -> int:
        response = self.client.claim_reanalysis(self.worker_id, max_requests, kinds=self.claim_kinds())
        for claimed in response.requests:
            self.handle(claimed.request, claimed.claim_token)
        return len(response.requests)

    def handle(self, request: ReanalysisRequest, claim_token: str) -> Optional[str]:
        try:
            body = self.build(request, claim_token)
        except (PlanError, CatalogError) as exc:
            return self._reject(request, claim_token, str(exc))
        except StoreUnavailable:
            raise
        except StoreError as exc:
            if exc.retryable or (exc.status or 0) >= 500:
                self.last_error = f"reanalysis {request.id}: {exc.code}"
                return None  # the claim expires and the request returns to pending
            return self._reject(request, claim_token, f"Store refused a read the graph needs ({exc.code})")
        try:
            graph = self.client.create_job_graph(request.conversation_id, body)
        except StoreUnavailable:
            raise
        except StoreError as exc:
            if exc.code == "claim_token_stale" or exc.retryable or (exc.status or 0) >= 500:
                self.last_error = f"reanalysis {request.id}: {exc.code}"
                return None
            return self._reject(request, claim_token, f"Store refused the graph ({exc.code})")
        self.handled += 1
        self.ledger.record(conversation_id=request.conversation_id, call_id=request.call_id, graph_id=graph.graph_id,
                           reason=f"reanalysis:{request.kind.value}")
        return graph.graph_id

    def _reject(self, request: ReanalysisRequest, claim_token: str, reason: str) -> None:
        self.rejected += 1
        self.last_error = f"reanalysis {request.id} rejected: {reason}"
        try:
            self.client.reject_reanalysis(request.id, claim_token, reason[:500])
        except StoreError as exc:
            log.warning("could not reject reanalysis request %s: %s", request.id, exc.code)
        return None

    # --- building ----------------------------------------------------------------------------

    def _linked(self, conversation_id: str) -> Dict[Tuple[ArtifactKind, str], Artifact]:
        latest: Dict[Tuple[ArtifactKind, str], Artifact] = {}
        for artifact in self.client.list_artifacts(conversation_id):
            key = (artifact.kind, artifact.slot)
            if artifact.linked and artifact.superseded_by is None and (key not in latest or (artifact.version or 0) > (latest[key].version or 0)):
                latest[key] = artifact
        return latest

    def _json(self, artifact: Artifact):
        return json.loads(self.client.download(artifact).decode("utf-8"))  # type: ignore[union-attr]

    def _rubric(self, request: ReanalysisRequest, arts) -> RubricVersion:
        if request.rubric is not None:
            return self.client.get_rubric_version(request.rubric.rubric_id, request.rubric.version)
        rubric_id = self.config.rubric_id
        card = arts.get((ArtifactKind.QA_SCORECARD, ""))
        if card is not None:
            rubric_id = QaScorecardContent.model_validate(self._json(card)).rubric.rubric_id
        return self.client.get_rubric(rubric_id)

    def _audio(self, arts) -> Tuple[Optional[Artifact], AudioInfo, Optional[int]]:
        source = arts.get((ArtifactKind.SOURCE_AUDIO, ""))
        report = arts.get((ArtifactKind.VALIDATION_REPORT, ""))
        channels = duration = rate = agent_channel = None
        if report is not None:
            content = AudioValidationContent.model_validate(self._json(report))
            channels, duration, rate, agent_channel = content.channels, content.duration_seconds, content.sample_rate, content.agent_channel
        content_type = source.content_type if source is not None else "audio/wav"
        container = next((k for k, v in CONTENT_TYPES.items() if v == content_type), "wav")
        return source, AudioInfo(container=container, content_type=content_type, size_bytes=source.size_bytes if source else 0,
                                 channels=channels, sample_rate=rate, duration_seconds=duration), agent_channel

    def build(self, request: ReanalysisRequest, claim_token: str) -> JobGraphRequest:
        arts = self._linked(request.conversation_id)
        key = f"reanalysis.{request.id}"
        source, audio, agent_channel = self._audio(arts)
        seconds = audio.duration_seconds

        def need(kind: ArtifactKind) -> Artifact:
            found = arts.get((kind, ""))
            if found is None:
                raise PlanError(f"the call has no {kind.value} yet")
            return found

        def pinned(kind: ArtifactKind) -> Optional[Src]:
            found = arts.get((kind, ""))
            return Src.pinned(found) if found is not None else None

        if request.kind is ReanalysisKind.FULL:
            if source is None:
                raise PlanError("the call has no source audio")
            rubric = self._rubric(request, arts)
            snapshot = self.client.mint_rubric_snapshot(request.conversation_id, rubric.ref.rubric_id, rubric.ref.version)
            signals = request_signals(self.client, request, arts, full=True) if self.config.stages.contact_signals else None
            vocabulary = self.planner.asr_vocabulary(read_vocabulary(self.client))
            return self.planner.ingest(conversation_id=request.conversation_id, source=source, audio=audio, rubric_ref=rubric.ref,
                                       rubric=rubric.definition, snapshot=snapshot, reason=GraphReason.REANALYSIS, key=key,
                                       reanalysis_request_id=request.id, reanalysis_claim_token=claim_token, agent_channel=agent_channel,
                                       signals=signals, priority_offset=request.priority, asr_vocabulary=vocabulary)

        jobs = new_jobs(key, request.priority)
        transcript = Src.pinned(need(ArtifactKind.TRANSCRIPT))
        speaker = pinned(ArtifactKind.SPEAKER_ATTRIBUTION)
        existing = QaSources(transcript=transcript, speaker=speaker, enrichment=pinned(ArtifactKind.ENRICHMENT),
                             sentiment=pinned(ArtifactKind.TEXT_SENTIMENT), tone=pinned(ArtifactKind.TONE_BLOCKS),
                             pii=pinned(ArtifactKind.PII_FINDINGS))

        if request.kind is ReanalysisKind.QA:
            rubric = self._rubric(request, arts)
            snapshot = self.client.mint_rubric_snapshot(request.conversation_id, rubric.ref.rubric_id, rubric.ref.version)
            self.planner.add_qa(jobs, existing, rubric=rubric.definition, rubric_ref=rubric.ref, snapshot=Src.pinned(snapshot), audio_seconds=seconds)
        elif request.kind is ReanalysisKind.SUMMARY:
            content = TranscriptContent.model_validate(self._json(need(ArtifactKind.TRANSCRIPT)))
            self.planner.summary_graph(jobs, content, transcript, speaker, seconds, pii=existing.pii)
        elif request.kind in (ReanalysisKind.CONTACT_SIGNALS, ReanalysisKind.CONTACT_SIGNALS_PREVIEW):
            signals = request_signals(self.client, request, arts, full=False)
            self.planner.add_contact_signals(jobs, existing, seconds, signals=signals)
        elif request.kind is ReanalysisKind.SPEAKER_CORRECTION:
            if request.speaker_correction is None:
                raise PlanError("a speaker correction request carries the correction")
            corrected = self.planner.speaker_correction_jobs(jobs, request.speaker_correction, transcript, speaker, seconds)
            sources = self.planner.add_analysis_stages(jobs, transcript, corrected, seconds, audio_src=Src.pinned(source) if source else None,
                                                       include_enrichment=False, include_embeddings=False)
            sources.enrichment = existing.enrichment
            sources.pii = existing.pii
            if sources.tone is None:
                sources.tone = existing.tone
            rubric = self._rubric(request, arts)
            snapshot = self.client.mint_rubric_snapshot(request.conversation_id, rubric.ref.rubric_id, rubric.ref.version)
            self.planner.add_qa(jobs, sources, rubric=rubric.definition, rubric_ref=rubric.ref, snapshot=Src.pinned(snapshot), audio_seconds=seconds)
            if self.config.stages.summary:
                content = TranscriptContent.model_validate(self._json(need(ArtifactKind.TRANSCRIPT)))
                self.planner.summary_graph(jobs, content, transcript, corrected, seconds, pii=existing.pii)
            if self.config.stages.contact_signals:
                self.planner.add_contact_signals(jobs, sources, seconds, signals=request_signals(self.client, request, arts, full=True))
        elif request.kind is ReanalysisKind.EMBEDDINGS:
            if not self.config.stages.embeddings:
                raise PlanError("the embeddings stage is turned off on this Process (stages.embeddings)")
            self.planner.add_embeddings(jobs, transcript, speaker, seconds)
        elif request.kind is ReanalysisKind.QA_DRAFT_TEST:
            draft = request.draft_rubric
            if draft is None:
                raise PlanError("a draft test carries its draft snapshot")
            snapshot = self.client.get_artifact(draft.snapshot_artifact_id)
            definition = RubricSnapshotContent.model_validate(self._json(snapshot)).definition
            self.planner.add_qa(jobs, existing, rubric=definition, draft=draft, snapshot=Src.pinned(snapshot), draft_request_id=request.id,
                                audio_seconds=seconds)
        else:  # pragma: no cover - the enum is closed
            raise PlanError(f"unknown reanalysis kind {request.kind.value}")
        return JobGraphRequest(idempotency_key=bounded_key(key), reason=GraphReason.REANALYSIS, reanalysis_request_id=request.id,
                               reanalysis_claim_token=claim_token, jobs=jobs.defs)


__all__ = ["ReanalysisConsumer"]
