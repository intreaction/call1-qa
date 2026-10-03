"""A label's source job, rebuilt as the ``HandlerJob`` its handler saw (docs/OnDeviceTraining.md 1.1).

``getJob`` supplies the job's parameters and frozen selection; ``getArtifactContent`` supplies the
artifacts the label names in ``sources`` (existing ``jobs:write`` and ``artifacts:read`` scopes).
Downloads go to ``work/<run_id>/inputs/`` (0700), are checksum-verified, and are shared between
labels of the same call; the runner deletes the directory as soon as the masked examples are
written. The job is ``ClaimedJob.model_construct(job=..., attempt_number=0, ...)``: it is never
claimed, and nothing here writes to Store.

Masking needs the pinned ``pii_findings`` of the transcript revision: ``has_findings`` says whether
the replayed job has them. Training never runs the PII model, so a label without them is skipped
(``no_pii_findings``) and never trained unmasked.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Optional

from call1.contracts.artifacts import Artifact, ArtifactKind
from call1.contracts.jobs import ClaimedJob, Job
from call1.contracts.training import TrainingSourceRef

from ..handlers.base import HandlerJob, InputArtifact


class SourceUnavailable(Exception):
    """A source job or artifact could not be read (``source_unavailable``)."""


class Sources:
    """Cached reads of source jobs and artifact content for one run."""

    def __init__(self, client, inputs_dir: Path, catalog=None) -> None:
        self.client = client
        self.inputs_dir = Path(inputs_dir)
        self.catalog = catalog
        self._jobs: Dict[str, Job] = {}
        self._files: Dict[str, Path] = {}

    def job(self, job_id: str) -> Job:
        if job_id not in self._jobs:
            self._jobs[job_id] = self.client.get_job(job_id)
        return self._jobs[job_id]

    def _record(self, ref: TrainingSourceRef) -> Artifact:
        return Artifact.model_construct(id=ref.artifact_id, kind=ref.kind, slot="", content_type="application/json", size_bytes=0,
                                        checksum=ref.checksum, content_contract="", sensitivity="derived", conversation_id="", linked=True,
                                        version=1, labels={})

    def fetch(self, artifact: Artifact, dest: Optional[Path]):
        """``InputArtifact``'s fetch: the verified download, from the run's inputs directory."""
        path = self._files.get(artifact.id)
        if path is None:
            if not self.inputs_dir.exists():
                self.inputs_dir.mkdir(parents=True, exist_ok=True)
                os.chmod(self.inputs_dir, 0o700)
            path = self.inputs_dir / f"{artifact.id}.json"
            self.client.download(artifact, path)
            os.chmod(path, 0o600)
            self._files[artifact.id] = path
        data = path.read_bytes()
        if dest is None:
            return data
        dest.write_bytes(data)
        return dest

    def content(self, ref: TrainingSourceRef):
        """A source artifact's parsed contract content."""
        item = InputArtifact(ref.role, self._record(ref), self.fetch, self.inputs_dir)
        return item.content()

    def handler_job(self, job: Job, sources: Iterable[TrainingSourceRef], roles: Iterable[str], *, catalog_entry=None) -> HandlerJob:
        """The replayed ``HandlerJob``: ``job`` with the named ``roles`` of ``sources`` as its inputs."""
        wanted = set(roles)
        inputs: Dict[str, Optional[InputArtifact]] = {}
        scratch = self.inputs_dir / "scratch"
        for ref in sources:
            if ref.role in wanted:
                inputs[ref.role] = InputArtifact(ref.role, self._record(ref), self.fetch, scratch)
        claimed = ClaimedJob.model_construct(job=job, attempt_number=0, final_attempt=False, upstream=[], inputs=[])
        if catalog_entry is None and self.catalog is not None and job.selection is not None:
            catalog_entry = self.catalog.by_ref(job.selection.catalog_entry)
        return HandlerJob(claimed, inputs, scratch, catalog_entry=catalog_entry)


def has_findings(job: HandlerJob) -> bool:
    """The job has a ``pii_findings`` input made from its own ``transcript`` input, so masking uses
    the findings and never runs the PII model."""
    item = job.input("pii_findings")
    transcript = job.input("transcript")
    if item is None or transcript is None:
        return False
    try:
        findings = item.content()
    except Exception:
        return False
    return getattr(getattr(findings, "transcript", None), "checksum", None) == transcript.artifact.checksum


def source_map(sources: Iterable[TrainingSourceRef]) -> Dict[str, TrainingSourceRef]:
    return {ref.role: ref for ref in sources}


def is_kind(ref: Optional[TrainingSourceRef], kind: ArtifactKind) -> bool:
    return ref is not None and ref.kind is kind


__all__ = ["SourceUnavailable", "Sources", "has_findings", "is_kind", "source_map"]
