"""The model PII layer's findings as a Process output (contract 1.2.0, team decision 19).

The ``enrichment`` job writes ``pii_findings`` once per transcript revision; masked text-model jobs
pin them and use them instead of loading the model again.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from call1 import pii_model
from call1.contracts.catalog import ModelPurpose
from call1.contracts.common import ArtifactRef
from call1.contracts.contents import PiiFindingsContent, PiiSpanContent, SpeakerRole, TranscriptContent, TranscriptTurnContent, TurnPiiFindings
from call1.contracts.errors import JobErrorCode
from call1.process.audio import AudioInfo
from call1.process.catalog import seeded_catalog
from call1.process.config import ProcessConfig
from call1.process.graph import GraphPlanner, Src, new_jobs
from call1.process.handlers.base import HandlerError
from call1.process.handlers.real import masking
from call1.process.transcripts import plan_segments

from .test_process_units import _artifact_model, _rubric

CHECKSUM = "sha256:" + "a" * 64
OTHER = "sha256:" + "b" * 64
TURNS = [
    TranscriptTurnContent(turn_id=0, speaker=SpeakerRole.AGENT, start_time=0, end_time=2, text="Thank you for calling, this is Sam."),
    TranscriptTurnContent(turn_id=1, speaker=SpeakerRole.CALLER, start_time=2, end_time=4,
                          text="Hi, my name is Maria Lopez and my email is maria@example.com."),
]


@pytest.fixture(autouse=True)
def fresh_cache():
    masking.clear_cache()
    yield
    masking.clear_cache()


def _graph(channels: int = 2, config: ProcessConfig = None):
    planner = GraphPlanner(seeded_catalog(mode="fake"), config or ProcessConfig())
    definition, ref = _rubric()
    audio = AudioInfo(container="wav", content_type="audio/wav", size_bytes=10, channels=channels, duration_seconds=90)
    graph = planner.ingest(conversation_id="conv_1", source=_artifact_model(), audio=audio, rubric_ref=ref, rubric=definition,
                           snapshot=_artifact_model("rubric_snapshot", "art_snap"))
    return planner, {j.ref: j for j in graph.jobs}


def _pii_input(job):
    return next((i for i in job.inputs if i.role == "pii_findings"), None)


def test_masked_text_jobs_pin_the_enrichment_jobs_findings():
    planner, by_ref = _graph()
    for ref in ("qa-0-reg-01", "cs-lifecycle", "cs-resolution", "summary"):
        found = _pii_input(by_ref[ref])
        assert found is not None and found.upstream.ref == "enrichment" and found.upstream.output_role == "pii_findings", ref
        assert "enrichment" in by_ref[ref].requires_refs, ref
    for ref in ("asr", "tone", "sentiment", "embeddings", "scorecard", "cs-merge"):
        assert _pii_input(by_ref[ref]) is None, ref
    # mono: the findings see the diarized speaker labels (agent self-introductions stay visible)
    _, mono = _graph(channels=1)
    assert {i.role for i in mono["enrichment"].inputs} == {"transcript", "speaker_attribution"} and "speaker" in mono["enrichment"].requires_refs
    # summary segments (planned at ASR completion) pin them when masked
    jobs = new_jobs("k.sum")
    planner.add_summary_jobs(jobs, plan_segments(TranscriptContent(duration_seconds=4, is_redacted=False, turns=TURNS), 60),
                             Src.job("job_asr", "transcript"), None, planner.pick(ModelPurpose.SUMMARY), pii=Src.job("job_enr", "pii_findings"))
    assert _pii_input(jobs.defs[0]).upstream.job_id == "job_enr" and jobs.defs[0].requires_job_ids == ["job_asr", "job_enr"]
    # unmasked text jobs need no findings
    _, plain = _graph(config=ProcessConfig(mask_model_text="off"))
    assert not any(_pii_input(plain[r]) for r in ("qa-0-reg-01", "cs-lifecycle", "summary"))


class _Input:
    def __init__(self, content, checksum):
        self._content = content
        self.artifact = SimpleNamespace(id="art_t", checksum=checksum)

    def content(self):
        return self._content

    @property
    def ref(self):
        return ArtifactRef(artifact_id="art_t", checksum=self.artifact.checksum)


def _job(inputs, agent_display_name=None):
    transcript = TranscriptContent(duration_seconds=4, is_redacted=False, turns=TURNS)
    items = {"transcript": _Input(transcript, CHECKSUM), **inputs}
    return SimpleNamespace(job=SimpleNamespace(conversation_id="conv_1"), input=items.get, require=items.__getitem__,
                           transcript=lambda: transcript, call_metadata=lambda: SimpleNamespace(agent_display_name=agent_display_name),
                           check_cancelled=lambda: None, log=logging.getLogger("test"))


def test_enrichment_findings_are_filtered_and_bound_to_the_transcript_revision():
    findings = masking.pii_findings(_job({}), backend="stub")
    assert findings.transcript.checksum == CHECKSUM and findings.detector == "stub" and findings.detector_revision == pii_model.STUB_REVISION
    by_turn = {t.turn_id: t.spans for t in findings.turns}
    assert by_turn[0] == []  # "this is Sam" is the agent's own introduction
    assert {(s.category, s.text) for s in by_turn[1]} == {("private_person", "Maria Lopez"), ("private_email", "maria@example.com")}
    for span in by_turn[1]:
        assert TURNS[1].text[span.start:span.end] == span.text
    PiiFindingsContent.model_validate(findings.model_dump(mode="json"))


def test_pinned_findings_are_used_instead_of_the_model(monkeypatch):
    def boom(backend=None):
        raise AssertionError("the model must not load when the findings are pinned")

    monkeypatch.setattr(pii_model, "detector", boom)
    pinned = PiiFindingsContent(transcript=ArtifactRef(artifact_id="art_t", checksum=CHECKSUM), detector="stub", detector_revision="stub-v1",
                                turns=[TurnPiiFindings(turn_id=1, spans=[PiiSpanContent(start=15, end=26, category="private_person", text="Maria Lopez")])])
    values = masking.sensitive_values([t.model_dump() for t in TURNS], job=_job({"pii_findings": _Input(pinned, OTHER)}))
    assert "Maria Lopez" in values


def test_findings_for_another_revision_fall_back_to_the_model(monkeypatch):
    stale = PiiFindingsContent(transcript=ArtifactRef(artifact_id="art_old", checksum=OTHER), detector="stub", detector_revision="stub-v1", turns=[])
    values = masking.model_values(_job({"pii_findings": _Input(stale, OTHER)}), [t.model_dump() for t in TURNS], backend="stub")
    assert {"Maria Lopez", "maria@example.com"} <= values


def test_missing_weights_fail_the_enrichment_job_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("CALL1_PII_MODEL_BACKEND", "privacy-filter")
    monkeypatch.setenv("CALL1_PII_MODEL_PATH", str(tmp_path))
    with pytest.raises(HandlerError) as caught:
        masking.pii_findings(_job({}))
    assert caught.value.code is JobErrorCode.MODEL_UNAVAILABLE
