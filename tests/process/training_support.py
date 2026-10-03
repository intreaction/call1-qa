"""Shared builders for the on-device training tests (``test_training_*.py``): a fake Store client that
serves the label log, source jobs and artifact content (and records any write), and a "world" of
calls whose QA, Contact Signals and speaker artifacts were made by the real prompt builders over the
fake-handler cascade. No MLX, torch or real model runs; no tests live here."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest.mock import patch

from call1.contracts.artifacts import Artifact, ArtifactKind
from call1.contracts.catalog import ModelPurpose
from call1.contracts.common import canonical_digest, canonical_json
from call1.contracts.contents import SpeakerAssignment, SpeakerAttributionContent, SpeakerRole, VerdictStatus
from call1.contracts.jobs import Job, JobParameters, JobType
from call1.contracts.rubrics import CheckType, RubricCheck, RubricCriterion, RubricDefinition, RubricSnapshotContent, RubricVersionRef
from call1.contracts.signals import builtin_signal_taxonomy
from call1.contracts.training import (
    QaVerdictLabel,
    SignalHitLabel,
    SignalSpanAt,
    SpeakerRoleLabel,
    TrainingLabel,
    TrainingLabelKind,
    TrainingLabelPage,
    TrainingSourceRef,
    speaker_subject_key,
    training_label_subject,
)
from call1.process.handlers.fake import CALLER_NAME_SCRIPT
from call1.process.training.examples import split_of
from call1.process.training.registry import AdapterRegistry
from call1.process.training.settings import TrainingSettings

from .test_signals_support import findings_for, make_job, run_v2, script_transcript

INSTALLATION = "inst_training_tests"
SENSITIVE = ("Maria Lopez", "1234")
"""Fixture values the masking must hide: the caller's name (stub PII detector) and the digits."""
COLORS = ["amber", "blue", "coral", "denim", "ember", "fern", "gold", "hazel", "indigo", "jade", "khaki", "lilac", "maroon", "navy",
          "olive", "pearl", "quartz", "rose", "sage", "teal", "umber", "violet", "wheat", "xenon", "yellow", "zinc"]


def script_for(i: int) -> List[Tuple[SpeakerRole, str]]:
    """The caller-name script with one call-specific word, so calls do not dedupe into one example."""
    script = list(CALLER_NAME_SCRIPT)
    role, text = script[6]
    script[6] = (role, text.replace("today?", f"today, {COLORS[i % len(COLORS)]} {i}?"))
    return script


def calls_in(split: str, n: int, start: int = 0) -> List[str]:
    """``n`` call IDs that land in ``split`` for ``INSTALLATION``."""
    out = []
    i = start
    while len(out) < n:
        call_id = f"call_{i:04d}"
        if split_of(INSTALLATION, call_id) == split:
            out.append(call_id)
        i += 1
    return out


def timestamp() -> datetime:
    return datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)


class FakeStore:
    """What the runner reads from Store, in memory. Any other method call is recorded in ``writes``
    (and answered with None): the runner must make none."""

    def __init__(self) -> None:
        self.artifacts: Dict[str, Tuple[ArtifactKind, bytes]] = {}
        self.jobs: Dict[str, Job] = {}
        self.labels: List[TrainingLabel] = []
        self.queued: List[Dict[str, Any]] = []
        self.reads: List[Tuple[str, Any]] = []
        self.writes: List[Tuple[str, Any]] = []
        self.taxonomy = builtin_signal_taxonomy()
        self.fail_labels: Optional[Exception] = None

    # --- world building
    def add(self, kind: ArtifactKind, content: Any, role: str) -> TrainingSourceRef:
        data = canonical_json(content) if not isinstance(content, (bytes, bytearray)) else bytes(content)
        digest = hashlib.sha256(data).hexdigest()
        artifact_id = f"art_{digest[:16]}"
        self.artifacts[artifact_id] = (kind, data)
        return TrainingSourceRef(role=role, artifact_id=artifact_id, checksum="sha256:" + digest, kind=kind)

    def add_label(self, **fields) -> TrainingLabel:
        seq = len(self.labels) + 1
        label = TrainingLabel(seq=seq, recorded_at=timestamp(), **fields)
        self.labels.append(label)
        return label

    # --- reads the runner makes
    def list_training_labels(self, *, after: int = 0, limit: int = 200, kinds=None, retry: bool = True, timeout=None) -> TrainingLabelPage:
        self.reads.append(("list_training_labels", (after, limit)))
        if self.fail_labels is not None:
            raise self.fail_labels
        rows = [label for label in self.labels if label.seq > after and (not kinds or label.kind.value in [getattr(k, "value", k) for k in kinds])]
        items = rows[:limit] if limit else []
        high = max((label.seq for label in self.labels), default=0)
        return TrainingLabelPage(items=items, next_after=items[-1].seq if items else after, count_after=len(rows), high_water=max(high, after))

    def get_job(self, job_id: str) -> Job:
        self.reads.append(("get_job", job_id))
        return self.jobs[job_id]

    def download(self, artifact: Artifact, dest: Optional[Path] = None):
        self.reads.append(("download", artifact.id))
        kind, data = self.artifacts[artifact.id]
        if "sha256:" + hashlib.sha256(data).hexdigest() != artifact.checksum:
            raise AssertionError("checksum mismatch")
        if dest is None:
            return data
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return dest

    def get_signal_taxonomy(self):
        self.reads.append(("get_signal_taxonomy", None))
        return SimpleNamespace(current=SimpleNamespace(taxonomy=self.taxonomy))

    def list_jobs(self, *, status=None, memory_slot=None, limit=None, retry=True, timeout=None, **kw) -> List[Any]:
        self.reads.append(("list_jobs", (status, memory_slot)))
        return [j for j in self.queued if (status is None or j["status"] == status) and (memory_slot is None or j.get("memory_slot") == memory_slot)][:limit or None]

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)

        def write(*args, **kwargs):
            self.writes.append((name, args))
            return None
        return write


# --- the three label kinds ---------------------------------------------------------------------------


def job_record(job_id: str, conversation_id: str, job_type: JobType, parameters: JobParameters, selection=None) -> Job:
    return Job.model_construct(id=job_id, conversation_id=conversation_id, graph_id=f"graph_{conversation_id}", job_type=job_type,
                               parameters=parameters, selection=selection)


def add_signal_call(store: FakeStore, tmp_path: Path, call_id: str, index: int, *, verdicts: Optional[Sequence[str]] = ("confirmed", "dismissed"),
                    taxonomy=None, script=None, stereo: bool = True):
    """One call through the fake v2 cascade and, with ``verdicts``, one category label per hit
    (cycling ``verdicts``). Returns ``SimpleNamespace(run, refs, job_id, call_id)``."""
    work = tmp_path / f"signals-{call_id}"
    work.mkdir(parents=True, exist_ok=True)
    run = run_v2(work, script or script_for(index), taxonomy, stereo=stereo)
    merge = run.jobs["merge"]
    refs = []
    for role, item in merge.inputs.items():
        if item is not None:
            refs.append(store.add(item.artifact.kind, item.read_bytes(), role))
    refs.append(store.add(ArtifactKind.CONTACT_SIGNALS, run.result, "contact_signals"))
    job_id = f"job_merge_{call_id}"
    store.jobs[job_id] = job_record(job_id, f"conv_{call_id}", JobType.CONTACT_SIGNALS_MERGE, merge.job.parameters)
    call = SimpleNamespace(run=run, refs=refs, job_id=job_id, call_id=call_id)
    for n, hit in enumerate(run.result.signals if verdicts else []):
        add_hit_label(store, call, hit, category_verdict=verdicts[n % len(verdicts)])
    return call


def add_hit_label(store: FakeStore, call, hit, **verdicts) -> TrainingLabel:
    spans = [SignalSpanAt(turn_id=hit.turn_id, block=hit.span.block)] + [SignalSpanAt(turn_id=p.turn_id, block=p.block) for p in hit.parts or []]
    withdrawn = all(v is None for k, v in verdicts.items() if k in ("category_verdict", "subcategory_verdict"))
    signal = SignalHitLabel(hit_id=hit.id, category_id=hit.category_id, signals_version=1, spans=spans, feedback_version=1, **verdicts)
    return store.add_label(kind=TrainingLabelKind.SIGNAL_HIT, subject=training_label_subject(TrainingLabelKind.SIGNAL_HIT, call.call_id, hit.id),
                           call_id=call.call_id, conversation_id=f"conv_{call.call_id}", source_job_id=None if withdrawn else call.job_id,
                           sources=[] if withdrawn else call.refs, signal=signal, withdrawn=withdrawn)


SEMANTIC = RubricCheck(check_type=CheckType.SEMANTIC_JUDGEMENT, pass_when="The agent discloses that the call is recorded.",
                       fail_when="The agent never mentions recording.", not_applicable_when="Never.", speaker=SpeakerRole.AGENT)
RUBRIC = RubricDefinition(rubric_id="training_rubric", name="Training", criteria=[
    RubricCriterion(criterion_id="REG-01", name="Recording disclosure", check=SEMANTIC),
    RubricCriterion(criterion_id="GREET-01", name="Greeting", check=RubricCheck(check_type=CheckType.PHRASE_ANY, phrases=["thank you for calling"])),
])
QUOTE = "This call may be recorded for quality assurance."


def qa_answer(verdict: str = "pass", quote: str = QUOTE, assessment: str = "The agent disclosed recording.") -> str:
    return json.dumps({"assessment": assessment, "verdict": verdict, "quote": quote})


def qa_primary(tmp_path: Path, transcript, findings, *, reply: Optional[str] = None, criterion: str = "REG-01"):
    """Run the real QA handler (masked route, pinned findings) with a scripted model answer.
    Returns (handler job, result, the prompts the model saw)."""
    from call1.process.catalog import seeded_catalog
    from call1.process.handlers.real.qa import RealQaCriterion

    snapshot = RubricSnapshotContent(source="published", rubric_id=RUBRIC.rubric_id, rubric_version=1, digest=canonical_digest(RUBRIC), definition=RUBRIC)
    ref = RubricVersionRef(rubric_id=RUBRIC.rubric_id, version=1, digest=canonical_digest(RUBRIC))
    inputs = {"transcript": (ArtifactKind.TRANSCRIPT, transcript), "rubric": (ArtifactKind.RUBRIC_SNAPSHOT, snapshot),
              "pii_findings": (ArtifactKind.PII_FINDINGS, findings)}
    job = make_job(tmp_path, JobType.QA_CRITERION, inputs, parameters=JobParameters(rubric=ref, criterion_id=criterion), entry_id="call1-bundled",
                   purpose=ModelPurpose.SEMANTIC_QA, masked=True, catalog=seeded_catalog(mode="fake"))
    prompts: List[Tuple[str, str]] = []

    def generate_text(model, system, prompt, allow_external=False, response_schema=None, schema_name="answer", max_tokens=None, synthetic=False,
                      text_model_path=None):
        prompts.append((system, prompt))
        return reply or qa_answer(), {}

    with patch("call1.question_models.generate_text", generate_text):
        result = RealQaCriterion().run(job)
    return job, result, prompts


def add_qa_call(store: FakeStore, tmp_path: Path, call_id: str, index: int, *, status: VerdictStatus = VerdictStatus.FAIL,
                original: VerdictStatus = VerdictStatus.PASS, reason_code=None, reply: Optional[str] = None, escalation=None):
    work = tmp_path / f"qa-{call_id}"
    work.mkdir(parents=True, exist_ok=True)
    transcript = script_transcript(script_for(index))
    findings = findings_for(work, transcript)
    job, result, prompts = qa_primary(work, transcript, findings, reply=reply)
    refs = [store.add(item.artifact.kind, item.read_bytes(), role) for role, item in job.inputs.items() if item is not None]
    refs.append(store.add(ArtifactKind.QA_ASSESSMENT, result.outputs["assessment"].content, "assessment"))
    refs.append(store.add(ArtifactKind.PROMPT_INPUT, result.outputs["prompt_input"].content, "prompt_input"))
    if escalation is not None:
        refs.append(store.add(ArtifactKind.QA_ASSESSMENT, escalation, "escalation_assessment"))
    job_id = f"job_qa_{call_id}"
    store.jobs[job_id] = job_record(job_id, f"conv_{call_id}", JobType.QA_CRITERION, job.job.parameters, job.job.selection)
    qa = QaVerdictLabel(override_id=f"ovr_{call_id}", criterion_id="REG-01", evaluation_version=1, original_status=original, status=status,
                        reason_code=reason_code)
    label = store.add_label(kind=TrainingLabelKind.QA_VERDICT, subject=training_label_subject(TrainingLabelKind.QA_VERDICT, call_id, "REG-01"),
                            call_id=call_id, conversation_id=f"conv_{call_id}", source_job_id=job_id, sources=refs, qa=qa)
    return label, result, prompts


def mono_attribution(script, *, wrong: bool = True, confidence: float = 0.8) -> SpeakerAttributionContent:
    """Clusters per role (spk_0 agent turns, spk_1 caller turns); ``wrong`` names both AGENT."""
    out = []
    for i, (role, _) in enumerate(script):
        cluster = "spk_0" if role is SpeakerRole.AGENT else "spk_1"
        speaker = SpeakerRole.AGENT if (wrong or role is SpeakerRole.AGENT) else SpeakerRole.CALLER
        out.append(SpeakerAssignment(turn_id=i, speaker=speaker, speaker_cluster=cluster, confidence=confidence))
    return SpeakerAttributionContent(method="diarization", assignments=out)


def add_speaker_call(store: FakeStore, tmp_path: Path, call_id: str, index: int, *, apply_to_cluster: bool = True, confidence: float = 0.8):
    work = tmp_path / f"speaker-{call_id}"
    work.mkdir(parents=True, exist_ok=True)
    script = script_for(index)
    transcript = script_transcript(script, stereo=False)
    findings = findings_for(work, transcript)
    attribution = mono_attribution(script, confidence=confidence)
    refs = [store.add(ArtifactKind.TRANSCRIPT, transcript, "transcript"), store.add(ArtifactKind.PII_FINDINGS, findings, "pii_findings"),
            store.add(ArtifactKind.SPEAKER_ATTRIBUTION, attribution, "speaker_attribution")]
    job_id = f"job_speakers_{call_id}"
    store.jobs[job_id] = job_record(job_id, f"conv_{call_id}", JobType.SPEAKER_ATTRIBUTION, JobParameters())
    speaker = SpeakerRoleLabel(turn_id=1, speaker="CALLER", apply_to_cluster=apply_to_cluster, speaker_cluster="spk_1",
                               reanalysis_request_id=f"rr_{call_id}")
    key = speaker_subject_key(1, apply_to_cluster, "spk_1")
    return store.add_label(kind=TrainingLabelKind.SPEAKER_ROLE, subject=training_label_subject(TrainingLabelKind.SPEAKER_ROLE, call_id, key),
                           call_id=call_id, conversation_id=f"conv_{call_id}", source_job_id=job_id, sources=refs, speaker=speaker)


def fake_base(tmp_path: Path) -> Path:
    """A stand-in for data/models/gemma-4-e2b-it: config.json and one weight file (fingerprint only)."""
    base = tmp_path / "models" / "gemma-4-e2b-it"
    base.mkdir(parents=True, exist_ok=True)
    (base / "config.json").write_text(json.dumps({"model_type": "gemma4"}))
    (base / "model.safetensors").write_bytes(b"weights")
    return base


def loose_settings(**changes) -> TrainingSettings:
    """Settings with the minimums lowered for small fixture worlds."""
    raw = {"trainer": "fake", "min_labeled_calls": 1, "min_train_examples": 1, "min_eval_items": 1}
    raw.update(changes)
    return TrainingSettings.from_mapping(raw)


def world(store: FakeStore, tmp_path: Path, *, train_calls: int = 2, eval_calls: int = 2, kinds=("signal", "qa", "speaker")) -> Dict[str, List[str]]:
    """Calls in the training and held-out buckets with labels of each kind."""
    calls = {"train": calls_in("train", train_calls), "eval": calls_in("eval", eval_calls)}
    index = 0
    for split in ("train", "eval"):
        for call_id in calls[split]:
            if "signal" in kinds:
                add_signal_call(store, tmp_path, call_id, index)
            if "qa" in kinds:
                add_qa_call(store, tmp_path, call_id, index)
            if "speaker" in kinds:
                add_speaker_call(store, tmp_path, call_id, index)
            index += 1
    return calls


def registry_for(tmp_path: Path, base: Path) -> AdapterRegistry:
    return AdapterRegistry(tmp_path / "process" / "adapters", base=base)


__all__ = ["COLORS", "FakeStore", "INSTALLATION", "QUOTE", "RUBRIC", "SENSITIVE", "add_hit_label", "add_qa_call", "add_signal_call", "add_speaker_call", "calls_in",
           "fake_base", "job_record", "loose_settings", "mono_attribution", "qa_answer", "qa_primary", "registry_for", "script_for", "world"]
