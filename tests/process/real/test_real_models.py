"""The real model stack end to end (opt-in: ``CALL1_REAL_MODELS=1``, Apple Silicon, weights under
``data/models``): ``sample_audio/call_01_compliant.wav`` ingested through an in-process Store with
``CALL1_BACKEND=mlx`` and the real handlers, then a Nemotron-3-Diarization check on a mono downmix.

Run with ``-s`` to see the per-stage timings table.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections import defaultdict
from typing import Dict, List

import pytest

from call1.contracts.catalog import CatalogEntryStatus
from call1.contracts.jobs import JobStatus, JobType

from ..conftest import REPO, SAMPLE

pytestmark = [
    pytest.mark.real_models,
    pytest.mark.skipif(os.getenv("CALL1_REAL_MODELS") != "1", reason="set CALL1_REAL_MODELS=1 to run on the real models (Apple Silicon, data/models)"),
]

V = "/store/v1"
MODELS = REPO / "data" / "models"


@pytest.fixture
def mlx_env(monkeypatch):
    monkeypatch.setenv("CALL1_BACKEND", "mlx")
    monkeypatch.setenv("CALL1_MODELS_DIR", str(MODELS))
    monkeypatch.setenv("CALL1_SENTIMENT_MODELS", "1")
    for env in ("CALL1_MLX_ASR_PATH", "CALL1_MLX_DIARIZATION_PATH", "CALL1_MLX_TEXT_PATH", "CALL1_TONE_PATH", "CALL1_SENTIMENT_PATH"):
        monkeypatch.delenv(env, raising=False)


def _time_handlers(registry) -> Dict[str, List[float]]:
    timings: Dict[str, List[float]] = defaultdict(list)
    for job_type in list(registry.job_types()):
        handler = registry.get(job_type)
        original = handler.run

        def timed(job, original=original, name=job_type.value):
            started = time.monotonic()
            try:
                return original(job)
            finally:
                timings[name].append(time.monotonic() - started)

        handler.run = timed  # type: ignore[method-assign]
    return timings


def _report(title: str, timings: Dict[str, List[float]], total: float, extra: str = "") -> str:
    lines = [f"\n{title}", f"{'stage':<28}{'runs':>5}{'seconds':>10}{'max':>9}"]
    for name, values in sorted(timings.items(), key=lambda kv: -sum(kv[1])):
        lines.append(f"{name:<28}{len(values):>5}{sum(values):>10.1f}{max(values):>9.1f}")
    lines.append(f"{'wall clock (ingest to settled)':<33}{total:>10.1f}")
    if extra:
        lines.append(extra)
    return "\n".join(lines)


def test_the_sample_call_on_the_real_models(mlx_env, make_runtime, store_http, session, clock):
    runtime = make_runtime("real", handlers="real")
    assert runtime.registry.mode == "real" and runtime.registry.missing() == []
    defaults = {p.value: runtime.catalog.get(e) for p, e in runtime.catalog.defaults.items()}
    assert {k: runtime.catalog.status(e)[0] for k, e in defaults.items()} == {k: CatalogEntryStatus.AVAILABLE for k in defaults}
    timings = _time_handlers(runtime.registry)
    worker = runtime.connect()
    started = time.monotonic()
    result = runtime.ingestor.ingest_file(SAMPLE, agent_id="agent-real")
    for _ in range(4):  # retries back off on Store's clock; advance it between drains
        worker.drain()
        if runtime.client.get_progress(result.conversation_id).settled:
            break
        clock.advance(runtime.client.parameters.retry_backoff_max_seconds)
    total = time.monotonic() - started
    jobs = runtime.client.list_jobs(conversation_id=result.conversation_id)
    failures = [(j.job_type.value, j.status.value, j.attempt_count, j.error_code and j.error_code.value, j.error_detail)
                for j in jobs if j.status is not JobStatus.SUCCEEDED]
    attempts = {j.id: runtime.client.list_attempts(j.id) for j in jobs}
    retried = [(j.job_type.value, [(a.status.value, a.error_code and a.error_code.value) for a in attempts[j.id]]) for j in jobs if len(attempts[j.id]) > 1]

    reviewer = session()
    detail = store_http.get(f"{V}/calls/{result.call_id}", headers=reviewer.read_headers).json()
    transcript = store_http.get(f"{V}/calls/{result.call_id}/transcript", headers=reviewer.read_headers).json()
    evaluation = store_http.get(f"{V}/calls/{result.call_id}/evaluation", headers=reviewer.read_headers)
    summary = store_http.get(f"{V}/calls/{result.call_id}/summary", headers=reviewer.read_headers)
    signals = store_http.get(f"{V}/calls/{result.call_id}/contact-signals", headers=reviewer.read_headers)
    outcome = {
        "results": {g["kind"]: g["state"] for g in detail["results"]},
        "turns": [(t["turn_id"], t["speaker"], t["text"]) for t in transcript["turns"]],
        "tone_blocks": [(b["speaker"], b["status"], b["valence"]) for b in transcript["tone_blocks"]],
        "sentiment": [(t["turn_id"], t.get("text_sentiment")) for t in transcript["turns"]],
        "verdicts": [(v["criterion_id"], v["status"], v["quoted_evidence"], v["reasoning"][:160]) for v in evaluation.json().get("verdicts", [])]
        if evaluation.status_code == 200 else evaluation.json(),
        "score": evaluation.json().get("overall_score") if evaluation.status_code == 200 else None,
        "summary": summary.json() if summary.status_code == 200 else summary.json(),
        "signals": signals.json() if signals.status_code == 200 else signals.json(),
        "failures": failures, "retried": retried,
    }
    print(_report("Real models: sample_audio/call_01_compliant.wav (44 s stereo), CALL1_BACKEND=mlx", timings, total))
    print(json.dumps(outcome, indent=1, default=str)[:12000])

    assert failures == [], failures
    assert outcome["results"] == {k: "available" for k in ("transcript", "tone", "text_sentiment", "qa", "summary", "contact_signals")}
    assert transcript["turns"] and {t["speaker"] for t in transcript["turns"]} == {"AGENT", "CALLER"}
    assert any(b["status"] == "SCORED" for b in transcript["tone_blocks"])
    assert [v[0] for v in outcome["verdicts"]] == ["REG-01", "SEC-01", "COMP-01", "ETIQ-01"]
    assert outcome["summary"]["key_points"] and outcome["summary"]["catalog_entry_id"] == "call1-bundled"


def test_diarization_on_a_mono_downmix(mlx_env, tmp_path):
    """Real Parakeet on mono audio (UNKNOWN speakers), then real Nemotron-3-Diarization clusters per turn, and every
    turn gets a role (decision 26)."""
    from call1.contracts.artifacts import ArtifactKind
    from call1.contracts.jobs import JobParameters

    from call1.process.handlers.real.media import RealAsr, RealSpeakerAttribution

    from ..test_real_handlers import _TMP, make_job

    _TMP[:] = [tmp_path]
    mono = tmp_path / "mono.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(SAMPLE), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(mono)],
                   check=True, capture_output=True, timeout=120)
    timings: Dict[str, List[float]] = defaultdict(list)
    started = time.monotonic()
    asr_job = make_job(tmp_path, JobType.ASR, {"audio": (ArtifactKind.SOURCE_AUDIO, mono)}, parameters=JobParameters(extra={"channels": 1}),
                       entry_id="parakeet-tdt-0.6b-v3")
    t = time.monotonic()
    transcript = RealAsr().run(asr_job).outputs["transcript"].content
    timings["asr (mono)"].append(time.monotonic() - t)
    job = make_job(tmp_path, JobType.SPEAKER_ATTRIBUTION, {"audio": (ArtifactKind.SOURCE_AUDIO, mono), "transcript": (ArtifactKind.TRANSCRIPT, transcript)},
                   entry_id="nemotron-3-diarization")
    handler = RealSpeakerAttribution()
    handler.ready(job)
    t = time.monotonic()
    content = handler.run(job).outputs["speaker_attribution"].content
    timings["speaker_attribution"].append(time.monotonic() - t)
    clusters = [(a.turn_id, a.speaker_cluster) for a in content.assignments]
    texts = {t.turn_id: (round(t.start_time, 1), round(t.end_time, 1), " ".join(t.text.split()[:8])) for t in transcript.turns}
    print(_report("Real models: mono downmix, Parakeet + Nemotron-3-Diarization", timings, time.monotonic() - started,
                  "\n".join(f"  turn {tid} {texts[tid][0]}-{texts[tid][1]}s {cluster}: {texts[tid][2]}" for tid, cluster in clusters)))
    assert {t.speaker.value for t in transcript.turns} == {"UNKNOWN"} and len(content.assignments) == len(transcript.turns)
    assert {a.speaker.value for a in content.assignments} <= {"AGENT", "CALLER"}  # decision 26: no UNKNOWN turns
    assert len({c for _, c in clusters if c}) >= 2, clusters  # two voices on the sample call
