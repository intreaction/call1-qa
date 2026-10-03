"""The real handler registry end to end against an in-process Store, with only the model runtimes
faked (MLX ASR, Nemotron diarization, the torch tone/sentiment models, and the LLM behind
``call1.question_models.generate_text``). Everything else is the production path: the pre-split
pipeline code, the worker, contract validation in Store, result projections. No ports, no weights.
"""

from __future__ import annotations

import json
import math
import re
import struct
import wave
from collections import Counter

import pytest

from call1.contracts.jobs import JobStatus, JobType
from call1.process.handlers.real import paths

from .conftest import SAMPLE
from .test_real_handlers import FFMPEG, FakeModels

V = "/store/v1"

SCRIPT = [
    ("AGENT", "Thank you for calling. This call may be recorded for quality assurance. How can I help you today?"),
    ("CALLER", "Hi, I have a question about a fee on my account."),
    ("AGENT", "I can help with that. For security, please verify your date of birth."),
    ("CALLER", "Sure, it is May twelfth."),
    ("AGENT", "Thank you, you are verified. I have removed the monthly service fee today."),
    ("CALLER", "Great, that is resolved then. Thanks for explaining."),
    ("AGENT", "Is there anything else I can help you with today? Have a great day."),
]


PII_SSN = "442-89-1099"
PII_SCRIPT = [*SCRIPT[:3], ("CALLER", f"Sure, my social security number is {PII_SSN}."), *SCRIPT[4:]]


def _legacy_script(channels: int, script=SCRIPT):
    from call1.models.schemas import SpeakerRole, TranscriptTurn, WordTimestamp

    turns = []
    for i, (speaker, text) in enumerate(script):
        start, end = i * 6.0, i * 6.0 + 5.0
        tokens = text.split()
        step = (end - start) / len(tokens)
        words = [WordTimestamp(word=(" " if k else "") + w, start_time=start + k * step, end_time=start + (k + 1) * step, probability=0.93)
                 for k, w in enumerate(tokens)]
        stereo = channels == 2
        turns.append(TranscriptTurn(turn_id=i, speaker=SpeakerRole(speaker) if stereo else SpeakerRole.UNKNOWN, start_time=start, end_time=end,
                                    text=text, raw_text=text, channel=(0 if speaker == "AGENT" else 1) if stereo else 0, word_timestamps=words,
                                    confidence=0.93))
    return turns


class ScriptedLlm:
    """Answers each prompt the way a well-behaved model would, from the prompt itself."""

    def __init__(self):
        self.calls = Counter()
        self.prompts = []

    def __call__(self, model, system, prompt, allow_external=False, response_schema=None, schema_name="answer", max_tokens=None,
                 synthetic=False, text_model_path=None):
        self.calls[schema_name] += 1
        self.prompts.append((schema_name, prompt))
        if schema_name == "qa_answer":
            data = json.loads(prompt.split("Evaluate this input data:\n", 1)[1])
            wanted = data["criterion"]["quote_speaker"]
            turn = next(t for t in data["transcript"] if wanted in ("any speaker", t["speaker"]))
            return json.dumps({"assessment": "The quoted turn shows the behavior.", "verdict": "pass", "quote": turn["text"]}), {}
        if schema_name == "call_summary":
            if prompt.startswith("SEGMENT") or prompt.startswith("["):
                parts = [json.loads(block.split("\n", 1)[1]) for block in prompt.split("\n\n")] if prompt.startswith("SEGMENT") else json.loads(prompt)
                points = [p for part in parts for p in part["key_points"]][:3]
                return json.dumps({"narrative": " ".join(part["narrative"] for part in parts), "key_points": points}), {}
            lines = re.findall(r"^\[(\d+)\] \w+: (.*)$", prompt, flags=re.MULTILINE)
            points = [f"{' '.join(text.split()[:5])} (turn {tid})" for tid, text in lines[:3]]
            return json.dumps({"narrative": f"The call covers turns {lines[0][0]} to {lines[-1][0]}.", "key_points": points}), {}
        if schema_name.startswith("contact_signals_"):
            lines = re.findall(r"^\[Turn (\d+)\] (\w+): (.*)$", prompt, flags=re.MULTILINE)
            callers = [line for line in lines if line[1] == "Caller"]
            agents = [line for line in lines if line[1] == "Agent"]
            if not callers or len(agents) < 2:  # a mono call: nobody is identified yet
                return json.dumps({"observations": []}), {}
            if schema_name.endswith("lifecycle"):
                tid, _, text = callers[0]
                kind = "intent"
            else:
                tid, _, text = agents[-2]
                kind = "agent_reports_completed"
            quote = " ".join(text.split()[:6])
            return json.dumps({"observations": [{"kind": kind, "confidence": 0.8, "evidence_turn": int(tid), "evidence_quote": quote}]}), {}
        raise AssertionError(f"unexpected prompt kind {schema_name}")


@pytest.fixture
def real_runtime(make_runtime, monkeypatch, tmp_path):
    return _real_runtime(make_runtime, monkeypatch, tmp_path)


def _real_runtime(make_runtime, monkeypatch, tmp_path, script=SCRIPT, name="real", **overrides):
    monkeypatch.setenv("CALL1_BACKEND", "mlx")
    monkeypatch.setenv("CALL1_SENTIMENT_MODELS", "1")
    root = tmp_path / f"models-{name}"
    for directory, weights in [("parakeet-tdt-0.6b-v3", "weights.npz"), ("nemotron-3-diarization", "model.safetensors"), ("meralion-ser-v1", "model.safetensors"),
                               ("roberta-sentiment", "pytorch_model.bin"), ("gemma-4-e2b-it", "model.safetensors"),
                               ("nemotron-3-embed-1b", "model.safetensors")]:
        (root / directory).mkdir(parents=True)
        (root / directory / "config.json").write_text("{}")
        (root / directory / weights).write_bytes(b"x")
    monkeypatch.setenv("CALL1_MODELS_DIR", str(root))
    for env in paths.LEGACY_PATH_ENV.values():
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(paths, "_importable", lambda module: True)  # MLX need not exist where this test runs
    monkeypatch.setattr("call1.pipeline.sentiment.LocalSentimentModels.text", lambda self, text: FakeModels().text(text))
    monkeypatch.setattr("call1.pipeline.sentiment.LocalSentimentModels.tone", lambda self, samples: FakeModels().tone(samples))
    monkeypatch.setattr("call1.adapters.mlx.MLXAdapter.transcribe",
                        lambda self, path, channels, agent_channel=0: _legacy_script(channels, script))
    monkeypatch.setattr("call1.adapters.mlx_diarization.diarize",
                        lambda wav_path, model_path: [{"start": i * 6.0, "end": i * 6.0 + 5.5, "speaker": i % 2} for i in range(len(script))])
    llm = ScriptedLlm()
    monkeypatch.setattr("call1.question_models.generate_text", llm)
    runtime = make_runtime(name, handlers="real", **overrides)
    runtime.test_llm = llm  # type: ignore[attr-defined]
    return runtime


@FFMPEG
def test_the_real_registry_runs_the_sample_call_through_store(real_runtime, store_http, session):
    runtime = real_runtime
    assert runtime.registry.mode == "real" and runtime.registry.missing() == []
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE, agent_id="agent-9")
    worker.drain()
    jobs = runtime.client.list_jobs(conversation_id=result.conversation_id)
    assert {(j.job_type.value, j.status.value, j.error_code) for j in jobs if j.status is not JobStatus.SUCCEEDED} == set()
    attempts = {j.job_type: runtime.client.list_attempts(j.id)[0].provenance.adapter_id for j in jobs}
    assert attempts[JobType.ASR] == "call1.mlx.parakeet" and attempts[JobType.QA_CRITERION] == "call1.qa.semantic_judgement"
    assert attempts[JobType.SUMMARY_ASSEMBLY] == "call1.summarizer.assembly" and attempts[JobType.EMBEDDINGS] == "call1.fake.embeddings"  # CALL1_EMBEDDING_BACKEND=fake (tests/conftest.py)

    reviewer = session()
    detail = store_http.get(f"{V}/calls/{result.call_id}", headers=reviewer.read_headers).json()
    assert {g["kind"]: g["state"] for g in detail["results"]} == {k: "available" for k in
                                                                   ("transcript", "tone", "text_sentiment", "qa", "summary", "contact_signals")}
    assert detail["call"]["channels"] == 2 and detail["call"]["codec"] == "pcm_s16le"
    transcript = store_http.get(f"{V}/calls/{result.call_id}/transcript", headers=reviewer.read_headers).json()
    assert [t["speaker"] for t in transcript["turns"]][:2] == ["AGENT", "CALLER"] and transcript["tone_blocks"]
    evaluation = store_http.get(f"{V}/calls/{result.call_id}/evaluation", headers=reviewer.read_headers).json()
    verdicts = {v["criterion_id"]: v for v in evaluation["verdicts"]}
    assert verdicts["REG-01"]["status"] == "PASS" and verdicts["REG-01"]["quote_turn_id"] == 0
    # call1_standard_v2 needs a configured policy for SEC-01 and COMP-01: flagged without a model call, as before the split
    assert verdicts["SEC-01"]["status"] == "FLAGGED" and "business policy" in verdicts["SEC-01"]["reasoning"]
    assert evaluation["requires_human_review"] is True and evaluation["passed"] is False
    assert runtime.test_llm.calls["qa_answer"] == 2
    summary = store_http.get(f"{V}/calls/{result.call_id}/summary", headers=reviewer.read_headers).json()
    assert summary["key_points"] and summary["citations"] and summary["grounding"]["key_points_citations_checked"] is True
    signals = store_http.get(f"{V}/calls/{result.call_id}/contact-signals", headers=reviewer.read_headers).json()
    assert signals["completeness"] == "complete" and {s["kind"] for s in signals["signals"]} == {"intent", "agent_reports_completed"}


def _mono(path, seconds=44.0):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"".join(struct.pack("<h", int(6000 * math.sin(2 * math.pi * 200 * i / 16000)) if (i // 16000) % 6 < 5 else 0)
                               for i in range(int(16000 * seconds))))


@FFMPEG
def test_a_mono_call_gets_diarization_clusters_and_every_turn_a_role(real_runtime, store_http, session, tmp_path):
    runtime = real_runtime
    worker = runtime.connect()
    path = tmp_path / "mono.wav"
    _mono(path)
    result = runtime.ingestor.ingest_file(path)
    worker.drain()
    jobs = runtime.client.list_jobs(conversation_id=result.conversation_id)
    assert {(j.job_type.value, j.status.value, j.error_code, j.error_detail) for j in jobs if j.status is not JobStatus.SUCCEEDED} == set()
    assert Counter(j.job_type for j in jobs)[JobType.SPEAKER_ATTRIBUTION] == 1
    transcript = store_http.get(f"{V}/calls/{result.call_id}/transcript", headers=session().read_headers).json()
    # Decision 26: every turn gets agent or caller, so nothing stays UNKNOWN on a mono call.
    assert {t["speaker"] for t in transcript["turns"]} <= {"AGENT", "CALLER"}
    assert [t["speaker_cluster"] for t in transcript["turns"]][:3] == ["speaker_1", "speaker_2", "speaker_1"]
    evaluation = store_http.get(f"{V}/calls/{result.call_id}/evaluation", headers=session().read_headers).json()
    assert not any("Speaker identity is unknown" in (v.get("reasoning") or "") for v in evaluation["verdicts"])
    signals = store_http.get(f"{V}/calls/{result.call_id}/contact-signals", headers=session().read_headers).json()
    assert signals["completeness"] == "complete"


TEXT_MODEL_TYPES = {JobType.QA_CRITERION, JobType.SUMMARY_SEGMENT, JobType.CONTACT_SIGNALS_LIFECYCLE, JobType.CONTACT_SIGNALS_RESOLUTION}


@FFMPEG
@pytest.mark.parametrize("mode", ["store", "off"])
def test_appliance_prompts_are_masked_while_store_masks_text_as_before_the_split(make_runtime, monkeypatch, tmp_path, store_http, session, mode):
    """Judge condition 1 / team decision 3: the legacy app masked QA and summary prompts whenever
    ``redaction.text`` was on, even for the included local model. Store's text masking
    (``mask_reviewer_reads``, on by default) is that toggle now, so the appliance route's text-model
    prompts are masked by default; ``mask_model_text: off`` turns it off for this Process."""
    from call1.redaction import REDACTED

    runtime = _real_runtime(make_runtime, monkeypatch, tmp_path, script=PII_SCRIPT, name=f"pii-{mode}", mask_model_text=mode)
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE, agent_id="agent-9")
    worker.drain()
    jobs = runtime.client.list_jobs(conversation_id=result.conversation_id)
    assert {(j.job_type.value, j.status.value, j.error_code) for j in jobs if j.status is not JobStatus.SUCCEEDED} == set()
    masked = mode == "store"
    assert {j.job_type: j.selection.route.masked for j in jobs if j.job_type in TEXT_MODEL_TYPES} == {t: masked for t in TEXT_MODEL_TYPES}
    assert not any(j.selection.route.masked for j in jobs if j.job_type in (JobType.ASR, JobType.ACOUSTIC_TONE, JobType.TEXT_SENTIMENT))
    kinds = {kind for kind, _ in runtime.test_llm.prompts}
    assert {"qa_answer", "call_summary"} <= kinds and any(k.startswith("contact_signals_") for k in kinds)
    leaked = [kind for kind, prompt in runtime.test_llm.prompts if PII_SSN in prompt]
    redacted = [kind for kind, prompt in runtime.test_llm.prompts if REDACTED in prompt]
    if masked:
        assert leaked == [] and {"qa_answer", "call_summary"} <= set(redacted)
    else:
        assert "qa_answer" in leaked and "call_summary" in leaked and redacted == []

    reviewer = session()
    evaluation = store_http.get(f"{V}/calls/{result.call_id}/evaluation", headers=reviewer.read_headers).json()
    assert {v["criterion_id"]: v["status"] for v in evaluation["verdicts"]}["REG-01"] == "PASS"  # grounded in the text the model saw
    summary = store_http.get(f"{V}/calls/{result.call_id}/summary", headers=reviewer.read_headers).json()
    assert summary["key_points"] and summary["grounding"]["redacted_input"] is masked and summary["grounding"]["key_points_citations_checked"]
    assert PII_SSN not in json.dumps(summary)  # reviewer reads are masked either way


@FFMPEG
def test_a_qa_prompt_over_the_context_budget_is_flagged_and_the_scorecard_still_publishes(real_runtime, store_http, session, monkeypatch):
    """Judge condition 7: as the pre-split router did, a context overflow FLAGs the criterion
    (trigger provider_error) instead of failing the job and dead-blocking the scorecard."""
    runtime = real_runtime
    scripted = runtime.test_llm

    def overflowing(model, system, prompt, allow_external=False, response_schema=None, schema_name="answer", *args, **kwargs):
        if schema_name == "qa_answer":
            raise RuntimeError("Input exceeds the appliance context budget; split into smaller chunks.")
        return scripted(model, system, prompt, allow_external, response_schema, schema_name, *args, **kwargs)

    monkeypatch.setattr("call1.question_models.generate_text", overflowing)
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE, agent_id="agent-9")
    worker.drain()
    jobs = runtime.client.list_jobs(conversation_id=result.conversation_id)
    assert {(j.job_type.value, j.status.value, j.error_code) for j in jobs if j.status is not JobStatus.SUCCEEDED} == set()
    reviewer = session()
    detail = store_http.get(f"{V}/calls/{result.call_id}", headers=reviewer.read_headers).json()
    assert {g["kind"]: g["state"] for g in detail["results"]}["qa"] == "available"
    evaluation = store_http.get(f"{V}/calls/{result.call_id}/evaluation", headers=reviewer.read_headers).json()
    reg = {v["criterion_id"]: v for v in evaluation["verdicts"]}["REG-01"]
    assert reg["status"] == "FLAGGED" and "context budget" in reg["reasoning"] and evaluation["requires_human_review"] is True
    attempts = reg["model_attempts"]
    assert attempts and attempts[0]["error_code"] == "context_limit_exceeded" and attempts[0]["trigger"] == "provider_error"
