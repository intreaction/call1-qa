"""The real search embedder (opt-in: ``CALL1_REAL_MODELS=1`` and the Nemotron-3-Embed-1B weights
under ``data/models/nemotron-3-embed-1b``, fetched with ``python -m call1.embedding download``).
Runs on CPU, so it needs no Apple Silicon, only the weights.

    CALL1_REAL_MODELS=1 .venv-local/bin/python -m pytest tests/process/real/test_real_embedding.py -m real_models -s -p no:warnings
"""

from __future__ import annotations

import os
import time

import pytest

from call1 import embedding
from call1.contracts.common import ReviewerRole
from call1.contracts.jobs import JobStatus, JobType
from call1.process.handlers.fake import SCRIPT

from ..conftest import REPO, SAMPLE, write_headers

WEIGHTS = REPO / "data" / "models" / embedding.MODEL_DIRECTORY

pytestmark = [
    pytest.mark.real_models,
    pytest.mark.skipif(os.getenv("CALL1_REAL_MODELS") != "1", reason="set CALL1_REAL_MODELS=1 to run on the real models"),
    pytest.mark.skipif(not embedding.weights_installed(WEIGHTS), reason=f"the embedding weights are not installed at {WEIGHTS}"),
]

V = "/store/v1"
TURNS = [text for _, text in SCRIPT]


@pytest.fixture
def nemotron_env(monkeypatch):
    monkeypatch.setenv("CALL1_EMBEDDING_BACKEND", "nemotron")
    monkeypatch.setenv("CALL1_EMBEDDING_PATH", str(WEIGHTS))
    for name in ("CALL1_EMBEDDING_DEVICE", "CALL1_EMBEDDING_DTYPE"):
        monkeypatch.delenv(name, raising=False)
    return embedding.get_embedder()


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def test_the_encoder_is_bidirectional_and_matches_the_model_card(nemotron_env):
    embedder = nemotron_env
    embedder.load()
    model = embedder._model  # the loaded transformers model
    assert model.config.is_causal is False
    import torch

    # Bidirectional attention: the first token's state depends on a later token.
    tokenizer = embedder._tokenizer
    a = tokenizer(["passage: the cat sat on the mat"], return_tensors="pt")
    b = tokenizer(["passage: the cat sat on the rug"], return_tensors="pt")
    with torch.inference_mode():
        first_a = model(**a).last_hidden_state[0, 0].float()
        first_b = model(**b).last_hidden_state[0, 0].float()
    assert float((first_a - first_b).abs().max()) > 1e-3, "is_causal: false was not honored (the encoder ran causally)"

    # The model card's example: each query scores its own passage far above the others.
    query = embedder.embed_query("How can someone reduce exposure to pollen during allergy season?")
    docs = embedder.embed_documents([
        "Eczema commonly causes itchy, dry, inflamed patches of skin.",
        "People with pollen allergy can reduce exposure by staying indoors on dry, windy days and checking pollen forecasts.",
    ])
    assert len(query) == embedding.MODEL_DIMENSIONS and abs(_dot(query, query) - 1.0) < 1e-3
    assert _dot(query, docs[1]) > _dot(query, docs[0]) + 0.3


@pytest.mark.parametrize("query,turn", [
    ("call recording disclosure", 0),
    ("identity verification", 2),
    ("how long does the customer have to dispute the fee", 4),
    ("the caller gives the last four digits", 3),
])
def test_a_relevant_query_ranks_the_right_turn_first(nemotron_env, query, turn):
    embedder = nemotron_env
    docs = embedder.embed_documents(TURNS)
    scores = [_dot(embedder.embed_query(query), d) for d in docs]
    assert scores.index(max(scores)) == turn, scores


def test_process_and_store_share_the_model_end_to_end(nemotron_env, make_runtime, store_http, session):
    """Fake stages elsewhere, but the embeddings job and Store's query both run the real model."""
    runtime = make_runtime("nemotron-embed")
    assert runtime.registry.get(JobType.EMBEDDINGS).adapter_id == "call1.torch.nemotron_embed"
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE)
    started = time.monotonic()
    worker.drain()
    print(f"\ningest + drain with the real embedder: {time.monotonic() - started:.1f} s")
    jobs = runtime.client.list_jobs(conversation_id=result.conversation_id)
    embed_job = next(j for j in jobs if j.job_type is JobType.EMBEDDINGS)
    assert embed_job.status is JobStatus.SUCCEEDED, embed_job.error_code
    reviewer = session(ReviewerRole.REVIEWER)
    started = time.monotonic()
    body = store_http.post(f"{V}/search/semantic", json={"query": "how long does the customer have to dispute the fee", "top_k": 3},
                           headers=write_headers(reviewer)).json()
    print(f"Store query (model may already be resident in this interpreter): {time.monotonic() - started:.2f} s")
    assert body["embedding_scheme"] == embedding.MODEL_SCHEME and body["calls_needing_reembedding"] == 0
    assert body["results"] and body["results"][0]["call_id"] == result.call_id and "dispute" in body["results"][0]["text"]
