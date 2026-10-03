"""The ``embeddings`` job (contract 1.2.0, decision 18): one search vector per transcript turn.

A model stage since 1.2.0. Both handlers embed with ``call1.embedding``, the module Store embeds
search queries with, so the turn vectors and the queries share one vector space:

* ``NemotronEmbeddingsHandler`` (real mode): ``nvidia/Nemotron-3-Embed-1B-BF16`` at the pinned
  revision, scheme ``nemotron-3-embed-1b@c0c9fea``. Turns are embedded as ``passage: <text>``, in
  batches, with a cancellation check between batches; the model is dropped after each job (Store,
  not Process, keeps it resident). A claim is refused (``model_unavailable``) when the weights are
  not installed, and (``model_unqualified``) when the job's frozen selection names another revision
  than this build embeds with.
* ``FakeEmbeddingsHandler`` (fake mode, or ``CALL1_EMBEDDING_BACKEND=fake``): the deterministic
  test embedder, scheme ``fake-embedding-v1``. Store must be configured with the same backend.
"""

from __future__ import annotations

import time
from typing import Optional

from call1 import embedding
from call1.contracts.contents import EmbeddingsContent, TurnEmbedding
from call1.contracts.errors import JobErrorCode
from call1.contracts.jobs import JobType

from .base import Handler, HandlerJob, HandlerResult, Output, ReleaseJob, Usage

BATCH_TURNS = 16


class _EmbeddingsHandler(Handler):
    job_type = JobType.EMBEDDINGS
    adapter_version = "1"
    backend = "fake"

    def embedder(self) -> embedding.Embedder:
        # Process uses the GPU when it has one; Store stays on CPU. Vectors agree to cosine 0.9999+.
        return embedding.get_embedder(self.backend, device="auto")

    def run(self, job: HandlerJob) -> HandlerResult:
        transcript = job.transcript()
        embedder = self.embedder()
        started = time.monotonic()
        batch = int(job.parameters.extra.get("batch_turns") or BATCH_TURNS)
        try:
            vectors = embedder.embed_documents([t.text for t in transcript.turns], batch_size=max(1, batch), cancelled=job.check_cancelled)
        except embedding.EmbedderUnavailable as exc:
            raise ReleaseJob("reject", JobErrorCode.MODEL_UNAVAILABLE, str(exc)) from exc
        content = EmbeddingsContent(scheme=embedder.scheme, dimensions=embedder.dimensions,
                                    turn_vectors=[TurnEmbedding(turn_id=t.turn_id, vector=v) for t, v in zip(transcript.turns, vectors)])
        return HandlerResult(outputs={"embeddings": Output(content)}, usage=Usage(inference_seconds=round(time.monotonic() - started, 6)))


class NemotronEmbeddingsHandler(_EmbeddingsHandler):
    adapter_id = "call1.torch.nemotron_embed"
    backend = embedding.DEFAULT_BACKEND

    def run(self, job: HandlerJob) -> HandlerResult:
        # The embedder runs on the Apple GPU (torch MPS). MLX stages share that GPU, and concurrent
        # Metal work from two frameworks aborts the process ("A command encoder is already encoding
        # to this command buffer"), so embedding holds the same lock every local model stage holds.
        from call1.pipeline.inference import inference_lock
        try:
            with inference_lock:
                return super().run(job)
        finally:
            unload = getattr(self.embedder(), "unload", None)
            if unload is not None:
                unload()

    def ready(self, job: HandlerJob) -> None:
        selection = getattr(job, "selection", None)
        revision: Optional[str] = getattr(selection, "model_revision", None)
        if revision is not None and revision != embedding.MODEL_REVISION:
            raise ReleaseJob("reject", JobErrorCode.MODEL_UNQUALIFIED,
                             f"the job froze embedder revision {revision[:12]}, this Process embeds with {embedding.MODEL_REVISION[:12]}")
        if not embedding.weights_installed():
            raise ReleaseJob("reject", JobErrorCode.MODEL_UNAVAILABLE, "the search embedding weights are not installed on this host")


class FakeEmbeddingsHandler(_EmbeddingsHandler):
    adapter_id = "call1.fake.embeddings"
    backend = "fake"


def embeddings_handler(handlers_mode: str) -> _EmbeddingsHandler:
    """The handler for this Process: fake in fake mode unless ``CALL1_EMBEDDING_BACKEND`` says
    otherwise, Nemotron in real mode unless it says ``fake`` (``call1.embedding.configured_backend``)."""
    backend = embedding.configured_backend(handlers_mode)
    return FakeEmbeddingsHandler() if backend == "fake" else NemotronEmbeddingsHandler()


__all__ = ["BATCH_TURNS", "FakeEmbeddingsHandler", "NemotronEmbeddingsHandler", "embeddings_handler"]
