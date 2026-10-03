"""Semantic search over published transcripts (contract 1.2.0, team decision 18).

Store embeds the query with its configured search embedder (``call1.embedding``:
Nemotron-3-Embed-1B at a pinned revision, or the deterministic fake on fake-handler stacks), the
same module Process embeds turns with, and ranks only vectors of that embedder's scheme
(``results_search_vectors``, filled by the completion hook). Vectors of any other scheme (e.g.
``hashing-projection-v1`` from before 1.2.0) are never compared; the response counts their calls in
``calls_needing_reembedding`` and reanalysis kind ``embeddings`` re-embeds them.

The embedder is the only model Store runs. It loads lazily on the first query, before the read
snapshot opens, and stays resident; a missing or broken install answers 503 ``search_unavailable``
(not retryable), never a crash and never a silent fallback to another scheme.

Hit text, speaker and timing come from the call's current transcript view, so a hit is masked like
every reviewer read and follows the newest speaker attribution; a call whose text is withheld
(contract 1.2.0: no PII findings for its current transcript yet) contributes no hits. Cosine similarity below 0 counts as 0.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from call1 import embedding
from call1.contracts.calls import SemanticSearchHit, SemanticSearchQuery, SemanticSearchResponse
from call1.contracts.errors import ErrorCode

from ..db import StoreConnection, read_snapshot
from ..errors import StoreError
from . import records


def search_embedder() -> embedding.Embedder:
    """Store's search embedder, resolved from the environment (``call1.embedding.configured_backend``)."""
    try:
        return embedding.get_embedder()
    except embedding.EmbeddingConfigError as exc:
        raise StoreError(ErrorCode.SEARCH_UNAVAILABLE, "Semantic search is misconfigured on Store",
                         details={"reason": "misconfigured"}, retryable=False) from exc


def embed_query(query: str) -> Tuple[str, List[float]]:
    embedder = search_embedder()
    try:
        return embedder.scheme, embedder.embed_query(query)
    except embedding.EmbedderUnavailable as exc:
        message = ("Semantic search is unavailable: the search embedding model is not installed on Store"
                   if exc.reason == "not_installed" else "Semantic search is unavailable: the search embedding model failed to load")
        raise StoreError(ErrorCode.SEARCH_UNAVAILABLE, message, details={"reason": exc.reason, "embedding_scheme": embedder.scheme},
                         retryable=False) from exc


def search(conn: StoreConnection, query: SemanticSearchQuery) -> SemanticSearchResponse:
    scheme, query_vector = embed_query(query.query)
    qv = np.asarray(query_vector, dtype=np.float32)
    with read_snapshot(conn):
        scope, scope_args = "", []
        if query.call_id is not None:
            scope, scope_args = " AND call_id = ?", [query.call_id]
        rows = conn.execute("SELECT conversation_id, call_id, turn_id, dimensions, vector FROM results_search_vectors "
                            f"WHERE scheme = ? AND dimensions = ?{scope}", [scheme, len(query_vector), *scope_args]).fetchall()
        stale = conn.execute("SELECT COUNT(DISTINCT conversation_id) FROM results_search_vectors WHERE scheme != ?"
                             f"{scope}", [scheme, *scope_args]).fetchone()[0]
        scored: List[Tuple[float, str, str, int]] = []
        if rows:
            matrix = np.frombuffer(b"".join(row["vector"] for row in rows), dtype=np.float32).reshape(len(rows), len(query_vector))
            norms = np.linalg.norm(matrix, axis=1) * float(np.linalg.norm(qv))
            sims = np.divide(matrix @ qv, norms, out=np.zeros(len(rows), dtype=np.float32), where=norms > 0)
            for row, value in zip(rows, sims.tolist()):
                similarity = max(0.0, min(1.0, float(value)))
                if similarity >= query.min_score:
                    scored.append((similarity, row["call_id"], row["conversation_id"], int(row["turn_id"])))
        scored.sort(key=lambda s: (-s[0], s[1], s[3]))

        views: Dict[str, Optional[dict]] = {}
        hits: List[SemanticSearchHit] = []
        for similarity, call_id, conversation_id, turn_id in scored:
            if call_id not in views:
                view = records.transcript_view(conn, call_id, conversation_id)
                # A call whose text is withheld (no PII findings for its transcript yet) yields no hits.
                views[call_id] = {t.turn_id: t for t in view.turns} if view and not view.text_withheld else None
            turns = views[call_id]
            turn = turns.get(turn_id) if turns else None
            if turn is None:
                continue
            if query.speaker_filter is not None and turn.speaker is not query.speaker_filter:
                continue
            hits.append(SemanticSearchHit(call_id=call_id, turn_id=turn_id, speaker=turn.speaker, start_time=turn.start_time,
                                          end_time=turn.end_time, text=turn.text, similarity_score=round(similarity, 4)))
            if len(hits) >= query.top_k:
                break
    return SemanticSearchResponse(query=query.query, count=len(hits), results=hits, embedding_scheme=scheme,
                                  calls_needing_reembedding=int(stale))


def embedder_status() -> Optional[dict]:
    """``StoreStatus.search_embedder``: the configured embedder's state, never loading it."""
    try:
        status = embedding.get_embedder().status()
    except embedding.EmbeddingConfigError:
        return None
    return {k: status[k] for k in ("scheme", "model", "revision", "dimensions", "state", "detail")}


__all__ = ["embed_query", "embedder_status", "search", "search_embedder"]
