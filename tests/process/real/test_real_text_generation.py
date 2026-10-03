"""Gemma 4 E2B on MLX (opt-in: ``CALL1_REAL_MODELS=1``, Apple Silicon, ``data/models/gemma-4-e2b-it``):
the cached outlines index (compiled before ``inference_lock``) and ``keep_text_model_loaded`` (one
load for several prompts) give the same greedy answers, character for character, as the previous
path, which loaded the model and rebuilt the outlines backend for every prompt.

    CALL1_REAL_MODELS=1 .venv-local/bin/python -m pytest tests/process/real/test_real_text_generation.py -m real_models -s -p no:warnings
"""

from __future__ import annotations

import gc
import json
import os
import time

import pytest

from ..conftest import REPO

GEMMA = REPO / "data" / "models" / "gemma-4-e2b-it"

pytestmark = [
    pytest.mark.real_models,
    pytest.mark.skipif(os.getenv("CALL1_REAL_MODELS") != "1", reason="set CALL1_REAL_MODELS=1 to run on the real models"),
    pytest.mark.skipif(not (GEMMA / "config.json").is_file(), reason=f"Gemma 4 E2B is not installed at {GEMMA}"),
]

SYSTEM = "You label customer-service call segments. Answer only with the JSON the schema allows."
OPTIONS = ["price_complaint", "cancel_request", "delivery_issue", "greeting", "none"]
SEGMENTS = {
    "s1": "Hi, thanks for calling, my name is Sam, how can I help today?",
    "s2": "Yeah, my order was meant to arrive Tuesday and it still isn't here.",
    "s3": "Honestly the price went up again and I want to cancel the whole plan.",
}
SCHEMA = {"type": "object", "additionalProperties": False, "required": list(SEGMENTS),
          "properties": {k: {"type": "object", "additionalProperties": False, "required": ["labels"],
                             "properties": {"labels": {"type": "array", "maxItems": 2, "items": {"enum": OPTIONS}}}}
                         for k in SEGMENTS}}
PROMPTS = [json.dumps({"segments": [{"id": k, "text": v}]}) for k, v in SEGMENTS.items()] + [json.dumps({"segments": [
    {"id": k, "text": v} for k, v in SEGMENTS.items()]})]


def _schema_for(prompt: str) -> dict:
    keys = [s["id"] for s in json.loads(prompt)["segments"]]
    return dict(SCHEMA, required=keys, properties={k: SCHEMA["properties"][k] for k in keys})


def _previous_path(prompt: str) -> str:
    """The pre-cache path: load per prompt, outlines backend rebuilt for every generation."""
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from outlines.backends.outlines_core import OutlinesCoreBackend
    from outlines.models.mlxlm import from_mlxlm

    from call1.adapters.mlx import _gemma_final_answer

    model, tokenizer = load(str(GEMMA))
    try:
        tokens = tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                                               add_generation_prompt=True, enable_thinking=False)
        processors = [OutlinesCoreBackend(from_mlxlm(model, tokenizer)).get_json_schema_logits_processor(json.dumps(_schema_for(prompt)))]
        out = "".join(p.text for p in stream_generate(model, tokenizer, tokens, max_tokens=160, logits_processors=processors))
        return _gemma_final_answer(out)
    finally:
        del model, tokenizer
        gc.collect()
        mx.clear_cache()


def test_cached_index_and_resident_model_answer_exactly_as_before(monkeypatch):
    from call1.adapters import mlx as mlx_adapter

    monkeypatch.delenv("CALL1_TEXT_ADAPTER", raising=False)
    mlx_adapter.clear_outlines_cache()
    adapter = mlx_adapter.MLXAdapter()
    started = time.monotonic()
    before = [_previous_path(p) for p in PROMPTS]
    previous_seconds = time.monotonic() - started
    started = time.monotonic()
    per_prompt = [adapter.generate(SYSTEM, p, 160, response_schema=_schema_for(p), text_model_path=str(GEMMA)) for p in PROMPTS]
    per_prompt_seconds = time.monotonic() - started
    started = time.monotonic()
    with mlx_adapter.keep_text_model_loaded():
        resident = [adapter.generate(SYSTEM, p, 160, response_schema=_schema_for(p), text_model_path=str(GEMMA)) for p in PROMPTS]
    resident_seconds = time.monotonic() - started
    print(f"\n{len(PROMPTS)} prompts: previous {previous_seconds:.1f} s, cached index {per_prompt_seconds:.1f} s, "
          f"cached index + one load {resident_seconds:.1f} s")
    assert all(json.loads(answer) for answer in before)
    assert per_prompt == before
    assert resident == before
