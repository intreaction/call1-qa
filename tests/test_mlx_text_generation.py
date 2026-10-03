"""``MLXAdapter.generate`` without MLX: the outlines vocabulary and schema-index caches (the index
is compiled before ``inference_lock`` and matches what ``OutlinesCoreBackend`` builds), and
``keep_text_model_loaded`` (one load per multi-prompt job, freed after, never shared across jobs).
``mlx.core`` and ``mlx_lm`` are stubbed, so this runs on any host."""
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from call1.adapters import mlx as mlx_adapter
from call1.pipeline.inference import inference_lock

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["answer"],
          "properties": {"answer": {"enum": ["yes", "no"]}}}


class TinyTokenizer:
    """Enough of an mlx_lm TokenizerWrapper for outlines: characters plus a few words."""

    name_or_path = "/models/tiny"
    eos_token = "</s>"
    eos_token_id = 0

    def __init__(self):
        pieces = ["</s>", *'{}":, ', *"abcdefghijklmnopqrstuvwxyz", "yes", "no", "answer", '{"', '":', '"}']
        self.vocab = {p: i for i, p in enumerate(dict.fromkeys(pieces))}
        self.vocab_size = len(self.vocab)
        self.vocab_calls = 0

    def get_vocab(self):
        self.vocab_calls += 1
        return dict(self.vocab)

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)


@pytest.fixture(autouse=True)
def fresh_caches():
    mlx_adapter.clear_outlines_cache()
    yield
    mlx_adapter.clear_outlines_cache()


def _allowed_walk(index, steps=12):
    """The allowed-token sets along the greedy path that always takes the smallest allowed id."""
    from outlines_core import Guide

    guide, walk = Guide(index), []
    for _ in range(steps):
        allowed = sorted(guide.get_tokens())
        walk.append(allowed)
        if guide.is_finished() or not allowed:
            break
        guide.advance(allowed[0], return_tokens=False)
    return walk


def test_the_cached_index_is_the_one_outlines_core_backend_builds():
    pytest.importorskip("outlines_core")
    import json

    from outlines.backends.outlines_core import OutlinesCoreBackend
    from outlines_core import Index
    from outlines_core.json_schema import build_regex_from_schema

    tok = TinyTokenizer()
    ours = mlx_adapter.schema_index(("tokenizer", "tiny"), SCHEMA, tok)
    vocabulary = OutlinesCoreBackend.create_outlines_core_vocabulary(tok.get_vocab(), tok.eos_token_id, tok.eos_token,
                                                                     lambda t: tok.convert_tokens_to_string([t]))
    reference = Index(build_regex_from_schema(json.dumps(SCHEMA), None), vocabulary)
    assert _allowed_walk(ours) == _allowed_walk(reference)


def test_the_vocabulary_is_built_once_per_tokenizer_and_the_index_once_per_schema():
    pytest.importorskip("outlines_core")
    tok = TinyTokenizer()
    key = ("tokenizer", "tiny")
    first = mlx_adapter.schema_index(key, SCHEMA, tok)
    assert mlx_adapter.schema_index(key, SCHEMA) is first  # no tokenizer needed once cached
    other = dict(SCHEMA, properties={"answer": {"enum": ["no"]}})
    assert mlx_adapter.schema_index(key, other) is not first
    assert tok.vocab_calls == 1
    assert mlx_adapter.schema_index(("tokenizer", "unseen"), SCHEMA) is None  # never seen: the caller builds it later


@pytest.fixture
def stub_mlx(monkeypatch, tmp_path):
    """Stub ``mlx.core`` / ``mlx_lm`` and ``generate_loaded``; record loads, lock state and indexes."""
    record = SimpleNamespace(loads=[], cleared=0, index_under_lock=[], generations=[])
    core = ModuleType("mlx.core")
    core.clear_cache = lambda: setattr(record, "cleared", record.cleared + 1)
    package = ModuleType("mlx")
    package.core = core
    lm = ModuleType("mlx_lm")

    def load(path, adapter_path=None):
        model = SimpleNamespace(name=f"model-{len(record.loads)}", model_type="gemma4")
        record.loads.append((path, adapter_path))
        return model, TinyTokenizer()

    lm.load = load
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setitem(sys.modules, "mlx_lm", lm)
    real_index = mlx_adapter.schema_index

    def schema_index(key, schema, tokenizer=None):
        record.index_under_lock.append((inference_lock._is_owned(), tokenizer is not None))
        return real_index(key, schema, tokenizer)

    monkeypatch.setattr(mlx_adapter, "schema_index", schema_index)

    def generate_loaded(model, tokenizer, system, prompt, max_tokens, response_schema=None, index=None):
        record.generations.append((model.name, index))
        return f"{model.name}:{prompt}"

    monkeypatch.setattr(mlx_adapter, "generate_loaded", generate_loaded)
    return record


def test_each_prompt_loads_and_frees_the_model_by_default(stub_mlx):
    adapter = mlx_adapter.MLXAdapter()
    assert adapter.generate("s", "p1", 8, text_model_path="/models/tiny") == "model-0:p1"
    assert adapter.generate("s", "p2", 8, text_model_path="/models/tiny") == "model-1:p2"
    assert len(stub_mlx.loads) == 2 and stub_mlx.cleared == 2


def test_the_schema_index_is_compiled_before_the_lock_once_the_tokenizer_is_known(stub_mlx):
    pytest.importorskip("outlines_core")
    adapter = mlx_adapter.MLXAdapter()
    adapter.generate("s", "p1", 8, response_schema=SCHEMA, text_model_path="/models/tiny")
    # First prompt of the process: nothing cached before the lock, built once the tokenizer loaded.
    assert stub_mlx.index_under_lock == [(False, False), (True, True)]
    stub_mlx.index_under_lock.clear()
    adapter.generate("s", "p2", 8, response_schema=SCHEMA, text_model_path="/models/tiny")
    assert stub_mlx.index_under_lock == [(False, False)]  # later prompts: compiled (cached) outside the lock
    first, second = (index for _, index in stub_mlx.generations)
    assert first is not None and first is second


def test_keep_text_model_loaded_loads_once_per_block_and_frees_after(stub_mlx):
    adapter = mlx_adapter.MLXAdapter()
    with mlx_adapter.keep_text_model_loaded():
        assert inference_lock._is_owned()
        answers = [adapter.generate("s", f"p{i}", 8, text_model_path="/models/tiny") for i in range(3)]
        assert mlx_adapter._resident_text_models.get()
    assert answers == ["model-0:p0", "model-0:p1", "model-0:p2"]
    assert len(stub_mlx.loads) == 1
    assert mlx_adapter._resident_text_models.get() is None and not inference_lock._is_owned()
    adapter.generate("s", "after", 8, text_model_path="/models/tiny")
    assert len(stub_mlx.loads) == 2  # never resident between jobs


def test_a_different_model_or_adapter_in_one_block_loads_separately(stub_mlx, monkeypatch):
    adapter = mlx_adapter.MLXAdapter()
    with mlx_adapter.keep_text_model_loaded():
        adapter.generate("s", "a", 8, text_model_path="/models/tiny")
        adapter.generate("s", "b", 8, text_model_path="/models/other")
        adapter.generate("s", "c", 8, text_model_path="/models/tiny")
    assert stub_mlx.loads == [("/models/tiny", None), ("/models/other", None)]


def test_the_gemma_classifier_keeps_the_model_for_its_job_only(stub_mlx):
    from call1.process.handlers.real.signals_v2 import GemmaSegmentClassifier

    classifier = GemmaSegmentClassifier.__new__(GemmaSegmentClassifier)
    adapter = mlx_adapter.MLXAdapter()
    with inference_lock:  # run_classifier's one hold
        classifier.load()
        for prompt in ("batch 1", "batch 2"):
            adapter.generate("s", prompt, 8, text_model_path="/models/tiny")
        classifier.release()
        assert mlx_adapter._resident_text_models.get() is None
    classifier.release()  # idempotent
    assert len(stub_mlx.loads) == 1 and not inference_lock._is_owned()
