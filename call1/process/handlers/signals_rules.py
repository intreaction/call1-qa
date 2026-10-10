"""The Contact Signals rules engine inside the categorize job (docs/SignalsEmbeddings.md; contract 1.4.0).

``handlers/signal_stages.run_categorize`` wraps its stage-1 engine with ``maybe_rules_engine`` in
every mode (fake and real). When at least one active category's recipe engine is ``rules``,
the wrapper is a ``RulesClassifier``. Recipes determine detection independently of the legacy
taxonomy-wide setting:

* it embeds the job's masked ~7 s segments with the search embedder (``call1.embedding``:
  Nemotron-3-Embed-1B on the Apple GPU in real mode, the deterministic fake in fake mode), with the
  ``passage:`` prefix on both sides, inside the ``inference_lock`` hold ``run_classifier`` takes, then
  drops the embedder and frees the MPS cache before any Gemma work;
* it embeds the example bank (the pack the taxonomy pins by digest, plus entries written from the
  taxonomy's own text) once per bank and embedder scheme, and caches the vectors under Process's
  data directory (``<data>/signal-bank/``, mode 0700), keyed by a digest of the scheme and texts;
* it runs the pure engine (``call1.pipeline.signal_rules``) and answers the rules categories' options
  with the pick scores (0.9 / 0.7), and, for a mixed taxonomy, sends the same rows with only the
  other categories' options to the wrapped Gemma (or fake) engine, loading it only when such rows exist;
* after the spans are built it records ``SignalCategoriesContent.rules`` and one
  ``SignalRuleDecision`` per rules span (``finish``).

**Fail closed.** A missing embedder (``model_unavailable``), a missing bank pack or one whose digest
differs from the taxonomy's pin (``input_unavailable``) fails the job, so the result group fails
visibly; it never publishes "no signals". Nothing here reads raw text: rows carry masked segment
text, and taxonomy text is masked with the call's values before it is embedded.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from call1 import embedding
from call1.contracts.common import canonical_digest
from call1.contracts.contents import (
    SIGNAL_NONE_OPTION,
    SIGNAL_RULES_ENTRY_ID,
    SignalCategoriesContent,
    SignalExampleBankRef,
    SignalRulesProvenance,
    short_digest,
)
from call1.contracts.errors import JobErrorCode
from call1.contracts.signals import SignalKnnSettings, SignalRecipe, SignalRulesConfig, rules_categories
from call1.pipeline import signal_rules as engine_rules
from call1.pipeline.signals_v2 import ChoiceRow, EngineError, RowBudget, SegmentClassifier

from .base import HandlerError, HandlerJob

log = logging.getLogger("call1.process.handlers.signals_rules")

EMBED_BATCH = 16
"""Texts per embedder batch (``handlers/embeddings.BATCH_TURNS``, and the research harness's)."""
RULES_TEMPLATE = "signals.rules.v1"
BANK_DIR_ENV = "CALL1_SIGNAL_BANK_DIR"
CACHE_DIR_ENV = "CALL1_SIGNAL_BANK_CACHE"


# --- the bank pack ---------------------------------------------------------------------------------


def bank_dir() -> Path:
    """Where installed bank packs live: ``CALL1_SIGNAL_BANK_DIR``, else ``data/signal-banks`` (the
    working directory's, as the model weights' ``data/models``), else the repository's."""
    explicit = os.environ.get(BANK_DIR_ENV)
    if explicit:
        return Path(explicit)
    local = Path("data") / "signal-banks"
    if local.is_dir():
        return local
    repository = Path(__file__).resolve().parents[3] / "data" / "signal-banks"
    if repository.is_dir():
        return repository
    return Path(__file__).resolve().parents[1] / "resources" / "signal-banks"


_PACKS: Dict[Tuple[str, float, int], Dict[str, Any]] = {}
_PACKS_LOCK = threading.Lock()


def load_bank_pack(ref: SignalExampleBankRef) -> Dict[str, Any]:
    """The pinned pack, verified against the taxonomy's digest. Parsed packs are kept per file
    version, so a job does not re-read 0.5 MB of JSON."""
    path = bank_dir() / f"{ref.bank_id}.json"
    try:
        stat = path.stat()
    except OSError:
        raise HandlerError(JobErrorCode.INPUT_UNAVAILABLE, f"example bank {ref.bank_id} is not installed on this host") from None
    key = (str(path.resolve()), stat.st_mtime, stat.st_size)
    with _PACKS_LOCK:
        pack = _PACKS.get(key)
    if pack is None:
        try:
            pack = json.loads(path.read_text(encoding="utf-8"))
            digest = engine_rules.bank_pack_digest(pack)
        except (OSError, ValueError, KeyError, TypeError):
            raise HandlerError(JobErrorCode.INPUT_UNAVAILABLE, f"example bank {ref.bank_id} is unreadable") from None
        pack = {**pack, "_digest": digest}
        with _PACKS_LOCK:
            _PACKS[key] = pack
    if pack.get("bank_id") != ref.bank_id or pack["_digest"] != ref.digest:
        raise HandlerError(JobErrorCode.INPUT_UNAVAILABLE, f"example bank {ref.bank_id} does not match the digest the taxonomy pins")
    return pack


# --- the embedder and the vector cache ---------------------------------------------------------------


def embedder_for(device: str) -> embedding.Embedder:
    """The fake embedder on fake handlers (``device == 'fake'``) or when ``CALL1_EMBEDDING_BACKEND``
    says so; otherwise Nemotron on the Apple GPU when there is one (``device='auto'``)."""
    backend = "fake" if device == "fake" else embedding.configured_backend("real")
    return embedding.get_embedder(backend, device="auto")


def cache_dir(job: HandlerJob) -> Path:
    explicit = os.environ.get(CACHE_DIR_ENV)
    if explicit:
        return Path(explicit)
    scratch = Path(job.scratch_dir)
    root = scratch.parents[2] if scratch.parent.name == "jobs" and len(scratch.parents) > 2 else scratch
    return root / "signal-bank"


def _cached_vectors(job: HandlerJob, entries: Sequence[engine_rules.BankEntry], scheme: str,
                    encode: Callable[[List[str]], np.ndarray]) -> np.ndarray:
    if not entries:
        return np.zeros((0, 0), dtype=np.float32)
    key = engine_rules.texts_digest(entries, scheme).removeprefix("sha256:")[:40]
    folder = cache_dir(job)
    path = folder / f"{key}.npy"
    if path.is_file():
        try:
            found = np.load(path, allow_pickle=False)
            if found.shape[0] == len(entries):
                return found.astype(np.float32, copy=False)
        except (OSError, ValueError):
            log.warning("signal bank vector cache %s unreadable; re-embedding", path.name)
    vectors = encode([e.text for e in entries])
    try:
        folder.mkdir(parents=True, exist_ok=True)
        os.chmod(folder, 0o700)
        tmp = folder / f".{key}.{os.getpid()}.tmp.npy"
        np.save(tmp, vectors, allow_pickle=False)
        os.replace(tmp, path)
    except OSError:  # a cache that cannot be written only costs time
        log.warning("signal bank vector cache not written")
    return vectors


def _free_embedder(emb: embedding.Embedder) -> None:
    unload = getattr(emb, "unload", None)
    if unload is None:
        return
    try:
        unload()
    except Exception:  # pragma: no cover - only with torch on MPS
        log.warning("embedder unload failed")
    gc.collect()


# --- the engine --------------------------------------------------------------------------------------


def active_rules(ctx) -> List[Any]:
    """The categories the rules engine decides in this job (empty: today's pipeline)."""
    return rules_categories(ctx.taxonomy, ctx.snapshot.settings)


def maybe_rules_engine(job: HandlerJob, ctx, engine: Optional[SegmentClassifier], *, device: str) -> Optional[SegmentClassifier]:
    """``engine`` unchanged unless rules detection decides at least one category; then the
    ``RulesClassifier`` around it."""
    categories = active_rules(ctx)
    if getattr(engine, "semantic_candidates", False):
        # A category without a semantic recipe still goes through similarity. This runtime
        # policy is versioned in the cascade provenance; it does not edit the pinned taxonomy.
        categories = [c if c.recipe is not None and c.recipe.engine == "rules" else
                      c.model_copy(update={"recipe": SignalRecipe(engine="rules", threshold=.5, check="gemma")})
                      for c in ctx.taxonomy.categories if c.active]
        if not categories:
            return engine
        return SemanticSystemOneClassifier(job, ctx, engine, device=device, categories=categories)
    if not categories:
        return engine
    return RulesClassifier(job, ctx, engine, device=device, categories=categories)


class RulesClassifier:
    """Stage 1 with the rules engine (and the wrapped engine for the categories without rules)."""

    key_orders = 1

    def __init__(self, job: HandlerJob, ctx, gemma: Optional[SegmentClassifier], *, device: str, categories: Sequence[Any]) -> None:
        self.job = job
        self.ctx = ctx
        self.gemma = gemma
        self.device = device
        self.entry_id = SIGNAL_RULES_ENTRY_ID
        self.plans = engine_rules.plan_categories(categories)
        self.rules_ids = [p.category_id for p in self.plans]
        self.gemma_ids = [c.category_id for c in ctx.taxonomy.categories if c.active and c.category_id not in set(self.rules_ids)]
        self.config: SignalRulesConfig = ctx.taxonomy.rules or SignalRulesConfig()
        self.budget: RowBudget = getattr(gemma, "budget", None) or RowBudget()
        gemma_part = ""
        if self.gemma_ids and gemma is not None:
            gemma_part = "+" + str(getattr(gemma, "stage1_template", None) or getattr(gemma, "entry_id", "engine"))
            gemma_part += ":" + str(getattr(gemma, "calibration_id", ""))
        self.stage1_template = (RULES_TEMPLATE + gemma_part)[:200]
        identity = canonical_digest({"engine": engine_rules.ENGINE_VERSION, "recipes": {p.category_id: p.digest12 for p in self.plans},
                                     "bank": self.config.bank.model_dump(mode="json") if self.config.bank else None,
                                     "knn": self.config.knn.model_dump(mode="json"), "gemma": gemma_part})
        self.calibration_id = f"{engine_rules.ENGINE_VERSION}:{short_digest(identity)}"
        self.result: Optional[engine_rules.CallRules] = None
        self.index: Optional[engine_rules.KnnIndex] = None
        self.units: List[engine_rules.RuleUnit] = []
        self.bank_entries = 0
        self.taxonomy_entries = 0
        self.embed_seconds = 0.0
        self.rules_seconds = 0.0
        self.scheme = ""
        self._gemma_loaded = False
        self._embedder: Optional[embedding.Embedder] = None

    @property
    def model_revision(self) -> Optional[str]:
        return getattr(self.gemma, "model_revision", None) if self.gemma_ids else None

    @property
    def transport(self):  # the Gemma transport's usage, when Gemma answered part of the job
        return getattr(self.gemma, "transport", None)

    # --- SegmentClassifier ----------------------------------------------------------------------

    def load(self) -> None:
        """Nothing: the embedder loads on the first encode, Gemma only when it has rows."""
        return None

    def release(self) -> None:
        if self._embedder is not None:
            _free_embedder(self._embedder)
            self._embedder = None
        if self._gemma_loaded and self.gemma is not None:
            self._gemma_loaded = False
            self.gemma.release()

    def choose(self, rows: Sequence[ChoiceRow]) -> List[Dict[str, float]]:
        rules_set = set(self.rules_ids)
        segments = {s.index: s for s in self.ctx.segmentation.segments}
        rule_rows = [r for r in rows if any(o in rules_set for o, _ in r.options)]
        units = []
        for r in rule_rows:
            seg = segments[int(r.key)]
            units.append(engine_rules.RuleUnit(index=seg.index, turn_id=seg.turn_id, window=seg.window, block=seg.block,
                                               speaker=engine_rules.speaker_key(seg.speaker), text=str(r.state.get("turn") or ""),
                                               start=float(seg.start), end=float(seg.end)))
        self.units = units
        self._run_rules(units)
        kept = self.result.kept if self.result is not None else {}
        by_key = {u.index: kept.get(i, []) for i, u in enumerate(units)}
        # The other categories: the wrapped engine on the same rows with only their options.
        gemma_answers: Dict[str, Dict[str, float]] = {}
        reduced = []
        for r in rows:
            options = tuple(o for o in r.options if o[0] not in rules_set)
            if any(o != SIGNAL_NONE_OPTION for o, _ in options):
                reduced.append(ChoiceRow(key=r.key, question=r.question, options=options, state=r.state, truncated=r.truncated))
        if reduced:
            if self.gemma is None:
                raise HandlerError(JobErrorCode.MODEL_UNAVAILABLE, "no classifier for the categories without a rules recipe")
            self.gemma.load()
            self._gemma_loaded = True
            got = self.gemma.choose(reduced)
            if len(got) != len(reduced):
                raise HandlerError(JobErrorCode.VALIDATION_REJECTED, "the classifier did not answer every row")
            gemma_answers = {r.key: a for r, a in zip(reduced, got)}
        answers = []
        for r in rows:
            options = [o for o, _ in r.options]
            rule_options = [o for o in options if o in rules_set] + [SIGNAL_NONE_OPTION]
            answer = engine_rules.stage1_answer(rule_options, by_key.get(int(r.key), []), SIGNAL_NONE_OPTION)
            fired = answer[SIGNAL_NONE_OPTION] < 0.5
            other = gemma_answers.get(r.key)
            merged = {o: 0.0 for o in options}
            merged.update({o: v for o, v in answer.items() if o != SIGNAL_NONE_OPTION})
            none = answer[SIGNAL_NONE_OPTION]
            if other is not None:
                merged.update({o: float(v) for o, v in other.items() if o != SIGNAL_NONE_OPTION and o not in rules_set})
                none = min(none, float(other.get(SIGNAL_NONE_OPTION, 1.0)))
            merged[SIGNAL_NONE_OPTION] = min(none, engine_rules.FIRED_NONE) if fired else none
            # At most two categories per segment across the rules and the other engine: the rules'
            # fires first at equal scores, then the other engine's picks.
            picks = sorted((o for o in options if o != SIGNAL_NONE_OPTION and merged[o] >= 0.5),
                           key=lambda o: (-merged[o], 0 if o in rules_set else 1, options.index(o)))
            for o in picks[engine_rules.MAX_FIRES:]:
                merged[o] = 0.0
            answers.append(merged)
        return answers

    def _run_rules(self, units: Sequence[engine_rules.RuleUnit]) -> None:
        """Embed, look up, evaluate: ``self.result``, ``self.index``."""
        started = time.monotonic()
        taxonomy = self.ctx.taxonomy
        pack_entries: List[engine_rules.BankEntry] = []
        if self.config.bank is not None:
            pack_entries = engine_rules.pack_entries(load_bank_pack(self.config.bank))
        tax_entries = engine_rules.taxonomy_entries(taxonomy, self.config.knn.taxonomy_example_weight, mask=self.ctx.mask)
        self.bank_entries, self.taxonomy_entries = len(pack_entries), len(tax_entries)
        if not units:
            self.index = engine_rules.KnnIndex([], np.zeros((0, 1), np.float32), taxonomy)
            self.result = engine_rules.evaluate_call([], self.ctx.transcript.duration_seconds, self.plans,
                                                     self.index.scores(np.zeros((0, 1), np.float32), [], self.config.knn), self.index)
            return
        try:
            emb = embedder_for(self.device)
            self._embedder = emb
            self.scheme = emb.scheme

            def encode(texts: List[str]) -> np.ndarray:
                self.job.check_cancelled()
                return np.asarray(emb.embed_documents(texts, batch_size=EMBED_BATCH, cancelled=self.job.check_cancelled), dtype=np.float32)

            bank_vectors = _cached_vectors(self.job, pack_entries, emb.scheme, encode)
            tax_vectors = _cached_vectors(self.job, tax_entries, emb.scheme, encode)
            query = encode([u.text for u in units])
        except embedding.EmbedderUnavailable as exc:
            raise HandlerError(JobErrorCode.MODEL_UNAVAILABLE, f"the rules engine needs the search embedder: {exc}") from None
        finally:
            if self._embedder is not None:  # free the GPU before any Gemma work in this job
                _free_embedder(self._embedder)
                self._embedder = None
                _empty_mps_cache()
        self.embed_seconds = round(time.monotonic() - started, 3)
        started = time.monotonic()
        parts = [v for v in (bank_vectors, tax_vectors) if len(v)]
        vectors = np.concatenate(parts) if parts else np.zeros((0, query.shape[1]), np.float32)
        self.index = engine_rules.KnnIndex(pack_entries + tax_entries, vectors, taxonomy)
        knn = self.index.scores(query, [u.speaker for u in units], self.config.knn)
        self.result = engine_rules.evaluate_call(units, float(self.ctx.transcript.duration_seconds or 0.0), self.plans, knn, self.index)
        self.rules_seconds = round(time.monotonic() - started, 4)

    # --- after the spans are built -----------------------------------------------------------------

    def finish(self, content: SignalCategoriesContent) -> SignalCategoriesContent:
        """The categories artifact with the rules provenance and one decision per rules span."""
        decisions = []
        if self.result is not None and self.index is not None:
            fires = {(f.unit_index, f.category_id): f for row in self.result.kept.values() for f in row}
            units = {(u.turn_id, u.window): u for u in self.units}
            plans = {p.category_id: p for p in self.plans}
            for span in content.spans:
                plan = plans.get(span.category_id)
                if plan is None:
                    continue
                found = engine_rules.span_decision(span, fires, units, plan, self.index, self.result.knn)
                if found is not None:
                    decisions.append(found)
        provenance = SignalRulesProvenance(
            engine_version=engine_rules.ENGINE_VERSION, embedder_scheme=self.scheme or "none", bank=self.config.bank,
            bank_entries=self.bank_entries, taxonomy_entries=self.taxonomy_entries, knn=self.config.knn or SignalKnnSettings(),
            rules_categories=list(self.rules_ids), checked_categories=[p.category_id for p in self.plans if p.recipe.check == "gemma"],
            gemma_categories=list(self.gemma_ids), recipe_digests={p.category_id: p.digest12 for p in self.plans},
            counts=list(self.result.counts) if self.result is not None else [], embedded_segments=len(self.units),
            embed_seconds=self.embed_seconds, rules_seconds=self.rules_seconds)
        data = content.model_dump(mode="json")
        data["rules"] = provenance.model_dump(mode="json")
        data["rule_decisions"] = [d.model_dump(mode="json") for d in decisions]
        return SignalCategoriesContent.model_validate(data)


class SemanticSystemOneClassifier(RulesClassifier):
    """Semantic recipes propose candidates; Laya triages; Gemma confirms uncertainty.

    A shortcut needs every segment's Laya score >= .95, category vote >= .8, a real
    subcategory with >= .8 vote and a positive matching neighbour cosine >= .75.
    These are conservative, unfitted thresholds, not a promise of accuracy. A low score
    never vetoes a semantic candidate. Overflow/provider failures route to Gemma too.
    Categories without semantic recipes get a .5 semantic-share proposal threshold.
    """
    LAYA_KEEP = .95
    SEMANTIC_SHARE = .8
    SEMANTIC_COSINE = .75
    requires_exact_revision = True
    FALLBACK_CODES = {"context_limit_exceeded", "provider_error", "provider_timeout", "model_unavailable", "validation_rejected"}

    def __init__(self, job, ctx, laya, *, device, categories):
        super().__init__(job, ctx, laya, device=device, categories=categories)
        self.entry_id = laya.entry_id
        self.budget = RowBudget(max_len=8192, head_tokens=1024, count_tokens=lambda text: len(text.encode("utf-8")))
        self.stage1_template = "semantic-laya.v1:" + short_digest(canonical_digest({"laya": laya.stage1_template,
                                                                                "semantic": self.stage1_template}))
        self.calibration_id = "semantic-laya-unfitted-v1:" + short_digest(canonical_digest({"semantic": self.calibration_id,
            "keep": self.LAYA_KEEP, "share": self.SEMANTIC_SHARE, "cosine": self.SEMANTIC_COSINE}))
        self.scores = {}
        self.failures = {}

    @property
    def model_revision(self):
        return self.gemma.model_revision

    def choose(self, rows):
        answers = super().choose(rows)
        candidates = []
        for row, answer in zip(rows, answers):
            options = tuple((key, gloss) for key, gloss in row.options
                            if key == SIGNAL_NONE_OPTION or (key in self.rules_ids and answer.get(key, 0) >= .5))
            if len(options) > 1:
                candidates.append(ChoiceRow(row.key, row.question, options, row.state, row.truncated))
        for row in candidates:
            self.job.check_cancelled()
            try:
                if not self._gemma_loaded:
                    self.gemma.load()
                    self._gemma_loaded = True
                answer = self.gemma.choose([row])[0]
                for key, _ in row.options:
                    if key != SIGNAL_NONE_OPTION:
                        self.scores[(int(row.key), key)] = answer[key]
            except EngineError as exc:
                if exc.code not in self.FALLBACK_CODES:
                    raise
                for key, _ in row.options:
                    if key != SIGNAL_NONE_OPTION:
                        self.failures[(int(row.key), key)] = JobErrorCode(exc.code)
        # Preserve semantic candidates, including Laya's low scores, for Gemma to decide.
        return answers

    def finish(self, content):
        content = super().finish(content)
        spans = {s.span_key: s for s in content.spans}
        decisions = []
        for decision in content.rule_decisions:
            span = spans[decision.span_key]
            members = [s.index for s in content.segments if s.turn_id == span.turn_id and s.block == span.block
                       and span.first_window <= s.window <= span.last_window]
            scores = [self.scores.get((i, decision.category_id)) for i in members]
            failure = next((self.failures[(i, decision.category_id)] for i in members
                            if (i, decision.category_id) in self.failures), None)
            score = min(scores) if scores and all(value is not None for value in scores) else None
            matching = [n.cosine for n in decision.neighbours if n.carries_category]
            kept = (failure is None and score is not None and score >= self.LAYA_KEEP
                    and decision.knn_share >= self.SEMANTIC_SHARE and decision.subcategory_id is not None
                    and decision.subcategory_share >= self.SEMANTIC_SHARE
                    and bool(matching) and max(matching) >= self.SEMANTIC_COSINE)
            decisions.append(decision.model_copy(update={"check": not kept, "system_one_score": score,
                                                        "system_one_kept": kept, "system_one_fallback": failure}))
        rules = content.rules.model_copy(update={"checked_categories": sorted({d.category_id for d in decisions if d.check})})
        return content.model_copy(update={"rule_decisions": decisions, "rules": rules})


def _empty_mps_cache() -> None:
    try:  # pragma: no cover - only with torch on MPS
        import sys

        torch = sys.modules.get("torch")
        if torch is not None and hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def rules_ready(job: HandlerJob) -> None:
    """Claim-time check for real handlers: under rules detection this host needs the embedder
    weights. Raises ``ReleaseJob('reject', model_unavailable)`` so the job is refused before any
    inference rather than publishing without the rules categories."""
    from .base import ReleaseJob

    item = job.input("taxonomy") if hasattr(job, "input") else None
    if item is None:
        return
    try:
        snapshot = item.content()
    except Exception:  # the stage itself reports an unreadable input
        return
    from call1.process.system_one import ENTRY_ID

    cascade = job.catalog_entry is not None and job.catalog_entry.entry_id == ENTRY_ID
    if not rules_categories(snapshot.taxonomy, snapshot.settings) and not (cascade and any(c.active for c in snapshot.taxonomy.categories)):
        return
    if embedding.configured_backend("real") != "fake" and not embedding.weights_installed():
        raise ReleaseJob("reject", JobErrorCode.MODEL_UNAVAILABLE, "rules detection needs the search embedding weights on this host")


__all__ = ["RulesClassifier", "active_rules", "bank_dir", "embedder_for", "load_bank_pack", "maybe_rules_engine", "rules_ready"]
