"""Default System One cascade: endpoint protocol, full targets, frozen digests and rules bypass."""
import json
from pathlib import Path

import httpx
import pytest

from call1.contracts.artifacts import ArtifactKind
from call1.contracts.catalog import ModelPurpose
from call1.contracts.contents import SIGNAL_NONE_OPTION, SpeakerRole
from call1.contracts.jobs import JobType
from call1.contracts.signals import SignalSettings, SignalTaxonomy
from call1.pipeline.signals_v2 import ChoiceRow, EngineError
from call1.process.catalog import CatalogError, replace_entry, seeded_catalog
from call1.process.config import ConfigError, ProcessConfig
from call1.process.handlers.base import JobCancelled, ReleaseJob
from call1.process.handlers.real.signals_v2 import GemmaSegmentClassifier, classifier_for
from call1.process.handlers.real.system_one import SystemOneClassifier
from call1.process.handlers.signal_stages import run_categorize
from call1.process.system_one import ENTRY_ID, SystemOneUnavailable, catalog_entry, discover

from .test_signals_support import findings_for, make_job, script_transcript, signal_params, snapshot

URL = "http://127.0.0.1:11434"
DIGEST = "a" * 64


class Server:
    def __init__(self):
        self.digest = DIGEST
        self.calls = []
        self.reply = None
        self.after = None
        self.client = httpx.Client(transport=httpx.MockTransport(self.handle))

    def handle(self, request):
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.40.0"})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "laya:latest", "digest": self.digest, "size": 800000000}]})
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["decision"], "model_info": {"laya.context_length": 512}})
        assert request.url.path == "/v1/systemone"
        payload = json.loads(request.content)
        self.calls.append(payload)
        answers = {key: {"type": "noul", "noul": 0.9 if key == "intent" else 0.1} for key in payload["questions"]}
        if self.after:
            self.after()
        return httpx.Response(200, json=self.reply or {"model": "laya", "answers": answers,
                                                     "usage": {"input_tokens": 80, "output_tokens": 0}})


@pytest.fixture
def setup(tmp_path, monkeypatch):
    server = Server()
    monkeypatch.setattr("call1.process.system_one.discover", lambda url, model, **kw: discover(url, model, http=server.client, fresh=True))
    entry = catalog_entry(URL, "laya")
    catalog = replace_entry(seeded_catalog(mode="fake"), entry)
    job = make_job(tmp_path, JobType.CONTACT_SIGNALS_CATEGORIZE, {}, catalog=catalog, entry_id=ENTRY_ID)
    yield SystemOneClassifier(job, http=server.client), server, catalog
    server.client.close()


def row(text="Can you hold this jacket?", previous=None):
    return ChoiceRow("1", "What is the caller doing?", (("intent", "Requests an outcome"), ("issue", "Reports a problem"),
                                                      (SIGNAL_NONE_OPTION, "None")),
                     {"speaker": "caller", "turn": text, "previous": previous or []})


def test_endpoint_binary_decisions_usage_and_frozen_digest(setup):
    engine, server, catalog = setup
    engine.ready()
    engine.load()
    assert engine.choose([row()]) == [{"intent": 0.9, "issue": 0.1, "none": pytest.approx(0.1)}]
    payload = server.calls[0]
    assert payload["keep_alive"] == 0 and "Target caller: Can you hold this jacket?" in payload["state"]
    assert set(payload["questions"]) == {"intent", "issue"}
    assert all(question["type"] == "noul" for question in payload["questions"].values())
    assert engine.usage().tokens_input.count == 80 and engine.usage().tokens_output.count == 0
    assert engine.model_revision == "sha256:" + DIGEST
    assert catalog.get(ENTRY_ID).selection(ModelPurpose.SIGNAL_CATEGORY).mutable_alias
    assert catalog.defaults[ModelPurpose.SIGNAL_CATEGORY] != ENTRY_ID
    with pytest.raises(CatalogError):
        catalog.select(ModelPurpose.SIGNAL_SUBCATEGORY, ENTRY_ID)


@pytest.mark.parametrize("value", [-1, 2, True, "0.9", float("inf")])
def test_bad_decision_scores_fail_closed(setup, value):
    engine, server, _ = setup
    server.reply = {"model": "laya", "answers": {"intent": {"type": "noul", "noul": value}, "issue": {"type": "noul", "noul": 0.1}}}
    with pytest.raises(EngineError, match="validation_rejected"):
        engine.choose([row()])
    assert engine.usage().tokens_input is None and engine.usage().tokens_output is None


def test_missing_category_is_not_treated_as_none(setup):
    engine, server, _ = setup
    server.reply = {"model": "laya", "answers": {"intent": {"type": "noul", "noul": 0.9}}}
    with pytest.raises(EngineError, match="validation_rejected"):
        engine.choose([row()])


def test_overflow_never_cuts_target_or_sends_partial_rows(setup):
    engine, server, _ = setup
    with pytest.raises(EngineError, match="context_limit_exceeded"):
        engine.choose([row(), row("x" * 600)])
    assert server.calls == []
    engine.choose([row(previous=[{"speaker": "agent", "text": "old " * 200}, {"speaker": "agent", "text": "Which item?"}])])
    assert "Which item?" in server.calls[0]["state"] and "old old" not in server.calls[0]["state"]
    assert server.calls[0]["state"].endswith("Target caller: Can you hold this jacket?")


def test_digest_change_before_or_during_inference_rejects_results(setup):
    engine, server, _ = setup
    server.digest = "b" * 64
    with pytest.raises(ReleaseJob, match="frozen revision"):
        engine.ready()
    assert not server.calls
    server.digest = DIGEST
    engine.load()
    server.after = lambda: setattr(server, "digest", "b" * 64)
    with pytest.raises(EngineError, match="frozen revision"):
        engine.choose([row()])


def test_cancellation_stops_before_request(setup):
    engine, server, _ = setup
    engine.job._cancel.set()
    with pytest.raises(JobCancelled):
        engine.choose([row()])
    assert not server.calls


def test_misreported_model_and_timeout_are_explicit_errors(setup):
    engine, server, _ = setup
    server.reply = {"model": "other", "answers": {}}
    with pytest.raises(EngineError, match="validation_rejected"):
        engine.choose([row()])
    server.after = lambda: (_ for _ in ()).throw(httpx.ReadTimeout("test"))
    with pytest.raises(EngineError, match="provider_timeout"):
        engine.choose([row()])


def test_semantic_candidates_then_laya_uncertainty_reaches_gemma(setup, tmp_path, monkeypatch):
    _, server, catalog = setup
    seed = json.loads((Path(__file__).parents[2] / "call1/store/seeds/signals_retail_v1.json").read_text())
    seed["taxonomy"]["rules"]["bank"] = None
    snap = snapshot(SignalTaxonomy.model_validate(seed["taxonomy"]), SignalSettings(pipeline="v2", detection="rules"))
    transcript = script_transcript([(SpeakerRole.CALLER, "Can you hold this jacket?")])
    inputs = {"transcript": (ArtifactKind.TRANSCRIPT, transcript),
              "pii_findings": (ArtifactKind.PII_FINDINGS, findings_for(tmp_path, transcript)),
              "taxonomy": (ArtifactKind.SIGNAL_TAXONOMY_SNAPSHOT, snap)}
    job = make_job(tmp_path, JobType.CONTACT_SIGNALS_CATEGORIZE, inputs, catalog=catalog, entry_id=ENTRY_ID,
                   parameters=signal_params(snap))
    engine = SystemOneClassifier(job, http=server.client)
    # Semantic recipes now propose candidates before the Laya endpoint.
    monkeypatch.setattr("call1.process.handlers.signal_stages.run_classifier",
                        lambda eng, rows, job, **kw: eng.choose(rows))
    result = run_categorize(job, engine, adapter_version="1", device="fake")
    assert server.calls and result.provenance.catalog_entry_id == ENTRY_ID
    assert result.rules is not None and result.rule_decisions
    assert all(d.check and not d.system_one_kept for d in result.rule_decisions)
    assert result.rule_decisions[0].system_one_score == .9
    assert [span.category_id for span in result.spans] == ["intent"]
    sub_job = make_job(tmp_path, JobType.CONTACT_SIGNALS_SUBCATEGORIZE,
                       {**inputs, "categories": (ArtifactKind.SIGNAL_CATEGORIES, result)},
                       catalog=catalog, entry_id="call1-bundled", parameters=signal_params(snap))
    assert isinstance(classifier_for(sub_job), GemmaSegmentClassifier)
    assert catalog.defaults[ModelPurpose.SIGNAL_SUBCATEGORY] != ENTRY_ID
    assert catalog.defaults[ModelPurpose.SIGNAL_EXTRACTION] != ENTRY_ID


@pytest.mark.parametrize("url", ["http://evil.example", "http://127.0.0.1:11434/v1", "http://user:secret@localhost:11434",
                                  "http://localhost:11434?token=x", "http://127.0.0.1:0", "ftp://localhost"])
def test_endpoint_config_is_loopback_only(url):
    with pytest.raises(ConfigError):
        ProcessConfig(system_one_url=url)


def test_environment_endpoint_and_tag_are_supported():
    config = ProcessConfig.from_mapping({}, env={"CALL1_SYSTEM_ONE_URL": "http://localhost:11435/",
                                                 "CALL1_SYSTEM_ONE_MODEL": "laya:421m-typed-decisions-mlx-fp16"})
    assert config.system_one_url == "http://localhost:11435"
    assert config.system_one_model == "laya:421m-typed-decisions-mlx-fp16"


def test_missing_model_and_old_ollama_are_not_available():
    for path, body in [("/api/version", {"version": "0.39.0"}), ("/api/tags", {"models": []})]:
        server = Server()
        original = server.handle
        with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=body) if req.url.path == path else original(req))) as client:
            with pytest.raises(SystemOneUnavailable):
                discover(URL, "laya", http=client)


def test_unavailable_optional_endpoint_is_cached_without_hiding_fresh_checks(monkeypatch):
    from call1.process import system_one

    system_one._CACHE.clear()
    seen = []
    client_type = httpx.Client
    def unavailable(request):
        seen.append(request)
        raise httpx.ConnectError("test", request=request)
    monkeypatch.setattr(system_one.httpx, "Client", lambda **kw: client_type(transport=httpx.MockTransport(unavailable)))
    for _ in range(3):
        with pytest.raises(SystemOneUnavailable):
            discover(URL, "laya")
    assert len(seen) == 1
    with pytest.raises(SystemOneUnavailable):
        discover(URL, "laya", fresh=True)
    assert len(seen) == 2
    system_one._CACHE.clear()


def cascade_case(setup, tmp_path, monkeypatch):
    from call1.process.handlers.signal_stages import SignalContext
    from call1.process.handlers.signals_rules import SemanticSystemOneClassifier
    _, server, catalog = setup
    seed = json.loads((Path(__file__).parents[2] / "call1/store/seeds/signals_retail_v1.json").read_text())
    seed["taxonomy"]["rules"]["bank"] = None
    snap = snapshot(SignalTaxonomy.model_validate(seed["taxonomy"]), SignalSettings(pipeline="v2", detection="rules"))
    transcript = script_transcript([(SpeakerRole.CALLER, "Can you hold this jacket?")])
    inputs = {"transcript": (ArtifactKind.TRANSCRIPT, transcript), "pii_findings": (ArtifactKind.PII_FINDINGS, findings_for(tmp_path, transcript)),
              "taxonomy": (ArtifactKind.SIGNAL_TAXONOMY_SNAPSHOT, snap)}
    job = make_job(tmp_path, JobType.CONTACT_SIGNALS_CATEGORIZE, inputs, catalog=catalog, entry_id=ENTRY_ID, parameters=signal_params(snap))
    raw = SystemOneClassifier(job, http=server.client)
    wrapper = SemanticSystemOneClassifier(job, SignalContext(job), raw, device="fake", categories=snap.taxonomy.categories)
    monkeypatch.setattr("call1.process.handlers.signals_rules.maybe_rules_engine", lambda *a, **kw: wrapper)
    content = run_categorize(job, raw, adapter_version="1", device="fake")
    return wrapper, content, inputs, snap, catalog, server


def test_confident_laya_needs_semantic_agreement_and_skips_only_confirmation(setup, tmp_path, monkeypatch):
    from call1.contracts.contents import SignalNeighbour
    from call1.process.handlers.fake import FakeSignalClassifier
    from call1.process.handlers.signal_stages import run_subcategorize
    wrapper, content, inputs, snap, catalog, _ = cascade_case(setup, tmp_path, monkeypatch)
    category = snap.taxonomy.category("intent")
    decision = content.rule_decisions[0].model_copy(update={"knn_share": .9, "subcategory_share": .9,
        "subcategory_id": category.subcategories[0].subcategory_id,
        "neighbours": [SignalNeighbour(entry_id="test:1", cosine=.9, carries_category=True)]})
    content = content.model_copy(update={"rule_decisions": [decision]})
    monkeypatch.setattr("call1.process.handlers.signals_rules.RulesClassifier.finish", lambda self, content: content)
    wrapper.scores = {(s.index, "intent"): .99 for s in content.segments}
    kept = wrapper.finish(content)
    assert kept.rule_decisions[0].system_one_kept and not kept.rule_decisions[0].check
    sub_job = make_job(tmp_path, JobType.CONTACT_SIGNALS_SUBCATEGORIZE,
                       {**inputs, "categories": (ArtifactKind.SIGNAL_CATEGORIES, kept)}, catalog=catalog, parameters=signal_params(snap))
    engine = FakeSignalClassifier(snap.taxonomy, stage=2)
    monkeypatch.setattr(engine, "load", lambda: pytest.fail("confident agreed category should bypass Gemma confirmation"))
    sub = run_subcategorize(sub_job, engine, adapter_version="1", device="fake")
    assert len(sub.decisions) == 1 and sub.provenance.rows == 0
    assert sub.decisions[0].subcategory_id == decision.subcategory_id
    # Strong Laya alone is insufficient: a weak example vote routes to Gemma.
    weak = content.model_copy(update={"rule_decisions": [decision.model_copy(update={"knn_share": .2})]})
    assert wrapper.finish(weak).rule_decisions[0].check


@pytest.mark.parametrize("failure", ["context_limit_exceeded", "provider_timeout", "validation_rejected"])
def test_failed_laya_preserves_candidate_for_gemma(setup, tmp_path, monkeypatch, failure):
    _, server, _ = setup
    monkeypatch.setattr(SystemOneClassifier, "choose", lambda self, rows: (_ for _ in ()).throw(EngineError(failure, "safe test error")))
    _, content, _, _, _, _ = cascade_case(setup, tmp_path, monkeypatch)
    assert [span.category_id for span in content.spans] == ["intent"]
    decision = content.rule_decisions[0]
    assert decision.check and not decision.system_one_kept and decision.system_one_score is None
    assert decision.system_one_fallback.value == failure


def test_real_catalog_migrates_standard_to_default_cascade(setup, monkeypatch):
    engine, _, _ = setup
    monkeypatch.setattr("call1.process.system_one.catalog_entry", lambda *args: engine.entry)
    catalog = seeded_catalog(mode="real", system_one_url=URL, overrides={"signal_category": "call1-bundled"})
    assert catalog.defaults[ModelPurpose.SIGNAL_CATEGORY] == ENTRY_ID
    assert catalog.defaults[ModelPurpose.SIGNAL_SUBCATEGORY] == "call1-bundled"
    assert catalog.defaults[ModelPurpose.SIGNAL_EXTRACTION] == "call1-bundled"


def test_category_without_recipe_still_gets_semantic_first_pass(setup, tmp_path, monkeypatch):
    from call1.process.handlers.signal_stages import SignalContext
    from call1.process.handlers.signals_rules import SemanticSystemOneClassifier, maybe_rules_engine
    from .test_signals_support import cancel_taxonomy
    _, server, catalog = setup
    snap = snapshot(cancel_taxonomy(), SignalSettings(pipeline="v2"))
    transcript = script_transcript([(SpeakerRole.CALLER, "I want to cancel this order.")])
    inputs = {"transcript": (ArtifactKind.TRANSCRIPT, transcript), "pii_findings": (ArtifactKind.PII_FINDINGS, findings_for(tmp_path, transcript)),
              "taxonomy": (ArtifactKind.SIGNAL_TAXONOMY_SNAPSHOT, snap)}
    job = make_job(tmp_path, JobType.CONTACT_SIGNALS_CATEGORIZE, inputs, catalog=catalog, entry_id=ENTRY_ID, parameters=signal_params(snap))
    raw = SystemOneClassifier(job, http=server.client)
    wrapped = maybe_rules_engine(job, SignalContext(job), raw, device="fake")
    assert isinstance(wrapped, SemanticSystemOneClassifier) and not wrapped.gemma_ids
    assert all(plan.recipe.engine == "rules" and plan.recipe.threshold == .5 for plan in wrapped.plans)
    result = run_categorize(job, raw, adapter_version="1", device="fake")
    assert result.rules is not None and result.rules.embedded_segments > 0
    assert all(decision.system_one_score is not None for decision in result.rule_decisions)
    assert all(category.recipe is None for category in snap.taxonomy.categories)  # snapshot not rewritten
