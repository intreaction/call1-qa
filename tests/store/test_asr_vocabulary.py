"""The ASR vocabulary for dual transcription in Store (contract 1.3.0, team decision 33;
docs/DualAsr.md sections 4, 8, 9 and 10): the admin document and its save rules, the pack install
and ``--demo``, the graph cap, completion with the ``asr`` job's optional outputs, and the masked
``TranscriptView.vocabulary_correction``.

No model runs here: Process is played by ``QueueHarness`` over HTTP, or by ``FakeQueue`` driving the
real projections, with scripted transcripts that already carry the merge's replacements.
"""

from __future__ import annotations

import io
import itertools
import json
import os
import sqlite3
import string
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from call1 import launch
from call1.contracts.artifacts import ArtifactKind
from call1.contracts.common import ContractParameters, ServiceScope
from call1.contracts.contents import (
    TranscriptContent,
    TranscriptReplacement,
    TranscriptTurnContent,
    VocabularyCorrection,
    VocabularyCorrectionStatus,
)
from call1.contracts.errors import JobErrorCode
from call1.contracts.jobs import JobType
from call1.contracts.vocabulary import (
    AsrVocabularyPack,
    AsrVocabularyParameters,
    AsrVocabularyTerm,
    VocabularyMergeRule,
    VocabularyTermSource,
    vocabulary_digest,
    vocabulary_term_problem,
)
from call1.redaction import find_pii
from call1.store import __main__ as store_cli
from call1.store import audit
from call1.store.config import StoreConfig
from call1.store.errors import StoreError
from call1.store.results import vocabulary as store_vocabulary

from .test_queue_harness import V, QueueHarness, content, hooks, job, pinned, q, rubrics, upstream_input  # noqa: F401  (fixtures)
from .test_results_harness import accounts, fq  # noqa: F401  (fixtures)

VOCAB = f"{V}/vocabulary"
SEED = Path(__file__).resolve().parents[2] / "call1" / "store" / "seeds" / "asr_vocabulary_retail_v1.json"
CAPS = ContractParameters(max_vocabulary_terms=5, max_vocabulary_pack_terms=100)
PACK = AsrVocabularyPack(pack_id="retail", version=1, industry="Retail", title="Test pack", terms=["Stanley cup", "Target", "Afterpay", "Wi-Fi"])
WHISPER = {"entry_id": "whisper-small-vocab", "entry_version": 1}


@pytest.fixture
def store_config(tmp_path) -> StoreConfig:
    """Small caps, so the cap refusals are cheap to reach: 5 customer terms, 100 pack terms."""
    return StoreConfig.for_tests(tmp_path / "store", parameters=CAPS)


@pytest.fixture
def real(client, store, clock, service_key, mint_service_key, mint_session) -> QueueHarness:
    """Process over HTTP with the real results projections (no hook recorders)."""
    return QueueHarness(client, store, clock, service_key, mint_service_key, mint_session)


def _settings(customer=(), disabled=(), enabled=True) -> Dict[str, Any]:
    return {"enabled": enabled, "customer_terms": list(customer), "disabled_pack_terms": list(disabled)}


def _put(client, session, settings: Dict[str, Any], version: int, *, expect: int = 200):
    response = client.put(VOCAB, json={"expected_record_version": version, "settings": settings}, headers=session.headers)
    assert response.status_code == expect, response.text
    return response.json() if expect == 200 else response


def _install(store, pack: AsrVocabularyPack = PACK):
    with store.connection() as conn:
        return store_vocabulary.install_pack(conn, pack, actor=audit.installer_actor("tester"), parameters=store.config.parameters)


def _audits(store) -> List[Dict[str, Any]]:
    with store.connection() as conn:
        rows = conn.execute("SELECT actor_kind, details_json FROM audit_events WHERE action = 'asr_vocabulary_saved' ORDER BY sequence").fetchall()
    return [{"actor_kind": r["actor_kind"], "details": json.loads(r["details_json"])} for r in rows]


def _events(store) -> List[Dict[str, Any]]:
    with store.connection() as conn:
        return [dict(r) for r in conn.execute("SELECT resource_id, version, status FROM change_events WHERE kind = 'asr_vocabulary' ORDER BY seq")]


def _terms(*terms: str, source: VocabularyTermSource = VocabularyTermSource.CUSTOMER) -> List[AsrVocabularyTerm]:
    return [AsrVocabularyTerm(term=t, source=source) for t in terms]


def _asr_vocabulary(terms: List[AsrVocabularyTerm]) -> Dict[str, Any]:
    return AsrVocabularyParameters(digest=vocabulary_digest(terms), terms=terms, candidate_entry=WHISPER).model_dump(mode="json")


# --- the document and its routes ------------------------------------------------------------------


def test_before_any_save_the_vocabulary_is_empty_and_inactive(client, admin_session, reviewer_session, supervisor_session,
                                                              service_key_headers, mint_service_key):
    record = client.get(VOCAB, headers=admin_session.read_headers)
    assert record.status_code == 200, record.text
    assert record.json() == {"record_version": 0, "settings": _settings(), "pack": None, "effective_terms": [], "effective_digest": None,
                             "active": False, "updated_at": None, "updated_by_account_id": None}
    # Process reads it (jobs:write) to freeze the terms into new asr jobs; reviewers and supervisors never need it.
    assert client.get(VOCAB, headers=service_key_headers).json() == record.json()
    assert client.get(VOCAB, headers=reviewer_session.read_headers).status_code == 403
    assert client.get(VOCAB, headers=supervisor_session.read_headers).status_code == 403
    assert client.get(VOCAB, headers=mint_service_key([ServiceScope.ARTIFACTS_READ]).headers).status_code == 403
    assert client.put(VOCAB, json={"expected_record_version": 0, "settings": _settings(["Target"])},
                      headers=supervisor_session.headers).status_code == 403
    assert client.put(VOCAB, json={"expected_record_version": 0, "settings": _settings(["Target"])},
                      headers=service_key_headers).status_code == 403


def test_save_round_trip_audits_and_announces_without_terms(client, store, admin_session, service_key_headers):
    saved = _put(client, admin_session, _settings(["Stanley cup", "Afterpay"]), 0)
    expected = _terms("Stanley cup", "Afterpay")
    assert saved["record_version"] == 1 and saved["active"] is True and saved["pack"] is None
    assert saved["effective_terms"] == [t.model_dump(mode="json") for t in expected]
    assert saved["effective_digest"] == vocabulary_digest(expected)
    assert saved["updated_by_account_id"] == admin_session.account_id and saved["updated_at"] is not None
    assert client.get(VOCAB, headers=service_key_headers).json() == saved

    [entry] = _audits(store)
    assert entry["actor_kind"] == "reviewer"
    assert entry["details"] | {} == {**entry["details"], "record_version": 1, "old_digest": None, "new_digest": saved["effective_digest"],
                                     "enabled": True, "active": True, "customer_terms": 2, "effective_customer_terms": 2,
                                     "effective_pack_terms": 0, "pack_id": None, "change": "saved"}
    assert "Stanley" not in json.dumps(entry) and "Afterpay" not in json.dumps(entry)
    assert _events(store) == [{"resource_id": "asr_vocabulary", "version": 1, "status": "saved"}]
    # Process's feed carries it, so Process refreshes the vocabulary it freezes into new asr jobs.
    feed = client.get(f"{V}/changes", headers=service_key_headers).json()["events"]
    assert [(e["resource_id"], e["version"], e["status"]) for e in feed if e["kind"] == "asr_vocabulary"] == [("asr_vocabulary", 1, "saved")]

    # A save that changes nothing returns the record unchanged: no new version, audit or event.
    assert _put(client, admin_session, _settings(["Stanley cup", "Afterpay"]), 1) == saved
    assert len(_audits(store)) == 1 and len(_events(store)) == 1

    # Switched off: the terms stay listed, dual transcription stops for new graphs.
    off = _put(client, admin_session, _settings(["Stanley cup", "Afterpay"], enabled=False), 1)
    assert off["record_version"] == 2 and off["active"] is False and off["effective_terms"] == saved["effective_terms"]
    assert _audits(store)[-1]["details"]["enabled"] is False
    # On but empty is not active either.
    empty = _put(client, admin_session, _settings(), 2)
    assert empty["active"] is False and empty["effective_terms"] == [] and empty["effective_digest"] is None
    assert _audits(store)[-1]["details"]["old_digest"] == saved["effective_digest"]


def test_a_stale_record_version_is_409_with_the_current_version(client, admin_session):
    _put(client, admin_session, _settings(["Target"]), 0)
    stale = _put(client, admin_session, _settings(["Afterpay"]), 0, expect=409).json()
    assert stale["code"] == "conflict" and stale["details"]["current_version"] == 1
    assert client.get(VOCAB, headers=admin_session.read_headers).json()["settings"]["customer_terms"] == ["Target"]


@pytest.mark.parametrize("settings, field, reason, index, secret", [
    (_settings(["Stanley cup", "Galaxy S24"]), "settings.customer_terms", "digit", 1, "S24"),
    (_settings(["Target", "Two", "Line ٣"]), "settings.customer_terms", "digit", 2, "٣"),
    (_settings(["Q" * 61]), "settings.customer_terms", "too_long", 0, "Q" * 61),
    (_settings(["alpha beta gamma delta epsilon zeta eta"]), "settings.customer_terms", "too_many_words", 0, "epsilon"),
    (_settings(["Shop@Home"]), "settings.customer_terms", "character", 0, "Shop@Home"),
    (_settings(["-Target"]), "settings.customer_terms", "must_start_with_letter", 0, "-Target"),
    (_settings([" Target"]), "settings.customer_terms", "not_normalized", 0, " Target"),
    (_settings(["X"]), "settings.customer_terms", "too_few_letters", 0, None),
    (_settings(["Stanley cup", "stanley-cup"]), "settings.customer_terms", "duplicate_term", None, "stanley-cup"),
    (_settings(["pin one two three four"]), "settings.customer_terms", "pii_detected", 0, "one two three four"),
    (_settings(["Target", "Afterpay", "Zelle", "Klarna", "Venmo", "PayPal"]), "settings.customer_terms", "too_many_terms", None, "PayPal"),
    (_settings([], ["Nike"]), "settings.disabled_pack_terms", "unknown_pack_term", 0, "Nike"),
])
def test_refusals_name_the_field_and_reason_never_the_term(client, admin_session, settings, field, reason, index, secret):
    refused = _put(client, admin_session, settings, 0, expect=422)
    body = refused.json()
    assert body["code"] == "validation_failed"
    assert body["details"]["field"] == field and body["details"]["reason"] == reason
    if index is not None:
        assert body["details"]["index"] == index
    if secret is not None:
        assert secret not in refused.text
    assert client.get(VOCAB, headers=admin_session.read_headers).json()["record_version"] == 0


def test_pack_terms_come_first_can_be_switched_off_and_customer_duplicates_add_nothing(client, store, admin_session):
    installed, changed = _install(store)
    assert changed and installed.record_version == 1 and installed.active
    record = client.get(VOCAB, headers=admin_session.read_headers).json()
    assert record["pack"] == PACK.model_dump(mode="json")
    assert record["effective_terms"] == [t.model_dump(mode="json") for t in _terms(*PACK.terms, source=VocabularyTermSource.INDUSTRY_PACK)]
    saved = _put(client, admin_session, _settings(["Wi Fi", "Zelle"], ["target"]), 1)
    assert [(t["term"], t["source"]) for t in saved["effective_terms"]] == [
        ("Stanley cup", "industry_pack"), ("Afterpay", "industry_pack"), ("Wi-Fi", "industry_pack"), ("Zelle", "customer")]
    assert saved["settings"]["customer_terms"] == ["Wi Fi", "Zelle"]  # kept as the admin wrote them
    details = _audits(store)[-1]["details"]
    assert (details["pack_id"], details["pack_version"], details["pack_terms"], details["disabled_pack_terms"]) == ("retail", 1, 4, 1)
    assert (details["effective_pack_terms"], details["effective_customer_terms"]) == (3, 1)
    unknown = _put(client, admin_session, _settings([], ["Nike"]), 2, expect=422).json()
    assert unknown["details"] == {**unknown["details"], "field": "settings.disabled_pack_terms", "reason": "unknown_pack_term", "index": 0}


# --- pack install, the retail seed and --demo ------------------------------------------------------


def test_the_retail_seed_passes_the_install_rules():
    pack = AsrVocabularyPack.model_validate(json.loads(SEED.read_text(encoding="utf-8")))
    assert (pack.pack_id, pack.version, len(pack.terms)) == ("retail", 1, 73)
    store_vocabulary.check_pack(pack, ContractParameters())
    assert all(vocabulary_term_problem(t) is None and not find_pii(t) for t in pack.terms)
    # Section 10: fictional or call-specific shop names and people's names were removed.
    for removed in ("Suit Haberdashery", "Lucinda's Boutique", "Harry Potter", "Coldplay", "Michael's", "Jack Frost"):
        assert removed not in pack.terms
    assert "Stanley cup" in pack.terms


def test_a_pack_install_checks_its_caps_and_drops_disabled_terms_it_no_longer_has(client, store, admin_session):
    with pytest.raises(StoreError) as refused:
        store_vocabulary.check_pack(PACK, ContractParameters(max_vocabulary_pack_terms=2))
    assert refused.value.details["reason"] == "too_many_terms" and refused.value.details["limit"] == 2
    with pytest.raises(StoreError) as refused:
        _install(store, PACK.model_copy(update={"terms": ["Target", "security code one two three"]}))
    assert refused.value.details == {**refused.value.details, "field": "terms", "reason": "pii_detected", "index": 1}
    assert "one two three" not in json.dumps(refused.value.details)

    _install(store)
    _put(client, admin_session, _settings(["Zelle"], ["Stanley cup", "Target"]), 1)
    assert _install(store)[1] is False  # the same pack again: a no-op
    newer = PACK.model_copy(update={"version": 2, "terms": ["Target", "Afterpay", "Klarna"]})
    record, changed = _install(store, newer)
    assert changed and record.record_version == 3 and record.pack.version == 2
    assert record.settings.disabled_pack_terms == ["Target"] and record.settings.customer_terms == ["Zelle"]
    assert [t.term for t in record.effective_terms] == ["Afterpay", "Klarna", "Zelle"]
    installs = [a for a in _audits(store) if a["details"]["change"] == "pack_installed"]
    assert [a["actor_kind"] for a in installs] == ["installer", "installer"]
    assert "Klarna" not in json.dumps(installs)
    assert [e["status"] for e in _events(store)] == ["pack_installed", "saved", "pack_installed"]


def test_apply_vocabulary_seed_is_audited_and_idempotent(tmp_path, monkeypatch, capsys):
    data = tmp_path / "cli-store"
    monkeypatch.setenv("CALL1_STORE_DATA", str(data))
    monkeypatch.setenv("CALL1_STORE_MAINTENANCE_SECONDS", "0")
    assert store_cli.main(["apply-vocabulary-seed", str(SEED)]) == 0
    assert "retail v1 installed (73 terms" in capsys.readouterr().err
    assert store_cli.main(["apply-vocabulary-seed", str(SEED)]) == 0
    assert "already installed" in capsys.readouterr().err
    from call1.store.db import Database

    with Database(data / "store.db").connection() as conn:
        record = store_vocabulary.record(conn)
        assert record.record_version == 1 and record.active and len(record.effective_terms) == 73
        assert {t.source for t in record.effective_terms} == {VocabularyTermSource.INDUSTRY_PACK}
        rows = conn.execute("SELECT action, actor_kind FROM audit_events ORDER BY sequence").fetchall()
        assert [(r["action"], r["actor_kind"]) for r in rows] == [("asr_vocabulary_saved", "installer")]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**json.loads(SEED.read_text()), "terms": ["Target", "Galaxy S24"]}))
    assert store_cli.main(["apply-vocabulary-seed", str(bad)]) == 2
    assert "S24" not in capsys.readouterr().err


def test_demo_installs_the_retail_vocabulary_at_every_start(tmp_path):
    """Section 10: ``--demo`` installs the retail pack before any call is ingested; the install is
    idempotent, so every start runs it and on-stage edits survive. A pack that cannot install is reported."""
    root = tmp_path / "demo"
    root.mkdir()
    env = dict(os.environ, CALL1_STORE_DATA=str(root / "store"), CALL1_PROCESS_CONFIG=str(root / "process" / "config.json"),
               CALL1_STORE_MAINTENANCE_SECONDS="0")
    out = io.StringIO()
    launcher = launch.Launcher(python=sys.executable, env=env, log_path=tmp_path / "launch.log", out=out)
    assert launcher.run_step("store", ["-m", "call1.store", "migrate"]) == 0

    def state():
        with sqlite3.connect(f"file:{root / 'store' / 'store.db'}?mode=ro", uri=True) as conn:
            version, pack = conn.execute("SELECT record_version, pack_json FROM results_asr_vocabulary WHERE id = 1").fetchone()
        return version, json.loads(pack)["pack_id"] if pack else None

    assert launch.DEMO_VOCABULARY_SEED == SEED.resolve()
    assert launch.seed_demo_vocabulary(launcher)
    assert state() == (1, "retail")
    assert launch.seed_demo_vocabulary(launcher)  # a restart: the same pack is a no-op
    assert state() == (1, "retail") and "already installed" in out.getvalue()
    missing = tmp_path / "missing.json"
    assert not launch.seed_demo_vocabulary(launcher, missing)
    assert "was not installed" in out.getvalue() and state() == (1, "retail")


# --- graph creation and completion (the queue area) ----------------------------------------------


def test_an_asr_job_carries_at_most_the_capped_number_of_terms(q):
    conversation = q.register()
    audio = q.upload_audio(conversation["id"])
    cap = CAPS.max_vocabulary_terms + CAPS.max_vocabulary_pack_terms
    names = [f"Brand {a}{b}" for a, b in itertools.product(string.ascii_lowercase, repeat=2)][:cap + 1]
    over = _asr_vocabulary(_terms(*names))
    refused = q.graph(conversation["id"], [job("asr", JobType.ASR, inputs=[pinned("audio", audio)], parameters={"asr_vocabulary": over})],
                      expect=422).json()
    assert refused["code"] == "graph_invalid" and refused["details"]["reason"] == "asr_vocabulary_cap"
    assert refused["details"]["limit"] == cap and "Brand" not in json.dumps(refused)
    # The digest need not be the current vocabulary's (the record is empty here): a graph reproduces its plan.
    fine = _asr_vocabulary(_terms(*names[:cap]))
    created = q.graph(conversation["id"], [job("asr", JobType.ASR, inputs=[pinned("audio", audio)], parameters={"asr_vocabulary": fine})])
    frozen = q.job(created["jobs"][0]["job_id"])["parameters"]["asr_vocabulary"]
    assert frozen["digest"] == fine["digest"] and len(frozen["terms"]) == cap


def _asr_claim(q, *, vocabulary: bool = True):
    conversation = q.register()
    audio = q.upload_audio(conversation["id"])
    params = {"asr_vocabulary": _asr_vocabulary(_terms("Stanley cup"))} if vocabulary else {}
    graph = q.graph(conversation["id"], [
        job("asr", JobType.ASR, inputs=[pinned("audio", audio)], parameters=params),
        job("enrich", JobType.ENRICHMENT, requires=["asr"], inputs=[upstream_input("transcript", "asr", "transcript")]),
    ])
    return conversation, q.ids(graph), q.claim_one(q.ids(graph)["asr"])


def _optional(q, claimed, role: str, kind: ArtifactKind) -> Dict[str, Any]:
    art = q.inline(claimed["job"]["conversation_id"], kind, content(kind), job_id=claimed["job"]["id"], token=claimed["claim_token"]).json()
    return {"role": role, "artifact_id": art["id"], "checksum": art["checksum"]}


def test_asr_completion_links_its_optional_outputs_at_most_once(q, hooks):
    conversation, ids, claimed = _asr_claim(q)
    required = q.outputs_for(claimed)
    base = _optional(q, claimed, "base_transcript", ArtifactKind.ASR_BASE_TRANSCRIPT)
    candidates = _optional(q, claimed, "vocabulary_pass", ArtifactKind.ASR_VOCABULARY_PASS)

    unknown = q.complete(claimed, outputs=required + [{**base, "role": "whisper_transcript"}], expect=422).json()
    assert unknown["code"] == "graph_invalid" and unknown["details"]["reason"] == "outputs_mismatch"
    assert unknown["details"]["optional_roles"] == "base_transcript,vocabulary_pass"
    twice = q.complete(claimed, outputs=required + [base, base], expect=422).json()  # the contract model refuses a repeated role
    assert twice["code"] == "validation_failed" and "one output per role" in twice["details"]["reason"]
    missing = q.complete(claimed, outputs=[base, candidates], expect=422).json()
    assert missing["details"]["reason"] == "outputs_mismatch"
    wrong_kind = q.complete(claimed, outputs=required + [{**candidates, "role": "base_transcript"}], expect=422).json()
    assert wrong_kind["details"]["reason"] == "output_kind"

    receipt = q.complete(claimed, outputs=required + [base, candidates])
    assert len(receipt["linked_artifact_ids"]) == 3 and receipt["released_job_ids"] == [ids["enrich"]]
    done = q.job(ids["asr"])
    assert [o["role"] for o in done["outputs"]] == ["transcript", "base_transcript", "vocabulary_pass"]
    assert [o.role for o in hooks.completions[-1].outputs] == ["transcript", "base_transcript", "vocabulary_pass"]


def test_optional_outputs_are_optional(q):
    _conv, ids, claimed = _asr_claim(q)  # base_only: the base transcript, no vocabulary pass
    q.complete(claimed, outputs=q.outputs_for(claimed) + [_optional(q, claimed, "base_transcript", ArtifactKind.ASR_BASE_TRANSCRIPT)])
    assert [o["role"] for o in q.job(ids["asr"])["outputs"]] == ["transcript", "base_transcript"]
    _conv, ids, claimed = _asr_claim(q, vocabulary=False)  # Parakeet only, exactly as before 1.3.0
    q.complete(claimed)
    assert [o["role"] for o in q.job(ids["asr"])["outputs"]] == ["transcript"]


def test_only_the_asr_job_writes_the_raw_vocabulary_kinds(q):
    conversation, ids, claimed = _asr_claim(q)
    q.complete(claimed)
    enrich = q.claim_one(ids["enrich"])
    refused = q.inline(conversation["id"], ArtifactKind.ASR_BASE_TRANSCRIPT, job_id=enrich["job"]["id"], token=enrich["claim_token"],
                       expect=422).json()
    assert refused["details"]["reason"] == "kind_not_an_output"
    # An optional role is never an upstream input of another job.
    reads_base = q.graph(conversation["id"], [
        job("asr2", JobType.ASR, inputs=[pinned("audio", q.upload_audio(conversation["id"]))], key="asr-2"),
        job("enrich2", JobType.ENRICHMENT, requires=["asr2"], inputs=[upstream_input("transcript", "asr2", "base_transcript")], key="enrich-2"),
    ], expect=422)
    assert reads_base.json()["code"] in ("graph_invalid", "validation_failed")


# --- the masked view (results) ---------------------------------------------------------------------

AGENT_LINE = "Thanks for calling, this is Sam."
CALLER_LINE = "My name is Maria Lopez and I bought a Stanley cup at Target."
PAYMENT_LINE = "I paid with Afterpay today."
NAME = "Maria Lopez"


def _words(text: str, start: float) -> List[Dict[str, Any]]:
    out, t = [], start
    for w in text.split():
        out.append({"word": " " + w, "start_time": round(t, 2), "end_time": round(t + 0.1, 2), "probability": 0.9})
        t += 0.12
    return out


def _replacement(turn_id: int, text: str, words: List[Dict[str, Any]], term: str, heard: str) -> TranscriptReplacement:
    char_start = text.index(term)
    word_start = len(text[:char_start].split())
    word_end = word_start + len(term.split())
    return TranscriptReplacement(
        turn_id=turn_id, word_start=word_start, word_end=word_end, char_start=char_start, char_end=char_start + len(term), term=term,
        source=VocabularyTermSource.INDUSTRY_PACK, heard=heard, candidate_text=f"whisper said {term}",
        start_time=words[word_start]["start_time"], end_time=words[word_end - 1]["end_time"],
        candidate_start_time=words[word_start]["start_time"], candidate_end_time=words[word_end - 1]["end_time"],
        phonetic_similarity=0.8, character_similarity=0.7)


def _merged(correction: Optional[str] = "applied") -> TranscriptContent:
    lines = [("AGENT", AGENT_LINE), ("CALLER", CALLER_LINE), ("CALLER", PAYMENT_LINE)]
    turns, t = [], 0.0
    for i, (speaker, text) in enumerate(lines):
        turns.append(TranscriptTurnContent(turn_id=i, speaker=speaker, start_time=t, end_time=t + 2.0, text=text, word_timestamps=_words(text, t)))
        t += 2.0
    vocabulary = None
    if correction == "applied":
        w1, w2 = [w.model_dump(mode="json") for w in turns[1].word_timestamps], [w.model_dump(mode="json") for w in turns[2].word_timestamps]
        vocabulary = VocabularyCorrection(
            status=VocabularyCorrectionStatus.APPLIED, vocabulary_digest="sha256:" + "a" * 64, term_count=4, glossary_term_count=4,
            base_engine="parakeet-tdt-0.6b-v3", candidate_engine="whisper-small", rule=VocabularyMergeRule(), candidates=5,
            replacements=[
                _replacement(1, CALLER_LINE, w1, NAME, "Mario Lopez"),     # the term itself is masked: withheld
                _replacement(1, CALLER_LINE, w1, "Stanley cup", "standing cup"),  # shown, offsets shifted by the mask before it
                _replacement(1, CALLER_LINE, w1, "Target", NAME),          # heard is masked by value: null
                _replacement(2, PAYMENT_LINE, w2, "Afterpay", "after pay 4"),  # heard holds a digit: null
            ])
    elif correction == "base_only":
        vocabulary = VocabularyCorrection(
            status=VocabularyCorrectionStatus.BASE_ONLY, note="The vocabulary model is not installed; provision it to correct transcripts.",
            failure_code=JobErrorCode.MODEL_UNAVAILABLE, vocabulary_digest="sha256:" + "a" * 64, term_count=4,
            base_engine="parakeet-tdt-0.6b-v3", rule=VocabularyMergeRule())
    return TranscriptContent(duration_seconds=t, language="en", is_redacted=False, turns=turns, vocabulary_correction=vocabulary)


def _base() -> TranscriptContent:
    merged = _merged(None)
    turns = [t.model_copy(update={"text": t.text.replace("Stanley cup", "standing cup")}) if t.turn_id == 1 else t for t in merged.turns]
    return merged.model_copy(update={"turns": turns})


def _call(fq, correction: Optional[str] = "applied", *, findings: bool = True):
    fq.auto_pii = False
    conv = fq.register()
    graph = fq.graph(conv, [JobType.ASR, JobType.ENRICHMENT])
    outputs: Dict[str, object] = {"transcript": _merged(correction)}
    if correction is not None:
        outputs["base_transcript"] = _base()
    fq.complete(conv, graph, JobType.ASR, outputs)
    if findings:
        published = max((a for a in conv.artifacts if a.kind is ArtifactKind.TRANSCRIPT), key=lambda a: a.version)
        fq.link_pii(conv, published, {1: [("private_person", NAME, CALLER_LINE.index(NAME))]})
    return conv


def _view(client, reviewer_session, conv) -> Dict[str, Any]:
    response = client.get(f"{V}/calls/{conv.call_id}/transcript", headers=reviewer_session.read_headers)
    assert response.status_code == 200, response.text
    return response.json()


def test_the_view_shows_corrections_masked_like_the_transcript(fq, client, reviewer_session):
    view = _view(client, reviewer_session, _call(fq))
    caller, payment = view["turns"][1]["text"], view["turns"][2]["text"]
    assert NAME not in caller and "[REDACTED]" in caller and "Stanley cup" in caller
    correction = view["vocabulary_correction"]
    assert (correction["status"], correction["note"], correction["replacement_count"], correction["withheld_count"]) == ("applied", None, 4, 1)
    stanley, target, afterpay = correction["replacements"]
    assert (stanley["term"], stanley["heard"], stanley["source"]) == ("Stanley cup", "standing cup", "industry_pack")
    assert caller[stanley["char_start"]:stanley["char_end"]] == "Stanley cup"
    assert stanley["char_start"] == CALLER_LINE.index("Stanley cup") + len("[REDACTED]") - len(NAME)
    assert (stanley["word_start"], stanley["word_end"]) == (9, 11)
    assert [w["word"].strip() for w in view["turns"][1]["word_timestamps"][9:11]] == ["Stanley", "cup"]
    assert (target["term"], target["heard"]) == ("Target", None) and caller[target["char_start"]:target["char_end"]] == "Target"
    assert (afterpay["heard"], payment[afterpay["char_start"]:afterpay["char_end"]]) == (None, "Afterpay")
    assert afterpay["start_time"] == view["turns"][2]["word_timestamps"][3]["start_time"]
    # Nothing raw leaves Store: not the masked term's original words, not Whisper's text, not the digit.
    raw = json.dumps(view)
    assert "Mario" not in raw and NAME not in raw and "candidate_text" not in raw and "whisper said" not in raw and "after pay 4" not in raw


def test_withheld_text_withholds_every_correction(fq, client, reviewer_session):
    view = _view(client, reviewer_session, _call(fq, findings=False))
    assert view["text_withheld"] is True
    assert view["vocabulary_correction"] == {"status": "applied", "note": None, "replacement_count": 4, "withheld_count": 4, "replacements": []}
    assert "standing" not in json.dumps(view)


def test_base_only_and_plain_transcripts(fq, client, reviewer_session):
    base_only = _view(client, reviewer_session, _call(fq, "base_only"))["vocabulary_correction"]
    assert base_only == {"status": "base_only", "note": "The vocabulary model is not installed; provision it to correct transcripts.",
                         "replacement_count": 0, "withheld_count": 0, "replacements": []}
    assert _view(client, reviewer_session, _call(fq, None))["vocabulary_correction"] is None


def test_dual_transcription_end_to_end_through_the_queue(real, client, reviewer_session):
    """Process's view: an asr job frozen with the vocabulary completes with the merged transcript,
    the base transcript and the vocabulary pass; enrichment masks the merged text; the reviewer sees
    the correction, never the raw artifacts."""
    conversation = real.register()
    audio = real.upload_audio(conversation["id"])
    graph = real.graph(conversation["id"], [
        job("asr", JobType.ASR, inputs=[pinned("audio", audio)], parameters={"asr_vocabulary": _asr_vocabulary(_terms("Stanley cup"))}),
        job("enrich", JobType.ENRICHMENT, requires=["asr"], inputs=[upstream_input("transcript", "asr", "transcript")]),
    ])
    ids = real.ids(graph)
    claimed = real.claim_one(ids["asr"])
    merged = _merged().model_dump(mode="json")
    merged["vocabulary_correction"]["replacements"] = [r for r in merged["vocabulary_correction"]["replacements"] if r["term"] == "Stanley cup"]
    outputs = real.outputs_for(claimed, payloads={"transcript": merged})
    base = real.inline(conversation["id"], ArtifactKind.ASR_BASE_TRANSCRIPT, _base().model_dump(mode="json"), job_id=claimed["job"]["id"],
                       token=claimed["claim_token"]).json()
    real.complete(claimed, outputs=outputs + [{"role": "base_transcript", "artifact_id": base["id"], "checksum": base["checksum"]}])
    enrich = real.claim_one(ids["enrich"])
    transcript_input = next(i["artifact"] for i in enrich["inputs"] if i["role"] == "transcript")
    assert transcript_input["kind"] == "transcript"  # enrichment (and so every masked job) reads the merged text
    real.complete(enrich)
    view = _view(client, reviewer_session, type("Conv", (), {"call_id": conversation["call_id"]}))
    assert view["text_withheld"] is False and "Stanley cup" in view["turns"][1]["text"]
    assert "standing cup" not in json.dumps(view["turns"])
    [shown] = view["vocabulary_correction"]["replacements"]
    assert shown["heard"] == "standing cup" and view["turns"][1]["text"][shown["char_start"]:shown["char_end"]] == "Stanley cup"
