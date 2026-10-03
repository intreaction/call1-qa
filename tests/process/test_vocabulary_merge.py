"""The dual transcription merge (``call1/pipeline/vocabulary_merge.py``, docs/DualAsr.md sections 5 and
9): the port of the research unit tests (``scripts/research/asr_merge/test_merge.py``), the research's
TEST decisions on cached word lists, Double Metaphone parity with the Metaphone package, and the product
merge over contract transcripts (text and word rebuild, turns, channels, never inserting).

Pure code: no model, no Store.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pytest

from call1.contracts.contents import (
    AsrPassWord,
    SpeakerRole,
    TranscriptContent,
    TranscriptTurnContent,
    VocabularyCorrection,
    VocabularyCorrectionStatus,
    WordTimestampView,
)
from call1.contracts.vocabulary import AsrVocabularyTerm, VocabularyMergeRule, VocabularyTermSource
from call1.pipeline._vendor.doublemetaphone import doublemetaphone
from call1.pipeline.vocabulary_merge import (
    Candidate,
    apply_replacements,
    generate_candidates,
    locate_words,
    merge_transcript,
    normalize_text,
    pack_glossary,
    phonetic_similarity,
    rank_terms,
    rule_accepts,
    term_relevance,
    term_tokens,
    vocab_hits,
)

REPO = Path(__file__).resolve().parents[2]
FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "dual_asr_test_calls.json").read_text())
RESEARCH_VOCAB = [t["term"] for t in json.loads((Path(__file__).parent / "fixtures" / "vocabulary.json").read_text())["terms"]]
RETAIL_SEED = json.loads((REPO / "call1" / "store" / "seeds" / "asr_vocabulary_retail_v1.json").read_text())["terms"]


def w(word, start, end, **kw):
    return {"word": word, "start": start, "end": end, **kw}


VOCAB = ["Ticketek", "Ticketek VIP", "Wi-Fi", "Chadstone", "Liffey Valley"]


# --- the research unit tests, ported ------------------------------------------------------------


def test_normalize_and_term_tokens():
    assert normalize_text("(um,) I'm at f~ Chadstone, $85!") == ["im", "at", "chadstone", "eighty", "five", "dollars"]
    assert term_tokens("Wi-Fi") == ["wi", "fi"] and term_tokens("Stouffer's") == ["stouffers"]


def test_vocab_hits_prefers_longest_term():
    tokens = normalize_text("my Ticketek VIP pass and Ticketek")
    assert [(s, e, VOCAB[k]) for s, e, k in vocab_hits(tokens, VOCAB)] == [(1, 3, "Ticketek VIP"), (5, 6, "Ticketek")]
    assert [VOCAB[k] for _, _, k in vocab_hits(normalize_text("the wifi is down"), VOCAB)] == ["Wi-Fi"]


def test_phonetic_similarity_across_word_boundaries():
    assert phonetic_similarity(["chadstone"], ["chad", "stone"]) == 1.0
    assert phonetic_similarity(["liffey", "valley"], ["lifey", "valley"]) > 0.9
    assert phonetic_similarity(["ticketek"], ["banana"]) < 0.5


def test_candidate_replaces_misheard_span_keeping_timings():
    parakeet = [w("I", 0.0, 0.2), w("was", 0.2, 0.4), w("at", 0.4, 0.5), w("Chad", 0.6, 0.9), w("Stone", 0.9, 1.3), w("today", 1.4, 1.8)]
    whisper = [w("I", 0.0, 0.2), w("was", 0.2, 0.4), w("at", 0.4, 0.5), w("Chadstone", 0.62, 1.25), w("today", 1.4, 1.8)]
    cands = generate_candidates(parakeet, whisper, VOCAB)
    assert len(cands) == 1
    c = cands[0]
    assert (c.p_first, c.p_last, c.p_text) == (3, 5, "Chad Stone")
    assert not c.already_correct and rule_accepts(c)
    words, records = apply_replacements(parakeet, cands, rule_accepts)
    assert [x["word"] for x in words] == ["I", "was", "at", "Chadstone", "today"]
    assert (words[3]["start"], words[3]["end"]) == (0.6, 1.3)
    assert records[0]["parakeet"] == "Chad Stone"


def test_never_inserts_without_overlapping_parakeet_words():
    parakeet = [w("hello", 0.0, 0.5)]
    whisper = [w("hello", 0.0, 0.5), w("Ticketek", 3.0, 3.6)]
    cands = generate_candidates(parakeet, whisper, VOCAB)
    assert cands[0].no_overlap
    words, records = apply_replacements(parakeet, cands, lambda c: True)
    assert [x["word"] for x in words] == ["hello"] and records == []


def test_rule_rejects_dissimilar_span_and_already_correct_is_noop():
    parakeet = [w("the", 0.0, 0.2), w("banana", 0.3, 0.8), w("Wi-Fi", 1.0, 1.4)]
    whisper = [w("the", 0.0, 0.2), w("Ticketek", 0.3, 0.8), w("WiFi", 1.0, 1.4)]
    cands = generate_candidates(parakeet, whisper, VOCAB)
    assert [c.term for c in cands] == ["Ticketek", "Wi-Fi"]
    assert not rule_accepts(cands[0])
    assert cands[1].already_correct and not rule_accepts(cands[1])
    words, _ = apply_replacements(parakeet, cands, rule_accepts)
    assert [x["word"] for x in words] == ["the", "banana", "Wi-Fi"]


def test_overlapping_accepted_spans_keep_the_earlier_listed_hit():
    """The research kept the higher score; under the rule every score is 0, so the earlier-listed hit wins."""
    parakeet = [w("liffy", 0.0, 0.4), w("valley", 0.4, 0.9)]
    a = Candidate("Liffey Valley", 4, 0.0, 0.9, "", p_first=0, p_last=2, p_text="liffy valley", p_start=0.0, p_end=0.9)
    b = Candidate("Chadstone", 3, 0.0, 0.4, "", p_first=0, p_last=1, p_text="liffy", p_start=0.0, p_end=0.4)
    words, records = apply_replacements(parakeet, [a, b], lambda c: True)
    assert [x["word"] for x in words] == ["Liffey", "Valley"] and len(records) == 1
    words, records = apply_replacements(parakeet, [b, a], lambda c: True)
    assert [x["word"] for x in words] == ["Chadstone", "valley"] and records[0]["term"] == "Chadstone"


def test_the_rule_reads_its_thresholds_from_the_frozen_rule():
    parakeet = [w("standing", 0.0, 0.5), w("cup.", 0.5, 0.9)]
    whisper = [w("Stanley", 0.0, 0.5), w("cup.", 0.5, 0.9)]
    cand = generate_candidates(parakeet, whisper, ["Stanley cup"])[0]
    assert cand.features["phonetic_sim"] == pytest.approx(0.625)
    assert not rule_accepts(cand)
    assert rule_accepts(cand, VocabularyMergeRule(phonetic_min=0.6))


# --- the research's TEST decisions on committed word lists ----------------------------------------


def test_double_metaphone_matches_the_metaphone_package_on_every_vocabulary_token():
    codes = FIXTURE["double_metaphone"]
    tokens = {t for term in RESEARCH_VOCAB + RETAIL_SEED for t in term_tokens(term)}
    assert tokens <= set(codes)
    assert {token: list(doublemetaphone(token)) for token in codes} == codes


@pytest.mark.parametrize("call_id", sorted(FIXTURE["calls"]))
def test_candidates_reproduce_the_research_decisions(call_id):
    call = FIXTURE["calls"][call_id]
    candidates = generate_candidates(call["parakeet"], call["whisper_prompted"], RESEARCH_VOCAB)
    got = [{"term": c.term, "parakeet": c.p_text, "whisper": c.w_text, "no_overlap": c.no_overlap, "already_correct": c.already_correct,
            "rule": rule_accepts(c), "phonetic_sim": round(c.features.get("phonetic_sim", 0), 4), "char_sim": round(c.features.get("char_sim", 0), 4)}
           for c in candidates]
    assert got == FIXTURE["decisions"][call_id]


def test_the_rule_merge_on_test_makes_six_replacements_and_rejects_standing_cup():
    accepted, rejected = [], []
    hits = 0
    for call in FIXTURE["calls"].values():
        candidates = generate_candidates(call["parakeet"], call["whisper_prompted"], RESEARCH_VOCAB)
        hits += len(candidates)
        for c in candidates:
            if c.no_overlap or c.already_correct:
                continue
            (accepted if rule_accepts(c) else rejected).append(c)
    assert hits == 19
    assert sorted((c.term, c.p_text) for c in accepted) == sorted([
        ("Stanley cup", "standy cup."), ("Stanley cup", "Standy Cup."), ("Chadstone", "chatstone"), ("Chadstone", "chatstone"),
        ("Chadstone", "Chatstone"), ("Chadstone", "chatston")])
    assert [(c.term, c.p_text) for c in rejected] == [("Stanley cup", "standing cup.")]
    assert round(rejected[0].features["phonetic_sim"], 2) == 0.62


def test_glossary_ranking_is_the_research_score_and_packs_within_the_limit():
    call = FIXTURE["calls"]["en_IE_Retail_1592989"]
    ranked = rank_terms(RESEARCH_VOCAB, call["parakeet"])
    scores = {term: score for term, score in ranked}
    for term in ("Stanley cup", "Chadstone", "Wi-Fi", "click and collect"):
        assert scores[term] == pytest.approx(term_relevance(term, call["parakeet"]))
    assert ranked[0][0] == "Stanley cup"
    # the prefiltered ranking (large catalogues) still finds the obvious term
    assert rank_terms(RESEARCH_VOCAB, call["parakeet"], exact_limit=0)[0][0] == "Stanley cup"
    text, chosen = pack_glossary([t for t, _ in ranked], lambda s: len(s.split()), 20)
    assert text.startswith("Glossary: Stanley cup") and len((" " + text).split()) <= 20 and chosen[0] == "Stanley cup"


# --- the product merge over contract transcripts --------------------------------------------------


def _turn(turn_id: int, text: str, start: float, words: Optional[List[tuple]] = None, channel: Optional[int] = None,
          speaker: SpeakerRole = SpeakerRole.UNKNOWN) -> TranscriptTurnContent:
    if words is None:
        tokens = text.split()
        words = [(tok, start + i * 0.4, start + i * 0.4 + 0.35) for i, tok in enumerate(tokens)]
    timed = [WordTimestampView(word=wd, start_time=s, end_time=e, probability=0.8) for wd, s, e in words]
    return TranscriptTurnContent(turn_id=turn_id, speaker=speaker, start_time=words[0][1] if words else start,
                                 end_time=words[-1][2] if words else start, text=text, channel=channel, word_timestamps=timed)


def _pass(words: List[tuple], channel: Optional[int] = None) -> List[AsrPassWord]:
    return [AsrPassWord(word=wd, start_time=s, end_time=e, probability=0.7, channel=channel) for wd, s, e in words]


TERMS = [AsrVocabularyTerm(term=t, source=VocabularyTermSource.INDUSTRY_PACK) for t in ("Stanley cup", "Chadstone", "Afterpay")] + [
    AsrVocabularyTerm(term="Ticketek", source=VocabularyTermSource.CUSTOMER)]


def test_merge_rewrites_text_and_words_keeping_outer_punctuation():
    turn = _turn(3, "I bought a (standy cup.) today", 10.0, [("I", 10.0, 10.2), ("bought", 10.2, 10.5), ("a", 10.5, 10.6),
                                                              ("(standy", 10.7, 11.0), ("cup.)", 11.0, 11.4), ("today", 11.5, 11.9)])
    whisper = _pass([("I", 10.0, 10.2), ("bought", 10.2, 10.5), ("a", 10.5, 10.6), ("Stanley", 10.72, 11.0), ("cup,", 11.0, 11.35),
                     ("today", 11.5, 11.9)])
    out = merge_transcript([turn], whisper, TERMS)
    merged = out.turns[0]
    assert merged.text == "I bought a (Stanley cup.) today"
    assert [x.word for x in merged.word_timestamps] == ["I", "bought", "a", "(Stanley", "cup.)", "today"]
    assert (merged.word_timestamps[3].start_time, merged.word_timestamps[4].end_time) == (10.7, 11.4)
    assert merged.word_timestamps[3].end_time == pytest.approx(11.05) and merged.word_timestamps[3].probability == pytest.approx(0.8)
    assert out.candidates == 1 and len(out.replacements) == 1
    r = out.replacements[0]
    assert (r.turn_id, r.word_start, r.word_end) == (3, 3, 5)
    assert merged.text[r.char_start:r.char_end] == "Stanley cup"
    assert (r.heard, r.candidate_text, r.term, r.source) == ("(standy cup.)", "Stanley cup,", "Stanley cup", VocabularyTermSource.INDUSTRY_PACK)
    assert (r.start_time, r.end_time, r.candidate_start_time, r.candidate_end_time) == (10.7, 11.4, 10.72, 11.35)
    assert r.phonetic_similarity == pytest.approx(0.8333, abs=1e-4) and r.character_similarity == pytest.approx(0.8)


def test_merge_changes_word_counts_and_offsets_for_later_replacements_in_the_turn():
    text = "at Chad Stone paid with after pay ok"
    turn = _turn(0, text, 0.0)
    base = turn.word_timestamps
    whisper = _pass([("at", base[0].start_time, base[0].end_time), ("Chadstone", base[1].start_time, base[2].end_time),
                     ("paid", base[3].start_time, base[3].end_time), ("with", base[4].start_time, base[4].end_time),
                     ("Afterpay", base[5].start_time, base[6].end_time), ("ok", base[7].start_time, base[7].end_time)])
    out = merge_transcript([turn], whisper, TERMS)
    merged = out.turns[0]
    assert merged.text == "at Chadstone paid with Afterpay ok"
    assert [x.word for x in merged.word_timestamps] == ["at", "Chadstone", "paid", "with", "Afterpay", "ok"]
    assert [(r.word_start, r.word_end, merged.text[r.char_start:r.char_end]) for r in out.replacements] == [
        (1, 2, "Chadstone"), (4, 5, "Afterpay")]
    assert [r.heard for r in out.replacements] == ["Chad Stone", "after pay"]


def test_merge_never_inserts_and_leaves_turns_without_words_alone():
    words_turn = _turn(0, "hello there", 0.0)
    silent = TranscriptTurnContent(turn_id=1, speaker=SpeakerRole.UNKNOWN, start_time=5.0, end_time=6.0, text="Mm.")
    whisper = _pass([("hello", 0.0, 0.35), ("there", 0.4, 0.75), ("Ticketek", 5.0, 5.8)])
    out = merge_transcript([words_turn, silent], whisper, TERMS)
    assert out.candidates == 1 and out.replacements == []
    assert out.turns == [words_turn, silent]


def test_a_span_that_crosses_a_turn_is_rejected():
    first = _turn(0, "I went to Chad", 0.0, [("I", 0.0, 0.2), ("went", 0.2, 0.4), ("to", 0.4, 0.5), ("Chad", 0.6, 0.9)])
    second = _turn(1, "Stone yesterday", 0.9, [("Stone", 0.9, 1.3), ("yesterday", 1.4, 1.9)])
    whisper = _pass([("I", 0.0, 0.2), ("went", 0.2, 0.4), ("to", 0.4, 0.5), ("Chadstone", 0.62, 1.25), ("yesterday", 1.4, 1.9)])
    out = merge_transcript([first, second], whisper, TERMS)
    assert out.replacements == [] and out.rejected == {"crosses_turn": 1}
    assert [t.text for t in out.turns] == ["I went to Chad", "Stone yesterday"]


def test_a_candidate_whose_words_are_not_in_the_turn_text_is_rejected():
    turn = _turn(0, "I was at the mall", 0.0, [("I", 0.0, 0.2), ("was", 0.2, 0.4), ("Chad", 0.6, 0.9), ("Stone", 0.9, 1.3)])
    whisper = _pass([("I", 0.0, 0.2), ("was", 0.2, 0.4), ("Chadstone", 0.62, 1.25)])
    out = merge_transcript([turn], whisper, TERMS)
    assert out.replacements == [] and out.rejected == {"unlocated": 1}
    assert locate_words("a b a", ["a", "a", "c"]) == [(0, 1), (4, 5), None]


def test_stereo_candidates_are_generated_per_channel():
    agent = _turn(0, "welcome to Chad Stone", 0.0, [("welcome", 0.0, 0.4), ("to", 0.4, 0.5), ("Chad", 0.6, 0.9), ("Stone", 0.9, 1.3)], channel=0,
                  speaker=SpeakerRole.AGENT)
    caller = _turn(1, "a standy cup", 0.5, [("a", 0.5, 0.6), ("standy", 0.62, 0.95), ("cup", 0.95, 1.3)], channel=1, speaker=SpeakerRole.CALLER)
    # Whisper on channel 1 heard "Stanley cup" over the same seconds as the agent's "Chad Stone"; it may
    # only correct the caller's words, and channel 0's "Chadstone" only the agent's.
    whisper = _pass([("Chadstone", 0.62, 1.25)], channel=0) + _pass([("a", 0.5, 0.6), ("Stanley", 0.62, 0.95), ("cup", 0.95, 1.3)], channel=1)
    out = merge_transcript([agent, caller], whisper, TERMS, channels=2)
    assert [t.text for t in out.turns] == ["welcome to Chadstone", "a Stanley cup"]
    assert [(r.turn_id, r.term) for r in out.replacements] == [(0, "Chadstone"), (1, "Stanley cup")]
    # merged into one group, the overlapping channels garble each other and neither correction survives
    mono = merge_transcript([agent, caller], whisper, TERMS, channels=1)
    assert [t.text for t in mono.turns] == ["welcome to Chad Stone", "a standy cup"]


def test_the_merge_output_is_valid_contract_provenance():
    turn = _turn(0, "a standy cup", 0.0)
    base = turn.word_timestamps
    whisper = _pass([("a", base[0].start_time, base[0].end_time), ("Stanley", base[1].start_time, base[1].end_time),
                     ("cup", base[2].start_time, base[2].end_time)])
    out = merge_transcript([turn], whisper, TERMS)
    correction = VocabularyCorrection(status=VocabularyCorrectionStatus.APPLIED, vocabulary_digest="sha256:" + "0" * 64, term_count=4,
                                      base_engine="parakeet", candidate_engine="whisper-small", rule=VocabularyMergeRule(),
                                      candidates=out.candidates, replacements=out.replacements)
    content = TranscriptContent(duration_seconds=2.0, is_redacted=False, turns=out.turns, vocabulary_correction=correction)
    again = TranscriptContent.model_validate(json.loads(content.model_dump_json()))
    assert again.turns[0].text == "a Stanley cup" and again.vocabulary_correction.replacements[0].heard == "standy cup"
