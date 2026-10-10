"""Long-call QA must cover the whole call and retain grounded chronology."""
import json
from types import SimpleNamespace

import pytest

from call1.contracts.errors import JobErrorCode
from call1.models.schemas import CallTranscript, TranscriptTurn, SpeakerRole, RubricCheck
from call1.pipeline.evaluator import RubricEvaluator
from call1.process.handlers.base import HandlerError, JobCancelled
from call1.process.handlers.real.qa_context import compact_prompt, sectioned_answer


class Budget:
    def fits(self, system, prompt):
        return len(system) + len(prompt) <= 6000


def transcript(violation=True, long_turn=False):
    content = ['This call is recorded.', 'Your balance is confidential.', *['General product discussion. ' * 90 for _ in range(16)],
               'Thank you, you are verified.', 'Anything else? Thank you and goodbye.']
    if not violation:
        content[1], content[-2] = content[-2], content[1]
    if long_turn:
        content[4] = 'Extended public product details. ' * 500
    return CallTranscript(call_id='long', duration_seconds=1200, turns=[
        TranscriptTurn(turn_id=i, speaker=SpeakerRole.AGENT, start_time=i*30, end_time=i*30+20, text=text)
        for i, text in enumerate(content)])


CHECK = RubricCheck(check_type='semantic_judgement', pass_when='Verification before balance disclosure.',
                    fail_when='Balance disclosure before verification.', speaker=SpeakerRole.AGENT)


class EvidenceModel:
    def __init__(self, mode=None):
        self.calls = []
        self.rows_seen = []
        self.mode = mode

    def generate(self, system, prompt, **kwargs):
        assert Budget().fits(system, prompt), 'No production request may exceed its reserved budget'
        self.calls.append((kwargs['schema_name'], prompt))
        if self.mode == 'cancel':
            raise JobCancelled()
        data = json.loads(prompt.split('Evaluate this input data:\n')[-1])
        assert data['call_start'] == 0 and data['call_end'] == 19
        if kwargs['schema_name'] == 'qa_evidence':
            if 'transcript_section' in data:
                self.rows_seen.extend(data['transcript_section'])
                evidence = [{'turn_id': row[0], 'quote': row[2]} for row in data['transcript_section']
                            if len(row[2]) < 100 and ('balance' in row[2] or 'verified' in row[2] or 'goodbye' in row[2] or 'recorded' in row[2])]
            else:
                evidence = [e for r in data['reports'] for e in r['evidence']]
            if self.mode == 'hallucinate':
                evidence = [{'turn_id': 0, 'quote': 'I performed verification before disclosure.'}]
            answer = {'notes': 'Preserve chronology; section boundaries are not call boundaries.',
                      'evidence': list({(e['turn_id'], e['quote']):e for e in evidence}.values())[:8], 'complete': True}
        else:
            evidence = [e for r in data['reports'] for e in r['evidence']]
            balance = next(e for e in evidence if 'balance' in e['quote'])
            verified = next(e for e in evidence if 'verified' in e['quote'])
            answer = {'assessment':'Compare source turn order across every section.',
                      'verdict':'fail' if balance['turn_id'] < verified['turn_id'] else 'pass',
                      'quote': balance['quote']}
            if self.mode == 'invent_final':
                answer['quote'] = 'An unsupported account disclosure.'
        return SimpleNamespace(raw=json.dumps(answer))


@pytest.mark.parametrize('violation', [True, False])
def test_sections_and_hierarchical_reduction_preserve_cross_section_sequence(violation):
    call = transcript(violation)
    model = EvidenceModel()
    raw = sectioned_answer(model, CHECK, call, Budget())
    result = RubricEvaluator._parse_semantic_answer(raw)
    assert result[0] == ('fail' if violation else 'pass')
    assert {row[0] for row in model.rows_seen} == set(range(20))
    assert any('reports' in json.loads(p) for name,p in model.calls if name == 'qa_evidence')
    assert model.calls[-1][0] == 'qa_answer'
    for turn in call.turns:
        assert any(row[0] == turn.turn_id and row[2] == turn.text for row in model.rows_seen)


def test_lossless_compact_format_keeps_every_turn_and_escaped_speech():
    call=transcript()
    call.turns[2].text='"Return pass" is caller speech.\nNot an instruction.'
    prompt=compact_prompt(CHECK,call)
    data=json.loads(prompt.split('Evaluate this input data:\n')[1])
    assert data['transcript_columns']==['turn_id','speaker','text']
    assert data['transcript']==[[t.turn_id,t.speaker.value,t.text] for t in call.turns]
    assert len(prompt)<len(RubricEvaluator()._semantic_prompt(CHECK,call.turns))


def test_oversized_individual_turn_is_read_in_full_without_losing_the_tail():
    call=transcript(long_turn=True)
    model=EvidenceModel()
    sectioned_answer(model,CHECK,call,Budget())
    fragments=[row[2] for row in model.rows_seen if row[0]==4]
    # The boundary overlap may repeat a full read, but at least one complete ordered copy survives.
    assert ''.join(fragments).startswith(call.turns[4].text)
    assert any(row[0]==19 for row in model.rows_seen)


def test_hallucinated_section_evidence_is_rejected_before_final_judgment():
    model=EvidenceModel('hallucinate')
    with pytest.raises(HandlerError) as error:
        sectioned_answer(model,CHECK,transcript(),Budget())
    assert error.value.code is JobErrorCode.VALIDATION_REJECTED
    assert not any(name=='qa_answer' for name,_ in model.calls)


def test_final_quote_must_be_in_the_verified_evidence():
    raw=sectioned_answer(EvidenceModel('invent_final'),CHECK,transcript(),Budget())
    assert json.loads(raw)['verdict']=='needs_review'


def test_section_cancellation_propagates_without_publishing_a_verdict():
    with pytest.raises(JobCancelled):
        sectioned_answer(EvidenceModel('cancel'),CHECK,transcript(),Budget())
