"""Uncertain evidence is neither a failure nor passing credit."""
import pytest

from call1.contracts.contents import VerdictView
from call1.contracts.rubrics import RubricDefinition
from call1.process.handlers.code import score
from call1.process.handlers.real.convert import legacy_rubric
from call1.pipeline.evaluator import RubricEvaluator
from call1.models.schemas import CallTranscript, RubricVerdict


@pytest.mark.parametrize('weights,statuses,confidence,critical,expected,passed,review', [
    ([85,15], ['PASS','FLAGGED'], [.9,0], False, 100, False, True),
    ([40,30,30], ['PASS','FAIL','FLAGGED'], [.9,.9,0], False, 57.1, False, True),
    ([50,50], ['FLAGGED','FLAGGED'], [0,0], False, 0, False, True),
    ([50,50], ['PASS','PASS'], [.9,.3], False, 100, False, True),
    ([85,15], ['PASS','FAIL'], [.9,.9], True, 85, False, True),
    ([85,15], ['PASS','FAIL'], [.9,.9], False, 85, True, False),
    ([85,15], ['PASS','NOT_APPLICABLE'], [.9,.9], False, 100, True, False),
])
def test_provisional_and_final_scores_agree_across_engines(weights,statuses,confidence,critical,expected,passed,review):
    rubric=RubricDefinition.model_validate({
        'rubric_id':'scoring-test','name':'Scoring test','category':'GENERAL','pass_threshold':80,
        'criteria':[{'criterion_id':f'C{i}','name':f'Check {i}','category':'GENERAL','weight':weight,
                     'critical':critical and i==1,'check':{'check_type':'semantic_judgement'}}
                    for i,weight in enumerate(weights)]})
    verdicts={c.criterion_id:VerdictView(criterion_id=c.criterion_id,criterion_name=c.name,
              status=status,confidence=conf,speaker='AGENT',reasoning='Observed evidence.')
              for c,status,conf in zip(rubric.criteria,statuses,confidence)}
    _,overall,did_pass,did_critical,needs_review,_=score(rubric.criteria,verdicts,80)
    assert (overall,did_pass,did_critical,needs_review)==(expected,passed,critical,review)
    engine=RubricEvaluator(legacy_rubric(rubric))
    engine._evaluate_criterion=lambda criterion,*args: RubricVerdict(
        criterion_id=criterion.criterion_id,criterion_name=criterion.name,
        status=verdicts[criterion.criterion_id].status.value,
        confidence=verdicts[criterion.criterion_id].confidence,speaker='AGENT',reasoning='Observed evidence.')
    result=engine.evaluate_deterministic(CallTranscript(call_id='test',duration_seconds=10,turns=[]))
    assert (result.overall_score,result.passed,result.critical_failure,result.requires_human_review)==(expected,passed,critical,review)
