"""Synthetic history stays demo-only, idempotent and usable through the real Store reads."""
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from call1.demo_setup import sign_in
from call1.store.app import create_app
from call1.store.config import StoreConfig
from call1.store.results.demo_history import seed_demo_history
from .conftest import STORE_BASE_URL


def test_history_refuses_normal_store_and_invalid_count(store, tmp_path, clock):
    with pytest.raises(ValueError, match="DEMO"):
        seed_demo_history(store)
    demo = create_app(StoreConfig.for_tests(tmp_path / "demo", demo_mode=True), clock=clock).state.store
    for count in (0, -1, 5001):
        with pytest.raises(ValueError, match="between"):
            seed_demo_history(demo, count=count)


def test_history_has_real_projections_without_inference_or_duplicate_sessions(tmp_path, clock):
    app = create_app(StoreConfig.for_tests(tmp_path / "demo", demo_mode=True), clock=clock)
    store = app.state.store
    assert seed_demo_history(store) == 560
    with TestClient(app, base_url=STORE_BASE_URL) as client:
        headers = sign_in(client)
        def get(path, **kwargs):
            r = client.get('/store/v1' + path, headers=headers, **kwargs)
            assert r.status_code == 200, r.text
            return r.json()

        before = get('/metrics/executive')
        assert before['total_audited_calls'] == 560
        assert 0 < before['pass_rate_pct'] < 100
        assert before['calls_pending_analysis'] == 0
        rubric = get('/metrics/rubrics/call1_standard_v2')
        assert len(rubric['daily']) == 56
        assert sum(row['evaluated'] for row in rubric['daily']) == 560
        assert len({row['mean_score'] for row in rubric['daily']}) > 5
        assert sum(row['mean_score'] for row in rubric['daily'][-14:]) > sum(row['mean_score'] for row in rubric['daily'][:14])
        narrowed = get('/metrics/executive', params={'start': (clock.now() - timedelta(days=7)).isoformat()})
        assert 0 < narrowed['total_audited_calls'] < 560
        call = get('/calls')['items'][0]
        cid = call['call_id']
        detail = get('/calls/' + cid)
        assert detail['pending_work']['settled']
        transcript = get('/calls/' + cid + '/transcript')
        assert transcript['text_withheld'] is False and len(transcript['turns']) == 7
        assert get('/calls/' + cid + '/summary')['narrative'].startswith('Synthetic demo session')
        evaluation = get('/calls/' + cid + '/evaluation')
        for verdict in evaluation['verdicts']:
            assert verdict['quoted_evidence'] == transcript['turns'][verdict['quote_turn_id']]['text']
            assert verdict['model_attempts'] == []
        clock.advance(86400)
        headers = sign_in(client)
        assert seed_demo_history(store) == 0
        assert get('/metrics/executive')['total_audited_calls'] == 560
        assert seed_demo_history(store, count=600) == 40
        assert get('/metrics/executive')['total_audited_calls'] == 600
        with store.connection() as conn:
            assert conn.execute('SELECT COUNT(*) FROM q_jobs').fetchone()[0] == 0
            assert conn.execute('SELECT COUNT(DISTINCT agent_id) FROM results_calls').fetchone()[0] == 12
