"""Contact Signals v2 routes of the results area (contract 1.3.0; docs/ContactSignalsV2.md section 7.7):
the taxonomy and its versions, settings, text redaction, alert rules, hit feedback and metrics.
Previews, backfills and taxonomy snapshots are the queue area's (``call1/store/queue/signals.py``)."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Query

from call1.contracts.common import Page
from call1.contracts.metrics import SignalMetrics, SignalMetricsQuery
from call1.contracts.signals import (
    SignalAlertRuleRecord,
    SignalAlertRuleSave,
    SignalHitFeedback,
    SignalHitFeedbackSave,
    SignalSettingsSave,
    SignalTaxonomyRecord,
    SignalTaxonomyRedaction,
    SignalTaxonomySave,
    SignalTaxonomyVersion,
)

from .. import audit
from ..context import Store
from ..db import StoreConnection, read_snapshot, transaction
from ..deps import current_principal, get_conn, get_store
from ..errors import not_found
from ..principals import Principal, require_session
from . import metrics, records, signal_store, signals
from .routes import router


@router.operation("getSignalTaxonomy")
def get_signal_taxonomy(conn: StoreConnection = Depends(get_conn)) -> SignalTaxonomyRecord:
    with read_snapshot(conn):
        return signal_store.record(conn)


@router.operation("listSignalTaxonomyVersions")
def list_signal_taxonomy_versions(conn: StoreConnection = Depends(get_conn)) -> Page[SignalTaxonomyVersion]:
    with read_snapshot(conn):
        return Page[SignalTaxonomyVersion](items=signal_store.list_versions(conn), next_page_token=None)


@router.operation("getSignalTaxonomyVersion")
def get_signal_taxonomy_version(version: int, conn: StoreConnection = Depends(get_conn)) -> SignalTaxonomyVersion:
    with read_snapshot(conn):
        found = signal_store.get_version(conn, version)
    if found is None:
        raise not_found("Signal taxonomy version", version=version)
    return found


@router.operation("saveSignalTaxonomy")
def save_signal_taxonomy(body: SignalTaxonomySave, store: Store = Depends(get_store), conn: StoreConnection = Depends(get_conn),
                         principal: Principal = Depends(current_principal)) -> SignalTaxonomyRecord:
    session = require_session(principal)
    with transaction(conn):
        saved, _version = signal_store.save_taxonomy(conn, body, actor=audit.actor_for(principal), account_id=session.account_id,
                                                     parameters=store.config.parameters)
        return saved


@router.operation("saveSignalSettings")
def save_signal_settings(body: SignalSettingsSave, conn: StoreConnection = Depends(get_conn),
                         principal: Principal = Depends(current_principal)) -> SignalTaxonomyRecord:
    session = require_session(principal)
    with transaction(conn):
        return signal_store.save_settings(conn, body, actor=audit.actor_for(principal), account_id=session.account_id)


@router.operation("redactSignalTaxonomyText")
def redact_signal_taxonomy_text(version: int, body: SignalTaxonomyRedaction, conn: StoreConnection = Depends(get_conn),
                                principal: Principal = Depends(current_principal)) -> SignalTaxonomyVersion:
    session = require_session(principal)
    with transaction(conn):
        return signal_store.redact(conn, version, body, actor=audit.actor_for(principal), account_id=session.account_id)


@router.operation("listSignalAlertRules")
def list_signal_alert_rules(conn: StoreConnection = Depends(get_conn)) -> Page[SignalAlertRuleRecord]:
    with read_snapshot(conn):
        return Page[SignalAlertRuleRecord](items=signal_store.list_alert_rules(conn), next_page_token=None)


@router.operation("saveSignalAlertRule")
def save_signal_alert_rule(rule_id: str, body: SignalAlertRuleSave, store: Store = Depends(get_store), conn: StoreConnection = Depends(get_conn),
                           principal: Principal = Depends(current_principal)) -> SignalAlertRuleRecord:
    session = require_session(principal)
    with transaction(conn):
        return signal_store.save_alert_rule(conn, rule_id, body, actor=audit.actor_for(principal), account_id=session.account_id,
                                            parameters=store.config.parameters)


@router.operation("saveSignalHitFeedback")
def save_signal_hit_feedback(call_id: str, hit_id: str, body: SignalHitFeedbackSave, conn: StoreConnection = Depends(get_conn),
                             principal: Principal = Depends(current_principal)) -> SignalHitFeedback:
    session = require_session(principal)
    with transaction(conn):
        row = records.call_row(conn, call_id)
        if row is None:
            raise not_found("Call", call_id=call_id)
        return signals.save_feedback(conn, call_id, hit_id, body, account_id=session.account_id, actor=audit.actor_for(principal),
                                     conversation_id=row["conversation_id"])


@router.operation("getSignalMetrics")
def get_signal_metrics(query: Annotated[SignalMetricsQuery, Query()], conn: StoreConnection = Depends(get_conn)) -> SignalMetrics:
    return metrics.signal_metrics(conn, query)
