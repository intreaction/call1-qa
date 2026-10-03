"""Functions the results area offers the other Store areas.

Other areas import only this module (and ``projections``) from ``call1.store.results``. Every
function takes the caller's connection and runs inside the caller's transaction when there is one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from call1.contracts.calls import ResultGroup
from call1.contracts.contents import ResultKind
from call1.contracts.contents import QaScorecardContent
from call1.contracts.rubrics import RubricSnapshotContent, RubricVersion

from .. import db
from ..db import StoreConnection
from . import records, rubric_store


def rubric_version(conn: StoreConnection, rubric_id: str, version: int) -> Optional[RubricVersion]:
    """One published version, active or retired (``mintRubricSnapshot`` accepts both)."""
    return rubric_store.get_version(conn, rubric_id, version)


def rubric_snapshot_content(conn: StoreConnection, rubric_id: str, version: int) -> Optional[RubricSnapshotContent]:
    """The ``rubric_snapshot`` content Store mints for a published version (queue stores it as a
    linked artifact in slot ``rubric:<rubric_id>:v<version>``)."""
    return rubric_store.published_snapshot(conn, rubric_id, version)


def draft_snapshot_content(conn: StoreConnection, rubric_id: str, *, expected_draft_revision: int) -> RubricSnapshotContent:
    """The stored draft as snapshot content for ``testRubricDraft``. Raises
    ``StoreError(RUBRIC_VERSION_CONFLICT, details={current_version, draft_revision})`` when the
    draft revision moved, ``StoreError(NOT_FOUND)`` when there is no draft."""
    return rubric_store.draft_snapshot(conn, rubric_id, expected_draft_revision=expected_draft_revision)


def result_groups(conn: StoreConnection, conversation_id: str) -> List[ResultGroup]:
    """Every result group of a conversation with its derived state (``calls.derive_result_state``
    over ``queue.api.group_stakes`` and the published versions). The one implementation of the
    rule; the queue area's ``getGroupProgress`` uses it for ``GroupProgress.state``."""
    return records.result_groups(conn, conversation_id)


def masked_scorecard(conn: StoreConnection, conversation_id: str, scorecard: QaScorecardContent) -> QaScorecardContent:
    """A scorecard masked like the call's evaluation reads (rule values plus the current PII
    findings, or fully redacted while those are missing). The queue area's draft-test result uses it."""
    return QaScorecardContent.model_validate(records.masked_scorecard_data(scorecard, records.masker_for(conn, conversation_id)))


# --- Contact Signals v2 (contract 1.3.0). The queue area's snapshots, previews, backfills and
# shadow-mode companions read the taxonomy and results only through these. -----------------------


def signal_taxonomy_version(conn: StoreConnection, version: int):
    """One published ``SignalTaxonomyVersion`` (redacted or not), or None."""
    from . import signal_store

    return signal_store.get_version(conn, version)


def current_signal_taxonomy(conn: StoreConnection):
    """The current published ``SignalTaxonomyVersion``."""
    from . import signal_store

    return signal_store.current_version(conn)


def signal_settings(conn: StoreConnection):
    """The current ``SignalSettings`` (pipeline v1, shadow or v2)."""
    from . import signal_store

    return signal_store.settings(conn)


def check_signal_taxonomy(taxonomy, parameters) -> None:
    """The save-time caps and definition-text detectors, for a preview's unsaved taxonomy
    (``StoreError(VALIDATION_FAILED, details.field=<path>)``)."""
    from . import signal_store

    signal_store.check_taxonomy(taxonomy, parameters)


@dataclass(frozen=True)
class SignalBackfillCandidate:
    call_id: str
    conversation_id: str
    needs_update: bool
    """rescore: a digest-driven update would rerun something (or ``rescore_signals``); compare: the
    current result is v1."""


def signal_backfill_candidates(conn: StoreConnection, created_after, created_before, limit: int, mode: str,
                               rescore_signals: bool = False) -> List[SignalBackfillCandidate]:
    """Calls created in the window with a published contact_signals result, newest first, at most
    ``limit``. For a rescore, a call needs an update when ``rescore_signals`` is set, or when the
    settings build v2 and its current result is v1 or outdated against the current taxonomy
    (``signals.taxonomy_status``). Under v1 or shadow a v1 result never needs a taxonomy update (v1
    ignores the taxonomy). For a compare, a call needs one when its current result is v1."""
    from call1.contracts.contents import ContactSignalsContent

    from . import content, signal_store, signals

    where, args = ["c.created_at >= ?"], [db.ts(created_after)]
    if created_before is not None:
        where.append("c.created_at < ?")
        args.append(db.ts(created_before))
    rows = conn.execute(
        "SELECT c.call_id, c.conversation_id FROM results_calls c WHERE " + " AND ".join(where) +
        " AND EXISTS (SELECT 1 FROM results_group_versions g WHERE g.conversation_id = c.conversation_id AND g.kind = ?) "
        "ORDER BY c.created_at DESC, c.call_id DESC LIMIT ?", (*args, ResultKind.CONTACT_SIGNALS.value, limit)).fetchall()
    builds_v2 = signal_store.settings(conn).pipeline == "v2"
    out: List[SignalBackfillCandidate] = []
    for row in rows:
        pub = records.latest_publication(conn, row["conversation_id"], ResultKind.CONTACT_SIGNALS)
        published = content.read_model(conn, pub.checksum, ContactSignalsContent)
        if mode == "compare":
            needs = published.pipeline == "v1"
        elif rescore_signals:
            needs = True
        elif not builds_v2:
            needs = False
        elif published.pipeline == "v1":
            needs = True
        else:
            status = signals.taxonomy_status(conn, published)
            needs = bool(status.outdated_stages or status.thresholds_changed)
        out.append(SignalBackfillCandidate(row["call_id"], row["conversation_id"], needs))
    return out


def published_signal_pipeline(conn: StoreConnection, conversation_id: str) -> Optional[str]:
    """``pipeline`` of the call's current contact_signals result, or None."""
    from . import signals

    published = signals.published_content(conn, conversation_id)
    return None if published is None else published.pipeline


def masked_signal_result(conn: StoreConnection, conversation_id: str, result):
    """A preview's ``ContactSignalsContent`` masked like the published view (quotes and field text;
    ``[REDACTED]`` while the call's PII findings are pending)."""
    from call1.contracts.contents import ContactSignalsContent

    from . import signals

    return ContactSignalsContent.model_validate(signals.masked_content_data(result, records.masker_for(conn, conversation_id)))


def signal_preview_diff(conn: StoreConnection, conversation_id: str, result):
    """``SignalPreviewDiff`` of a preview result against the call's published contact signals."""
    from . import signals

    return signals.preview_diff(signals.published_content(conn, conversation_id), result)
