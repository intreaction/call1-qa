"""Per-group job accounting: the stakes ``calls.result_state_inputs`` needs, pending work, the
compact progress view, and the failure code of a group's newest publisher.

Jobs of draft-test graphs never count toward a call's groups, ``pending_work`` or ``settled``; their
progress is ``DraftTestResult.state``. The derived state itself is the results area's single
implementation (``results.api.result_groups``); this module only supplies inputs and counts.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from call1.contracts.calls import GroupGraphStake, PendingWorkIndicator
from call1.contracts.contents import ResultKind, ResultState
from call1.contracts.errors import JobErrorCode
from call1.contracts.jobs import JOB_TYPE_RULES, TERMINAL_STATUSES, GroupProgress, JobGroupProgress, JobStatus, WaitingReason

from .. import db
from .records import ConversationJobs, JobNode, require_conversation_row, waiting_reason


def _graphs(conn, conversation_id: str):
    return conn.execute("SELECT id, created_at, draft_test_request_id FROM q_graphs WHERE conversation_id = ? ORDER BY created_at, id",
                        (conversation_id,)).fetchall()


def _newest(nodes: List[JobNode]) -> JobNode:
    return max(nodes, key=lambda n: (n.created_at, n.id))


def group_stakes(conn, conversation_id: str, view: Optional[ConversationJobs] = None) -> Dict[ResultKind, List[GroupGraphStake]]:
    view = view or ConversationJobs(conn, conversation_id)
    by_graph: Dict[str, List[JobNode]] = {}
    for node in view.nodes.values():
        by_graph.setdefault(node.graph_id, []).append(node)
    stakes: Dict[ResultKind, List[GroupGraphStake]] = {kind: [] for kind in ResultKind}
    for graph in _graphs(conn, conversation_id):
        nodes = by_graph.get(graph["id"], [])
        for kind in ResultKind:
            members = [n for n in nodes if JOB_TYPE_RULES[n.job_type].group is kind]
            publishers = [n for n in nodes if JOB_TYPE_RULES[n.job_type].publishes is kind]
            if not members and not publishers:
                continue
            stakes[kind].append(GroupGraphStake(
                graph_id=graph["id"], graph_created_at=db.parse_ts(graph["created_at"]), draft_test=graph["draft_test_request_id"] is not None,
                has_group_jobs=bool(members), publisher=view.publisher_state(_newest(publishers).id) if publishers else None,
            ))
    return stakes


def group_failure_codes(conn, conversation_id: str, view: Optional[ConversationJobs] = None) -> Dict[ResultKind, Optional[JobErrorCode]]:
    """Per group: when the newest live graph's publisher ended without a result, its error code or
    the first ended upstream's; otherwise None. Draft-test graphs never count."""
    view = view or ConversationJobs(conn, conversation_id)
    out: Dict[ResultKind, Optional[JobErrorCode]] = {kind: None for kind in ResultKind}
    graphs = {g["id"]: g for g in _graphs(conn, conversation_id)}
    for kind in ResultKind:
        publishers = [n for n in view.nodes.values() if not n.draft_test and JOB_TYPE_RULES[n.job_type].publishes is kind]
        if not publishers:
            continue
        newest_graph = max({n.graph_id for n in publishers}, key=lambda g: (graphs[g]["created_at"], g))
        publisher = _newest([n for n in publishers if n.graph_id == newest_graph])
        if view.publisher_state(publisher.id).value == "ended_without_result":
            out[kind] = view.failure_code(publisher.id)
    return out


def pending_work(conn, conversation_id: str, view: Optional[ConversationJobs] = None) -> PendingWorkIndicator:
    view = view or ConversationJobs(conn, conversation_id)
    live = [n for n in view.nodes.values() if not n.draft_test]
    count = lambda status: sum(1 for n in live if n.status is status)  # noqa: E731
    return PendingWorkIndicator(
        jobs_total=len(live), jobs_succeeded=count(JobStatus.SUCCEEDED), jobs_running=count(JobStatus.RUNNING),
        jobs_queued=count(JobStatus.QUEUED), jobs_blocked=count(JobStatus.BLOCKED), jobs_failed=count(JobStatus.FAILED),
        jobs_cancelled=count(JobStatus.CANCELLED), settled=view.settled(),
    )


def group_progress(conn, conversation_id: str) -> JobGroupProgress:
    """``getGroupProgress``: counts per group plus the derived state from the results area."""
    from ..results import api as results_api

    conversation = require_conversation_row(conn, conversation_id)
    view = ConversationJobs(conn, conversation_id)
    states = {group.kind: group.state for group in results_api.result_groups(conn, conversation_id)}
    now = conn.now()
    rows = {r["id"]: r for r in conn.execute("SELECT * FROM q_jobs WHERE conversation_id = ? AND draft_test_request_id IS NULL", (conversation_id,))}
    groups = []
    for kind in ResultKind:
        members = [view.nodes[job_id] for job_id in rows if JOB_TYPE_RULES[view.nodes[job_id].job_type].group is kind]
        count = lambda status: sum(1 for n in members if n.status is status)  # noqa: E731
        dead = sum(1 for n in members if view.dead(n.id))
        reason = None
        if members and not any(n.status is JobStatus.RUNNING for n in members):
            waiting = [waiting_reason(rows[n.id], view, now) for n in sorted(members, key=lambda n: (n.created_at, n.id)) if n.status not in TERMINAL_STATUSES]
            queued = [w for w in waiting if w in (WaitingReason.WAITING_FOR_WORKER, WaitingReason.RETRY_BACKOFF, WaitingReason.DEFERRED_BY_WORKER)]
            if queued:
                reason = queued[0]
            elif WaitingReason.WAITING_FOR_DEPENDENCIES in waiting:
                reason = WaitingReason.WAITING_FOR_DEPENDENCIES
            elif WaitingReason.DEAD_BLOCKED in waiting:
                reason = WaitingReason.DEAD_BLOCKED
        groups.append(GroupProgress(
            kind=kind, state=states.get(kind, ResultState.DISABLED), total=len(members), succeeded=count(JobStatus.SUCCEEDED), running=count(JobStatus.RUNNING),
            queued=count(JobStatus.QUEUED), blocked=count(JobStatus.BLOCKED), dead_blocked=dead, failed=count(JobStatus.FAILED),
            cancelled=count(JobStatus.CANCELLED), waiting_reason=reason,
        ))
    supporting = sum(1 for job_id in rows if JOB_TYPE_RULES[view.nodes[job_id].job_type].group is None)
    updated = max((db.parse_ts(r["updated_at"]) for r in rows.values()), default=db.parse_ts(conversation["created_at"]))
    return JobGroupProgress(conversation_id=conversation_id, groups=groups, supporting_jobs_total=supporting, settled=view.settled(), updated_at=updated)
