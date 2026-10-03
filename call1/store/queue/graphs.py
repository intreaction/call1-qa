"""Job graphs: validation, insertion, initial status, dependent release and input resolution.

Used by ``createJobGraph`` and by a completion's follow-on jobs. A job starts BLOCKED; it becomes
QUEUED (``try_release``) when every ``requires`` upstream SUCCEEDED and every ``after`` upstream
is terminal, at which point each upstream-output input is resolved to the upstream's linked output
and pinned by checksum (``Job.resolved_inputs``). A dependent on a FAILED or CANCELLED ``requires``
upstream stays BLOCKED (dead-blocked) until that upstream is retried; it is never failed for it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from call1.contracts.artifacts import ArtifactKind
from call1.contracts.common import canonical_digest
from call1.contracts.custody import RouteClass
from call1.contracts.errors import ErrorCode
from call1.contracts.jobs import (
    JOB_TYPE_RULES,
    EdgeKind,
    JobDefinition,
    JobEdge,
    JobGraph,
    JobInput,
    JobRef,
    JobStatus,
    JobType,
    ResolvedInputRef,
    UpstreamOutput,
)
from call1.contracts.rubrics import RUBRIC_INPUT_ROLE, DraftRubricRef, RubricSnapshotContent
from call1.contracts.signals import SIGNAL_TAXONOMY_INPUT_ROLE, SignalTaxonomySnapshotContent

from .. import db
from ..errors import StoreError
from ..ids import new_id
from .changes import ChangeLog
from .records import ConversationJobs, graph_invalid, job_inputs, job_outputs, require_job_row

# Stage 2 applies the contract's admin-state defaults (admin state is deferred): only the appliance
# route is enabled. Every other route class is refused at graph creation and at admission.
PERMITTED_ROUTE_CLASSES = frozenset({RouteClass.APPLIANCE})


@dataclass
class PreparedJob:
    definition: JobDefinition
    job_id: str
    existing: bool
    digest: str
    inputs: List[JobInput] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)
    after: List[str] = field(default_factory=list)


def definition_digest(definition: JobDefinition) -> str:
    return canonical_digest(definition.model_dump(mode="json"))


def route_permitted(route_class: Optional[str]) -> bool:
    return route_class is None or RouteClass(route_class) in PERMITTED_ROUTE_CLASSES


def _check_rubric_input(conn, store, definition: JobDefinition, draft_rubric: Optional[DraftRubricRef]) -> None:
    """A QA job pins a rubric_snapshot under role ``rubric`` naming its rubric, version or draft
    revision, and digest (graph_invalid otherwise)."""
    params = definition.parameters
    if draft_rubric is not None:
        if params.draft_rubric is None or params.draft_rubric.model_dump() != draft_rubric.model_dump():
            raise graph_invalid(f"{definition.ref}: a draft-test graph's QA jobs score with the request's draft snapshot", ref=definition.ref, reason="draft_rubric_mismatch")
    elif params.draft_rubric is not None:
        raise graph_invalid(f"{definition.ref}: only a draft-test graph scores with a draft", ref=definition.ref, reason="draft_rubric_outside_draft_test")
    pinned = [i for i in definition.inputs if i.role == RUBRIC_INPUT_ROLE]
    if len(pinned) != 1 or pinned[0].artifact is None:
        raise graph_invalid(f"{definition.ref}: QA jobs pin their rubric_snapshot artifact under input role 'rubric'", ref=definition.ref, reason="rubric_input_missing")
    ref = pinned[0].artifact
    row = conn.execute("SELECT * FROM q_artifacts WHERE id = ?", (ref.artifact_id,)).fetchone()
    if row is None or row["kind"] != ArtifactKind.RUBRIC_SNAPSHOT.value:
        raise graph_invalid(f"{definition.ref}: the rubric input is not a rubric_snapshot", ref=definition.ref, reason="rubric_input_kind")
    try:
        content = RubricSnapshotContent.model_validate(json.loads(store.objects.read_bytes(row["checksum"])))
    except (ValueError, TypeError):
        raise graph_invalid(f"{definition.ref}: the rubric snapshot is unreadable", ref=definition.ref, reason="rubric_snapshot_unreadable") from None
    if params.rubric is not None:
        ok = (content.source == "published" and content.rubric_id == params.rubric.rubric_id
              and content.rubric_version == params.rubric.version and content.digest == params.rubric.digest)
    else:
        draft = params.draft_rubric
        ok = (content.source == "draft" and content.rubric_id == draft.rubric_id and content.draft_revision == draft.draft_revision
              and content.digest == draft.digest and draft.snapshot_artifact_id == ref.artifact_id)
    if not ok:
        raise graph_invalid(f"{definition.ref}: the rubric snapshot does not name the job's rubric, version and digest", ref=definition.ref, reason="rubric_snapshot_mismatch")


def _check_signals_input(conn, store, definition: JobDefinition, signal_request) -> None:
    """Contract 1.3.0: a Contact Signals v2 job (``needs_signal_taxonomy``), and a v2 merge that pins
    one, reads a ``signal_taxonomy_snapshot`` under role ``taxonomy`` whose digest is
    ``parameters.signals.taxonomy_digest`` (graph_invalid otherwise, like ``_check_rubric_input``).
    In a ``contact_signals_preview`` graph (``signal_request``) it is the request's own snapshot; a
    preview snapshot is refused anywhere else."""
    rule = JOB_TYPE_RULES[definition.job_type]
    params = definition.parameters.signals
    pinned = [i for i in definition.inputs if i.role == SIGNAL_TAXONOMY_INPUT_ROLE and i.artifact is not None]
    if not rule.needs_signal_taxonomy and not (params is not None and pinned):
        return
    ref = definition.ref
    if params is None or len(pinned) != 1:
        raise graph_invalid(f"{ref}: v2 jobs pin one signal_taxonomy_snapshot under input role 'taxonomy'", ref=ref, reason="signal_taxonomy_input_missing")
    artifact_id = pinned[0].artifact.artifact_id
    row = conn.execute("SELECT * FROM q_artifacts WHERE id = ?", (artifact_id,)).fetchone()
    if row is None or row["kind"] != ArtifactKind.SIGNAL_TAXONOMY_SNAPSHOT.value:
        raise graph_invalid(f"{ref}: the taxonomy input is not a signal_taxonomy_snapshot", ref=ref, reason="signal_taxonomy_input_kind")
    try:
        content = SignalTaxonomySnapshotContent.model_validate(json.loads(store.objects.read_bytes(row["checksum"])))
    except (ValueError, TypeError):
        raise graph_invalid(f"{ref}: the signal taxonomy snapshot is unreadable", ref=ref, reason="signal_snapshot_unreadable") from None
    if content.taxonomy_ref.digest != params.taxonomy_digest:
        raise graph_invalid(f"{ref}: the taxonomy snapshot's digest differs from parameters.signals.taxonomy_digest", ref=ref,
                            reason="signal_taxonomy_mismatch")
    if signal_request is not None:
        if artifact_id != signal_request["signal_taxonomy_snapshot_artifact_id"]:
            raise graph_invalid(f"{ref}: a preview graph runs with its request's taxonomy snapshot", ref=ref, reason="signal_preview_snapshot_mismatch")
        if params.preview_id is not None and params.preview_id != signal_request["signal_preview_id"]:
            raise graph_invalid(f"{ref}: parameters.signals.preview_id is not the request's preview", ref=ref, reason="signal_preview_mismatch")
    elif content.source != "published" or params.preview_id is not None:
        raise graph_invalid(f"{ref}: only a contact_signals_preview graph runs with a preview taxonomy", ref=ref, reason="signal_preview_outside_preview")


def prepare_jobs(conn, store, *, conversation_id: str, installation_id: str, definitions: Sequence[JobDefinition],
                 draft_rubric: Optional[DraftRubricRef] = None, draft_test_request_id: Optional[str] = None,
                 signal_request=None) -> List[PreparedJob]:
    """Validate a list of job definitions against Store's records and give each a job ID.

    A definition whose ``idempotency_key`` already names a job of this installation and
    conversation is a replay: the same definition returns that job, a different one is 409
    ``idempotency_key_reused``. Refs resolve to job IDs; upstream ``job_id``s must be jobs of the
    same conversation (no cross-conversation edges) and pinned artifacts must match their checksum.
    """
    prepared: List[PreparedJob] = []
    ref_map: Dict[str, str] = {}
    keys = [d.idempotency_key for d in definitions]
    if len(keys) != len(set(keys)):
        raise graph_invalid("Each job definition in a request has its own idempotency key", reason="duplicate_idempotency_key")
    for definition in definitions:
        digest = definition_digest(definition)
        row = conn.execute("SELECT id, definition_digest FROM q_jobs WHERE installation_id = ? AND conversation_id = ? AND idempotency_key = ?",
                           (installation_id, conversation_id, definition.idempotency_key)).fetchone()
        if row is not None and row["definition_digest"] != digest:
            raise StoreError(ErrorCode.IDEMPOTENCY_KEY_REUSED, "A job with this idempotency key has a different definition",
                             details={"original_id": row["id"], "ref": definition.ref})
        job_id = row["id"] if row is not None else new_id("job")
        ref_map[definition.ref] = job_id
        prepared.append(PreparedJob(definition=definition, job_id=job_id, existing=row is not None, digest=digest))

    for item in prepared:
        definition = item.definition
        item.requires = [ref_map[r] for r in definition.requires_refs] + list(definition.requires_job_ids)
        item.after = [ref_map[r] for r in definition.after_refs] + list(definition.after_job_ids)
        if item.existing:
            continue
        rule = JOB_TYPE_RULES[definition.job_type]
        if definition.selection is not None and definition.selection.route.route_class not in PERMITTED_ROUTE_CLASSES:
            raise StoreError(ErrorCode.ROUTE_NOT_PERMITTED, "This route class is not enabled in admin state (Stage 2: appliance only)",
                             details={"ref": definition.ref, "route_class": definition.selection.route.route_class.value})
        for upstream_id in list(definition.requires_job_ids) + list(definition.after_job_ids):
            up = conn.execute("SELECT conversation_id FROM q_jobs WHERE id = ?", (upstream_id,)).fetchone()
            if up is None:
                raise graph_invalid(f"{definition.ref}: upstream job {upstream_id} does not exist", ref=definition.ref, reason="unknown_job", job_id=upstream_id)
            if up["conversation_id"] != conversation_id:
                raise graph_invalid(f"{definition.ref}: jobs of another conversation cannot be upstream", ref=definition.ref, reason="cross_conversation", job_id=upstream_id)
        inputs: List[JobInput] = []
        for job_input in definition.inputs:
            if job_input.upstream is not None:
                upstream = job_input.upstream
                if upstream.job_id is not None:
                    up = conn.execute("SELECT job_type FROM q_jobs WHERE id = ?", (upstream.job_id,)).fetchone()
                    if upstream.output_role not in JOB_TYPE_RULES[JobType(up["job_type"])].outputs:
                        raise graph_invalid(f"{definition.ref}.{job_input.role}: {up['job_type']} has no output role {upstream.output_role}",
                                            ref=definition.ref, reason="unknown_output_role")
                    target = upstream.job_id
                else:
                    target = ref_map[upstream.ref]
                inputs.append(JobInput(role=job_input.role, upstream=UpstreamOutput(job_id=target, output_role=upstream.output_role), optional=job_input.optional))
                continue
            pinned = job_input.artifact
            row = conn.execute("SELECT conversation_id, checksum, linked FROM q_artifacts WHERE id = ?", (pinned.artifact_id,)).fetchone()
            if row is None or row["conversation_id"] != conversation_id:
                raise graph_invalid(f"{definition.ref}.{job_input.role}: artifact {pinned.artifact_id} is not an artifact of this conversation",
                                    ref=definition.ref, reason="unknown_artifact")
            if row["checksum"] != pinned.checksum:
                raise StoreError(ErrorCode.CHECKSUM_MISMATCH, f"{definition.ref}.{job_input.role}: the pinned checksum differs from the artifact's",
                                 details={"ref": definition.ref, "artifact_id": pinned.artifact_id, "reason": "input_checksum_differs"})
            if not row["linked"]:
                raise graph_invalid(f"{definition.ref}.{job_input.role}: artifact {pinned.artifact_id} was never linked", ref=definition.ref, reason="unlinked_input")
            inputs.append(job_input)
        item.inputs = inputs
        if rule.needs_rubric:
            _check_rubric_input(conn, store, definition, draft_rubric)
        elif definition.parameters.draft_rubric is not None and draft_rubric is None:
            raise graph_invalid(f"{definition.ref}: only a draft-test graph scores with a draft", ref=definition.ref, reason="draft_rubric_outside_draft_test")
        _check_signals_input(conn, store, definition, signal_request)
        _check_asr_vocabulary(store, definition)
    return prepared


def _check_asr_vocabulary(store, definition: JobDefinition) -> None:
    """1.3.0 (decision 33, docs/DualAsr.md section 4): an ``asr`` job's frozen vocabulary holds at
    most ``max_vocabulary_terms + max_vocabulary_pack_terms`` terms. The contract model already
    checks the digest, each term and that only ``asr`` carries it. The digest need not be the
    current vocabulary's: a graph reproduces what it was planned with."""
    frozen = definition.parameters.asr_vocabulary
    if frozen is None:
        return
    params = store.config.parameters
    cap = params.max_vocabulary_terms + params.max_vocabulary_pack_terms
    if len(frozen.terms) > cap:
        raise graph_invalid(f"{definition.ref}: parameters.asr_vocabulary holds at most {cap} terms", ref=definition.ref,
                            reason="asr_vocabulary_cap", limit=cap, actual=len(frozen.terms))


def insert_prepared(conn, *, prepared: Sequence[PreparedJob], graph_id: str, conversation_id: str, installation_id: str,
                    draft_test_request_id: Optional[str], changes: ChangeLog, default_max_attempts: int) -> Tuple[List[str], List[str]]:
    """Insert the new jobs (BLOCKED), their edges and the graph's membership rows, then give each
    new job its initial status. Returns (new job IDs, those that started QUEUED)."""
    now = db.ts(conn.now())
    position = conn.execute("SELECT COALESCE(MAX(position), -1) + 1 AS p FROM q_graph_jobs WHERE graph_id = ?", (graph_id,)).fetchone()["p"]
    created: List[str] = []
    for item in prepared:
        definition = item.definition
        if item.existing:
            continue
        selection = definition.selection
        estimate = definition.resource_estimate
        rule = JOB_TYPE_RULES[definition.job_type]
        conn.execute(
            "INSERT INTO q_jobs (id, conversation_id, graph_id, ref, installation_id, idempotency_key, definition_digest, job_type, execution_class, "
            "status, priority, max_attempts, inputs_json, selection_json, resource_estimate_json, parameters_json, memory_slot, "
            "outbound_connection_ref, route_class, catalog_entry_id, catalog_entry_version, draft_test_request_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'BLOCKED', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (item.job_id, conversation_id, graph_id, definition.ref, installation_id, definition.idempotency_key, item.digest,
             definition.job_type.value, rule.execution_class.value, definition.priority, definition.max_attempts or default_max_attempts,
             db.dumps([i.model_dump(mode="json") for i in item.inputs]), db.dumps(selection) if selection is not None else None,
             db.dumps(estimate), db.dumps(definition.parameters), estimate.memory_slot.value, estimate.outbound_connection_ref,
             selection.route.route_class.value if selection is not None else None,
             selection.catalog_entry.entry_id if selection is not None else None,
             selection.catalog_entry.entry_version if selection is not None else None,
             draft_test_request_id, now, now),
        )
        created.append(item.job_id)
    for item in prepared:
        conn.execute("INSERT INTO q_graph_jobs (graph_id, position, ref, job_id) VALUES (?, ?, ?, ?)", (graph_id, position, item.definition.ref, item.job_id))
        position += 1
        if item.existing:
            continue
        for upstream_id in item.requires:
            conn.execute("INSERT INTO q_edges (job_id, upstream_job_id, kind) VALUES (?, ?, 'requires')", (item.job_id, upstream_id))
        for upstream_id in item.after:
            conn.execute("INSERT INTO q_edges (job_id, upstream_job_id, kind) VALUES (?, ?, 'after')", (item.job_id, upstream_id))
    queued = []
    for job_id in created:
        if try_release(conn, job_id):
            queued.append(job_id)
        row = require_job_row(conn, job_id)
        changes.job(job_id, row["status"], conversation_id)
    return created, queued


def _upstream_rows(conn, job_id: str):
    return conn.execute(
        "SELECT e.kind, u.id, u.status, u.outputs_json FROM q_edges e JOIN q_jobs u ON u.id = e.upstream_job_id WHERE e.job_id = ?",
        (job_id,),
    ).fetchall()


def try_release(conn, job_id: str) -> bool:
    """BLOCKED -> QUEUED when every edge is satisfied; resolves and pins upstream-output inputs."""
    row = require_job_row(conn, job_id)
    if row["status"] != JobStatus.BLOCKED.value:
        return False
    upstream = {}
    for up in _upstream_rows(conn, job_id):
        kind, status = EdgeKind(up["kind"]), JobStatus(up["status"])
        if not ConversationJobs.satisfied(status, kind):
            return False
        upstream[up["id"]] = (status, {o.role: o for o in job_outputs(up)})
    resolved: List[ResolvedInputRef] = []
    for job_input in job_inputs(row):
        if job_input.artifact is not None:
            resolved.append(ResolvedInputRef(role=job_input.role, artifact_id=job_input.artifact.artifact_id, checksum=job_input.artifact.checksum))
            continue
        status, outputs = upstream.get(job_input.upstream.job_id, (None, {}))
        output = outputs.get(job_input.upstream.output_role) if status is JobStatus.SUCCEEDED else None
        if output is None:
            if job_input.optional:
                continue
            return False  # a required upstream succeeded without this output: cannot happen after verified completions
        resolved.append(ResolvedInputRef(role=job_input.role, artifact_id=output.artifact_id, checksum=output.checksum))
    now = db.ts(conn.now())
    conn.execute(
        "UPDATE q_jobs SET status = 'QUEUED', resolved_inputs_json = ?, queued_at = ?, next_run_at = NULL, wait_cause = NULL, updated_at = ? "
        "WHERE id = ? AND status = 'BLOCKED'",
        (db.dumps([r.model_dump(mode="json") for r in resolved]), now, now, job_id),
    )
    return True


def release_dependents(conn, upstream_job_id: str, changes: ChangeLog) -> List[str]:
    """Release every BLOCKED direct dependent of a job whose status just changed."""
    released = []
    for dep in conn.execute("SELECT e.job_id, j.conversation_id FROM q_edges e JOIN q_jobs j ON j.id = e.job_id WHERE e.upstream_job_id = ? ORDER BY j.created_at, j.id",
                            (upstream_job_id,)).fetchall():
        if try_release(conn, dep["job_id"]):
            released.append(dep["job_id"])
            changes.job(dep["job_id"], JobStatus.QUEUED, dep["conversation_id"])
    return released


def downstream_closure(conn, job_id: str) -> List[str]:
    """Every job that depends on ``job_id``, transitively, in breadth-first order."""
    seen: Set[str] = set()
    order: List[str] = []
    frontier = [job_id]
    while frontier:
        nxt = []
        for current in frontier:
            for row in conn.execute("SELECT job_id FROM q_edges WHERE upstream_job_id = ?", (current,)):
                if row["job_id"] not in seen:
                    seen.add(row["job_id"])
                    order.append(row["job_id"])
                    nxt.append(row["job_id"])
        frontier = nxt
    return order


def reaches(conn, start_ids: Sequence[str], target: str, extra: Dict[str, List[str]]) -> bool:
    """Whether ``target`` is an upstream (transitively) of any job in ``start_ids``; ``extra``
    adds not-yet-inserted upstream lists."""
    seen: Set[str] = set()
    stack = list(start_ids)
    while stack:
        current = stack.pop()
        if current == target:
            return True
        if current in seen:
            continue
        seen.add(current)
        stack.extend(extra.get(current, []))
        stack.extend(r["upstream_job_id"] for r in conn.execute("SELECT upstream_job_id FROM q_edges WHERE job_id = ?", (current,)))
    return False


def graph_model(conn, graph_id: str, *, created: bool) -> JobGraph:
    graph = conn.execute("SELECT * FROM q_graphs WHERE id = ?", (graph_id,)).fetchone()
    members = conn.execute(
        "SELECT g.ref, j.id, j.job_type, j.status FROM q_graph_jobs g JOIN q_jobs j ON j.id = g.job_id WHERE g.graph_id = ? ORDER BY g.position",
        (graph_id,),
    ).fetchall()
    job_ids = [m["id"] for m in members]
    edges: List[JobEdge] = []
    if job_ids:
        marks = ",".join("?" for _ in job_ids)
        for row in conn.execute(f"SELECT job_id, upstream_job_id, kind FROM q_edges WHERE job_id IN ({marks}) ORDER BY job_id, upstream_job_id", job_ids):
            edges.append(JobEdge(job_id=row["job_id"], upstream_job_id=row["upstream_job_id"], kind=EdgeKind(row["kind"])))
    return JobGraph(
        graph_id=graph["id"], conversation_id=graph["conversation_id"], reason=graph["reason"],
        reanalysis_request_id=graph["reanalysis_request_id"], created_at=db.parse_ts(graph["created_at"]), created=created,
        jobs=[JobRef(ref=m["ref"], job_id=m["id"], job_type=JobType(m["job_type"]), status=JobStatus(m["status"])) for m in members],
        edges=edges,
    )
