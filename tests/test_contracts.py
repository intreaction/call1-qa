"""Tests of the frozen Store contract in call1/contracts: drift, invariants and round-trips."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Dict, Set

import pytest
from pydantic import ValidationError

import call1.contracts as contracts
from call1.contracts import admin, api, artifacts, auth, calls, catalog, common, contents, custody, errors, events, jobs, metrics, release_trust, reviews, rubrics, signals, usage
from call1.contracts.generate import OPENAPI_PATH, REPO, TS_BIN, TS_PATH, constraint_pins, generator_version_problems, pins_required, render_openapi, render_typescript, typescript_required

NOW = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
D0 = "sha256:" + "0" * 64
D1 = "sha256:" + "1" * 64
D2 = "sha256:" + "2" * 64
TOKEN = "t" * 43
E = errors.ErrorCode
P = auth.Permission


# --- drift ---------------------------------------------------------------------------------


def _require_pinned_generators():
    """Byte-for-byte drift is only meaningful under the pinned generators. Elsewhere skip, unless
    CI demands the pins (CALL1_REQUIRE_GENERATOR_PINS or CALL1_REQUIRE_TS_CHECK)."""
    problems = generator_version_problems()
    if problems:
        message = "; ".join(problems) + " (pip install -r requirements.txt -c call1/contracts/constraints.txt)"
        if pins_required():
            pytest.fail(message)
        pytest.skip(message)


def test_committed_openapi_matches_regeneration():
    _require_pinned_generators()
    assert OPENAPI_PATH.read_text() == render_openapi(), "run: python -m call1.contracts.generate"


def test_committed_typescript_matches_regeneration():
    _require_pinned_generators()
    if not TS_BIN.exists():
        if typescript_required():
            pytest.fail("CALL1_REQUIRE_TS_CHECK is set but openapi-typescript is not installed (npm --prefix frontend ci)")
        pytest.skip("openapi-typescript not installed (npm --prefix frontend ci)")
    assert TS_PATH.read_text() == render_typescript(), "run: python -m call1.contracts.generate"


def test_generate_check_cli_passes():
    _require_pinned_generators()
    env = {k: v for k, v in os.environ.items() if k != "CALL1_REQUIRE_TS_CHECK"}
    result = subprocess.run([sys.executable, "-m", "call1.contracts.generate", "--check", "--no-typescript"], cwd=REPO, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr


def test_generator_versions_match_pins():
    assert api.build_openapi()["x-call1"]["generator"] == api.GENERATOR_PINS
    assert constraint_pins() == {k: v for k, v in api.GENERATOR_PINS.items() if k in ("fastapi", "pydantic")}
    _require_pinned_generators()
    assert generator_version_problems() == []


def test_drift_tests_skip_under_unpinned_generators_unless_required(monkeypatch):
    import call1.contracts.generate as generate

    monkeypatch.setattr(sys.modules[__name__], "generator_version_problems", lambda: ["fastapi 0.142.0 is installed; the contract is generated with 0.141.x"])
    for name in ("CALL1_REQUIRE_GENERATOR_PINS", "CALL1_REQUIRE_TS_CHECK"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(pytest.skip.Exception):
        _require_pinned_generators()
    monkeypatch.setenv("CALL1_REQUIRE_GENERATOR_PINS", "1")
    assert generate.pins_required()
    with pytest.raises(pytest.fail.Exception):
        _require_pinned_generators()


def test_contract_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", contracts.CONTRACT_VERSION)
    doc = api.build_openapi()
    assert doc["info"]["version"] == contracts.CONTRACT_VERSION
    assert doc["x-call1"]["contract_version"] == contracts.CONTRACT_VERSION


# --- canonical JSON ------------------------------------------------------------------------


def test_canonical_json_matches_rfc8785_vectors():
    sample = {"numbers": [333333333.33333329, 1e30, 4.50, 2e-3, 0.000000000000000000000000001], "string": "\u20ac$\u000f\nA'B\"\\\\\"/", "literals": [None, True, False]}
    assert common.canonical_json(sample).decode() == '{"literals":[null,true,false],"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27],"string":"€$\\u000f\\nA\'B\\"\\\\\\\\\\"/"}'
    keys = {"\u20ac": 1, "\r": 2, "\ufb33": 3, "1": 4, "\U0001F600": 5, "\u0080": 6, "\u00f6": 7}
    assert common.canonical_json(keys).decode() == '{"\\r":2,"1":4,"\u0080":6,"\u00f6":7,"\u20ac":1,"\U0001F600":5,"\ufb33":3}'
    for value, text in [(1e21, "1e+21"), (1e20, "100000000000000000000"), (1e-7, "1e-7"), (1e-6, "0.000001"), (-0.0, "0"), (123.0, "123"), (0.1, "0.1"), (5e-324, "5e-324")]:
        assert common.canonical_json(value).decode() == text, value
    for bad in (float("nan"), float("inf"), 2 ** 60):
        with pytest.raises(ValueError):
            common.canonical_json(bad)
    assert common.canonical_digest({"b": 1, "a": [1.0]}) == common.canonical_digest({"a": [1], "b": 1.0})


def test_digest_fields_are_canonical_digests():
    policy = _policy()
    state = _admin_state(policy)
    assert state.attestation_policy_digest == common.canonical_digest(policy)
    with pytest.raises(ValidationError):
        admin.AdminState(**{**state.model_dump(), "attestation_policy_digest": D0})
    definition = rubrics.RubricDefinition(rubric_id="std", name="Standard", criteria=[])
    rubrics.RubricVersion(ref={"rubric_id": "std", "version": 1, "digest": common.canonical_digest(definition)}, definition=definition, status="active", published_at=NOW)
    with pytest.raises(ValidationError):
        rubrics.RubricVersion(ref={"rubric_id": "std", "version": 1, "digest": D0}, definition=definition, status="active", published_at=NOW)
    profile = _hardware()
    with pytest.raises(ValidationError):
        usage.HardwareProfileInput(**{**profile.model_dump(), "fingerprint": D0})
    event = _audit_event()
    with pytest.raises(ValidationError):
        events.AuditEvent(**{**event.model_dump(), "event_digest": D0})
    assert "RFC 8785" in api.build_openapi()["x-call1"]["canonical_json"]


# --- route table ----------------------------------------------------------------------------


def test_every_route_declares_principals_errors_and_lives_under_prefix():
    seen_ops: Set[str] = set()
    seen_paths: Set[tuple] = set()
    for route in api.ROUTES:
        assert route.principals, route.operation_id
        assert route.all_errors(), route.operation_id
        assert route.path.startswith(contracts.STORE_API_PREFIX + "/"), route.path
        assert route.operation_id not in seen_ops, route.operation_id
        seen_ops.add(route.operation_id)
        assert (route.method, route.path) not in seen_paths, route.path
        seen_paths.add((route.method, route.path))
        if route.request is not None:
            assert issubclass(route.request, common.ContractModel)
        if route.response is None:
            assert route.status_code == 204, route.operation_id
        for code in route.all_errors():
            assert code in errors.ERROR_HTTP_STATUS
    doc = api.build_openapi()
    for path, methods in doc["paths"].items():
        for method, op in methods.items():
            for key in ("x-call1-principals", "x-call1-idempotency", "x-call1-audited", "x-call1-errors"):
                assert key in op, f"{method} {path} lacks {key}"


def test_console_credential_is_never_a_store_principal():
    for route in api.ROUTES:
        assert all(p.kind is not common.PrincipalKind.CONSOLE_CREDENTIAL for p in route.principals), route.operation_id
    assert auth.ConsoleCredentialDescriptor.model_fields["store_rights"].default == "none"
    assert api.contract_extensions()["console_credential"]["store_principal"] is False


def test_console_cannot_set_provider_connections():
    ops = {o.value for o in auth.ConsoleOperation}
    assert "set_provider_connections" not in ops and "set_local_runtime" in ops
    assert "never changes" in auth.ConsoleOperation.__doc__


def test_review_and_admin_writes_require_reviewer_sessions():
    for route in api.ROUTES:
        session_only = any(seg in route.path for seg in ("/review-queue", "/review", "/verdicts/", "/escalation", "/rubrics/", "/auth/"))
        under_admin = "/admin/" in route.path
        if (session_only or under_admin) and route.method != "GET":
            allowed = {common.PrincipalKind.REVIEWER_SESSION} | ({common.PrincipalKind.ANONYMOUS} if "/auth/" in route.path else set())
            assert all(p.kind in allowed for p in route.principals), route.operation_id
        if under_admin:
            for p in route.principals:
                if p.kind is common.PrincipalKind.REVIEWER_SESSION:
                    assert p.min_role in (common.ReviewerRole.ADMIN, common.ReviewerRole.SUPERVISOR), route.operation_id
                else:
                    assert route.method == "GET" and p.scope is common.ServiceScope.ADMIN_STATE_READ, route.operation_id


def test_admin_state_and_trust_changes_are_audited_admin_actions():
    trust_paths = ("/admin/state/changes", "/admin/pro1/clear-block", "/admin/releases/approvals", "/admin/releases/pending/decline", "/admin/updates/packages/{package_id}/install")
    found = 0
    for route in api.ROUTES:
        if any(route.path.startswith(contracts.STORE_API_PREFIX + p) for p in trust_paths) and route.method != "GET":
            found += 1
            assert route.audited, route.operation_id
            assert route.idempotency is api.Idempotency.EXPECTED_VERSION, route.operation_id
            assert all(p.kind is common.PrincipalKind.REVIEWER_SESSION and p.min_role is common.ReviewerRole.ADMIN for p in route.principals), route.operation_id
    assert found >= 6
    assert admin.AdminState.model_fields["changed_only_by"].default == "audited_admin_action"
    for route in api.ROUTES:
        for p in route.principals:
            if p.kind is common.PrincipalKind.PROCESS_SERVICE_KEY:
                assert route.request is not admin.AdminStateChange
                assert not (route.path.startswith(contracts.STORE_API_PREFIX + "/admin/") and route.method != "GET"), route.operation_id


def test_admin_changes_carry_settings_only():
    for model in (admin.RouteOptIns, admin.CustomerLanOptIn, admin.Pro1OptIn, admin.ByokOptIn, admin.MaskingSettings, admin.KeyManagerConfig, release_trust.AttestationPolicySettings):
        for name in ("changed_at", "changed_by_account_id", "audit_event_id", "connection", "status"):
            assert name not in model.model_fields, f"{model.__name__}.{name}"
    assert "minimum_release_svn" not in release_trust.AttestationPolicySettings.model_fields
    with pytest.raises(ValidationError):
        admin.AdminStateChange(section="route_opt_ins", expected_state_version=1, reason="x", route_opt_ins={"call1_confidential": {"connection": {"status": "ready"}}})
    assert "connection" not in admin.Pro1OptIn.model_fields and "pro1_connection" not in admin.AdminState.model_fields


def test_expected_version_and_key_idempotency_are_backed_by_fields():
    for route in api.ROUTES:
        if route.idempotency is api.Idempotency.EXPECTED_VERSION:
            fields = route.request.model_fields
            assert any(name.startswith("expected_") for name in fields), route.operation_id
            assert any(code.value.endswith("_conflict") or code is E.CONFLICT for code in route.all_errors()), route.operation_id
        if route.idempotency is api.Idempotency.BODY_KEY:
            fields = route.request.model_fields
            assert "idempotency_key" in fields or "completion_key" in fields, route.operation_id
    doc = api.build_openapi()
    header_ops = [r for r in api.ROUTES if r.idempotency is api.Idempotency.HEADER]
    assert header_ops
    for route in header_ops:
        params = doc["paths"][route.path][route.method.lower()]["parameters"]
        assert any(p["in"] == "header" and p["name"] == "Idempotency-Key" and p["required"] for p in params), route.operation_id


def test_session_rules_only_name_permissions_the_role_holds():
    with pytest.raises(ValueError):
        api.session(common.ReviewerRole.REVIEWER, P.MANAGE_ADMIN_STATE)
    assert auth.ROLE_PERMISSIONS[common.ReviewerRole.REVIEWER] < auth.ROLE_PERMISSIONS[common.ReviewerRole.SUPERVISOR] < auth.ROLE_PERMISSIONS[common.ReviewerRole.ADMIN]


def test_every_scope_and_permission_is_used():
    scopes = {p.scope for r in api.ROUTES for p in r.principals if p.scope}
    assert scopes == set(common.ServiceScope)
    permissions = {p.permission for r in api.ROUTES for p in r.principals if p.permission} | {q for r in api.ROUTES for q in r.object_permissions}
    assert permissions == set(auth.Permission)
    resolve = api.routes_by_operation()["resolveReview"]
    assert P.RESOLVE_ANY_REVIEW in resolve.object_permissions
    assert api.build_openapi()["paths"][resolve.path]["post"]["x-call1-object-rule"]["permissions"] == ["resolve_any_review"]
    assert api.routes_by_operation()["listInvitations"].query is auth.InvitationListQuery


def test_process_routes_take_one_scope_and_act_as_their_installation():
    for route in api.ROUTES:
        for p in route.principals:
            if p.kind is common.PrincipalKind.PROCESS_SERVICE_KEY:
                assert p.scope is not None and p.min_role is None and p.permission is None, route.operation_id
        process_writes = [p for p in route.principals if p.kind is common.PrincipalKind.PROCESS_SERVICE_KEY]
        if process_writes and route.method != "GET":
            assert "{installation_id}" not in route.path, route.operation_id
    ops = api.routes_by_operation()
    assert ops["publishCatalogSnapshot"].path.endswith("/catalog-snapshot")
    for op in ("claimJobs", "completeJob", "failJob", "publishCatalogSnapshot"):
        assert "installation" in ops[op].object_rule, op
    assert "{log_id}" in ops["putLogScanState"].path and "installation" in ops["putLogScanState"].object_rule
    assert "installation_id" in release_trust.LogScanState.model_fields


def test_service_key_hashing_scopes_and_rotation():
    record = dict(id="key_2", installation_id="inst_1", label="primary", scopes=["jobs:claim"], key_prefix="c1sk_ab12CD", key_hash=D0, created_at=NOW)
    auth.ServiceKeyRecord(**record)
    auth.ServiceKeyRecord(**{**record, "superseded_by_key_id": "key_3", "grace_until": NOW + timedelta(hours=24)})
    with pytest.raises(ValidationError):
        auth.ServiceKeyRecord(**{**record, "grace_until": NOW})
    with pytest.raises(ValidationError):
        auth.ServiceKeyRecord(**{**record, "superseded_by_key_id": "key_3"})
    with pytest.raises(ValidationError):
        auth.ServiceKeyRecord(**{**record, "scopes": []})
    assert auth.ServiceKeyRecord.model_fields["hash_algorithm"].default == "sha256"
    with pytest.raises(ValidationError):
        auth.ServiceKeyRotate(grace_seconds=30 * 24 * 3600)
    ops = api.routes_by_operation()
    assert ops["rotateServiceKey"].audited and ops["rotateServiceKey"].response is auth.ServiceKeyIssued
    assert ops["createServiceKey"].request is auth.ServiceKeyCreate and E.NOT_FOUND in ops["createServiceKey"].errors
    assert ops["registerInstallation"].audited and ops["retireInstallation"].audited
    assert "token" not in auth.ServiceKeyRecord.model_fields


def test_session_routes_can_report_expiry_and_disablement():
    for route in api.ROUTES:
        if any(p.kind is common.PrincipalKind.REVIEWER_SESSION for p in route.principals):
            assert E.SESSION_EXPIRED in route.all_errors() and E.ACCOUNT_DISABLED in route.all_errors(), route.operation_id


def test_anonymous_status_carries_no_build_or_activity():
    ops = api.routes_by_operation()
    assert ops["getStatus"].principals == [api.ANONYMOUS] and ops["getStatus"].response is admin.StoreHealth
    assert not {"running_build", "admin_state_version", "latest_change_cursor", "tls", "feed_epoch"} & set(admin.StoreHealth.model_fields)
    assert api.ANONYMOUS not in ops["getStatusDetail"].principals


def test_change_feed_is_filtered_by_principal():
    byp = events.CHANGE_KINDS_BY_PRINCIPAL
    assert events.ChangeKind.REVIEW not in byp["process_service_key"] and events.ChangeKind.REVIEW_QUEUE not in byp["process_service_key"]
    assert events.ChangeKind.JOB not in byp["reviewer"] and events.ChangeKind.ADMIN_STATE not in byp["reviewer"]
    assert set(byp["admin"]) == set(events.ChangeKind)
    assert "CHANGE_KINDS_BY_PRINCIPAL" in api.routes_by_operation()["listChanges"].object_rule


def test_updater_verdicts_have_no_write_route():
    for route in api.ROUTES:
        assert route.request not in (release_trust.UpdaterVerification, release_trust.UpdaterVerificationRecord), route.operation_id
    ops = api.routes_by_operation()
    for op in ("stageUpdatePackage", "commitUpdatePackage", "installUpdatePackage"):
        assert all(p.kind is common.PrincipalKind.REVIEWER_SESSION for p in ops[op].principals), op
    assert "installed" in release_trust.UpdaterVerification.__doc__ and "evidence only" in release_trust.UpdaterVerification.__doc__


# --- queue separation --------------------------------------------------------------------


JOB_QUEUE_MODELS = {"Job", "JobDefinition", "JobGraphRequest", "JobGraph", "ClaimRequest", "ClaimedJob", "ClaimResponse", "HeartbeatRequest", "HeartbeatResponse", "CompletionRequest", "CompletionReceipt", "FailureRequest", "FailureReceipt", "LeaseInfo", "Attempt", "WorkerCapabilities", "JobReleaseRequest", "JobReleaseReceipt"}
REVIEW_QUEUE_MODELS = {"ReviewQueueItem", "ReviewQueueRule", "ReviewQueueRuleRecord", "ReviewQueueQuery", "ReviewQueueStats", "AssignRequest", "ClaimNextResponse", "StartReviewRequest", "ReleaseRequest", "ResolveRequest"}


def _schema_refs(model) -> Set[str]:
    return set(model.model_json_schema().get("$defs", {}))


def test_job_queue_and_review_queue_share_no_type_status_or_endpoint():
    assert not {s.value for s in jobs.JobStatus} & {s.value for s in reviews.ReviewQueueStatus}
    all_models = contracts.all_models()
    for name in REVIEW_QUEUE_MODELS:
        assert not _schema_refs(all_models[name]) & JOB_QUEUE_MODELS, name
    for name in JOB_QUEUE_MODELS:
        assert not _schema_refs(all_models[name]) & REVIEW_QUEUE_MODELS, name
    for route in api.ROUTES:
        assert not ("/jobs" in route.path and "review" in route.path), route.path
        if "/review-queue" in route.path:
            assert all(p.kind is common.PrincipalKind.REVIEWER_SESSION for p in route.principals), route.operation_id
            assert route.request not in {all_models[n] for n in JOB_QUEUE_MODELS}
        if route.path.startswith(contracts.STORE_API_PREFIX + "/jobs") and route.method != "GET":
            assert all(p.kind is common.PrincipalKind.PROCESS_SERVICE_KEY for p in route.principals), route.operation_id
    assert {t.trigger.value for t in jobs.JOB_TRANSITIONS}.isdisjoint({t.trigger.value for t in reviews.REVIEW_QUEUE_TRANSITIONS})


# --- secrets and location neutrality ------------------------------------------------------


_SECRET_LIKE = re.compile(r"(^|_)(secret|password|passwd|token|api_key|access_key|secret_key|private_key|bearer|session_key|cookie|setup_code)($|_)")
_NON_SECRET_SUFFIXES = ("_hash", "_digest", "_env_name", "_prefix", "_id", "_ref", "_hash_algorithm", "_seconds", "_limit", "page_token")
_CAPABILITY_FIELDS = {"claim_token", "reanalysis_claim_token"}  # per-claim capabilities, sent in flight; stored records keep hashes only


def test_no_field_carries_a_secret_value():
    offenders = []
    for name, model in contracts.all_models().items():
        for field_name in model.model_fields:
            if not _SECRET_LIKE.search(field_name):
                continue
            if field_name.endswith(_NON_SECRET_SUFFIXES) or field_name in _CAPABILITY_FIELDS:
                continue
            if (name, field_name) in auth.ONE_TIME_SECRET_FIELDS | auth.SESSION_BOUND_SECRET_FIELDS:
                continue
            offenders.append(f"{name}.{field_name}")
    assert not offenders, offenders
    secret_models = {m for m, _ in auth.ONE_TIME_SECRET_FIELDS | auth.SESSION_BOUND_SECRET_FIELDS}
    carriers = auth.ONE_TIME_CARRIER_MODELS
    assert secret_models <= carriers
    for name, model in contracts.all_models().items():
        if name in carriers:
            continue
        assert not _schema_refs(model) & carriers, name
    for name in carriers - {"SessionInfo"}:
        used = [r for r in api.ROUTES if r.request is getattr(auth, name) or r.response is getattr(auth, name)]
        assert used, name
    session_info_routes = {r.operation_id for r in api.ROUTES if r.response in (auth.SessionInfo, auth.SignInResponse, auth.RegistrationFinishResponse)}
    assert session_info_routes == {"getSession", "signInFinish", "enrollFinish", "addAuthenticatorFinish"}
    assert "csrf_token" not in auth.SessionListItem.model_fields
    for model in (jobs.Job, jobs.LeaseInfo, jobs.Attempt):
        assert "claim_token" not in model.model_fields
    for field_name in ("credential_env_name", "anchor_credential_env_name"):
        assert field_name in admin.KeyManagerConfig.model_fields or field_name in custody.RouteRecord.model_fields


_LOCATION_LIKE = re.compile(r"(^|_)(path|dir|directory|filename|file_path|mount|socket|local_path|volume)($|_)")
_LOCATION_ALLOWED = {"SessionCookieSpec.path", "TlsState.trust_path", "BackupManifest.trust_path", "ContractParameters.max_fields_per_path"}  # a category + subcategory path of the signal taxonomy (1.3.0), not a file path


def test_no_field_assumes_shared_files_or_colocation():
    offenders = []
    for name, model in contracts.all_models().items():
        for field_name, info in model.model_fields.items():
            if _LOCATION_LIKE.search(field_name) and f"{name}.{field_name}" not in _LOCATION_ALLOWED:
                offenders.append(f"{name}.{field_name}")
            if info.description and re.search(r"\bfile ?path\b|\bshared (folder|filesystem)\b|\blocalhost\b", info.description or "", re.I) and name not in ("RouteRecord",):
                offenders.append(f"{name}.{field_name} description")
    assert not offenders, offenders
    assert {"id", "checksum", "storage"} <= set(artifacts.Artifact.model_fields)
    assert not any(k in artifacts.Artifact.model_fields for k in ("object_key", "url", "bucket"))
    for model in (artifacts.UploadGrant, artifacts.ContentGrant):
        assert "expires_at" in model.model_fields
        with pytest.raises(ValidationError):
            model.model_validate({**_upload_grant().model_dump(), "url": "http://store.example.com/x"} if model is artifacts.UploadGrant else {**_content_grant().model_dump(), "url": "http://x"})
    assert not any("process" in r.path.lower() for r in api.ROUTES)
    assert "object_store_kind" in admin.StoreStatus.model_fields and "object_store_location" not in admin.StoreStatus.model_fields


# --- status set and transitions -----------------------------------------------------------


def test_status_set_reserves_waiting_provider():
    assert jobs.JobStatus.WAITING_PROVIDER in jobs.JobStatus
    assert {s.value for s in jobs.JobStatus} == {"BLOCKED", "QUEUED", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED", "WAITING_PROVIDER"}
    active = [t for t in jobs.JOB_TRANSITIONS if not t.reserved]
    assert all(t.to_status is not jobs.JobStatus.WAITING_PROVIDER and t.from_status is not jobs.JobStatus.WAITING_PROVIDER for t in active)
    assert any(t.reserved and t.to_status is jobs.JobStatus.WAITING_PROVIDER for t in jobs.JOB_TRANSITIONS)
    for status in jobs.JobStatus:
        if status in jobs.TERMINAL_STATUSES or status in jobs.RESERVED_STATUSES:
            continue
        assert any(t.from_status is status for t in active), status
    reachable = {jobs.JobStatus.BLOCKED, jobs.JobStatus.QUEUED} | {t.to_status for t in active}
    assert reachable >= (set(jobs.JobStatus) - jobs.RESERVED_STATUSES)
    assert {t.from_status for t in active if t.from_status in jobs.TERMINAL_STATUSES} == {jobs.JobStatus.FAILED}
    assert all(t.trigger is jobs.JobTrigger.MANUAL_RETRY for t in active if t.from_status is jobs.JobStatus.FAILED)


def _edge(frm, to, trigger):
    return [t for t in jobs.JOB_TRANSITIONS if t.from_status.value == frm and t.to_status.value == to and t.trigger.value == trigger]


def test_every_described_transition_is_in_the_table():
    S, PR, O = jobs.TransitionActor.STORE, jobs.TransitionActor.PROCESS, jobs.TransitionActor.OPERATOR
    for frm, to, trigger, actor in [
        ("QUEUED", "FAILED", "admission_rejected", S),
        ("RUNNING", "QUEUED", "release_requeue", PR),
        ("RUNNING", "FAILED", "release_reject", PR),
        ("RUNNING", "CANCELLED", "cancel_acknowledged", PR),
        ("RUNNING", "CANCELLED", "lease_expired_after_cancel", S),
        ("RUNNING", "QUEUED", "lease_expired_retryable", S),
        ("RUNNING", "FAILED", "lease_exhausted", S),
        ("BLOCKED", "QUEUED", "dependencies_satisfied", S),
        ("FAILED", "QUEUED", "manual_retry", O),
    ]:
        found = _edge(frm, to, trigger)
        assert len(found) == 1 and found[0].actor is actor, (frm, to, trigger)
    assert [t.trigger for t in jobs.JOB_TRANSITIONS if t.consumes_attempt] == [jobs.JobTrigger.CLAIM]
    assert not _edge("QUEUED", "BLOCKED", "dependency_added")
    assert "BLOCKED" in jobs.NewDependency.__doc__


def test_review_queue_transitions_are_closed():
    statuses = {t.from_status for t in reviews.REVIEW_QUEUE_TRANSITIONS} | {t.to_status for t in reviews.REVIEW_QUEUE_TRANSITIONS}
    assert statuses == set(reviews.ReviewQueueStatus)
    final = (reviews.ReviewQueueStatus.APPROVED, reviews.ReviewQueueStatus.OVERRIDDEN, reviews.ReviewQueueStatus.SUPERSEDED)
    assert not any(t.from_status in final for t in reviews.REVIEW_QUEUE_TRANSITIONS)
    assert {t.from_status for t in reviews.REVIEW_QUEUE_TRANSITIONS if t.to_status is reviews.ReviewQueueStatus.SUPERSEDED} == {reviews.ReviewQueueStatus.PENDING, reviews.ReviewQueueStatus.IN_REVIEW}
    for model in (reviews.ResolveRequest, reviews.VerdictOverride, reviews.EscalationResolution):
        assert "current" in model.model_fields["evaluation_version"].description, model.__name__
    item = dict(id="rvw_1", call_id="call_1", conversation_id="conv_1", rule_id="triage", rule_name="Triage", stream="TRIAGE", reason="r", urgency_score=10, evaluation_version=2, item_version=2, created_at=NOW)
    reviews.ReviewQueueItem(**item, status="SUPERSEDED", stale=True, superseded_by_item_id="rvw_2")
    with pytest.raises(ValidationError):
        reviews.ReviewQueueItem(**item, status="SUPERSEDED", stale=True)
    with pytest.raises(ValidationError):
        reviews.ReviewQueueItem(**item, status="PENDING", stale=True, superseded_by_item_id="rvw_2")


def test_job_error_codes_are_classified_and_pro1_definitive_codes_never_escalate():
    assert set(errors.JOB_ERROR_CLASSES) == set(errors.JobErrorCode)
    for code in errors.PRO1_ATTESTATION_CLASS_CODES:
        assert code.value.startswith("pro1_")
    assert errors.JobErrorCode.PRO1_UNREACHABLE not in errors.PRO1_ATTESTATION_CLASS_CODES
    assert errors.JobErrorCode.PRO1_SERVICE_ERROR not in errors.PRO1_ATTESTATION_CLASS_CODES
    assert not errors.PROVIDER_FAILURE_CODES & errors.PRO1_ATTESTATION_CLASS_CODES
    assert errors.JOB_ERROR_CLASSES[errors.JobErrorCode.PRO1_ATTESTATION_INVALID] is errors.JobErrorClass.DEFINITIVE
    assert set(errors.ERROR_HTTP_STATUS) == set(E)


def test_pro1_connection_has_one_writer_and_a_sticky_block():
    T = custody.PRO1_CONNECTION_TRANSITIONS
    S = custody.Pro1ConnectionStatus
    assert {t.trigger for t in T if t.from_status is S.BLOCKED} == {custody.Pro1ConnectionTrigger.BLOCK_CLEARED}
    assert {t.to_status for t in T if t.from_status is S.BLOCKED} == {S.PENDING_VERIFICATION}
    assert {(t.from_status, t.trigger) for t in T if t.to_status is S.READY} == {(S.PENDING_VERIFICATION, custody.Pro1ConnectionTrigger.VERIFICATION_PASSED)}
    assert all(t.audited for t in T)
    custody.Pro1Connection(status="blocked", connection_version=3, blocked_reason="pro1_attestation_invalid", blocked_since=NOW, updated_at=NOW)
    with pytest.raises(ValidationError):
        custody.Pro1Connection(status="blocked", connection_version=3, updated_at=NOW)
    with pytest.raises(ValidationError):
        custody.Pro1Connection(status="blocked", connection_version=3, blocked_reason="pro1_unreachable", blocked_since=NOW, updated_at=NOW)
    with pytest.raises(ValidationError):
        custody.Pro1VerificationReport(outcome="failed", policy_version=D0, verified_at=NOW)
    with pytest.raises(ValidationError):
        custody.Pro1VerificationReport(outcome="passed", policy_version=D0, verified_at=NOW)
    custody.Pro1VerificationReport(outcome="passed", policy_version=D0, evidence_digest=D1, release_id="r1", manifest_digest=D2, verified_at=NOW)
    ops = api.routes_by_operation()
    assert ops["reportPro1Verification"].request is custody.Pro1VerificationReport
    assert not any(r.request is custody.Pro1Connection for r in api.ROUTES)
    assert ops["clearPro1Block"].request is custody.Pro1BlockClear and "expected_connection_version" in custody.Pro1BlockClear.model_fields
    assert "in this transaction" in jobs.FailureReceipt.model_fields["pro1_connection_blocked"].description


def test_reanalysis_requests_have_a_state_machine_and_one_graph():
    T = jobs.REANALYSIS_TRANSITIONS
    assert {(t.from_status.value, t.to_status.value) for t in T} == {("pending", "claimed"), ("claimed", "pending"), ("claimed", "fulfilled"), ("claimed", "rejected")}
    assert not any(r.path.endswith("/fulfil") for r in api.ROUTES)
    good = _job_def("asr", "asr", inputs=[{"role": "audio", "artifact": {"artifact_id": "art_a", "checksum": D0}}], selection=_selection("asr", "transcript.v1").model_dump())
    jobs.JobGraphRequest(idempotency_key="conv1:reanalysis:1", reason="reanalysis", reanalysis_request_id="rq_1", reanalysis_claim_token=TOKEN, jobs=[good])
    with pytest.raises(ValidationError):
        jobs.JobGraphRequest(idempotency_key="conv1:reanalysis:1", reason="reanalysis", reanalysis_request_id="rq_1", jobs=[good])
    with pytest.raises(ValidationError):
        jobs.JobGraphRequest(idempotency_key="conv1:ingest:1", reason="ingest", reanalysis_claim_token=TOKEN, jobs=[good])
    assert E.CLAIM_TOKEN_STALE in api.routes_by_operation()["createJobGraph"].errors


# --- data custody --------------------------------------------------------------------------


def _route(route_class: custody.RouteClass, provider: custody.ProviderType, host: str, **kw) -> custody.RouteRecord:
    if route_class is not custody.RouteClass.APPLIANCE:
        kw.setdefault("provider_connection_ref", "pro1" if route_class is custody.RouteClass.CALL1_CONFIDENTIAL else "conn_1")
    return custody.RouteRecord(route_class=route_class, provider_type=provider, destination_host=host, masked=True, **kw)


def test_route_record_enforces_route_classes():
    _route(custody.RouteClass.APPLIANCE, custody.ProviderType.MLX, "in-process")
    _route(custody.RouteClass.APPLIANCE, custody.ProviderType.OLLAMA, "127.0.0.1")
    _route(custody.RouteClass.CUSTOMER_LAN, custody.ProviderType.OLLAMA, "ollama.corp.example")
    _route(custody.RouteClass.CUSTOMER_LAN, custody.ProviderType.OLLAMA, "10.0.0.5")
    _route(custody.RouteClass.CUSTOMER_DIRECTED, custody.ProviderType.BYOK, "api.openai.com")
    _route(custody.RouteClass.CALL1_CONFIDENTIAL, custody.ProviderType.PRO1, "pro1.call1.cc", attestation_policy_version=D0)
    for bad in (
        dict(route_class=custody.RouteClass.APPLIANCE, provider=custody.ProviderType.OLLAMA, host="ollama.corp.example"),
        dict(route_class=custody.RouteClass.APPLIANCE, provider=custody.ProviderType.MLX, host="127.0.0.1"),
        dict(route_class=custody.RouteClass.CUSTOMER_LAN, provider=custody.ProviderType.OLLAMA, host="127.0.0.1"),
        dict(route_class=custody.RouteClass.CUSTOMER_DIRECTED, provider=custody.ProviderType.BYOK, host="api.call1.cc"),
        dict(route_class=custody.RouteClass.CUSTOMER_DIRECTED, provider=custody.ProviderType.BYOK, host="8.8.8.8"),
        dict(route_class=custody.RouteClass.CUSTOMER_DIRECTED, provider=custody.ProviderType.BYOK, host="2001:4860:4860::8888"),
        dict(route_class=custody.RouteClass.CUSTOMER_LAN, provider=custody.ProviderType.OLLAMA, host="[2001:4860:4860::8888]"),
        dict(route_class=custody.RouteClass.CUSTOMER_LAN, provider=custody.ProviderType.BYOK, host="10.0.0.5"),
        dict(route_class=custody.RouteClass.CALL1_CONFIDENTIAL, provider=custody.ProviderType.PRO1, host="pro1.call1.cc"),
        dict(route_class=custody.RouteClass.CALL1_CONFIDENTIAL, provider=custody.ProviderType.BYOK, host="pro1.call1.cc"),
    ):
        with pytest.raises(ValidationError):
            _route(bad["route_class"], bad["provider"], bad["host"])
    with pytest.raises(ValidationError):
        custody.RouteRecord(route_class=custody.RouteClass.CUSTOMER_DIRECTED, provider_type=custody.ProviderType.BYOK, destination_host="api.openai.com", masked=True)
    with pytest.raises(ValidationError):
        _route(custody.RouteClass.CUSTOMER_DIRECTED, custody.ProviderType.BYOK, "api.openai.com", provider_connection_ref="pro1")
    with pytest.raises(ValidationError):
        _route(custody.RouteClass.CALL1_CONFIDENTIAL, custody.ProviderType.PRO1, "pro1.call1.cc", attestation_policy_version=D0, provider_connection_ref="conn_1")
    with pytest.raises(ValidationError):
        custody.RouteRecord(route_class=custody.RouteClass.CUSTOMER_DIRECTED, provider_type=custody.ProviderType.BYOK, destination_host="api.openai.com", provider_connection_ref="conn_1", masked=False)
    custody.RouteRecord(route_class=custody.RouteClass.CUSTOMER_DIRECTED, provider_type=custody.ProviderType.BYOK, destination_host="api.openai.com", provider_connection_ref="conn_1", masked=False, masking_override_audit_event_id="evt_1")


PRO1_ENDPOINT_HOST = "front.pro1-host.net"
STAGE0_HOSTS = [
    "call1.cc", "API.CALL1.CC.", "sub.call1.cc", "notcall1.cc", "api.openai.com", "", "10.0.0.5", "8.8.8.8",
    "127.0.0.1", "169.254.1.1", "fe80::1", "::1", "fd00::5", "2001:4860:4860::8888", "2606:4700::1111", "::ffff:8.8.8.8",
    PRO1_ENDPOINT_HOST, "x." + PRO1_ENDPOINT_HOST, "pro1-host.net",
]


@pytest.mark.parametrize("pro1_endpoint", [None, "https://" + PRO1_ENDPOINT_HOST])
@pytest.mark.parametrize("host", STAGE0_HOSTS)
def test_route_host_rule_is_the_stage0_guard(host, pro1_endpoint, monkeypatch):
    """The contract's pure rule and the Stage 0 guard agree on the same host, IPv6 included."""
    from call1.models.schemas import QuestionModel
    from call1.question_models import is_call1_operated

    if pro1_endpoint:
        monkeypatch.setenv("CALL1_PRO1_ENDPOINT", pro1_endpoint)
    else:
        monkeypatch.delenv("CALL1_PRO1_ENDPOINT", raising=False)
    url_host = f"[{host}]" if ":" in host else host
    if host:
        legacy = QuestionModel(id="probe", name="probe", source="byok", model="m", endpoint=f"https://{url_host}")
    else:  # QuestionModel refuses a hostless endpoint at validation; the guard still sees one.
        legacy = SimpleNamespace(local_model_id=None, source="byok", endpoint="https://")
    pro1_hosts = [PRO1_ENDPOINT_HOST] if pro1_endpoint else []
    expected = is_call1_operated(legacy)
    assert custody.call1_operated_host(host, pro1_hosts) is expected
    assert custody.call1_operated_host(url_host, pro1_hosts) is expected
    assert custody.call1_operated_host(custody.host_of(f"https://{url_host}:8443/v1"), pro1_hosts) is expected


def test_host_rule_is_environment_independent(monkeypatch):
    monkeypatch.setenv("CALL1_PRO1_ENDPOINT", "https://" + PRO1_ENDPOINT_HOST)
    assert custody.call1_operated_host(PRO1_ENDPOINT_HOST) is False
    assert custody.call1_operated_host(PRO1_ENDPOINT_HOST, [PRO1_ENDPOINT_HOST]) is True
    import call1.contracts.custody as module
    assert "os" not in vars(module) and "question_models" not in open(module.__file__).read().split('"""', 2)[2]
    assert api.build_openapi()["x-call1"]["call1_operated_domains"] == ["call1.cc"]


def _attestation(**overrides) -> custody.Pro1AttestationRecord:
    base = dict(evidence_digest=D0, evidence_artifact_id="art_ev", policy_version=D1, trust_anchor_bundle_digest=D2, approval_id="apr_1", release_id="pro1-2026.09", manifest_digest=D2, release_svn=3, log_id="call1-log", log_index=42, checkpoint_digest=D1, revocation_list_sequence=7, platform="azure-snp-paravisor", gpus=[{"model": "H100", "driver_version": "550.90", "vbios_version": "96.00", "confidential_mode": True}], session_id="sess_p1", key_id="k-one-way", key_release_ref="kr_1", model_id="gemma-4-e4b", weights_digest=D0, verified_at=NOW, expires_at=NOW + timedelta(minutes=10))
    return custody.Pro1AttestationRecord(**{**base, **overrides})


def test_pro1_attempts_record_evidence_whether_they_succeed_or_fail():
    pro1 = _route(custody.RouteClass.CALL1_CONFIDENTIAL, custody.ProviderType.PRO1, "pro1.call1.cc", attestation_policy_version=D1)
    base = dict(worker_id="w1", installation_id="inst_1", adapter_id="pro1", adapter_version="1", model_revision=D0, route=pro1)
    with pytest.raises(ValidationError):
        jobs.AttemptProvenance(**base)
    with pytest.raises(ValidationError):
        jobs.AttemptProvenance(**base, attestation=_attestation())
    with pytest.raises(ValidationError):
        jobs.AttemptProvenance(**base, attestation=_attestation(), key_release_ref="kr_other")
    ok = jobs.AttemptProvenance(**base, attestation=_attestation(), key_release_ref="kr_1")
    assert ok.attestation.manifest_digest == D2
    rejected = custody.Pro1AttemptFailureEvidence(failed_check="measurements", error_code="pro1_measurement_rejected", policy_version=D1, evidence_digest=D0, evidence_artifact_id="art_ev", release_id="pro1-2026.10")
    failed = jobs.AttemptProvenance(**base, pro1_failure=rejected)
    jobs.FailureRequest(claim_token=TOKEN, completion_key="job_1:1:fail", error_code="pro1_measurement_rejected", usage=_usage(outcome="failed", error_code="pro1_measurement_rejected"), provenance=failed)
    after_release = jobs.AttemptProvenance(**base, attestation=_attestation(), key_release_ref="kr_1", pro1_failure=custody.Pro1AttemptFailureEvidence(failed_check="response", error_code="pro1_response_rejected", policy_version=D1))
    with pytest.raises(ValidationError):
        jobs.CompletionRequest(claim_token=TOKEN, completion_key="job_1:1:c", outputs=[{"role": "assessment", "artifact_id": "a", "checksum": D0}], usage=_usage(), provenance=after_release)
    with pytest.raises(ValidationError):
        jobs.CompletionRequest(claim_token=TOKEN, completion_key="job_1:1:c", outputs=[{"role": "assessment", "artifact_id": "a", "checksum": D0}], usage=_usage(), provenance=failed)
    with pytest.raises(ValidationError):
        custody.Pro1AttemptFailureEvidence(failed_check="measurements", error_code="provider_error", policy_version=D1)
    local = _route(custody.RouteClass.APPLIANCE, custody.ProviderType.MLX, "in-process")
    for extra in (dict(attestation=_attestation(), key_release_ref="kr_1"), dict(pro1_failure=rejected)):
        with pytest.raises(ValidationError):
            jobs.AttemptProvenance(**{**base, "route": local}, **extra)
    with pytest.raises(ValidationError):
        catalog.FrozenSelection(catalog_entry={"entry_id": "pro1-gemma", "entry_version": 1}, purpose="semantic_qa", model_family="gemma", model_revision="r", adapter_id="a", adapter_version="1", output_contract="qa_assessment.v1", provider_model_id="gemma-4-e4b", route=pro1)
    rule = custody.ROUTE_CLASS_RULES[custody.RouteClass.CALL1_CONFIDENTIAL]
    assert rule.requires_attestation and rule.requires_key_release and not rule.default_enabled
    key_release_routes = [r for r in api.ROUTES if r.request is custody.KeyReleaseInput]
    assert key_release_routes and key_release_routes[0].audited


def test_pro1_provider_failures_complete_flagged_with_their_evidence():
    """Pro1ConfidentialInference.md section 4, 'Escalation': pro1_unreachable and pro1_service_error
    still fire a configured escalation, so a final-attempt QA assessment completes FLAGGED."""
    pro1 = _route(custody.RouteClass.CALL1_CONFIDENTIAL, custody.ProviderType.PRO1, "pro1.call1.cc", attestation_policy_version=D1)
    base = dict(worker_id="w1", installation_id="inst_1", adapter_id="pro1", adapter_version="1", model_revision=D0, route=pro1)
    outputs = [{"role": "assessment", "artifact_id": "art_o", "checksum": D2}, {"role": "prompt_input", "artifact_id": "art_p", "checksum": D1}]
    complete = lambda prov, code: jobs.CompletionRequest(claim_token=TOKEN, completion_key="job_1:3:c", outputs=outputs, usage=_usage(outcome="failed", error_code=code), provenance=prov)  # noqa: E731
    evidence = lambda check, code: custody.Pro1AttemptFailureEvidence(failed_check=check, error_code=code, policy_version=D1)  # noqa: E731

    unreachable = jobs.AttemptProvenance(**base, pro1_failure=evidence("evidence_request", "pro1_unreachable"))
    complete(unreachable, "pro1_unreachable")
    service = jobs.AttemptProvenance(**base, attestation=_attestation(), key_release_ref="kr_1", pro1_failure=evidence("service", "pro1_service_error"))
    complete(service, "pro1_service_error")
    with pytest.raises(ValidationError, match="sealed response"):
        complete(jobs.AttemptProvenance(**base, pro1_failure=evidence("service", "pro1_service_error")), "pro1_service_error")
    with pytest.raises(ValidationError, match="same|usage row"):
        complete(unreachable, "pro1_service_error")
    with pytest.raises(ValidationError, match="usage row"):
        complete(jobs.AttemptProvenance(**base, attestation=_attestation(), key_release_ref="kr_1"), "pro1_service_error")
    with pytest.raises(ValidationError, match="pro1_unreachable or pro1_service_error"):
        complete(unreachable, "provider_timeout")
    with pytest.raises(ValidationError, match="PROVIDER_FAILURE_CODES"):
        complete(jobs.AttemptProvenance(**base, pro1_failure=evidence("measurements", "pro1_measurement_rejected")), "pro1_measurement_rejected")
    local = jobs.AttemptProvenance(**{**base, "route": _route(custody.RouteClass.APPLIANCE, custody.ProviderType.MLX, "in-process")})
    complete(local, "provider_timeout")
    with pytest.raises(ValidationError, match="call1_confidential route"):
        complete(local, "pro1_unreachable")
    with pytest.raises(ValidationError, match="PROVIDER_FAILURE_CODES"):
        complete(local, "configuration_error")

    assert errors.PRO1_PROVIDER_FAILURE_CODES == {errors.JobErrorCode.PRO1_UNREACHABLE, errors.JobErrorCode.PRO1_SERVICE_ERROR}
    assert not errors.PROVIDER_FAILURE_CODES & errors.PRO1_ATTESTATION_CLASS_CODES
    assert api.contract_extensions()["pro1_provider_failure_codes"] == ["pro1_service_error", "pro1_unreachable"]


def test_pro1_failure_evidence_matches_the_failure_and_covers_interruptions():
    pro1 = _route(custody.RouteClass.CALL1_CONFIDENTIAL, custody.ProviderType.PRO1, "pro1.call1.cc", attestation_policy_version=D1)
    base = dict(worker_id="w1", installation_id="inst_1", adapter_id="pro1", adapter_version="1", model_revision=D0, route=pro1)
    fail = lambda code, prov: jobs.FailureRequest(claim_token=TOKEN, completion_key="job_1:1:fail", error_code=code, usage=_usage(outcome=usage.usage_outcome_for(errors.JobErrorCode(code)).value, error_code=code), provenance=prov)  # noqa: E731

    # Cancelled or crashed before verification: the evidence says so, with the same code.
    for code in ("cancelled", "worker_crashed"):
        fail(code, jobs.AttemptProvenance(**base, pro1_failure=custody.Pro1AttemptFailureEvidence(failed_check="interrupted", error_code=code, policy_version=D1)))
    # After verification the attestation record is enough.
    fail("cancelled", jobs.AttemptProvenance(**base, attestation=_attestation(), key_release_ref="kr_1"))
    with pytest.raises(ValidationError, match="interrupted"):
        custody.Pro1AttemptFailureEvidence(failed_check="interrupted", error_code="pro1_unreachable", policy_version=D1)
    with pytest.raises(ValidationError, match="interrupted"):
        custody.Pro1AttemptFailureEvidence(failed_check="measurements", error_code="cancelled", policy_version=D1)
    with pytest.raises(ValidationError, match="interruption"):
        custody.Pro1AttemptFailureEvidence(failed_check="interrupted", error_code="validation_rejected", policy_version=D1)
    measured = jobs.AttemptProvenance(**base, pro1_failure=custody.Pro1AttemptFailureEvidence(failed_check="measurements", error_code="pro1_measurement_rejected", policy_version=D1))
    with pytest.raises(ValidationError, match="failure's error code"):
        fail("pro1_attestation_invalid", measured)
    with pytest.raises(ValidationError, match="carries its pro1_failure"):
        fail("pro1_unreachable", jobs.AttemptProvenance(**base, attestation=_attestation(), key_release_ref="kr_1"))
    with pytest.raises(ValidationError, match="carries its pro1_failure"):
        fail("pro1_unreachable", None)
    assert not any(c.value.startswith("pro1_") for c in custody.PRO1_INTERRUPTION_CODES)
    assert errors.JobErrorCode.LEASE_EXPIRED not in custody.PRO1_INTERRUPTION_CODES


def test_attestation_evidence_is_global_and_readable_by_auditors():
    assert artifacts.ArtifactKind.ATTESTATION_EVIDENCE in artifacts.GLOBAL_ARTIFACT_KINDS
    ev = dict(id="art_ev", conversation_id=None, linked=True, version=1, storage="object", committed_at=NOW, kind="attestation_evidence", content_type="application/octet-stream", size_bytes=10, checksum=D0, content_contract="attestation_evidence.v1", sensitivity="non_customer")
    artifacts.Artifact(**ev)
    with pytest.raises(ValidationError):
        artifacts.Artifact(**{**ev, "conversation_id": "conv_1"})
    ops = api.routes_by_operation()
    assert any(p.kind is common.PrincipalKind.REVIEWER_SESSION and p.permission is P.READ_AUDIT for p in ops["getAttestationEvidenceContent"].principals)
    assert ops["uploadAttestationEvidence"].principals[0].scope is common.ServiceScope.KEY_RELEASE_WRITE
    with pytest.raises(ValidationError):
        artifacts.InlineArtifactCreate(kind="attestation_evidence", content_type="application/json", size_bytes=2, checksum=D0, content_contract="attestation_evidence.v1", sensitivity="non_customer", payload={})


def test_key_source_and_key_manager_never_hold_values():
    custody.KeySourceRef(kind="local")
    with pytest.raises(ValidationError):
        custody.KeySourceRef(kind="customer_anchor")
    anchor = custody.KeySourceRef(kind="customer_anchor", anchor_provider="aws_kms", anchor_key_ref="arn:aws:kms:us-east-1:1:key/abc")
    with pytest.raises(ValidationError):
        admin.KeyManagerConfig(key_source=anchor)
    admin.KeyManagerConfig(key_source=anchor, anchor_credential_env_name="CALL1_KEY_ANCHOR_CREDENTIAL")
    with pytest.raises(ValidationError):
        admin.KeyManagerConfig(key_source=anchor, anchor_credential_env_name="AWS_SECRET")


def test_key_release_preconditions_are_listed_with_errors():
    reasons = [p.reason for p in custody.KEY_RELEASE_PRECONDITIONS]
    assert reasons[:2] == ["route_disabled", "pro1_connection_blocked"]
    assert {"approval_not_active", "approval_mismatch", "policy_version_stale", "key_source_mismatch", "evidence_not_stored"} <= set(reasons)
    route = api.routes_by_operation()["recordKeyRelease"]
    assert {p.error for p in custody.KEY_RELEASE_PRECONDITIONS} <= set(route.all_errors())
    assert len(api.build_openapi()["x-call1"]["key_release_preconditions"]) == len(reasons)


def test_admin_endpoints_are_full_urls_and_reject_call1_hosts():
    byok = admin.ByokProvider(connection_ref="conn_1", label="OpenAI", base_url="https://api.openai.com/v1/", credential_env_name="CALL1_MODEL_KEY_OPENAI")
    assert byok.base_url == "https://api.openai.com/v1" and byok.destination_host == "api.openai.com"
    lan = admin.CustomerLanHost(connection_ref="conn_2", base_url="https://ollama.corp.example:8443/v1", label="lan")
    assert lan.host == "ollama.corp.example"
    admin.CustomerLanHost(connection_ref="conn_2", base_url="https://[fd00::5]:8443/v1", label="lan")
    for bad in ("http://api.openai.com/v1", "https://proxy.call1.cc/v1", "https://8.8.8.8/v1", "https://[2001:4860:4860::8888]/v1", "https://127.0.0.1:11434/v1", "https://user:pw@api.openai.com/v1", "https://"):
        with pytest.raises(ValidationError):
            admin.ByokProvider(connection_ref="conn_1", label="x", base_url=bad, credential_env_name="CALL1_MODEL_KEY_X")
    with pytest.raises(ValidationError):
        admin.CustomerLanHost(connection_ref="conn_2", base_url="https://[2606:4700::1111]/v1", label="lan")
    with pytest.raises(ValidationError):
        admin.RouteOptIns(call1_confidential={"enabled": True, "endpoint_base_url": "https://" + PRO1_ENDPOINT_HOST}, customer_directed={"providers": [{"connection_ref": "conn_1", "label": "x", "base_url": "https://api." + PRO1_ENDPOINT_HOST + "/v1", "credential_env_name": "CALL1_MODEL_KEY_X"}]})
    with pytest.raises(ValidationError):
        admin.Pro1OptIn(enabled=True)
    with pytest.raises(ValidationError):
        admin.RouteOptIns(customer_directed={"providers": [byok.model_dump(), {**byok.model_dump(), "label": "dup"}]})
    for model in (admin.ByokProvider, admin.CustomerLanHost, admin.Pro1OptIn):
        assert not {"host", "destination_host", "port", "endpoint_host"} & set(model.model_fields), model.__name__


# --- graph, claim, completion -------------------------------------------------------------


def _selection(purpose="semantic_qa", output_contract="qa_assessment.v1") -> catalog.FrozenSelection:
    return catalog.FrozenSelection(catalog_entry={"entry_id": "gemma4-e2b", "entry_version": 1}, purpose=purpose, model_family="gemma", model_revision="gemma-4-e2b-it-4bit@abc", adapter_id="mlx_lm", adapter_version="1", output_contract=output_contract, provider_model_id="gemma-4-e2b-it-4bit", route=_route(custody.RouteClass.APPLIANCE, custody.ProviderType.MLX, "in-process"))


RUBRIC_REF = {"rubric_id": "std", "version": 1, "digest": D0}
SCORECARD_RUBRIC = {"rubric_id": "std", "rubric_version": 1, "digest": D0}


def _job_def(ref: str, job_type: str, **kw) -> Dict[str, Any]:
    base = dict(ref=ref, job_type=job_type, idempotency_key=f"conv1:{ref}:v1", resource_estimate={"size_class": "s", "memory_slot": "cpu"})
    return {**base, **kw}


def _ingest_jobs():
    asr = _job_def("asr", "asr", inputs=[{"role": "audio", "artifact": {"artifact_id": "art_a", "checksum": D0}}], selection=_selection("asr", "transcript.v1").model_dump())
    crit = _job_def("crit", "qa_criterion", requires_refs=["asr"], inputs=[{"role": "transcript", "upstream": {"ref": "asr", "output_role": "transcript"}}], selection=_selection().model_dump(), parameters={"criterion_id": "greeting", "rubric": RUBRIC_REF}, resource_estimate={"size_class": "m", "memory_slot": "local_memory"})
    card = _job_def("card", "qa_scorecard", requires_refs=["crit"], inputs=[{"role": "assessment:greeting", "upstream": {"ref": "crit", "output_role": "assessment"}}], parameters={"rubric": RUBRIC_REF})
    return asr, crit, card


def test_job_graph_rejects_cycles_unknown_refs_and_bad_selections():
    asr, crit, card = _ingest_jobs()
    jobs.JobGraphRequest(idempotency_key="conv1:ingest:v1", reason="ingest", jobs=[asr, crit, card])
    with pytest.raises(ValidationError, match="cycle"):
        jobs.JobGraphRequest(idempotency_key="conv1:ingest:cycle", reason="ingest", jobs=[{**asr, "requires_refs": ["card"]}, crit, card])
    with pytest.raises(ValidationError, match="unknown"):
        jobs.JobGraphRequest(idempotency_key="conv1:ingest:unknown", reason="ingest", jobs=[asr, {**crit, "requires_refs": ["missing"], "inputs": []}])
    with pytest.raises(ValidationError, match="code stage"):
        jobs.JobDefinition(**_job_def("card2", "qa_scorecard", selection=_selection().model_dump(), parameters={"rubric": RUBRIC_REF}))
    with pytest.raises(ValidationError, match="criterion"):
        jobs.JobDefinition(**_job_def("c2", "qa_criterion", selection=_selection().model_dump(), parameters={"rubric": RUBRIC_REF}))
    with pytest.raises(ValidationError, match="rubric"):
        jobs.JobDefinition(**_job_def("c3", "qa_criterion", selection=_selection().model_dump(), parameters={"criterion_id": "greeting"}))
    lan = _selection("asr", "transcript.v1").model_dump()
    lan["route"] = _route(custody.RouteClass.CUSTOMER_LAN, custody.ProviderType.OLLAMA, "10.0.0.5").model_dump()
    with pytest.raises(ValidationError, match="primary Process host"):
        jobs.JobDefinition(**_job_def("asr2", "asr", selection=lan))


def test_dependents_bind_upstream_outputs_by_role():
    asr, crit, card = _ingest_jobs()
    with pytest.raises(ValidationError, match="does not depend"):
        jobs.JobDefinition(**{**crit, "requires_refs": []})
    with pytest.raises(ValidationError, match="no output role"):
        jobs.JobGraphRequest(idempotency_key="conv1:ingest:role", reason="ingest", jobs=[asr, {**crit, "inputs": [{"role": "transcript", "upstream": {"ref": "asr", "output_role": "summary"}}]}, card])
    with pytest.raises(ValidationError):
        jobs.JobInput(role="x")
    with pytest.raises(ValidationError):
        jobs.JobInput(role="x", artifact={"artifact_id": "a", "checksum": D0}, upstream={"ref": "asr", "output_role": "transcript"})
    with pytest.raises(ValidationError):
        jobs.UpstreamOutput(ref="asr", job_id="job_1", output_role="transcript")
    assert {"requires_job_ids", "after_job_ids", "resolved_inputs"} <= set(jobs.Job.model_fields)
    assert "edges" in jobs.JobGraph.model_fields


def test_merges_use_after_edges_and_optional_inputs():
    passes = [_job_def(f"cs-{k}", f"contact_signals_{k}", selection=_selection("contact_signals", "contact_signals_pass.v1").model_dump(), parameters={"pass_kind": k}, resource_estimate={"size_class": "m", "memory_slot": "local_memory"}) for k in ("lifecycle", "resolution")]
    merge = _job_def("cs-merge", "contact_signals_merge", after_refs=["cs-lifecycle", "cs-resolution"], inputs=[{"role": f"pass:{k}", "upstream": {"ref": f"cs-{k}", "output_role": "pass"}, "optional": True} for k in ("lifecycle", "resolution")])
    jobs.JobGraphRequest(idempotency_key="conv1:signals:v1", reason="ingest", jobs=[*passes, merge])
    with pytest.raises(ValidationError, match="optional"):
        jobs.JobDefinition(**{**merge, "inputs": [{"role": "pass:lifecycle", "upstream": {"ref": "cs-lifecycle", "output_role": "pass"}}]})
    with pytest.raises(ValidationError, match="one edge kind"):
        jobs.JobDefinition(**{**merge, "requires_refs": ["cs-lifecycle"]})
    claimed_fields = jobs.ClaimedJob.model_fields
    assert "upstream" in claimed_fields and jobs.ResolvedInput.model_fields["artifact"].is_required()


def test_every_job_type_has_outputs_a_group_and_each_group_one_publisher():
    rules = jobs.JOB_TYPE_RULES
    assert set(rules) == set(jobs.JobType)
    for rule in rules.values():
        assert rule.outputs, rule.job_type
        assert rule.publishes in (None, rule.group), rule.job_type
        for kind in rule.outputs.values():
            assert kind not in artifacts.GLOBAL_ARTIFACT_KINDS
    for group in calls.ResultKind:
        assert [r.job_type for r in rules.values() if r.publishes is group], group
        assert len([r for r in rules.values() if r.publishes is group]) == 1, group
    assert [r.job_type for r in rules.values() if r.group is None] == [jobs.JobType.EMBEDDINGS]
    assert set(rules[jobs.JobType.VALIDATION_VAD].outputs.values()) == {artifacts.ArtifactKind.VALIDATION_REPORT, artifacts.ArtifactKind.VAD_METRICS}
    assert set(rules[jobs.JobType.QA_CRITERION].completion_outcomes) == {usage.UsageOutcome.SUCCEEDED, usage.UsageOutcome.VALIDATION_REJECTED, usage.UsageOutcome.FAILED}
    assert rules[jobs.JobType.QA_SCORECARD].completion_outcomes == [usage.UsageOutcome.SUCCEEDED]


def _job(**overrides) -> jobs.Job:
    base = dict(id="job_1", conversation_id="conv_1", graph_id="graph_1", job_type="qa_criterion", execution_class="llm_route", status="RUNNING", priority=0, attempt_count=1, claim_count=1, max_attempts=3, retry_generation=0, lease={"worker_id": "w1", "installation_id": "inst_1", "attempt_number": 1, "granted_at": NOW, "expires_at": NOW + timedelta(seconds=300), "claim_token_hash": D1}, inputs=[{"role": "transcript", "upstream": {"job_id": "job_asr", "output_role": "transcript"}}], resolved_inputs=[{"role": "transcript", "artifact_id": "art_t", "checksum": D0}], requires_job_ids=["job_asr"], selection=_selection(), resource_estimate={"size_class": "m", "memory_slot": "local_memory"}, parameters={"criterion_id": "greeting", "rubric": RUBRIC_REF}, idempotency_key="conv_1:crit:greeting:v1", created_at=NOW, updated_at=NOW)
    return jobs.Job(**{**base, **overrides})


def test_job_record_consistency():
    _job()
    with pytest.raises(ValidationError):
        _job(status="QUEUED")
    with pytest.raises(ValidationError):
        _job(status="SUCCEEDED", lease=None)
    with pytest.raises(ValidationError):
        _job(attempt_count=2, claim_count=1)
    _job(status="SUCCEEDED", lease=None, outputs=[{"role": "assessment", "artifact_id": "art_o", "checksum": D2}], result_version=1, completed_at=NOW)


def _usage(**overrides) -> usage.UsageRecordInput:
    base = dict(tokens_input={"count": 1200, "source": "local_tokenizer"}, tokens_output={"count": 80, "source": "local_tokenizer"}, inference_seconds=4.2, total_seconds=4.9, hardware_profile_id="hw_1", outcome="succeeded")
    return usage.UsageRecordInput(**{**base, **overrides})


def _hardware() -> usage.HardwareProfileInput:
    fields = usage.HardwareProfileFields(kind="process_host", source="measured", chip="Apple M4 Pro", memory_bytes=48 * 2 ** 30, os_name="macOS", os_version="26.0", runtime_versions={"mlx": "0.29"})
    return usage.HardwareProfileInput(**fields.model_dump(), fingerprint=usage.hardware_fingerprint(fields))


def test_usage_and_hardware_rules():
    with pytest.raises(ValidationError, match="never estimate"):
        usage.TokenCount(count=5, source="unavailable")
    with pytest.raises(ValidationError):
        usage.TokenCount(count=None, source="provider_reported")
    with pytest.raises(ValidationError):
        _usage(outcome="failed")
    _usage(outcome="failed", error_code="provider_timeout")
    with pytest.raises(ValidationError):
        _usage(error_code="provider_timeout")
    with pytest.raises(ValidationError):
        _usage(outcome="validation_rejected", error_code="provider_error")
    with pytest.raises(ValidationError):
        _usage(outcome="abandoned")
    with pytest.raises(ValidationError):
        usage.HardwareProfileFields(kind="process_host", source="declared", chip="Apple M4", memory_bytes=1, os_name="macOS", os_version="26")
    _hardware()


def test_completion_request_carries_usage_provenance_and_follow_on():
    prov = jobs.AttemptProvenance(worker_id="w1", installation_id="inst_1", adapter_id="mlx_lm", adapter_version="1", model_revision="r", route=_route(custody.RouteClass.APPLIANCE, custody.ProviderType.MLX, "in-process"))
    escalation = _job_def("esc", "qa_escalation", requires_job_ids=["job_1", "job_asr"], inputs=[{"role": "transcript", "upstream": {"job_id": "job_asr", "output_role": "transcript"}}], selection=_selection().model_dump(), parameters={"criterion_id": "greeting", "escalation_trigger": "needs_review", "rubric": RUBRIC_REF}, resource_estimate={"size_class": "m", "memory_slot": "local_memory"})
    outputs = [{"role": "assessment", "artifact_id": "art_o", "checksum": D2}, {"role": "prompt_input", "artifact_id": "art_p", "checksum": D1}]
    binding = {"dependent_job_id": "job_card", "requires_ref": "esc", "input_role": "escalation:greeting", "output_role": "assessment"}
    req = jobs.CompletionRequest(claim_token=TOKEN, completion_key="job_1:1:complete", outputs=outputs, usage=_usage(), provenance=prov, follow_on={"jobs": [escalation], "add_dependencies": [binding]})
    assert req.follow_on.jobs[0].job_type is jobs.JobType.QA_ESCALATION
    assert req.follow_on.add_dependencies[0].input_role == "escalation:greeting"
    jobs.FollowOnJobs(jobs=[escalation], add_dependencies=[{"dependent_job_id": "job_card", "requires_ref": "esc"}])
    with pytest.raises(ValidationError):
        jobs.FollowOnJobs(jobs=[escalation], add_dependencies=[{"dependent_job_id": "job_card", "requires_ref": "nope"}])
    with pytest.raises(ValidationError, match="no output role"):
        jobs.FollowOnJobs(jobs=[escalation], add_dependencies=[{**binding, "output_role": "summary"}])
    with pytest.raises(ValidationError, match="both"):
        jobs.NewDependency(dependent_job_id="job_card", requires_ref="esc", input_role="escalation:greeting")
    with pytest.raises(ValidationError, match="one binding"):
        jobs.FollowOnJobs(jobs=[escalation], add_dependencies=[binding, binding])
    assert "input bindings" in next(s for s in jobs.COMPLETION_TRANSACTION_STEPS if "add_dependencies" in s)
    with pytest.raises(ValidationError):
        jobs.CompletionRequest(claim_token=TOKEN, completion_key="job_1:1:complete", outputs=outputs + [outputs[0]], usage=_usage(), provenance=prov)
    with pytest.raises(ValidationError):
        jobs.ResultPublication(kind="qa", state="stale")
    with pytest.raises(ValidationError):
        jobs.ResultPublication(kind="contact_signals", state="partial")


def test_failure_and_completion_usage_outcomes_match_the_ending_call():
    with pytest.raises(ValidationError, match="same error code"):
        jobs.FailureRequest(claim_token=TOKEN, completion_key="job_1:1:fail", error_code="provider_timeout", usage=_usage(outcome="failed", error_code="provider_error"))
    with pytest.raises(ValidationError, match="same error code"):
        jobs.FailureRequest(claim_token=TOKEN, completion_key="job_1:1:fail", error_code="provider_timeout", usage=_usage())
    with pytest.raises(ValidationError):
        jobs.FailureRequest(claim_token=TOKEN, completion_key="job_1:1:fail", error_code="validation_rejected", usage=_usage(outcome="failed", error_code="validation_rejected"))
    jobs.FailureRequest(claim_token=TOKEN, completion_key="job_1:1:fail", error_code="validation_rejected", usage=_usage(outcome="validation_rejected", error_code="validation_rejected"))
    jobs.FailureRequest(claim_token=TOKEN, completion_key="job_1:1:fail", error_code="cancelled", usage=_usage(outcome="cancelled", error_code="cancelled"))
    with pytest.raises(ValidationError, match="Store's code"):
        jobs.FailureRequest(claim_token=TOKEN, completion_key="job_1:1:fail", error_code="lease_expired", usage=_usage(outcome="failed", error_code="lease_expired"))
    prov = jobs.AttemptProvenance(worker_id="w1", installation_id="inst_1", adapter_id="a", adapter_version="1")
    with pytest.raises(ValidationError):
        jobs.CompletionRequest(claim_token=TOKEN, completion_key="k" * 8, outputs=[{"role": "r", "artifact_id": "a", "checksum": D0}], usage=_usage(outcome="cancelled", error_code="cancelled"), provenance=prov)
    for outcome, code in (("succeeded", None), ("failed", "provider_error"), ("validation_rejected", "validation_rejected"), ("cancelled", "cancelled")):
        assert usage.usage_outcome_for(code and errors.JobErrorCode(code)).value == outcome


def test_release_ends_a_claim_without_consuming_an_attempt():
    jobs.JobReleaseRequest(claim_token=TOKEN, completion_key="job_1:1:release", disposition="requeue", reason_code="resource_unavailable", not_before=NOW)
    jobs.JobReleaseRequest(claim_token=TOKEN, completion_key="job_1:1:release", disposition="reject", reason_code="pro1_key_unavailable")
    for bad in (dict(disposition="requeue", reason_code="credential_missing"), dict(disposition="reject", reason_code="resource_unavailable"), dict(disposition="reject", reason_code="context_limit_exceeded", not_before=NOW), dict(disposition="requeue", reason_code="pro1_unreachable")):
        with pytest.raises(ValidationError):
            jobs.JobReleaseRequest(claim_token=TOKEN, completion_key="job_1:1:release", **bad)
    assert jobs.JobErrorCode.CONTEXT_LIMIT_EXCEEDED in jobs.RELEASE_REJECT_CODES
    attempt = dict(job_id="job_1", attempt_number=2, started_at=NOW, worker_id="w1", installation_id="inst_1", claim_token_hash=D0, resource_estimate={"size_class": "s", "memory_slot": "cpu"})
    jobs.Attempt(**attempt, status="released", counts_as_attempt=False)
    with pytest.raises(ValidationError):
        jobs.Attempt(**attempt, status="released", counts_as_attempt=True)
    with pytest.raises(ValidationError):
        jobs.Attempt(**attempt, status="released", counts_as_attempt=False, usage_record_id="u1")
    assert "never reused" in jobs.LeaseInfo.model_fields["attempt_number"].description


def test_claims_consume_bounded_slot_offers():
    worker = dict(worker_id="w1", installation_id="inst_1", hardware_profile_id="hw_1", primary_host=True, job_types=["qa_criterion"], slot_offers=[{"memory_slot": "local_memory", "count": 1}, {"memory_slot": "outbound", "outbound_connection_ref": "pro1", "count": 2}, {"memory_slot": "outbound", "outbound_connection_ref": "conn_1", "count": 1}])
    jobs.ClaimRequest(worker=worker, max_jobs=4)
    with pytest.raises(ValidationError):
        jobs.ClaimRequest(worker=worker, max_jobs=5)
    with pytest.raises(ValidationError):
        jobs.SlotOffer(memory_slot="outbound", count=1)
    with pytest.raises(ValidationError):
        jobs.WorkerCapabilities(**{**worker, "slot_offers": [{"memory_slot": "cpu", "count": 1}, {"memory_slot": "cpu", "count": 2}]})
    assert "slot_offer_index" in jobs.ClaimedJob.model_fields and "final_attempt" in jobs.ClaimedJob.model_fields


def test_completion_transaction_order():
    steps = jobs.COMPLETION_TRANSACTION_STEPS
    index = lambda text: next(i for i, s in enumerate(steps) if text in s)  # noqa: E731
    assert index("completion_key") < index("claim token") < index("SUCCEEDED") < index("follow-on jobs and their edges") < index("initial status") < index("release every BLOCKED")
    assert "whatever the lease state" in steps[0]
    assert "job_cancelling" in steps[1]
    route = api.routes_by_operation()["completeJob"]
    assert E.JOB_CANCELLING in route.errors and "your_attempt_outcome" in route.object_rule


def test_draft_tests_never_publish_and_need_manage_rubrics():
    with pytest.raises(ValidationError):
        jobs.ReanalysisRequestCreate(kind="qa_draft_test")
    ops = api.routes_by_operation()
    test_route = ops["testRubricDraft"]
    assert test_route.request is jobs.DraftTestRequest and test_route.principals[0].permission is P.MANAGE_RUBRICS
    assert ops["getDraftTestResult"].principals[0].permission is P.MANAGE_RUBRICS and ops["getDraftTestResult"].response is jobs.DraftTestResult
    draft = {"rubric_id": "std", "draft_revision": 4, "snapshot_artifact_id": "art_snap", "digest": D1}
    jobs.JobDefinition(**_job_def("card", "qa_scorecard", parameters={"draft_rubric": draft}))
    with pytest.raises(ValidationError):
        jobs.JobDefinition(**_job_def("card", "qa_scorecard", parameters={"draft_rubric": draft, "rubric": RUBRIC_REF}))
    base = dict(id="rq_1", call_id="call_1", conversation_id="conv_1", status="pending", requested_at=NOW, idempotency_key="click-0001")
    jobs.ReanalysisRequest(**base, kind="qa_draft_test", draft_rubric=draft)
    with pytest.raises(ValidationError):
        jobs.ReanalysisRequest(**base, kind="qa_draft_test")
    with pytest.raises(ValidationError):
        jobs.ReanalysisRequest(**base, kind="qa", draft_result_artifact_id="art_x")
    assert "draft-test graph" in jobs.CompletionRequest.model_fields["result"].description
    assert jobs.REANALYSIS_KIND_AFFECTS[jobs.ReanalysisKind.QA_DRAFT_TEST] == frozenset()
    with pytest.raises(ValidationError):
        contents.ScorecardRubricRef(rubric_id="std", rubric_version=1, draft_revision=4, digest=D0)


# --- result groups ---------------------------------------------------------------------------


@pytest.mark.parametrize("inputs, expected", [
    (dict(has_work=False), "disabled"),
    (dict(has_work=True, newest_publisher="in_progress"), "pending"),
    (dict(has_work=True, newest_publisher="ended_without_result"), "failed"),
    (dict(has_work=True, published="available", newest_publisher="succeeded"), "available"),
    (dict(has_work=True, published="partial", newest_publisher="succeeded"), "partial"),
    (dict(has_work=True, published="available", reanalysis_pending=True), "stale"),
    (dict(has_work=True, published="available", newest_publisher="in_progress", newest_publisher_is_newer_than_published=True), "stale"),
    (dict(has_work=True, published="available", newest_publisher="ended_without_result", newest_publisher_is_newer_than_published=True), "available"),
    (dict(has_work=False, reanalysis_pending=True), "pending"),
])
def test_result_state_derivation(inputs, expected):
    assert calls.derive_result_state(calls.ResultStateInputs(**inputs)).value == expected


def test_result_state_rule_is_settled_and_reanalysis_scoped():
    with pytest.raises(ValidationError):
        calls.ResultStateInputs(has_work=True, published="stale")
    assert calls.SETTLED_RESULT_STATES == {calls.ResultState.AVAILABLE, calls.ResultState.PARTIAL, calls.ResultState.FAILED, calls.ResultState.DISABLED}
    affects = jobs.REANALYSIS_KIND_AFFECTS
    assert set(affects) == set(jobs.ReanalysisKind)
    assert affects[jobs.ReanalysisKind.QA] == {calls.ResultKind.QA} and affects[jobs.ReanalysisKind.FULL] == set(calls.ResultKind)
    assert calls.ResultKind.TRANSCRIPT not in affects[jobs.ReanalysisKind.SPEAKER_CORRECTION]
    assert "settled" in calls.PendingWorkIndicator.model_fields and "dead_blocked" in jobs.GroupProgress.model_fields


def _stake(minutes: int, publisher, draft_test=False, graph_id=None) -> calls.GroupGraphStake:
    return calls.GroupGraphStake(graph_id=graph_id or f"graph_{minutes}", graph_created_at=NOW + timedelta(minutes=minutes), draft_test=draft_test, has_group_jobs=True, publisher=publisher)


def test_draft_tests_never_change_the_calls_result_state():
    state = lambda stakes, published=None, at=None, pending=False: calls.derive_result_state(calls.result_state_inputs(stakes, published, at, pending)).value  # noqa: E731
    live = _stake(0, "succeeded")
    published_at = live.graph_created_at
    assert state([live], "available", published_at) == "available"
    # A draft test that runs, fails or succeeds on the call leaves the call's QA exactly as it was.
    for outcome in ("in_progress", "ended_without_result", "succeeded"):
        inputs = calls.result_state_inputs([live, _stake(10, outcome, draft_test=True)], calls.ResultState.AVAILABLE, published_at, False)
        assert inputs.newest_publisher is calls.PublisherState.SUCCEEDED and not inputs.newest_publisher_is_newer_than_published
        assert calls.derive_result_state(inputs).value == "available"
    assert state([_stake(10, "ended_without_result", draft_test=True)]) == "disabled"
    assert state([_stake(10, "in_progress", draft_test=True)]) == "disabled"
    # Live graphs still drive the state.
    assert state([live, _stake(10, "in_progress")], "available", published_at) == "stale"
    assert state([live, _stake(10, "ended_without_result")], "available", published_at) == "available"
    assert state([_stake(10, "in_progress")]) == "pending"
    assert state([_stake(10, "ended_without_result")]) == "failed"
    assert state([_stake(10, "in_progress")], "available", None) == "stale"  # a migrated version has no graph
    assert state([], "partial", None, pending=True) == "stale"
    # Draft-test outputs live in their own slots, so they never supersede the call's artifacts.
    assert artifacts.draft_test_slot("rq_1", "greeting") == "draft:rq_1:greeting"
    assert artifacts.is_draft_test_slot(artifacts.draft_test_slot("rq_1", "")) and not artifacts.is_draft_test_slot("greeting")
    assert artifacts.draft_rubric_snapshot_slot("rq_1", "std", 4) == "draft:rq_1:rubric:std:r4"
    with pytest.raises(ValueError):
        artifacts.draft_test_slot("rq_1", "draft:rq_0:greeting")
    longest = artifacts.draft_test_slot("r" * 128, "g" * artifacts.SLOT_MAX_LENGTH)
    art = dict(id="art_1", conversation_id="conv_1", storage="inline", committed_at=NOW, kind="qa_assessment", content_type="application/json", size_bytes=10, checksum=D0, content_contract="qa_assessment.v1", sensitivity="masked", producing_job_id="job_1", linked=False)
    artifacts.Artifact(**art, slot=longest)
    with pytest.raises(ValidationError, match="live slot"):
        artifacts.Artifact(**art, slot="g" * (artifacts.SLOT_MAX_LENGTH + 1))
    assert artifacts.ArtifactListQuery().include_draft_tests is False
    steps = jobs.COMPLETION_TRANSACTION_STEPS
    assert any("draft:<request_id>:" in step for step in steps)
    assert any("never supersedes" in step for step in steps) and any("no review-queue items" in step for step in steps)
    for model in (calls.PendingWorkIndicator, jobs.GroupProgress):
        assert "draft-test" in model.__doc__
    assert "draft-test" in jobs.JobGroupProgress.model_fields["settled"].description


def test_speaker_corrections_are_frozen_on_a_code_stage_job():
    correction = {"turn_id": 5, "speaker": "CALLER", "apply_to_cluster": True}
    inputs = [{"role": "transcript", "artifact": {"artifact_id": "art_t", "checksum": D0}}, {"role": "speaker_attribution", "artifact": {"artifact_id": "art_s", "checksum": D1}}]
    fix = jobs.JobDefinition(**_job_def("spk", "speaker_attribution", inputs=inputs, parameters={"speaker_correction": correction}))
    assert not fix.model_backed and fix.execution_class is jobs.ExecutionClass.PRIMARY_HOST
    with pytest.raises(ValidationError, match="code stage"):
        jobs.JobDefinition(**_job_def("spk", "speaker_attribution", inputs=inputs, selection=_selection("speaker_diarization", "speaker_attribution.v1").model_dump(), parameters={"speaker_correction": correction}))
    with pytest.raises(ValidationError, match="freezes a model selection"):
        jobs.JobDefinition(**_job_def("spk", "speaker_attribution", inputs=inputs))
    with pytest.raises(ValidationError, match="only a speaker_attribution"):
        jobs.JobDefinition(**_job_def("tone", "acoustic_tone", selection=_selection("acoustic_tone", "tone_blocks.v1").model_dump(), parameters={"speaker_correction": correction}))
    # Every group a kind makes stale has a publisher the fulfilling graph reruns; asr is not rerun
    # for a speaker correction, so the transcript group is not listed.
    publishers = {rule.publishes: rule.job_type for rule in jobs.JOB_TYPE_RULES.values() if rule.publishes}
    assert publishers[calls.ResultKind.TRANSCRIPT] is jobs.JobType.ASR
    assert calls.ResultKind.TRANSCRIPT not in jobs.REANALYSIS_KIND_AFFECTS[jobs.ReanalysisKind.SPEAKER_CORRECTION]


def test_qa_assessments_record_invalid_answers_and_provider_failures_as_flagged():
    attempt = dict(catalog_entry_id="gemma4-e2b", model_revision="r", route_class="appliance", destination_host="in-process", status="FLAGGED", reasoning="Provider failed.", latency_ms=10)
    base = dict(criterion_id="greeting", assessment_kind="primary", confidence=0, reasoning="x", escalation_requested=True)
    contents.QaAssessmentContent(**base, status="FLAGGED", trigger="provider_error", attempt={**attempt, "error_code": "provider_timeout"})
    contents.QaAssessmentContent(**base, status="FLAGGED", trigger="invalid_answer", attempt=attempt)
    with pytest.raises(ValidationError):
        contents.QaAssessmentContent(**base, status="PASS", trigger="invalid_answer", attempt=attempt)
    with pytest.raises(ValidationError):
        contents.QaAssessmentContent(**base, status="FLAGGED", trigger="provider_error", attempt=attempt)
    assert errors.JobErrorCode.PROVIDER_TIMEOUT in errors.PROVIDER_FAILURE_CODES


# --- artifacts and content -------------------------------------------------------------------


STORE_ASSIGNED = {"artifact_id", "version", "call_id", "conversation_id"}


def test_every_json_artifact_kind_has_a_content_model_without_store_ids():
    for kind, contract in artifacts.ARTIFACT_CONTENT_CONTRACTS.items():
        assert contract in artifacts.ARTIFACT_CONTENT_MODELS, kind
    opaque = {c for c, m in artifacts.ARTIFACT_CONTENT_MODELS.items() if m is None}
    assert opaque == {"audio.v1", "attestation_evidence.v1", "trust_anchor_bundle.v1"}
    for contract, model in artifacts.ARTIFACT_CONTENT_MODELS.items():
        if model is None:
            continue
        schema = model.model_json_schema()
        # The content never carries its own Store-assigned identity. References to other, already
        # committed artifacts (ArtifactRef: ID plus checksum) and to an already published signal
        # taxonomy version (SignalTaxonomyRef, 1.3.0) are fine: they exist before the job runs.
        own = set(schema.get("properties", {}))
        nested = {p for name, d in schema.get("$defs", {}).items() if name not in ("ArtifactRef", "SignalTaxonomyRef") for p in d.get("properties", {})}
        assert not (own | nested) & STORE_ASSIGNED, (contract, (own | nested) & STORE_ASSIGNED)
    schemas = api.build_openapi()["components"]["schemas"]
    for name in api.build_openapi()["x-call1"]["artifact_content_schemas"].values():
        assert name is None or name in schemas, name


SIGNALS_VIEW_READ_CONTEXT = {"taxonomy_status", "feedback", "alerts", "text_withheld", "comparison_preview_id"}
"""1.3.0: read-time context Store adds to the contact-signals view; none of it is in the artifact."""


def test_views_are_content_plus_store_ids():
    for view, content in ((calls.EvaluationView, contents.QaScorecardContent), (calls.SummaryView, contents.SummaryContent), (calls.ContactSignalsView, contents.ContactSignalsContent)):
        assert issubclass(view, content)
        extra = SIGNALS_VIEW_READ_CONTEXT if view is calls.ContactSignalsView else set()
        assert set(view.model_fields) - set(content.model_fields) == {"call_id", "artifact_id", "version"} | extra
    for name in SIGNALS_VIEW_READ_CONTEXT:
        assert not calls.ContactSignalsView.model_fields[name].is_required(), name
    assert "speaker_attribution" in calls.TranscriptView.__doc__


def test_inline_artifacts_hash_canonical_payloads():
    payload = contents.VadMetricsContent(total_speech_duration=10, total_silence_duration=2, silence_ratio=0.1666, overtalk_duration=0.5, overtalk_ratio=0.05).model_dump(mode="json")
    create = artifacts.InlineArtifactCreate(kind="vad_metrics", content_type="application/json", size_bytes=len(common.canonical_json(payload)), checksum=common.canonical_digest(payload), content_contract="vad_metrics.v1", sensitivity="derived", producing_job_id="job_v", claim_token=TOKEN, payload=payload)
    assert artifacts.content_model_for(create.kind).model_validate(create.payload)
    with pytest.raises(ValidationError):
        artifacts.InlineArtifactCreate(**{**create.model_dump(), "claim_token": None})
    with pytest.raises(ValidationError):
        artifacts.InlineArtifactCreate(**{**create.model_dump(), "kind": "source_audio", "content_contract": "audio.v1", "sensitivity": "raw"})


def test_inline_payloads_are_the_full_canonical_dump():
    """One checksum: the payload exactly as sent must already be the content model's full dump."""
    full = contents.SpeakerAttributionContent(method="channel", assignments=[{"turn_id": 0, "speaker": "AGENT"}]).model_dump(mode="json")
    assert artifacts.canonical_content("speaker_attribution.v1", full) == common.canonical_json(full)
    descriptor = dict(kind="speaker_attribution", content_type="application/json", content_contract="speaker_attribution.v1", sensitivity="derived", producing_job_id="job_s", claim_token=TOKEN)
    artifacts.InlineArtifactCreate(**descriptor, payload=full, size_bytes=len(common.canonical_json(full)), checksum=common.canonical_digest(full))
    sparse = {"method": "channel", "assignments": [{"turn_id": 0, "speaker": "AGENT"}]}  # defaults omitted
    assert common.canonical_digest(sparse) != common.canonical_digest(full)
    with pytest.raises(ValueError, match="canonical"):
        artifacts.canonical_content("speaker_attribution.v1", sparse)
    with pytest.raises(ValidationError, match="canonical"):
        artifacts.InlineArtifactCreate(**descriptor, payload=sparse, size_bytes=len(common.canonical_json(sparse)), checksum=common.canonical_digest(sparse))
    card = contents.QaScorecardContent(rubric=SCORECARD_RUBRIC, overall_score=90, passed=True, critical_failure=False, requires_human_review=False, verdicts=[], evaluated_at=NOW).model_dump(mode="json")
    artifacts.canonical_content("qa_scorecard.v1", card)
    with pytest.raises(ValueError, match="canonical"):
        artifacts.canonical_content("qa_scorecard.v1", {**card, "evaluated_at": card["evaluated_at"].replace("Z", "+00:00")})
    with pytest.raises(ValueError, match="not valid"):
        artifacts.canonical_content("speaker_attribution.v1", {"method": "guess", "assignments": []})
    assert "canonical_content" in artifacts.ArtifactDescriptor.model_fields["checksum"].description


def test_rubric_snapshots_are_minted_by_store():
    route = api.routes_by_operation()["mintRubricSnapshot"]
    assert route.request is rubrics.RubricSnapshotRequest and route.response is artifacts.Artifact
    assert route.idempotency is api.Idempotency.NATURAL and [p.scope for p in route.principals] == [common.ServiceScope.JOBS_WRITE]
    snapshot = rubrics.RubricSnapshotContent(source="published", rubric_id="std", rubric_version=1, digest=common.canonical_digest(rubrics.RubricDefinition(rubric_id="std", name="Standard")), definition=rubrics.RubricDefinition(rubric_id="std", name="Standard")).model_dump(mode="json")
    descriptor = dict(kind="rubric_snapshot", slot=artifacts.rubric_snapshot_slot("std", 1), content_type="application/json", size_bytes=len(common.canonical_json(snapshot)), checksum=common.canonical_digest(snapshot), content_contract="rubric_snapshot.v1", sensitivity="derived")
    assert descriptor["slot"] == "rubric:std:v1"
    with pytest.raises(ValidationError, match="minted by Store"):
        artifacts.InlineArtifactCreate(**descriptor, payload=snapshot)
    with pytest.raises(ValidationError, match="minted by Store"):
        artifacts.UploadGrantRequest(**descriptor)
    assert artifacts.STORE_MINTED_KINDS == {artifacts.ArtifactKind.RUBRIC_SNAPSHOT, artifacts.ArtifactKind.SIGNAL_TAXONOMY_SNAPSHOT}
    assert rubrics.RUBRIC_INPUT_ROLE == "rubric" and api.contract_extensions()["rubric_input_role"] == "rubric"
    assert "mintRubricSnapshot" in rubrics.RubricSnapshotContent.__doc__


def test_artifacts_version_per_slot_only_when_linked():
    base = dict(id="art_1", conversation_id="conv_1", storage="inline", committed_at=NOW, kind="qa_assessment", slot="greeting", content_type="application/json", size_bytes=10, checksum=D0, content_contract="qa_assessment.v1", sensitivity="masked", producing_job_id="job_1")
    artifacts.Artifact(**base, linked=False)
    artifacts.Artifact(**base, linked=True, version=1, linked_by_receipt_id="rcpt_1")
    with pytest.raises(ValidationError):
        artifacts.Artifact(**base, linked=False, version=1)
    with pytest.raises(ValidationError):
        artifacts.Artifact(**base, linked=True)
    with pytest.raises(ValidationError):
        artifacts.Artifact(**{**base, "slot": "bad slot!"}, linked=False)
    assert "slot" in artifacts.Artifact.model_fields["version"].description
    assert artifacts.ArtifactListQuery().include_unlinked is False
    assert "producing job" in artifacts.InlineArtifactCreate.__doc__


def test_call_record_media_fields_wait_for_validation():
    calls.CallRecordView(call_id="call_1", conversation_id="conv_1", agent_id="A1", created_at=NOW)
    calls.CallListItem(call_id="call_1", conversation_id="conv_1", agent_id="A1", created_at=NOW, transcript_state="pending", qa_state="pending", summary_state="pending", review_version=0)
    contents.AudioValidationContent(container="wav", codec="pcm_s16le", sample_rate=8000, channels=2, channel_layout="STEREO", duration_seconds=120)


def test_contact_signal_partials_say_what_is_missing():
    base = dict(signals=[], transcript_fingerprint=D0, generated_at=NOW)
    contents.ContactSignalsContent(**base, completeness="partial", partial_reason="resolution pass failed", passes=[{"pass_kind": "lifecycle", "included": True}, {"pass_kind": "resolution", "included": False, "failure_code": "validation_rejected"}])
    with pytest.raises(ValidationError):
        contents.ContactSignalsContent(**base, completeness="complete", passes=[{"pass_kind": "resolution", "included": False}])
    with pytest.raises(ValidationError):
        contents.ContactSignalsContent(**base, completeness="partial", passes=[])


# --- auth, hostname, TLS, backup ---------------------------------------------------------


@pytest.mark.parametrize("bad", ["localhost", "192.168.1.5", "::1", "store", "store.local", "qa.call1.cc", "x.localdomain", "-bad.example.com"])
def test_hostname_rule_rejects(bad):
    with pytest.raises(ValueError):
        common.validate_store_hostname(bad)


def test_hostname_rule_accepts_customer_dns_names_and_binds_rp_and_origins():
    assert common.validate_store_hostname("QA.Example.com.") == "qa.example.com"
    rp = admin.WebAuthnRelyingParty(rp_id="qa.example.com", allowed_origins=["https://qa.example.com", "https://qa.example.com:8443"])
    assert rp.rp_id == "qa.example.com"
    for origins in (["https://other.example.com"], ["http://qa.example.com"], ["https://qa.example.com/path"]):
        with pytest.raises(ValidationError):
            admin.WebAuthnRelyingParty(rp_id="qa.example.com", allowed_origins=origins)


def _cert(**kw) -> Dict[str, Any]:
    return {"subject": "CN=qa.example.com", "issuer": "CN=Call1 Store CA", "not_before": NOW, "not_after": NOW + timedelta(days=365), "fingerprint_sha256": D0, "san_dns": ["qa.example.com"], **kw}


def test_tls_paths():
    admin.TlsState(store_hostname="qa.example.com", trust_path="path_a_call1_ca", server_certificate=_cert(), ca_certificate=_cert(subject="CN=Call1 Store CA", san_dns=[]), ca_name_constraint="qa.example.com", renewal_owner="customer IT", health="ok", days_until_expiry=365)
    admin.TlsState(store_hostname="qa.example.com", trust_path="path_b_customer_cert", server_certificate=_cert(issuer="CN=Corp CA"), renewal_owner="customer IT", health="ok", days_until_expiry=90)
    with pytest.raises(ValidationError):
        admin.TlsState(store_hostname="qa.example.com", trust_path="path_a_call1_ca", server_certificate=_cert(), ca_certificate=_cert(), ca_name_constraint="example.com", renewal_owner="x", health="ok", days_until_expiry=1)
    with pytest.raises(ValidationError):
        admin.TlsState(store_hostname="qa.example.com", trust_path="path_b_customer_cert", server_certificate=_cert(san_dns=["other.example.com"]), renewal_owner="x", health="ok", days_until_expiry=1)


def test_restore_rule_same_hostname_keeps_enrollments_and_starts_a_new_epoch():
    admin.RestorePreflightResult(hostname_matches=True, enrollments_preserved=True, requires_reinvite_all=False)
    admin.RestorePreflightResult(hostname_matches=False, enrollments_preserved=False, requires_reinvite_all=True)
    with pytest.raises(ValidationError):
        admin.RestorePreflightResult(hostname_matches=False, enrollments_preserved=True, requires_reinvite_all=False)
    assert admin.BackupManifest.model_fields["includes_credential_records"].default is True
    assert admin.RestorePreflightResult.model_fields["new_feed_epoch"].default is True
    assert "feed_epoch" in admin.BackupManifest.model_fields and "feed_epoch" in admin.StoreStatus.model_fields and "feed_epoch" in events.ChangeFeed.model_fields


def test_change_cursors_are_typed_and_resume_without_gaps():
    for model, name in ((calls.CallDetail, "change_cursor"), (jobs.CompletionReceipt, "change_cursor"), (jobs.FailureReceipt, "change_cursor"), (jobs.JobReleaseReceipt, "change_cursor"), (reviews.ReviewWriteResult, "change_cursor"), (admin.StoreStatus, "latest_change_cursor"), (events.ChangeFeed, "next_cursor")):
        info = model.model_fields[name]
        assert any(getattr(m, "pattern", None) == r"^[A-Za-z0-9_-]{1,64}$" for m in info.metadata), f"{model.__name__}.{name}"
    assert "scan position" in events.ChangeFeed.model_fields["next_cursor"].description
    assert "commit order" in events.ChangeFeed.__doc__
    assert E.CURSOR_UNKNOWN in api.routes_by_operation()["listChanges"].errors


def test_passkey_only_ceremonies():
    opts = auth.CredentialCreationOptions(rp={"id": "qa.example.com", "name": "Call1 Store"}, user={"id": "dXNlcg", "name": "ann@example.com", "displayName": "Ann"}, challenge="Y2hhbGxlbmdl")
    assert opts.authenticatorSelection.userVerification == "required" and opts.attestation == "none"
    with pytest.raises(ValidationError):
        auth.CredentialCreationOptions(rp={"id": "qa.example.com", "name": "x"}, user={"id": "dXNlcg", "name": "ann@example.com", "displayName": "Ann"}, challenge="Y2hhbGxlbmdl", authenticatorSelection={"userVerification": "preferred"})
    with pytest.raises(ValidationError):
        auth.EnrollmentBeginRequest()
    with pytest.raises(ValidationError):
        auth.EnrollmentBeginRequest(invitation_token="x" * 20, setup_code="y" * 10)
    assert not any("password" in r.path for r in api.ROUTES)
    cookie = auth.SESSION_COOKIE
    assert cookie.http_only and cookie.secure and cookie.same_site == "Strict"
    assert cookie.name.startswith("__Host-") and cookie.path == "/" and cookie.domain is None and cookie.stored_as == "sha256"
    assert "Never the cookie value" in auth.SessionInfo.model_fields["session_id"].description


# Exactly what @simplewebauthn/browser v14 returns (esm/methods/startRegistration.js, startAuthentication.js).
SIMPLEWEBAUTHN_REGISTRATION = {
    "id": "-AbC_dEf", "rawId": "-AbC_dEf", "type": "public-key",
    "response": {"attestationObject": "o2NmbXRkbm9uZQ", "clientDataJSON": "eyJ0eXBlIjoid2ViYXV0aG4uY3JlYXRlIn0", "transports": ["usb", "nfc", "cable", "hybrid", "something-new"], "publicKeyAlgorithm": -7, "publicKey": "MFkwEwYHKoZIzj0CAQ", "authenticatorData": "SZYN5YgOjGh0NBcPZHZgW4_krrmihjLHmVzzuoMdl2M"},
    "clientExtensionResults": {"credProps": {"rk": True}}, "authenticatorAttachment": "cross-platform",
}
SIMPLEWEBAUTHN_AUTHENTICATION = {
    "id": "-AbC_dEf", "rawId": "-AbC_dEf", "type": "public-key",
    "response": {"authenticatorData": "SZYN5YgOjGh0NBcPZHZgW4_krrmihjLHmVzzuoMdl2M", "clientDataJSON": "eyJ0eXBlIjoid2ViYXV0aG4uZ2V0In0", "signature": "MEUCIQ", "userHandle": "dXNlcg"},
    "clientExtensionResults": {}, "authenticatorAttachment": "platform",
}


def test_simplewebauthn_browser_payloads_validate_unchanged():
    reg = auth.RegistrationFinishRequest(ceremony_id="cer_1", credential=SIMPLEWEBAUTHN_REGISTRATION)
    assert reg.credential.response.publicKeyAlgorithm == -7
    minimal = {**SIMPLEWEBAUTHN_REGISTRATION, "response": {k: v for k, v in SIMPLEWEBAUTHN_REGISTRATION["response"].items() if k in ("attestationObject", "clientDataJSON")}}
    minimal.pop("authenticatorAttachment")
    auth.RegistrationFinishRequest(ceremony_id="cer_1", credential=minimal)
    auth.AuthenticationFinishRequest(ceremony_id="cer_2", credential=SIMPLEWEBAUTHN_AUTHENTICATION)
    auth.AddAuthenticatorFinishRequest(ceremony_id="cer_3", reauthentication=SIMPLEWEBAUTHN_AUTHENTICATION, credential=SIMPLEWEBAUTHN_REGISTRATION)
    assert auth.AuthenticatorTransport.CABLE.value == "cable"


def test_csrf_token_is_session_bound_and_recoverable():
    ops = api.routes_by_operation()
    assert ops["getSession"].response is auth.SessionInfo and "csrf_token" in auth.SessionInfo.model_fields
    assert "csrf_token" not in auth.SignInResponse.model_fields
    assert ("SessionInfo", "csrf_token") in auth.SESSION_BOUND_SECRET_FIELDS
    assert not any(f == "csrf_token" for _, f in auth.ONE_TIME_SECRET_FIELDS)
    assert "ONE-TIME" not in (auth.SessionInfo.model_fields["csrf_token"].description or "")


def test_add_authenticator_is_a_step_up_ceremony():
    ops = api.routes_by_operation()
    assert ops["addAuthenticatorBegin"].response is auth.AddAuthenticatorBeginResponse
    assert ops["addAuthenticatorFinish"].request is auth.AddAuthenticatorFinishRequest
    assert auth.AddAuthenticatorFinishRequest.model_fields["reauthentication"].is_required()
    assert {"reauthentication", "options"} <= set(auth.AddAuthenticatorBeginResponse.model_fields)


def test_authenticator_paths_use_store_handles():
    for route in api.ROUTES:
        assert "{credential_id}" not in route.path, route.operation_id
    assert "{authenticator_id}" in api.routes_by_operation()["removeOwnAuthenticator"].path
    assert "authenticator_id_used" in auth.SessionInfo.model_fields and "credential_id_used" not in auth.SessionInfo.model_fields
    assert api.routes_by_operation()["revokeAccountAuthenticator"].audited


def test_sign_in_never_reveals_whether_an_account_exists():
    with pytest.raises(ValidationError):
        auth.CredentialRequestOptions(challenge="Y2g", rpId="qa.example.com", allowCredentials=[])
    doc = auth.CredentialRequestOptions.__doc__
    assert "decoy" in doc and "bound to the ceremony" in doc


def test_setup_codes_carry_the_identity_they_enroll():
    auth.SetupCodeIssueRequest(purpose="first_admin", email="it@example.com", display_name="IT", os_user="call1store")
    auth.SetupCodeIssueRequest(purpose="break_glass", email="it@example.com", display_name="IT", target_account_id="acct_1", os_user="call1store")
    with pytest.raises(ValidationError):
        auth.SetupCodeIssueRequest(purpose="first_admin", email="it@example.com", display_name="IT", target_account_id="acct_1", os_user="call1store")
    record = auth.SetupCodeRecord(id="sc_1", purpose="first_admin", email="it@example.com", display_name="IT", code_hash=D0, issued_at=NOW, expires_at=NOW, audit_event_id="evt_1")
    assert record.role == "admin"
    assert "revokes the account's credentials and sessions" in auth.SetupCodeIssueRequest.__doc__


def test_admin_state_change_is_one_section_at_a_time():
    with pytest.raises(ValidationError):
        admin.AdminStateChange(section="masking", expected_state_version=1, reason="x")
    with pytest.raises(ValidationError):
        admin.AdminStateChange(section="masking", expected_state_version=1, reason="x", masking={}, minimum_release_svn={"kind": "pro1_release", "value": 1})
    change = admin.AdminStateChange(section="masking", expected_state_version=1, reason="x", masking={})
    assert change.masking.mask_reviewer_reads is True
    admin.AdminStateChange(section="trust_anchors", expected_state_version=1, reason="x", trust_anchor_adoption={"pending_digest": D1})


def test_minimum_release_svn_has_one_home_each():
    assert "minimum_release_svn" in release_trust.AttestationPolicy.model_fields
    assert "minimum_release_svn" not in admin.AdminState.model_fields and "minimum_appliance_build_svn" in admin.AdminState.model_fields
    assert "pro1_session_lifetime_seconds" not in common.ContractParameters.model_fields
    assert "session_lifetime_seconds" in release_trust.AttestationPolicySettings.model_fields
    admin.MinimumSvnChange(kind="pro1_release", value=2, confirm_lower=True)
    policy = _policy()
    lowered = release_trust.AttestationPolicy(**{**policy.model_dump(), "minimum_release_svn": 0})
    assert common.canonical_digest(lowered) != common.canonical_digest(policy)


def test_attestation_policy_pins_platform_identity():
    pins = release_trust.PlatformIdentityPins(expected_id_key_digests=["a" * 96], expected_vm_configuration={"secure-boot": True, "tpm-enabled": True}, tdx={"mrconfigid": "0" * 96}, cvm_firmware_pcrs={"0": "b" * 64}, cvm_firmware_pcrs_source="policy")
    policy = _policy(platform_identity=pins.model_dump())
    assert common.canonical_digest(policy) != common.canonical_digest(_policy())
    with pytest.raises(ValidationError):
        release_trust.PlatformIdentityPins(cvm_firmware_pcrs={"9": "b" * 64}, cvm_firmware_pcrs_source="policy")
    with pytest.raises(ValidationError):
        release_trust.PlatformIdentityPins(cvm_firmware_pcrs_source="policy")


def test_updater_verification_needs_every_check_and_an_approval():
    checks = dict(signature_valid=True, digest_in_witnessed_log=True, approval_matches_digest=True, no_install_scripts=True, not_revoked=True, svn_at_or_above_minimum=True)
    release_trust.UpdaterVerification(package_digest=D0, release_id="r1", manifest_digest=D1, log_index=3, approval_id="apr_1", checks=checks, verdict="accept", verified_by_build_digest=D2, verified_at=NOW)
    with pytest.raises(ValidationError):
        release_trust.UpdaterVerification(package_digest=D0, release_id="r1", manifest_digest=D1, checks=checks, verdict="accept", verified_by_build_digest=D2, verified_at=NOW)
    with pytest.raises(ValidationError):
        release_trust.UpdaterVerification(package_digest=D0, release_id="r1", manifest_digest=D1, approval_id="apr_1", checks={**checks, "no_install_scripts": False}, verdict="accept", verified_by_build_digest=D2, verified_at=NOW)
    with pytest.raises(ValidationError):
        release_trust.UpdaterVerification(package_digest=D0, release_id="r1", manifest_digest=D1, checks={**checks, "no_install_scripts": False}, verdict="reject", verified_by_build_digest=D2, verified_at=NOW)


def test_approvals_name_an_observed_pending_release_and_can_be_declined():
    fields = release_trust.ReleaseApprovalRequest.model_fields
    assert "evidence" not in fields and {"kind", "release_id", "manifest_digest"} <= set(fields)
    assert "not_pending" in release_trust.ReleaseApprovalRequest.__doc__
    pending = dict(kind="pro1_release", release_id="r1", manifest_digest=D1, log_index=1, release_svn=1, first_seen_at=NOW, evidence=_evidence())
    release_trust.PendingRelease(**pending, decision="declined", declined_at=NOW, declined_by_account_id="acct_1", decline_reason="not now")
    with pytest.raises(ValidationError):
        release_trust.PendingRelease(**pending, decision="declined")
    assert api.routes_by_operation()["declineRelease"].request is release_trust.ReleaseDeclineRequest


def test_trust_anchor_bundles_are_artifacts_process_registers_and_admins_adopt():
    assert artifacts.ArtifactKind.TRUST_ANCHOR_BUNDLE in artifacts.GLOBAL_ARTIFACT_KINDS
    assert {"artifact_id", "digest"} <= set(release_trust.TrustAnchorBundleRef.model_fields)
    ops = api.routes_by_operation()
    assert ops["submitTrustAnchorBundle"].principals[0].scope is common.ServiceScope.RELEASE_TRUST_WRITE and ops["submitTrustAnchorBundle"].audited
    assert ops["getTrustAnchorBundle"].response is api.BINARY
    assert "adopted" in release_trust.TrustAnchorState.model_fields["pending"].description
    assert admin.AdminStateChange.model_fields["trust_anchor_adoption"].annotation is not None


def test_egress_entries_label_customer_data():
    with pytest.raises(ValidationError):
        release_trust.EgressEntry(destination="pro1.call1.cc", purpose="pro1_frontend", required_when="Pro1 enabled", carries_customer_data=True, ciphertext_only=False)
    with pytest.raises(ValidationError):
        release_trust.EgressEntry(destination="kdsintf.amd.com", purpose="amd_kds", required_when="Pro1 enabled", carries_customer_data=True)
    release_trust.EgressEntry(destination="smtp.corp.example", port=587, protocol="smtp+starttls", purpose="customer_smtp_relay", required_when="SMTP invitations", carries_customer_data=True)
    with pytest.raises(ValidationError):
        release_trust.EgressEntry(destination="smtp.corp.example", port=587, protocol="smtp+starttls", purpose="customer_smtp_relay", required_when="SMTP invitations", carries_customer_data=False)
    with pytest.raises(ValidationError):
        release_trust.EgressEntry(destination="api.openai.com", protocol="smtps", purpose="byok_provider", required_when="BYOK", carries_customer_data=True)


# --- usage -----------------------------------------------------------------------------------


def test_usage_report_has_csv_a_price_table_and_medians():
    ops = api.routes_by_operation()
    csv_route = ops["exportUsageRecordsCsv"]
    assert csv_route.response is api.CSV and csv_route.audited
    assert "text/csv" in api.build_openapi()["paths"][csv_route.path]["post"]["responses"]["200"]["content"]
    assert set(usage.USAGE_CSV_COLUMNS) >= {"job_id", "attempt_number", "outcome", "tokens_input", "tokens_input_source", "hardware_profile_id", "recorded_by"}
    assert ops["savePriceTable"].audited and ops["savePriceTable"].idempotency is api.Idempotency.EXPECTED_VERSION
    usage.PriceEntry(route_class="call1_confidential", basis="per_billing_unit", billing_unit="request", unit_price=0.01, effective_from=NOW)
    with pytest.raises(ValidationError):
        usage.PriceEntry(route_class="appliance", basis="per_billing_unit", unit_price=1, effective_from=NOW)
    assert usage.CostEstimate(amount=1.5, currency="USD", price_table_version=2, unpriced_attempts=0).label == "estimate"
    medians = ops["getUsageMedians"]
    assert any(p.scope is common.ServiceScope.USAGE_READ for p in medians.principals) and medians.response is usage.UsageMedians
    assert "UI" not in (usage.UsageReport.__doc__ or "")


def test_lease_expiry_produces_one_abandoned_usage_row():
    row = dict(id="u1", job_id="job_1", attempt_number=1, conversation_id="conv_1", job_type="qa_criterion", recorded_at=NOW, tokens_input={"source": "unavailable"}, tokens_output={"source": "unavailable"}, inference_seconds=0, total_seconds=300, hardware_profile_id="hw_1")
    usage.UsageRecord(**row, outcome="abandoned", recorded_by="store_synthesized")
    usage.UsageRecord(**row, outcome="abandoned", recorded_by="process_late")
    with pytest.raises(ValidationError):
        usage.UsageRecord(**row, outcome="abandoned")
    with pytest.raises(ValidationError):
        usage.UsageRecord(**row, outcome="succeeded", recorded_by="store_synthesized")
    route = api.routes_by_operation()["attachLateUsage"]
    assert route.request is usage.LateUsageReport and "once" in route.object_rule
    assert "abandoned" in [t.note for t in jobs.JOB_TRANSITIONS if t.trigger is jobs.JobTrigger.LEASE_EXPIRED_RETRYABLE][0]


# --- legacy carry-over ---------------------------------------------------------------------


def test_legacy_rubric_check_fields_carry_unchanged():
    from call1.models.schemas import RubricCheck as LegacyCheck, RubricDefinition as LegacyRubric

    assert set(rubrics.RubricCheck.model_fields) == set(LegacyCheck.model_fields)
    legacy = LegacyRubric(rubric_id="std", name="Standard", description="d", criteria=[{"criterion_id": "greet", "name": "Greeting", "check": {"check_type": "phrase_any", "phrases": ["hello"]}}])
    carried = rubrics.RubricDefinition.model_validate(legacy.model_dump(mode="json"))
    assert carried.criteria[0].check.phrases == ["hello"]


def test_legacy_transcript_and_verdict_projections_are_field_subsets():
    from call1.models.schemas import RubricVerdict as LegacyVerdict, TranscriptTurn as LegacyTurn

    assert set(calls.TranscriptTurnView.model_fields) <= set(LegacyTurn.model_fields)
    assert set(calls.VerdictView.model_fields) - {"quote_turn_id"} <= set(LegacyVerdict.model_fields)
    assert set(contents.TranscriptTurnContent.model_fields) <= set(LegacyTurn.model_fields)


# --- round trips ----------------------------------------------------------------------------


def _upload_grant() -> artifacts.UploadGrant:
    return artifacts.UploadGrant(upload_id="up_1", artifact_id="art_1", url="https://qa.example.com/store/v1/uploads/up_1", expires_at=NOW + timedelta(minutes=15), max_bytes=10)


def _content_grant() -> artifacts.ContentGrant:
    return artifacts.ContentGrant(artifact_id="art_1", url="https://qa.example.com/store/v1/objects/x", expires_at=NOW, checksum=D0, content_type="audio/wav", size_bytes=10)


def _policy(**overrides) -> release_trust.AttestationPolicy:
    base = {"minimum_release_svn": 1, "witnesses": [{"name": "w1", "key_id": "k1"}, {"name": "w2", "key_id": "k2"}], "witness_quorum": 2, "allowed_platforms": ["azure-snp-paravisor"]}
    return release_trust.AttestationPolicy(**{**base, **overrides})


def _admin_state(policy: release_trust.AttestationPolicy) -> admin.AdminState:
    return admin.AdminState(state_version=4, attestation_policy=policy, attestation_policy_digest=common.canonical_digest(policy), minimum_appliance_build_svn=0, trust_anchors={"adopted": {"artifact_id": "art_ta", "digest": D1, "description": "install bundle", "received_at": NOW}, "adopted_at": NOW}, key_manager={}, route_opt_ins={"call1_confidential": {"enabled": True, "endpoint_base_url": "https://pro1.call1.cc"}}, masking={}, section_changes={"route_opt_ins": {"changed_at": NOW, "changed_by_account_id": "acct_1", "audit_event_id": "evt_3", "state_version": 4}}, running_build={"manifest_digest": D2, "approved": True, "started_at": NOW}, updated_at=NOW)


def _audit_event() -> events.AuditEvent:
    body = events.AuditEventBody(id="evt_1", sequence=1, occurred_at=NOW, actor={"kind": "reviewer", "account_id": "acct_1"}, action="route_opt_in_changed", target={"kind": "admin_state", "id": "route_opt_ins"}, details={"route_class": "customer_directed", "enabled": True})
    return events.AuditEvent(**body.model_dump(), event_digest=events.audit_event_digest(body))


def _evidence() -> Dict[str, Any]:
    return {"manifest_digest": D1, "log_id": "log", "log_index": 1, "checkpoint_digest": D2, "witness_cosignatures": 2, "witness_quorum": 2, "rebuild_statements": 1, "source_commit": "abc", "sbom_digest": D0, "release_svn": 1, "log_age_seconds": 100}


def _registered_conversation(**metadata):
    return calls.Conversation(
        id="conv_1", ingestion_kind="call_audio", call_id="call_1", created_at=NOW,
        source={"kind": "api_upload", "content_digest": D0, "received_at": NOW},
        call_metadata={"agent_id": "A1", "agent_display_name": "Samantha Reyes", "agent_extension": "104", **metadata})


def _samples():
    pro1_route = _route(custody.RouteClass.CALL1_CONFIDENTIAL, custody.ProviderType.PRO1, "pro1.call1.cc", attestation_policy_version=D1)
    transcript = artifacts.Artifact(id="art_t", conversation_id="conv_1", linked=True, version=1, storage="object", committed_at=NOW, kind="transcript", content_type="application/json", size_bytes=10, checksum=D0, content_contract="transcript.v1", sensitivity="masked", producing_job_id="job_asr", linked_by_receipt_id="rcpt_1")
    yield _job()
    yield jobs.ClaimedJob(job=_job(), attempt_number=1, claim_token=TOKEN, lease_expires_at=NOW, slot_offer_index=0, final_attempt=False, inputs=[{"role": "transcript", "artifact": transcript}], upstream=[{"job_id": "job_asr", "job_type": "asr", "edge": "requires", "status": "SUCCEEDED"}])
    yield jobs.CompletionRequest(claim_token=TOKEN, completion_key="job_1:1:complete", outputs=[{"role": "assessment", "artifact_id": "art_o", "checksum": D2}, {"role": "prompt_input", "artifact_id": "art_p", "checksum": D1}], usage=_usage(session_id="sess_p1", session_setup_seconds=3.1, tokens_input={"count": 1200, "source": "provider_reported"}), provenance=jobs.AttemptProvenance(worker_id="w1", installation_id="inst_1", adapter_id="pro1", adapter_version="1", model_revision=D0, route=pro1_route, attestation=_attestation(), key_release_ref="kr_1"))
    yield jobs.CompletionRequest(claim_token=TOKEN, completion_key="job_1:3:complete", outputs=[{"role": "assessment", "artifact_id": "art_o", "checksum": D2}, {"role": "prompt_input", "artifact_id": "art_p", "checksum": D1}], usage=_usage(outcome="failed", error_code="pro1_unreachable", tokens_input={"count": None, "source": "unavailable"}, tokens_output={"count": None, "source": "unavailable"}), provenance=jobs.AttemptProvenance(worker_id="w1", installation_id="inst_1", adapter_id="pro1", adapter_version="1", model_revision=D0, route=pro1_route, pro1_failure={"failed_check": "evidence_request", "error_code": "pro1_unreachable", "policy_version": D1}), follow_on={"jobs": [_job_def("esc", "qa_escalation", requires_job_ids=["job_1"], selection=_selection().model_dump(), parameters={"criterion_id": "greeting", "escalation_trigger": "provider_error", "rubric": RUBRIC_REF}, resource_estimate={"size_class": "m", "memory_slot": "local_memory"})], "add_dependencies": [{"dependent_job_id": "job_card", "requires_ref": "esc", "input_role": "escalation:greeting", "output_role": "assessment"}]})
    yield calls.GroupGraphStake(graph_id="graph_2", graph_created_at=NOW, draft_test=True, has_group_jobs=True, publisher="in_progress")
    yield rubrics.RubricSnapshotRequest(rubric_id="std", version=2)
    yield custody.KeyReleaseRecord(id="kr_1", installation_id="inst_1", session_id="sess_p1", key_id="k1", key_source={"kind": "local"}, evidence_digest=D0, release_id="r", manifest_digest=D1, approval_id="apr_1", policy_version=D2, released_at=NOW, expires_at=NOW, audit_event_id="evt_1")
    yield custody.Pro1Connection(status="pending_verification", connection_version=4, cleared_at=NOW, cleared_by_account_id="acct_1", updated_at=NOW)
    yield reviews.ReviewQueueItem(id="rvw_1", call_id="call_1", conversation_id="conv_1", rule_id="triage", rule_name="Triage", stream="TRIAGE", reason="critical breach", urgency_score=80, evaluation_version=2, stale=False, status="PENDING", item_version=1, created_at=NOW)
    yield reviews.CallReviewState(call_id="call_1", review_version=3, current_evaluation_version=2, reviewed_evaluation_version=2, staleness="current", escalation_status="PENDING", updated_at=NOW)
    yield auth.Invitation(id="inv_1", email="ann@example.com", display_name="Ann", role="reviewer", status="pending", delivery="out_of_band", token_hash=D0, issued_at=NOW, expires_at=NOW)
    yield auth.ServiceKeyRecord(id="key_1", installation_id="inst_1", label="primary", scopes=["jobs:claim", "jobs:write"], key_prefix="c1sk_ab12CD", key_hash=D0, created_at=NOW)
    yield auth.SignInResponse(session={"session_id": "sess_1", "account_id": "acct_1", "email": "ann@example.com", "display_name": "Ann", "role": "reviewer", "permissions": ["read_calls"], "created_at": NOW, "last_seen_at": NOW, "idle_expires_at": NOW, "absolute_expires_at": NOW, "authenticator_id_used": "cred_1", "prompt_second_authenticator": True, "csrf_token": "c" * 32})
    yield release_trust.ReleaseApproval(id="apr_1", kind="pro1_release", release_id="r1", manifest_sha256=D1, log_id="log", log_index=1, checkpoint_digest=D2, release_svn=1, evidence=_evidence(), status="approved", approved_by_account_id="acct_1", approved_at=NOW, audit_event_id="evt_2")
    yield _admin_state(_policy())
    yield _audit_event()
    yield _hardware()
    yield calls.CallDetail(call={"call_id": "call_1", "conversation_id": "conv_1", "agent_id": "A1", "duration_seconds": 100, "channels": 2, "channel_layout": "STEREO", "sample_rate": 16000, "codec": "pcm_s16le", "silence_ratio": 0.1, "overtalk_duration": 1.0, "created_at": NOW}, results=[{"kind": "qa", "state": "pending"}], pending_work={"jobs_total": 5, "jobs_succeeded": 2, "jobs_running": 1, "jobs_queued": 1, "jobs_blocked": 1, "jobs_failed": 0, "jobs_cancelled": 0, "settled": False}, review_version=0, change_cursor="e2-00000000000000000042")
    yield calls.EvaluationView(call_id="call_1", artifact_id="art_s", version=2, rubric=SCORECARD_RUBRIC, overall_score=90, passed=True, critical_failure=False, requires_human_review=False, verdicts=[{"criterion_id": "greet", "criterion_name": "Greeting", "status": "PASS", "confidence": 0.9, "reasoning": "said hello", "quote_turn_id": 0}], evaluated_at=NOW)
    yield calls.ConversationRegistration(ingestion_kind="call_audio", source={"kind": "s3_event", "bucket": "recordings", "object_key": "2026/09/call.wav", "etag": "abc", "received_at": NOW}, call_metadata={"agent_id": "A1", "agent_channel": 0})
    yield calls.ConversationRegistered(conversation=_registered_conversation(), created=False, metadata_updated=True, updated_fields=["agent_display_name", "agent_extension"])
    yield contents.TranscriptContent(duration_seconds=12.5, is_redacted=False, turns=[{"turn_id": 0, "speaker": "AGENT", "start_time": 0, "end_time": 2.5, "text": "Hello", "word_timestamps": [{"word": "Hello", "start_time": 0, "end_time": 0.4, "probability": 0.98}]}])
    yield contents.QaScorecardContent(rubric={"rubric_id": "std", "draft_revision": 3, "digest": D1}, overall_score=50, passed=False, critical_failure=True, requires_human_review=True, verdicts=[], evaluated_at=NOW)
    yield usage.PriceTable(version=1, currency="USD", entries=[{"route_class": "appliance", "hardware_profile_id": "hw_1", "basis": "per_inference_hour", "unit_price": 0.12, "effective_from": NOW}], updated_at=NOW)
    yield _upload_grant()


@pytest.mark.parametrize("sample", list(_samples()), ids=lambda s: type(s).__name__)
def test_models_round_trip_through_json(sample):
    model = type(sample)
    restored = model.model_validate_json(sample.model_dump_json())
    assert restored == sample
    assert model.model_validate(sample.model_dump(mode="json")) == sample
    assert common.canonical_digest(restored) == common.canonical_digest(sample)


def test_extra_fields_are_rejected_everywhere():
    strict = [name for name, m in contracts.all_models().items() if m.model_config.get("extra") == "forbid"]
    assert len(strict) == len(contracts.all_models())
    with pytest.raises(ValidationError):
        calls.CallMetadata(agent_id="A", customer_phone="+15555550100")


def test_source_reference_dedup_identity():
    s3 = calls.SourceReference(kind="s3_event", bucket="b", object_key="k", etag="e", received_at=NOW)
    assert s3.dedup_identity == "s3:b/k#e"
    with pytest.raises(ValidationError):
        calls.SourceReference(kind="local_import", received_at=NOW)
    assert calls.SourceReference(kind="local_import", content_digest=D0, received_at=NOW).dedup_identity == f"local_import:{D0}"


def test_criterion_ids_are_slot_safe():
    """A criterion ID becomes the qa_assessment slot, so the rubric model must refuse any ID the
    artifact slot rules would reject, or that would land in the draft-test namespace."""
    def criterion(cid):
        return rubrics.RubricCriterion.model_validate({"criterion_id": cid, "name": "n", "check": {}})

    for ok in ("COMP-01", "greeting", "FDCPA-ARR-01", "C1", "x" * artifacts.SLOT_MAX_LENGTH):
        assert criterion(ok).criterion_id == ok
        assert not artifacts.is_draft_test_slot(ok)
        assert artifacts._slot(ok) == ok
    for bad in ("draft:x", "draft:x y", "a b", "segment:1", "", "-lead", "x" * (artifacts.SLOT_MAX_LENGTH + 1)):
        with pytest.raises(ValidationError):
            criterion(bad)


def test_every_route_has_a_delivery_stage():
    from call1.contracts import api
    assert set(api.STAGE_BY_OPERATION) <= {r.operation_id for r in api.ROUTES}
    assert all(r.stage in api.DELIVERY_STAGES for r in api.ROUTES)
    doc = api.build_openapi()
    stages = [op["x-call1-stage"] for item in doc["paths"].values() for op in item.values()]
    assert len(stages) == len(api.ROUTES) and set(stages) <= set(api.DELIVERY_STAGES)


def test_semantic_search_uses_the_shared_embedder_since_1_2_0():
    from call1.contracts import api
    route = next(r for r in api.ROUTES if r.operation_id == "semanticSearch")
    assert "pending" not in route.summary.lower() and "nvidia/Nemotron-3-Embed-1B-BF16" in route.description
    assert errors.ErrorCode.SEARCH_UNAVAILABLE in route.errors and errors.ERROR_HTTP_STATUS[errors.ErrorCode.SEARCH_UNAVAILABLE] == 503


# --- 1.1.0: agent identity and re-registration ---------------------------------------------------


def test_contract_1_1_0_is_a_minor_version_of_v1():
    assert contracts.CONTRACT_VERSION.startswith("1.") and contracts.CONTRACT_VERSION >= "1.1.0"
    assert contracts.STORE_API_PREFIX == "/store/v1"
    readme = (REPO / "call1" / "contracts" / "README.md").read_text()
    assert "Version 1.1.0" in readme and "Conversation re-registration" in readme


# --- 1.2.0: the search embedder (decision 18) ------------------------------------------------


def test_contract_1_2_0_adds_the_search_embedder_additively():
    assert contracts.CONTRACT_VERSION >= "1.2.0"
    readme = (REPO / "call1" / "contracts" / "README.md").read_text()
    assert "Version 1.2.0" in readme
    assert jobs.ReanalysisKind.EMBEDDINGS.value == "embeddings" and jobs.REANALYSIS_KIND_AFFECTS[jobs.ReanalysisKind.EMBEDDINGS] == frozenset()
    jobs.ReanalysisRequestCreate(kind=jobs.ReanalysisKind.EMBEDDINGS)  # no extra fields needed
    # Additive: a 1.1 response without the new fields still validates, and the defaults say "unknown / none".
    old = calls.SemanticSearchResponse.model_validate({"query": "q", "count": 0, "results": []})
    assert old.embedding_scheme is None and old.calls_needing_reembedding == 0
    assert admin.StoreStatus.model_fields["search_embedder"].default is None
    assert {s.value for s in admin.SearchEmbedderState} == {"not_installed", "installed", "loaded", "failed", "fake"}
    # The embeddings job is a model stage: it freezes an embeddings selection (unchanged rule).
    assert jobs.JOB_TYPE_RULES[jobs.JobType.EMBEDDINGS].purpose is catalog.ModelPurpose.EMBEDDINGS


def test_call_metadata_carries_optional_agent_display_name_and_extension():
    legacy = calls.CallMetadata.model_validate({"agent_id": "A1", "agent_channel": 0})
    assert legacy.agent_display_name is None and legacy.agent_extension is None
    meta = calls.CallMetadata(agent_id="A1", agent_display_name="Samantha Reyes", agent_extension="104")
    assert calls.CallMetadata.model_validate_json(meta.model_dump_json()) == meta
    for ok in ("104", "4410#2", "x-22", "*7", "1" * calls.AGENT_EXTENSION_MAX_LENGTH):
        assert calls.CallMetadata(agent_extension=ok).agent_extension == ok
    for bad in ("", "10 4", "1" * (calls.AGENT_EXTENSION_MAX_LENGTH + 1), "ext/4"):
        with pytest.raises(ValidationError):
            calls.CallMetadata(agent_extension=bad)
    for ok in ("Bob", "Samantha O'Neil-Reyes", "Ana María", "x" * calls.AGENT_DISPLAY_NAME_MAX_LENGTH):
        assert calls.CallMetadata(agent_display_name=ok).agent_display_name == ok
    for bad in ("", " Bob", "Bob ", "x" * (calls.AGENT_DISPLAY_NAME_MAX_LENGTH + 1)):
        with pytest.raises(ValidationError):
            calls.CallMetadata(agent_display_name=bad)


def test_agent_label_rule():
    assert calls.agent_label("A1", "Samantha Reyes", "104") == "Samantha Reyes (104)"
    assert calls.agent_label("A1", "Samantha Reyes", None) == "Samantha Reyes"
    assert calls.agent_label("A1", None, "104") == "A1 (104)"
    assert calls.agent_label("A1") == "A1"
    assert calls.agent_label("Unknown") == "Unknown"


def test_views_that_show_an_agent_carry_display_name_and_extension():
    for model in (calls.CallRecordView, calls.CallListItem, reviews.ReviewQueueItem, reviews.EscalationListItem):
        fields = model.model_fields
        assert {"agent_id", "agent_display_name", "agent_extension"} <= set(fields), model.__name__
        assert not fields["agent_display_name"].is_required() and not fields["agent_extension"].is_required()
    schemas = api.build_openapi()["components"]["schemas"]
    for name in ("CallListItem", "CallRecordView", "ReviewQueueItem", "EscalationListItem"):
        props = schemas[name]["properties"]
        assert {"agent_display_name", "agent_extension"} <= set(props), name
    assert "display name" in calls.CallListQuery.model_fields["text"].description


def test_merge_call_metadata_replaces_only_the_fields_the_body_sets():
    stored = calls.CallMetadata(agent_id="A1", agent_channel=0, external_call_ref="pbx-9", recorded_at=NOW)

    # identical body: a pure replay
    same = calls.CallMetadata.model_validate({"agent_id": "A1", "agent_channel": 0})
    merged, changed = calls.merge_call_metadata(stored, same)
    assert changed == [] and merged is stored

    # a re-upload that adds identity keeps everything it does not mention
    body = calls.CallMetadata.model_validate({"agent_display_name": "Bob Lee", "agent_extension": "202"})
    merged, changed = calls.merge_call_metadata(stored, body)
    assert changed == ["agent_display_name", "agent_extension"]
    assert (merged.agent_id, merged.agent_channel, merged.external_call_ref, merged.recorded_at) == ("A1", 0, "pbx-9", NOW)
    assert calls.agent_label(merged.agent_id, merged.agent_display_name, merged.agent_extension) == "Bob Lee (202)"

    # a body that omits agent_id never resets a known agent to the default
    merged, changed = calls.merge_call_metadata(stored, calls.CallMetadata.model_validate({}))
    assert changed == [] and merged.agent_id == "A1"

    # a body that sets agent_id changes it; an explicit null clears an optional field
    merged, changed = calls.merge_call_metadata(stored, calls.CallMetadata.model_validate({"agent_id": "A2", "external_call_ref": None}))
    assert changed == ["agent_id", "external_call_ref"]
    assert merged.agent_id == "A2" and merged.external_call_ref is None and merged.agent_channel == 0

    # the body's presence is what counts over the wire, too
    wire = calls.ConversationRegistration.model_validate_json(
        '{"ingestion_kind": "call_audio", "source": {"kind": "api_upload", "content_digest": "%s", "received_at": "2026-09-24T12:00:00Z"},'
        ' "call_metadata": {"agent_display_name": "Bob Lee"}}' % D0)
    assert wire.call_metadata.model_fields_set == {"agent_display_name"}
    merged, changed = calls.merge_call_metadata(stored, wire.call_metadata)
    assert changed == ["agent_display_name"] and merged.agent_id == "A1"

    # nothing stored before (e.g. a conversation registered without metadata)
    merged, changed = calls.merge_call_metadata(None, calls.CallMetadata(agent_id="A1", agent_extension="7"))
    assert merged.agent_extension == "7" and changed == ["agent_id", "agent_extension"]


def test_conversation_registered_reports_metadata_updates():
    conversation = _registered_conversation()
    plain = calls.ConversationRegistered(conversation=conversation, created=False)
    assert plain.metadata_updated is False and plain.updated_fields == []
    # a 1.0.x Store response (no new fields) still validates
    assert calls.ConversationRegistered.model_validate({"conversation": conversation.model_dump(mode="json"), "created": True}).metadata_updated is False
    updated = calls.ConversationRegistered(conversation=conversation, created=False, metadata_updated=True, updated_fields=["agent_extension"])
    assert updated.updated_fields == ["agent_extension"]
    for bad in (
        {"created": True, "metadata_updated": True, "updated_fields": ["agent_id"]},
        {"created": False, "metadata_updated": True, "updated_fields": []},
        {"created": False, "metadata_updated": False, "updated_fields": ["agent_id"]},
        {"created": False, "metadata_updated": True, "updated_fields": ["customer_phone"]},
    ):
        with pytest.raises(ValidationError):
            calls.ConversationRegistered(conversation=conversation, **bad)
    response = api.build_openapi()["components"]["schemas"]["ConversationRegistered"]
    assert {"metadata_updated", "updated_fields"} <= set(response["properties"])


def test_metadata_updates_are_audited_and_announced_without_reprocessing():
    assert events.AuditAction.CALL_METADATA_UPDATED.value == "call_metadata_updated"
    assert events.CALL_METADATA_UPDATED_STATUS == "metadata_updated"
    assert events.ChangeKind.CALL in events.CHANGE_KINDS_BY_PRINCIPAL["reviewer"]
    events.ChangeEvent(cursor="e2-00000000000000000043", occurred_at=NOW, kind="call", resource_id="call_1", conversation_id="conv_1", call_id="call_1", status=events.CALL_METADATA_UPDATED_STATUS)
    route = next(r for r in api.ROUTES if r.operation_id == "registerConversation")
    assert route.audited and route.idempotency is api.Idempotency.NATURAL
    for phrase in ("merge_call_metadata", "call_metadata_updated", "metadata_updated", "Nothing is reprocessed"):
        assert phrase in route.description, phrase
    op = api.build_openapi()["paths"]["/store/v1/conversations"]["post"]
    assert op["x-call1-audited"] is True


# --- 1.3.0: Contact Signals v2 (docs/ContactSignalsV2.md sections 6, 7 and 17 "F1") -------------

SPK = contents.SpeakerRole
SEED_PATH = REPO / "call1" / "store" / "seeds" / "signals_retail_v1.json"
T_CHECKSUM = "sha256:" + "ab" * 32


def _builtins(**overrides):
    out = []
    for cid, b in signals.BUILTIN_SIGNAL_CATEGORIES.items():
        out.append({"category_id": cid, "builtin": True, "name": b.name, "gloss": b.gloss, "speaker": b.speaker.value, **overrides.get(cid, {})})
    return out


REASON = {"field_id": "reason", "name": "Reason", "type": "enum", "description": "Why the caller wants to cancel.", "enum_values": ["price", "service quality", "moving", "other"], "pii_class": "none"}
CANCEL = {"subcategory_id": "cancel_account", "name": "Cancel account", "gloss": "Caller wants to cancel their account", "examples": ["cancel"], "fields": [REASON]}
UPSELL = {"category_id": "upsell_attempt", "builtin": False, "name": "Upsell attempt", "gloss": "Offers something beyond the ask", "description": "Agent offers an add-on the caller did not ask for.", "speaker": "AGENT",
          "fields": [{"field_id": "accepted", "name": "Accepted", "type": "boolean", "description": "Whether the caller accepted the offer.", "pii_class": "none"}]}


def _custom(cid: str, **kw):
    return {"category_id": cid, "builtin": False, "name": kw.pop("name", cid.replace("_", " ").title()), "gloss": kw.pop("gloss", "Some behavior worth flagging"), **kw}


def _taxonomy(*custom, **builtin_overrides) -> signals.SignalTaxonomy:
    return signals.SignalTaxonomy(categories=_builtins(**builtin_overrides) + list(custom))


def _version(t: signals.SignalTaxonomy, version: int) -> signals.SignalTaxonomyVersion:
    return signals.SignalTaxonomyVersion(version=version, digest=signals.taxonomy_digest(t), taxonomy=t, published_at=NOW, published_by_account_id="acct_admin" if version > 1 else None)


def _provenance(stage: str, **kw) -> Dict[str, Any]:
    return {"stage": stage, "catalog_entry_id": "fake-signal-classifier", "model_revision": "fake-1", "adapter_version": "1", "calibration_id": "fake-signals-v1", "device": "fake", "route_class": "appliance", "masked": True, "rows": 4, **kw}


def _span_view(block=0) -> Dict[str, Any]:
    return {"block": block, "first_window": 0, "last_window": 1, "timing": "interpolated", "context_start": 0.0, "context_end": 12.0}


def _v2_hit(**kw) -> Dict[str, Any]:
    t = _taxonomy(intent={"subcategories": [CANCEL]})
    intent = t.category("intent")
    hit_id = contents.signal_hit_id("intent", signals.category_digest(intent), T_CHECKSUM, 1, 0)
    base = {"id": hit_id, "kind": "intent", "label": "Caller objective", "start": 4.0, "end": 9.5, "speaker": "CALLER", "quote": "I want to cancel my account", "turn_id": 1, "char_start": 4, "char_end": 31, "confidence": 0.93,
            "category_id": "intent", "category_digest": contents.short_digest(signals.category_digest(intent)), "category_confidence": 0.88, "subcategory_id": "cancel_account", "subcategory_label": "Cancel account",
            "subcategory_digest": contents.short_digest(signals.subcategory_digest(intent.subcategories[0])), "subcategory_confidence": 0.81, "span": _span_view(),
            "fields": [{"field_id": "reason", "type": "enum", "status": "extracted", "value": "price", "name": "Reason"}]}
    return {**base, **kw}


def _segmentation() -> Dict[str, Any]:
    return {"segmenter_version": "seg-v1", "window_seconds": 7.0, "segments": 12, "scored_segments": 10, "skipped_unattributed": 0, "skipped_system": 0, "interpolated_turns": 5}


def _v2_content(stages=None, completeness="complete", partial_reason=None, hits=None, **kw) -> Dict[str, Any]:
    t = _taxonomy(intent={"subcategories": [CANCEL]})
    stages = stages if stages is not None else [{"stage": "categorize", "included": True, "provenance": _provenance("categorize")}, {"stage": "subcategorize", "included": True}, {"stage": "extract", "included": True}]
    return {"completeness": completeness, "partial_reason": partial_reason, "signals": hits if hits is not None else [_v2_hit()], "passes": [], "transcript_fingerprint": D0, "generated_at": NOW,
            "pipeline": "v2", "taxonomy": {"version": 2, "digest": signals.taxonomy_digest(t)}, "stages": stages, "segmentation": _segmentation(),
            "stage1_digests": {scope: signals.stage1_digest(t, SPK(scope)) for scope in ("AGENT", "CALLER", "UNKNOWN")}, **kw}


# F1: taxonomy validators --------------------------------------------------------------------


def test_signal_builtins_are_every_kind_but_custom_and_follow_the_v1_labels():
    from call1.pipeline.contact_signals import ALL_SIGNALS

    kinds = {k.value for k in contents.ContactSignalKind}
    assert set(signals.BUILTIN_SIGNAL_CATEGORIES) == kinds - {"custom"}
    assert contents.ContactSignalKind.CUSTOM.value == "custom"
    for cid, b in signals.BUILTIN_SIGNAL_CATEGORIES.items():
        label, speaker, _ = ALL_SIGNALS[cid]
        assert b.name == label and b.speaker.value == speaker.upper(), cid
        assert len(b.gloss) <= 80
    v1 = signals.builtin_signal_taxonomy()
    assert [c.category_id for c in v1.categories] == list(signals.BUILTIN_SIGNAL_CATEGORIES) and all(c.builtin and c.active for c in v1.categories)
    assert signals.BUILTIN_EDITABLE_FIELDS == ("threshold", "subcategory_threshold", "subcategories", "fields", "narrow_quote", "examples", "recipe")


def test_signal_taxonomy_keeps_builtins_present_and_fixed():
    _taxonomy()
    _taxonomy(intent={"threshold": 0.3, "subcategory_threshold": 0.6, "subcategories": [CANCEL], "fields": [], "narrow_quote": True, "examples": ["calling about"]})
    with pytest.raises(ValidationError):
        signals.SignalTaxonomy(categories=_builtins()[1:] + [_custom("extra")])  # intent missing
    with pytest.raises(ValidationError, match="unique"):
        signals.SignalTaxonomy(categories=_builtins() + [_builtins()[0]])
    for change in ({"name": "Why they called"}, {"gloss": "Caller states a goal"}, {"speaker": "AGENT"}, {"speaker": None}, {"active": False}, {"description": "Anything"}):
        with pytest.raises(ValidationError, match="built-in"):
            _taxonomy(intent=change)
    with pytest.raises(ValidationError, match="builtin is true"):
        _taxonomy(intent={"builtin": False})
    with pytest.raises(ValidationError, match="builtin is true"):
        _taxonomy({**_custom("cancellation"), "builtin": True})
    for speaker in ("SYSTEM", "UNKNOWN"):
        with pytest.raises(ValidationError, match="scoped"):
            _taxonomy(_custom("x1", speaker=speaker))
    for speaker in ("AGENT", "CALLER", None):
        _taxonomy(_custom("x1", speaker=speaker))
    with pytest.raises(ValidationError, match="active category names"):
        _taxonomy(_custom("dupe", name="caller OBJECTIVE"))
    _taxonomy(_custom("dupe", name="caller OBJECTIVE", active=False))  # a retired node keeps its name


def test_signal_taxonomy_ids_are_patterned_and_reserved_ids_refused():
    for ok in ("a", "cancel_account", "x-1", "a" * 40):
        _taxonomy(_custom(ok if ok not in signals.BUILTIN_SIGNAL_CATEGORIES else "zz"))
    for bad in ("Bad", "has.dot", "_lead", "a" * 41, "sp ace", ""):
        with pytest.raises(ValidationError):
            _taxonomy(_custom(bad))
    for reserved in ("other", "not"):
        with pytest.raises(ValidationError, match="reserved"):
            _taxonomy(intent={"subcategories": [{**CANCEL, "subcategory_id": reserved}]})
    with pytest.raises(ValidationError, match="reserved"):
        _taxonomy(intent={"fields": [{**REASON, "field_id": "quote"}]})
    assert signals.RESERVED_SUBCATEGORY_IDS == {"other", "not"} and signals.RESERVED_FIELD_IDS == {"quote"}
    # 'none' is stage 1's none-of-these option and 'custom' the kind of every custom hit
    assert signals.RESERVED_CATEGORY_IDS == {signals.SIGNAL_NONE_OPTION, contents.ContactSignalKind.CUSTOM.value} == {"none", "custom"}
    for reserved in ("none", "custom"):
        with pytest.raises(ValidationError, match="reserved"):
            _taxonomy(_custom(reserved))
        with pytest.raises(ValidationError, match="reserved"):
            signals.SignalCategory(**{**_custom(reserved), "builtin": True})
    for kind in signals.BUILTIN_SIGNAL_CATEGORIES:  # a built-in kind's ID is the built-in's alone
        with pytest.raises(ValidationError, match="builtin is true"):
            signals.SignalCategory(**_custom(kind))
    _taxonomy(_custom("other"), _custom("not"))  # stage-2 reserved IDs are fine as category IDs
    with pytest.raises(ValidationError, match="unique in their category"):
        _taxonomy(intent={"subcategories": [CANCEL, {**CANCEL, "name": "Close account"}]})
    with pytest.raises(ValidationError, match="active subcategory names"):
        _taxonomy(intent={"subcategories": [CANCEL, {**CANCEL, "subcategory_id": "close", "name": "CANCEL ACCOUNT"}]})
    with pytest.raises(ValidationError, match="unique on a category"):
        _taxonomy(intent={"fields": [REASON], "subcategories": [CANCEL]})  # 'reason' on both halves of one path
    _taxonomy(intent={"subcategories": [CANCEL, {**CANCEL, "subcategory_id": "close", "name": "Close account"}]})  # same field ID on sibling paths is fine


def test_signal_fields_refuse_forbidden_pii_and_keep_enum_values_to_enums():
    assert signals.FORBIDDEN_FIELD_PII_CLASSES == {signals.FieldPiiClass(v) for v in ("caller_name", "account_number", "card_number", "phone", "email", "address", "url", "secret", "government_id")}
    for cls in signals.FieldPiiClass:
        field = {**REASON, "pii_class": cls.value}
        if cls in signals.FORBIDDEN_FIELD_PII_CLASSES:
            with pytest.raises(ValidationError, match="masked before any model"):
                signals.SignalField(**field)
        else:
            signals.SignalField(**field)
    with pytest.raises(ValidationError, match="enum field lists"):
        signals.SignalField(**{**REASON, "enum_values": []})
    for t in ("string", "boolean", "number", "amount", "date"):
        with pytest.raises(ValidationError, match="enum field lists"):
            signals.SignalField(**{**REASON, "type": t})
        signals.SignalField(**{**REASON, "type": t, "enum_values": []})
    with pytest.raises(ValidationError, match="unique"):
        signals.SignalField(**{**REASON, "enum_values": ["Price", "price"]})
    for bad in (["x" * 41], [""], [" padded"], ["a"] * 13):
        with pytest.raises(ValidationError):
            signals.SignalField(**{**REASON, "enum_values": bad})
    with pytest.raises(ValidationError, match="one line"):
        signals.SignalField(**{**REASON, "description": "line one\nline two"})
    assert "decision 22" in signals.FieldPiiClass.__doc__  # an opaque order number is business data: pii_class none


def test_signal_taxonomy_outer_ceilings():
    many = [{**CANCEL, "subcategory_id": f"s{i}", "name": f"Sub {i}"} for i in range(13)]
    with pytest.raises(ValidationError, match="12 active"):
        _taxonomy(intent={"subcategories": many})
    _taxonomy(intent={"subcategories": many[:12] + [{**s, "active": False} for s in many[12:]]})
    with pytest.raises(ValidationError):
        _taxonomy(intent={"subcategories": [{**CANCEL, "subcategory_id": f"s{i}", "name": f"Sub {i}", "active": False} for i in range(25)]})
    with pytest.raises(ValidationError):  # at most 16 custom categories in total (24 categories)
        _taxonomy(*[_custom(f"c{i}", name=f"C {i}", active=False) for i in range(17)])
    _taxonomy(*[_custom(f"c{i}", name=f"C {i}", active=i < 8) for i in range(16)])
    with pytest.raises(ValidationError):
        _taxonomy(_custom("long", gloss="x" * 81))
    with pytest.raises(ValidationError):
        _taxonomy(intent={"fields": [{**REASON, "field_id": f"f{i}"} for i in range(9)]})
    with pytest.raises(ValidationError):
        _taxonomy(intent={"threshold": 0.99})
    for bad in (["x" * 121], ["two\nlines"], [" lead"], [{"text": "an object"}], ["a"] * 6):
        with pytest.raises(ValidationError):
            _taxonomy(intent={"examples": bad})
    _taxonomy(intent={"examples": ["x" * 120]})


def test_signal_caps_are_contract_parameters_the_save_validator_reads():
    p = common.CONTRACT_PARAMETERS
    assert (p.max_custom_signal_categories, p.max_active_subcategories, p.max_fields_per_path, p.max_option_gloss_chars, p.max_signal_alert_rules, p.signal_preview_max_calls, p.signal_backfill_max_calls, p.max_extraction_spans_per_call) == (8, 12, 12, 40, 50, 10, 500, 64)
    assert api.contract_extensions()["parameters"]["max_option_gloss_chars"] == 40
    check = signals.signal_taxonomy_cap_violations
    assert check(_taxonomy(UPSELL, intent={"subcategories": [CANCEL]})) == []
    assert check(signals.builtin_signal_taxonomy()) == []  # built-in glosses (up to 59 characters) are Call1's
    # custom gloss and subcategory gloss, by path, never by value
    t = _taxonomy(_custom("wordy", gloss="y" * 41), intent={"subcategories": [{**CANCEL, "gloss": "z" * 41}]})
    found = {(v.field, v.cap) for v in check(t)}
    assert found == {("categories[8].gloss", "max_option_gloss_chars"), ("categories[0].subcategories[0].gloss", "max_option_gloss_chars")}
    assert all("y" * 41 not in v.field for v in check(t))
    # active custom categories: the ninth active one is named; retired ones do not count
    t = _taxonomy(*[_custom(f"c{i}", name=f"C {i}") for i in range(9)], _custom("old", name="Old", active=False))
    assert [(v.field, v.cap, v.limit, v.actual) for v in check(t)] == [("categories[16]", "max_custom_signal_categories", 8, 9)]
    # fields per category + subcategory path
    wide = {"fields": [{**REASON, "field_id": f"c{i}"} for i in range(8)], "subcategories": [{**CANCEL, "fields": [{**REASON, "field_id": f"s{i}"} for i in range(5)]}]}
    assert [(v.field, v.actual) for v in check(_taxonomy(intent=wide))] == [("categories[0].subcategories[0].fields", 13)]
    # changing a parameter changes the refusal, with no shape change
    tighter = common.ContractParameters(max_option_gloss_chars=30, max_active_subcategories=1, max_fields_per_path=0)
    got = {(v.field, v.cap) for v in check(_taxonomy(intent={"subcategories": [CANCEL, {**CANCEL, "subcategory_id": "close", "name": "Close"}]}), tighter)}
    assert ("categories[0].subcategories[0].gloss", "max_option_gloss_chars") in got and ("categories[0].subcategories[1]", "max_active_subcategories") in got and ("categories[0].subcategories[0].fields", "max_fields_per_path") in got
    for bad in (dict(max_option_gloss_chars=81), dict(max_active_subcategories=13), dict(max_custom_signal_categories=17), dict(signal_preview_max_calls=11), dict(signal_backfill_max_calls=501)):
        with pytest.raises(ValidationError):
            common.ContractParameters(**bad)  # parameters stay inside the models' outer ceilings


def test_signal_definition_text_paths_name_every_text_by_path():
    t = _taxonomy(UPSELL, intent={"subcategories": [CANCEL], "examples": ["calling about", "i want to"]})
    paths = dict(signals.signal_taxonomy_text_paths(t))
    assert paths["categories[0].examples[1]"] == "i want to"
    assert paths["categories[0].subcategories[0].examples[0]"] == "cancel"
    assert paths["categories[0].subcategories[0].fields[0].enum_values[1]"] == "service quality"
    assert paths["categories[8].description"].startswith("Agent offers")
    assert paths["categories[8].fields[0].description"] and paths["categories[0].subcategories[0].gloss"] == CANCEL["gloss"]


# F1: digests and hit identity -------------------------------------------------------------------


def test_signal_digests_ignore_thresholds():
    base = _taxonomy(UPSELL, intent={"subcategories": [CANCEL]})
    tuned = _taxonomy({**UPSELL, "threshold": 0.3, "subcategory_threshold": 0.6}, intent={"subcategories": [CANCEL], "threshold": 0.2, "subcategory_threshold": 0.7})
    for scope in (SPK.AGENT, SPK.CALLER, SPK.UNKNOWN):
        assert signals.stage1_digest(base, scope) == signals.stage1_digest(tuned, scope)
    for a, b in zip(base.categories, tuned.categories):
        assert signals.category_digest(a) == signals.category_digest(b) and signals.stage2_digest(a) == signals.stage2_digest(b)
        assert signals.stage3_digest(a) == signals.stage3_digest(b)
        for sa, sb in zip(a.subcategories, b.subcategories):
            assert signals.subcategory_digest(sa) == signals.subcategory_digest(sb) and signals.stage3_digest(a, sa) == signals.stage3_digest(b, sb)
    assert signals.taxonomy_digest(base) != signals.taxonomy_digest(tuned)  # a threshold edit is still a new version


def test_signal_category_digest_ignores_siblings_name_and_description():
    base = _taxonomy(UPSELL, intent={"subcategories": [CANCEL]})
    grown = _taxonomy(UPSELL, _custom("competitor_mention", speaker="CALLER"), intent={"subcategories": [CANCEL, {**CANCEL, "subcategory_id": "close", "name": "Close"}], "fields": [{**REASON, "field_id": "other_reason"}]})
    for cid in ("intent", "upsell_attempt", "deferred"):
        assert signals.category_digest(base.category(cid)) == signals.category_digest(grown.category(cid)), cid
    renamed = _taxonomy({**UPSELL, "name": "Cross-sell attempt", "description": "Reworded."})
    old, new = base.category("upsell_attempt"), renamed.category("upsell_attempt")
    assert signals.category_digest(old) == signals.category_digest(new)
    for change in ({"gloss": "Offers an add-on"}, {"speaker": "CALLER"}, {"speaker": None}):
        assert signals.category_digest(signals.SignalCategory(**{**UPSELL, **change})) != signals.category_digest(old), change
    # the hit ID follows the category digest, the transcript revision, the turn and the block, never a threshold
    hid = contents.signal_hit_id("upsell_attempt", signals.category_digest(old), T_CHECKSUM, 12, 1)
    assert hid == f"upsell_attempt.{signals.category_digest(old)[7:19]}.{'ab' * 4}.t12b1"
    assert contents.signal_hit_id("upsell_attempt", signals.category_digest(new), T_CHECKSUM, 12, 1) == hid
    assert len(contents.signal_hit_id("a" * 40, D1, T_CHECKSUM, 9999, 9)) <= 75
    assert contents.signal_preview_hit_id("intent", 3, 0) == "intent.preview.t3b0" and contents.signal_span_key("intent", 3, 0) == "intent.t3b0"


def test_signal_stage_digests_follow_what_each_stage_reads():
    base = _taxonomy(UPSELL, intent={"subcategories": [CANCEL]})
    up = base.category("upsell_attempt")
    # stage 2 reads the category name and the active subcategories' IDs and glosses
    assert signals.stage2_digest(signals.SignalCategory(**{**UPSELL, "name": "Cross-sell attempt"})) != signals.stage2_digest(up)
    assert signals.stage2_digest(signals.SignalCategory(**{**UPSELL, "description": "Reworded."})) == signals.stage2_digest(up)
    intent = base.category("intent")
    for sub_change in ({"gloss": "Caller wants to close the account"}, {"subcategory_id": "close"}, {"active": False}):
        other = _taxonomy(intent={"subcategories": [{**CANCEL, **sub_change}]}).category("intent")
        assert signals.stage2_digest(other) != signals.stage2_digest(intent), sub_change
    for sub_change in ({"name": "Cancellation"}, {"description": "Now described."}, {"examples": ["close it"]}, {"fields": []}):
        other = _taxonomy(intent={"subcategories": [{**CANCEL, **sub_change}]}).category("intent")
        assert signals.stage2_digest(other) == signals.stage2_digest(intent), sub_change
    # stage 3 reads the path's fields, narrow_quote, and the names and descriptions
    assert signals.stage3_digest(signals.SignalCategory(**{**UPSELL, "description": "Reworded."})) != signals.stage3_digest(up)
    sub = intent.subcategories[0]
    for sub_change in ({"description": "Now described."}, {"name": "Cancellation"}, {"narrow_quote": True}, {"fields": [{**REASON, "description": "The stated reason."}]}):
        changed = signals.SignalSubcategory(**{**CANCEL, **sub_change})
        assert signals.stage3_digest(intent, changed) != signals.stage3_digest(intent, sub), sub_change
    assert signals.stage3_digest(intent, signals.SignalSubcategory(**{**CANCEL, "gloss": "Caller closes the account"})) == signals.stage3_digest(intent, sub)
    assert signals.subcategory_digest(signals.SignalSubcategory(**{**CANCEL, "name": "Cancellation", "description": "x"})) == signals.subcategory_digest(sub)
    # stage 1 reads the ordered option set of each speaker scope
    caller_custom = _taxonomy(UPSELL, _custom("cancellation_request", speaker="CALLER"), intent={"subcategories": [CANCEL]})
    either = _taxonomy(UPSELL, _custom("profanity", speaker=None), intent={"subcategories": [CANCEL]})
    assert signals.stage1_digest(caller_custom, SPK.CALLER) != signals.stage1_digest(base, SPK.CALLER)
    assert signals.stage1_digest(caller_custom, SPK.AGENT) == signals.stage1_digest(base, SPK.AGENT)
    assert signals.stage1_digest(caller_custom, SPK.UNKNOWN) == signals.stage1_digest(base, SPK.UNKNOWN)
    assert all(signals.stage1_digest(either, s) != signals.stage1_digest(base, s) for s in (SPK.AGENT, SPK.CALLER, SPK.UNKNOWN))
    assert [o[0] for o in signals.stage1_options(base, SPK.CALLER)] == ["intent", "issue", "friction", "caller_confirms_resolved", "caller_reports_unresolved"]
    assert [o[0] for o in signals.stage1_options(base, SPK.AGENT)] == ["fix_proposed", "agent_reports_completed", "deferred", "upsell_attempt"]
    assert signals.stage1_options(base, SPK.UNKNOWN) == [] and signals.stage1_options(either, SPK.UNKNOWN) == [("profanity", "Some behavior worth flagging")]
    assert signals.stage1_options(either, SPK.SYSTEM) == []
    assert signals.stage3_planned(intent, sub) and not signals.stage3_planned(base.category("issue"))


def test_signal_taxonomy_status_labels_what_each_edit_outdates():
    base = _taxonomy(UPSELL, intent={"subcategories": [CANCEL]})
    scored = _version(base, 2)
    digests = {s: signals.stage1_digest(base, SPK(s)) for s in ("AGENT", "CALLER", "UNKNOWN")}

    def status(t):
        return signals.signal_taxonomy_status(scored, _version(t, 3), digests)

    same = signals.signal_taxonomy_status(scored, scored, digests)
    assert (same.outdated_stages, same.thresholds_changed) == ([], False)
    threshold = status(_taxonomy({**UPSELL, "threshold": 0.4}, intent={"subcategories": [CANCEL], "subcategory_threshold": 0.7}))
    assert (threshold.outdated_stages, threshold.thresholds_changed, threshold.scored_version, threshold.current_version) == ([], True, 2, 3)
    assert status(_taxonomy(UPSELL, _custom("cancellation_request", speaker="CALLER"), intent={"subcategories": [CANCEL]})).outdated_stages == ["categorize"]
    assert status(_taxonomy({**UPSELL, "active": False}, intent={"subcategories": [CANCEL]})).outdated_stages == ["categorize"]
    assert status(_taxonomy(UPSELL, intent={"subcategories": [CANCEL, {**CANCEL, "subcategory_id": "close", "name": "Close", "fields": []}]})).outdated_stages == ["subcategorize"]
    assert status(_taxonomy(UPSELL, intent={"subcategories": [{**CANCEL, "fields": [REASON, {**REASON, "field_id": "competitor", "type": "string", "enum_values": [], "pii_class": "organization"}]}]})).outdated_stages == ["extract"]
    assert status(_taxonomy(UPSELL, intent={"subcategories": [{**CANCEL, "narrow_quote": True}]})).outdated_stages == ["extract"]
    assert status(_taxonomy({**UPSELL, "name": "Cross-sell attempt"}, intent={"subcategories": [CANCEL]})).outdated_stages == ["subcategorize", "extract"]  # upsell has a field, so the extractor reads the name
    assert status(_taxonomy(UPSELL, intent={"subcategories": [{**CANCEL, "description": "Reworded."}]})).outdated_stages == ["extract"]  # only the extractor reads descriptions
    no_fields = _taxonomy(_custom("quiet", speaker="AGENT", name="Quiet", description="One."))
    after = _taxonomy(_custom("quiet", speaker="AGENT", name="Quiet", description="Two."))
    s2 = signals.signal_taxonomy_status(_version(no_fields, 2), _version(after, 3), {s: signals.stage1_digest(no_fields, SPK(s)) for s in ("AGENT", "CALLER")})
    assert s2.outdated_stages == []  # nothing reads a description on a node without fields or narrowing
    v1 = signals.signal_taxonomy_status(None, _version(base, 3), {})
    assert (v1.scored_version, v1.outdated_stages, v1.thresholds_changed) == (None, [], False)
    with pytest.raises(ValidationError):
        signals.SignalTaxonomyStatus(scored_version=None, current_version=3, outdated_stages=["extract"], thresholds_changed=False)
    with pytest.raises(ValidationError):
        signals.SignalTaxonomyStatus(scored_version=2, current_version=3, outdated_stages=["extract", "categorize"], thresholds_changed=False)


def test_signal_taxonomy_redaction_tombstones_every_custom_text_path_uniquely():
    enum_field = {"field_id": "channel", "name": "Channel", "type": "enum", "description": "Where the caller came from.", "enum_values": ["web", "store", "phone"], "pii_class": "none"}
    close = {**CANCEL, "subcategory_id": "close", "name": "Close account", "gloss": "Caller wants the account closed", "description": "Closing, not pausing.", "examples": ["close it", "shut it down"]}
    t = _taxonomy(
        {**UPSELL, "examples": ["would you like", "add on"], "fields": [*UPSELL["fields"], enum_field]},
        _custom("loyalty", description="Mentions years as a customer.", subcategories=[CANCEL, close, {**CANCEL, "subcategory_id": "gone", "name": "Retired", "active": False}]),
        _custom("retired_one", name="Old one", active=False),
        intent={"subcategories": [CANCEL, {**close, "fields": []}], "fields": [enum_field], "examples": ["calling about"]},
    )
    v = _version(t, 4)
    r = signals.redact_signal_taxonomy_text(t)
    before, after = dict(signals.signal_taxonomy_text_paths(t)), dict(signals.signal_taxonomy_text_paths(r))
    assert list(before) == list(after)  # same paths, same structure
    builtin_constants = {f"categories[{i}].{k}" for i, c in enumerate(t.categories) if c.builtin for k in ("name", "gloss")}
    tombstones = [after[path] for path in after if path not in builtin_constants]
    assert tombstones == [signals.REDACTED_SIGNAL_TEXT.format(n=n) for n in range(1, len(tombstones) + 1)]  # numbered in text-path order
    assert all(len(x) <= 40 for x in tombstones) and all(after[p] == before[p] for p in builtin_constants)
    assert not set(tombstones) & set(before.values())  # no custom text survives
    assert [(c.category_id, c.builtin, c.active, c.speaker, c.threshold, [s.subcategory_id for s in c.subcategories], [(f.field_id, f.type, f.pii_class) for f in c.fields]) for c in r.categories] == \
        [(c.category_id, c.builtin, c.active, c.speaker, c.threshold, [s.subcategory_id for s in c.subcategories], [(f.field_id, f.type, f.pii_class) for f in c.fields]) for c in t.categories]
    assert signals.redact_signal_taxonomy_text(r) == r  # idempotent: redacting a redacted version returns it
    redacted = signals.SignalTaxonomyVersion(**{**v.model_dump(), "taxonomy": r, "text_redacted": True, "redacted_at": NOW, "redacted_by_account_id": "acct_admin"})
    assert redacted.digest == v.digest and redacted.ref == v.ref and signals.taxonomy_digest(r) != v.digest
    signals.SignalTaxonomyVersion.model_validate_json(redacted.model_dump_json())  # every later read validates
    assert not signals.signal_taxonomy_cap_violations(r)
    # a bare [REDACTED] on every path is what the numbering avoids
    with pytest.raises(ValidationError, match="unique"):
        _taxonomy({**_custom("a1"), "name": "[REDACTED]"}, {**_custom("a2"), "name": "[REDACTED]"})
    op = api.routes_by_operation()["redactSignalTaxonomyText"]
    assert "redact_signal_taxonomy_text" in op.description and "[REDACTED <n>]" in op.description


def test_signal_taxonomy_snapshot_is_reused_only_with_the_current_settings():
    t = _taxonomy(UPSELL)
    v = _version(t, 2)
    settings = signals.SignalSettings(pipeline="v2")
    snap = signals.SignalTaxonomySnapshotContent(source="published", taxonomy_ref=v.ref, taxonomy=t, settings=settings)
    assert signals.signal_taxonomy_snapshot_current(snap, v.ref, settings)
    for changed in ({"v1_fallback": True}, {"fallback_extraction_entry_id": "other-extractor"}, {"pipeline": "shadow"}):
        assert not signals.signal_taxonomy_snapshot_current(snap, v.ref, signals.SignalSettings(**{**settings.model_dump(), **changed})), changed
    assert not signals.signal_taxonomy_snapshot_current(snap, contents.SignalTaxonomyRef(version=3, digest=v.digest), settings)
    preview = signals.SignalTaxonomySnapshotContent(source="preview", taxonomy_ref={"version": None, "digest": v.digest}, taxonomy=t, settings=settings, preview_id="prv_1")
    assert not signals.signal_taxonomy_snapshot_current(preview, v.ref, settings)
    op = api.routes_by_operation()["mintSignalTaxonomySnapshot"]
    assert "current settings" in op.description and "signal_taxonomy_snapshot_current" in op.description


def test_signal_taxonomy_versions_and_snapshots_are_digest_checked():
    t = _taxonomy(UPSELL)
    v = _version(t, 2)
    assert v.ref == contents.SignalTaxonomyRef(version=2, digest=signals.taxonomy_digest(t))
    with pytest.raises(ValidationError, match="digest"):
        signals.SignalTaxonomyVersion(**{**v.model_dump(), "digest": D0})
    redacted = signals.SignalTaxonomyVersion(**{**v.model_dump(), "taxonomy": signals.redact_signal_taxonomy_text(t), "text_redacted": True, "redacted_at": NOW, "redacted_by_account_id": "acct_admin"})
    assert redacted.digest == v.digest  # the tombstone keeps the digest
    with pytest.raises(ValidationError, match="redact_signal_taxonomy_text"):  # a redacted version holds no custom text
        signals.SignalTaxonomyVersion(**{**redacted.model_dump(), "taxonomy": t})
    with pytest.raises(ValidationError, match="redaction"):
        signals.SignalTaxonomyVersion(**{**v.model_dump(), "redacted_at": NOW})
    settings = signals.SignalSettings()
    assert (settings.pipeline, settings.v1_fallback, settings.fallback_extraction_entry_id) == ("v2", False, None)
    published = signals.SignalTaxonomySnapshotContent(source="published", taxonomy_ref=v.ref, taxonomy=t, settings=settings)
    signals.SignalTaxonomySnapshotContent(source="preview", taxonomy_ref={"version": None, "digest": signals.taxonomy_digest(t)}, taxonomy=t, settings=settings, preview_id="prv_1")
    for bad in (dict(source="preview"), dict(preview_id="prv_1"), dict(taxonomy_ref={"version": None, "digest": v.digest}), dict(taxonomy_ref={"version": 2, "digest": D0})):
        with pytest.raises(ValidationError):
            signals.SignalTaxonomySnapshotContent(**{**published.model_dump(), **bad})
    payload = published.model_dump(mode="json")
    descriptor = dict(kind="signal_taxonomy_snapshot", slot=artifacts.signal_taxonomy_snapshot_slot(2), content_type="application/json", size_bytes=len(common.canonical_json(payload)), checksum=common.canonical_digest(payload), content_contract="signal_taxonomy_snapshot.v1", sensitivity="derived")
    assert descriptor["slot"] == "signals:v2" and artifacts.preview_signal_taxonomy_snapshot_slot("rq_9") == "draft:rq_9:signals:preview"
    artifacts.canonical_content("signal_taxonomy_snapshot.v1", payload)
    with pytest.raises(ValidationError, match="minted by Store"):
        artifacts.InlineArtifactCreate(**descriptor, payload=payload)
    with pytest.raises(ValidationError, match="minted by Store"):
        artifacts.UploadGrantRequest(**descriptor)
    with pytest.raises(ValidationError):
        artifacts.ArtifactDescriptor(**{**descriptor, "sensitivity": "masked"})
    route = api.routes_by_operation()["mintSignalTaxonomySnapshot"]
    assert route.request is signals.SignalTaxonomySnapshotRequest and route.response is artifacts.Artifact and route.idempotency is api.Idempotency.NATURAL
    assert [p.scope for p in route.principals] == [common.ServiceScope.JOBS_WRITE] and "redacted" in route.description
    assert signals.SIGNAL_TAXONOMY_INPUT_ROLE == "taxonomy" and api.contract_extensions()["signal_taxonomy_input_role"] == "taxonomy"


# F1: stage artifacts, provenance and the v2 result --------------------------------------------


def test_signal_stage_provenance_is_always_masked():
    signals_prov = contents.SignalStageProvenance(**_provenance("categorize"))
    assert signals_prov.masked is True
    with pytest.raises(ValidationError, match="masked"):
        contents.SignalStageProvenance(**_provenance("categorize", masked=False))
    with pytest.raises(ValidationError):
        contents.SignalStageProvenance(**_provenance("categorize", key_orders=3))
    assert catalog.ALWAYS_MASKED_PURPOSES == {catalog.ModelPurpose.SIGNAL_CATEGORY, catalog.ModelPurpose.SIGNAL_SUBCATEGORY, catalog.ModelPurpose.SIGNAL_EXTRACTION}
    assert catalog.ModelPurpose.SIGNAL_EXTRACTION in catalog.LLM_PURPOSES and not catalog.SIGNAL_CLASSIFIER_PURPOSES & catalog.LLM_PURPOSES


def _categories_content(**kw) -> Dict[str, Any]:
    t = _taxonomy(intent={"subcategories": [CANCEL]})
    segs = [{"index": i, "turn_id": 1, "window": i, "block": i // 4, "speaker": "CALLER", "char_start": i * 10, "char_end": i * 10 + 9, "start": i * 7.0, "end": i * 7.0 + 6.5, "timing": "interpolated"} for i in range(2)]
    base = {"mode": "run", "provenance": _provenance("categorize"), "segmenter_version": "seg-v1", "window_seconds": 7.0, "taxonomy_ref": {"version": 2, "digest": signals.taxonomy_digest(t)},
            "stage1_digests": {"CALLER": signals.stage1_digest(t, SPK.CALLER)}, "thresholds": {"intent": 0.3}, "transcript": {"artifact_id": "art_t", "checksum": T_CHECKSUM},
            "segments": segs, "scores": [{"index": 0, "probabilities": {"intent": 0.9, "none": 0.01}}, {"index": 1, "probabilities": {"intent": 0.5, "issue": 0.4, "none": 0.1}}],
            "spans": [{"span_key": "intent.t1b0", "category_id": "intent", "turn_id": 1, "block": 0, "first_window": 0, "last_window": 1, "peak_window": 0, "peak_probability": 0.9, "context_first": 0, "context_last": 1}],
            "skipped_unattributed": 0, "skipped_system": 0, "unscored_no_options": 0}
    return {**base, **kw}


def test_signal_stage_artifacts_validate_their_shape():
    cats = contents.SignalCategoriesContent(**_categories_content())
    artifacts.canonical_content("signal_categories.v1", cats.model_dump(mode="json"))
    good = _categories_content()
    for bad in (dict(scores=[{"index": 0, "probabilities": {"intent": 0.9}}]), dict(scores=[{"index": 0, "probabilities": {"intent": 0.01, "none": 0.99}}]),
                dict(scores=[{"index": 7, "probabilities": {"none": 1.0}}]), dict(stage1_digests={"SYSTEM": D0}), dict(provenance=_provenance("extract")),
                dict(spans=[{**good["spans"][0], "span_key": "intent.t2b0"}]), dict(spans=[{**good["spans"][0], "last_window": 4}]),
                dict(segments=[{**good["segments"][0], "block": 1}]), dict(segments=[{**good["segments"][0], "speaker": "SYSTEM"}])):
        with pytest.raises(ValidationError):
            contents.SignalCategoriesContent(**{**good, **bad})
    contents.SegmentScores(index=0, probabilities={"none": 0.001})  # 'none' is kept even under 0.02
    decision = {"span_key": "intent.t1b0", "stage2_digest": D1, "probabilities": {"cancel_account": 0.7, "other": 0.2, "not": 0.1}, "decision": "subcategory", "subcategory_id": "cancel_account", "confidence": 0.9, "factors": ["span", "context"], "status": "decided"}
    subs = contents.SignalSubcategoriesContent(provenance=_provenance("subcategorize"), decisions=[decision], carried_forward=["intent.t1b0"])
    artifacts.canonical_content("signal_subcategories.v1", subs.model_dump(mode="json"))
    for bad in (dict(subcategory_id=None), dict(decision="other"), dict(confidence=0.5), dict(status="error"), dict(error_code="model_unavailable"), dict(subcategory_id="other")):
        with pytest.raises(ValidationError):
            contents.SpanSubcategoryDecision(**{**decision, **bad})
    contents.SpanSubcategoryDecision(**{**decision, "decision": "rejected", "subcategory_id": None, "probabilities": {"not": 0.8, "other": 0.2}, "confidence": 0.2})
    contents.SpanSubcategoryDecision(**{**decision, "status": "error", "error_code": "model_unavailable", "confidence": 0.0})
    with pytest.raises(ValidationError, match="carried-forward"):
        contents.SignalSubcategoriesContent(provenance=_provenance("subcategorize"), decisions=[decision], carried_forward=["issue.t1b0"])
    fld = {"field_id": "reason", "type": "enum", "status": "extracted", "value": "price", "evidence": "the price went up", "char_start": 30, "char_end": 47}
    span = {"span_key": "intent.t1b0", "stage3_digest": D2, "status": "extracted", "fields": [fld], "narrowed_quote": {"char_start": 4, "char_end": 31, "text": "I want to cancel my account"}}
    ext = contents.SignalExtractionContent(provenance=_provenance("extract", device="mps", catalog_entry_id="call1-bundled"), spans=[span])
    artifacts.canonical_content("signal_extraction.v1", ext.model_dump(mode="json"))
    with pytest.raises(ValidationError, match="fallback"):
        contents.SignalExtractionContent(provenance=_provenance("extract"), spans=[{**span, "source": "fallback"}])
    contents.SignalExtractionContent(provenance=_provenance("extract"), fallback_provenance=_provenance("extract", catalog_entry_id="call1-bundled"), spans=[{**span, "source": "fallback"}])
    with pytest.raises(ValidationError, match="error"):
        contents.SpanExtraction(**{**span, "status": "error", "narrowed_quote": None})
    with pytest.raises(ValidationError, match="quote range"):
        contents.QuoteRange(char_start=4, char_end=10, text="cancel my")
    # absence remains absence, and only grounded values keep text
    for bad in (dict(value=None), dict(value=3), dict(status="absent"), dict(type="string", value="price"), dict(char_end=None)):
        with pytest.raises(ValidationError):
            contents.ExtractedField(**{**fld, **bad})
    contents.ExtractedField(field_id="reason", type="enum", status="absent")
    contents.ExtractedField(field_id="total", type="amount", status="extracted", value=49.99, surface="forty nine ninety nine", char_start=3, char_end=25)
    contents.ExtractedField(field_id="total", type="amount", status="invalid", surface="a lot", char_start=3, char_end=8)
    contents.ExtractedField(field_id="accepted", type="boolean", status="extracted", value=False)
    for bad in (dict(type="amount", status="extracted", value="49.99"), dict(type="boolean", status="extracted", value=1), dict(type="string", status="ungrounded", surface="x", char_start=0, char_end=1), dict(type="date", status="withheld_pii", evidence="x", char_start=0, char_end=1)):
        with pytest.raises(ValidationError):
            contents.ExtractedField(field_id="f", **bad)
    for kind, contract in ((artifacts.ArtifactKind.SIGNAL_CATEGORIES, "signal_categories.v1"), (artifacts.ArtifactKind.SIGNAL_SUBCATEGORIES, "signal_subcategories.v1"), (artifacts.ArtifactKind.SIGNAL_EXTRACTION, "signal_extraction.v1")):
        assert artifacts.ARTIFACT_CONTENT_CONTRACTS[kind] == contract
    assert artifacts.ALLOWED_SENSITIVITY[artifacts.ArtifactKind.SIGNAL_EXTRACTION] == {artifacts.Sensitivity.MASKED}
    assert artifacts.ALLOWED_SENSITIVITY[artifacts.ArtifactKind.SIGNAL_CATEGORIES] == artifacts.ALLOWED_SENSITIVITY[artifacts.ArtifactKind.SIGNAL_SUBCATEGORIES] == {artifacts.Sensitivity.DERIVED}


def test_contact_signals_v2_result_validator():
    contents.ContactSignalsContent(**_v2_content())
    partial = contents.ContactSignalsContent(**_v2_content(stages=[{"stage": "categorize", "included": True}, {"stage": "subcategorize", "included": True}, {"stage": "extract", "included": False, "failure_code": "model_unavailable"}], completeness="partial", partial_reason="extract: model_unavailable"))
    assert partial.stages[2].failure_code is errors.JobErrorCode.MODEL_UNAVAILABLE
    contents.ContactSignalsContent(**_v2_content(stages=[{"stage": "categorize", "included": True}, {"stage": "subcategorize", "included": True}]))  # extract not planned
    contents.ContactSignalsContent(**_v2_content(completeness="partial", partial_reason="extraction_cap"))  # spans over the cap
    bad_cases = [
        _v2_content(stages=[{"stage": "categorize", "included": True}, {"stage": "subcategorize", "included": True}, {"stage": "extract", "included": False}]),  # complete but a stage missing
        _v2_content(stages=[{"stage": "categorize", "included": False}, {"stage": "subcategorize", "included": True}], completeness="partial", partial_reason="x"),  # categorize missing: the merge fails instead
        _v2_content(stages=[{"stage": "subcategorize", "included": True}, {"stage": "categorize", "included": True}]),
        _v2_content(stages=[{"stage": "categorize", "included": True}]),
        _v2_content(stages=[{"stage": "categorize", "included": True}, {"stage": "subcategorize", "included": True}, {"stage": "subcategorize", "included": True}]),
        _v2_content(passes=[{"pass_kind": "lifecycle", "included": True}]),
        _v2_content(taxonomy=None),
        _v2_content(segmentation=None),
        _v2_content(hits=[{**_v2_hit(), "category_id": None, "category_digest": None, "category_confidence": None, "subcategory_id": None, "subcategory_label": None, "subcategory_digest": None, "subcategory_confidence": None, "span": None, "fields": []}]),
    ]
    for case in bad_cases:
        with pytest.raises(ValidationError):
            contents.ContactSignalsContent(**case)
    with pytest.raises(ValidationError, match="included stage"):
        contents.SignalStageOutcome(stage="extract", included=True, failure_code="model_unavailable")
    # a v1 result carries no v2 parts, and 1.2 artifacts (no new keys) still validate as v1
    v1_hit = {"id": "sig_1", "kind": "intent", "label": "Caller objective", "start": 1, "end": 2, "speaker": "CALLER", "quote": "q", "confidence": 0.9}
    old = contents.ContactSignalsContent.model_validate({"completeness": "complete", "signals": [v1_hit], "passes": [{"pass_kind": "lifecycle", "included": True}], "transcript_fingerprint": D0, "generated_at": NOW.isoformat()})
    assert old.pipeline == "v1" and old.stages == [] and old.taxonomy is None
    v1 = dict(completeness="complete", signals=[v1_hit], passes=[], transcript_fingerprint=D0, generated_at=NOW, pipeline_note="v2 selected; no qualified classifier on this host")
    contents.ContactSignalsContent(**v1)
    for extra in (dict(stages=[{"stage": "categorize", "included": True}]), dict(segmentation=_segmentation()), dict(stage1_digests={"CALLER": D0}), dict(signals=[_v2_hit()])):
        with pytest.raises(ValidationError, match="v1|v2"):
            contents.ContactSignalsContent(**{**v1, **extra})


def test_v2_hits_fill_the_v1_fields_for_1_2_clients():
    hit = contents.ContactSignalView(**_v2_hit())
    v1_fields = {"id", "kind", "label", "start", "end", "speaker", "quote", "turn_id", "char_start", "char_end", "confidence", "review_status"}
    dumped = hit.model_dump(mode="json")
    assert v1_fields <= set(dumped) and all(dumped[f] is not None for f in v1_fields) and dumped["review_status"] == "unreviewed"
    custom = contents.ContactSignalView(**_v2_hit(kind="custom", category_id="upsell_attempt", label="Upsell attempt", id="upsell_attempt.0123456789ab.abababab.t1b0", subcategory_id=None, subcategory_label=None, subcategory_digest=None, subcategory_confidence=None, fields=[]))
    assert custom.label == "Upsell attempt"  # an old client shows CONTACT_SIGNAL_LABEL[kind] ?? label
    contents.ContactSignalView(**_v2_hit(id="intent.preview.t1b0"))  # preview hit ID
    contents.ContactSignalView(**_v2_hit(subcategory_id="other", subcategory_label="Other", subcategory_digest=None))
    for bad in (dict(kind="custom"), dict(category_id="issue"), dict(kind="custom", category_id="intent"), dict(span=None), dict(category_digest=None), dict(char_start=None),
                dict(id="intent.x.t2b0"), dict(subcategory_id="not"), dict(subcategory_id=None), dict(fields=[{"field_id": "reason", "type": "enum", "status": "absent", "name": "Reason"}] * 2)):
        with pytest.raises(ValidationError):
            contents.ContactSignalView(**_v2_hit(**bad))
    with pytest.raises(ValidationError, match="custom-category hit"):
        contents.ContactSignalView(id="s1", kind="custom", label="x", start=0, end=1, speaker="AGENT", quote="q", confidence=0.5)
    with pytest.raises(ValidationError, match="v2 hit fields"):
        contents.ContactSignalView(id="s1", kind="intent", label="x", start=0, end=1, speaker="CALLER", quote="q", confidence=0.5, quote_narrowed=True)
    assert "masked turn text" in contents.ContactSignalView.__doc__


def _part(turn_id=3, block=0, start=12.0, end=15.5, quote="and I want it closed today", **kw) -> Dict[str, Any]:
    return {"turn_id": turn_id, "block": block, "start": start, "end": end, "quote": quote, "char_start": 0, "char_end": len(quote), **kw}


def test_contract_1_3_0_multi_segment_hits_carry_ordered_parts_and_a_span_end():
    """Decision 25 (ContactSignalsV2.md section 6.5): parts and span_end are additive, v2-only fields."""
    plain = contents.ContactSignalView(**_v2_hit())
    assert plain.parts == [] and plain.span_end is None
    merged = contents.ContactSignalView(**_v2_hit(parts=[_part(), _part(turn_id=5, start=20.0, end=24.0)], span_end=24.0))
    assert [p.turn_id for p in merged.parts] == [3, 5] and merged.span_end == 24.0
    assert contents.ContactSignalView.model_validate(merged.model_dump(mode="json")) == merged  # round-trips
    contents.ContactSignalView(**_v2_hit(parts=[_part(turn_id=1, block=1, start=10.0)], span_end=15.5))  # a later block of the anchor's turn
    contents.ContactSignalView(**_v2_hit(parts=[_part(end=9.0, start=9.0)], span_end=9.5))  # span_end covers the anchor's end
    content = contents.ContactSignalsContent.model_validate(_v2_content(hits=[_v2_hit(parts=[_part()], span_end=15.5)]))
    assert content.signals[0].parts[0].quote == "and I want it closed today"
    for bad, match in (
        (dict(parts=[_part()]), "span_end is set exactly"),
        (dict(span_end=15.5), "span_end is set exactly"),
        (dict(parts=[_part()], span_end=15.0), "span_end is at or after"),
        (dict(parts=[_part()], span_end=9.0), "span_end is at or after"),
        (dict(parts=[_part(start=3.0, end=3.5)], span_end=9.5), "call order"),
        (dict(parts=[_part(turn_id=5, start=20.0, end=24.0), _part()], span_end=24.0), "call order"),
        (dict(parts=[_part(turn_id=1, block=0)], span_end=15.5), "each span once"),
        (dict(parts=[_part(), _part()], span_end=15.5), "each span once"),
    ):
        with pytest.raises(ValidationError, match=match):
            contents.ContactSignalView(**_v2_hit(**bad))
    with pytest.raises(ValidationError, match="ordered"):
        contents.SignalHitPart(**_part(start=5.0, end=4.0))
    with pytest.raises(ValidationError, match="ordered"):
        contents.SignalHitPart(**{**_part(), "char_end": 0})
    with pytest.raises(ValidationError):
        contents.SignalHitPart(**_part(extra_field=1))
    # Parts exist only on v2 hits.
    for extra in (dict(parts=[_part()], span_end=15.5), dict(span_end=15.5)):
        with pytest.raises(ValidationError, match="v2 hit fields"):
            contents.ContactSignalView(id="s1", kind="intent", label="x", start=0, end=1, speaker="CALLER", quote="q", confidence=0.5, **extra)
    # Additive: the model defaults them (a payload without them parses), and the OpenAPI document
    # lists them like the other 1.3.0 additions (always present in responses).
    from call1.contracts.api import build_openapi

    schemas = build_openapi()["components"]["schemas"]
    view = schemas["ContactSignalView"]
    assert "parts" in view["properties"] and "span_end" in view["properties"]
    required = set(view.get("required", []))
    assert ("parts" in required) == ("fields" in required) and ("span_end" in required) == ("quote_narrowed" in required)
    assert set(schemas["SignalHitPart"]["required"]) == {"turn_id", "block", "start", "end", "quote", "char_start", "char_end"}
    assert "Multi-segment" in contents.ContactSignalView.__doc__


# F1: jobs, graphs and reanalysis ----------------------------------------------------------------


TAXONOMY_INPUT = {"role": "taxonomy", "artifact": {"artifact_id": "art_snap", "checksum": D2}}


def _v2_jobs():
    sig = {"taxonomy_digest": D1}
    local = {"size_class": "m", "memory_slot": "local_memory"}
    asr = _job_def("asr", "asr", inputs=[{"role": "audio", "artifact": {"artifact_id": "art_a", "checksum": D0}}], selection=_selection("asr", "transcript.v1").model_dump())
    enrich = _job_def("enrich", "enrichment", requires_refs=["asr"], inputs=[{"role": "transcript", "upstream": {"ref": "asr", "output_role": "transcript"}}])
    common_in = [{"role": "transcript", "upstream": {"ref": "asr", "output_role": "transcript"}}, {"role": "pii_findings", "upstream": {"ref": "enrich", "output_role": "pii_findings"}}, TAXONOMY_INPUT]
    cat = _job_def("cs-categorize", "contact_signals_categorize", requires_refs=["asr", "enrich"], inputs=common_in, selection=_selection("signal_category", "signal_categories.v1").model_dump(), parameters={"signals": sig}, resource_estimate=local)
    sub = _job_def("cs-subcategorize", "contact_signals_subcategorize", requires_refs=["asr", "enrich"], after_refs=["cs-categorize"], inputs=common_in + [{"role": "categories", "upstream": {"ref": "cs-categorize", "output_role": "categories"}, "optional": True}], selection=_selection("signal_subcategory", "signal_subcategories.v1").model_dump(), parameters={"signals": sig}, resource_estimate=local)
    ext = _job_def("cs-extract", "contact_signals_extract", requires_refs=["asr", "enrich"], after_refs=["cs-categorize", "cs-subcategorize"], inputs=common_in + [{"role": "categories", "upstream": {"ref": "cs-categorize", "output_role": "categories"}, "optional": True}, {"role": "subcategories", "upstream": {"ref": "cs-subcategorize", "output_role": "subcategories"}, "optional": True}], selection=_selection("signal_extraction", "signal_extraction.v1").model_dump(), parameters={"signals": {**sig, "fallback_entry_id": "call1-bundled"}}, resource_estimate=local)
    merge = _job_def("cs-merge", "contact_signals_merge", requires_refs=["asr", "enrich"], after_refs=["cs-categorize", "cs-subcategorize", "cs-extract"], inputs=common_in + [{"role": f"stage:{k}", "upstream": {"ref": f"cs-{k}", "output_role": role}, "optional": True} for k, role in (("categorize", "categories"), ("subcategorize", "subcategories"), ("extract", "extraction"))], parameters={"signals": sig})
    return asr, enrich, cat, sub, ext, merge


def test_check_graph_accepts_a_v2_graph():
    graph = jobs.JobGraphRequest(idempotency_key="conv1:signals-v2:v1", reason="ingest", jobs=list(_v2_jobs()))
    by_ref = {j.ref: j for j in graph.jobs}
    assert by_ref["cs-categorize"].execution_class is jobs.ExecutionClass.PRIMARY_HOST and by_ref["cs-extract"].execution_class is jobs.ExecutionClass.LLM_ROUTE
    assert all(by_ref[r].model_backed for r in ("cs-categorize", "cs-subcategorize", "cs-extract")) and not by_ref["cs-merge"].model_backed
    with pytest.raises(ValidationError, match="no output role"):
        asr, enrich, cat, sub, ext, merge = _v2_jobs()
        jobs.JobGraphRequest(idempotency_key="conv1:signals-v2:bad", reason="ingest", jobs=[asr, enrich, cat, {**sub, "inputs": sub["inputs"][:-1] + [{"role": "categories", "upstream": {"ref": "cs-categorize", "output_role": "subcategories"}, "optional": True}]}, ext, merge])
    with pytest.raises(ValidationError, match="primary Process host"):
        lan = _selection("signal_category", "signal_categories.v1").model_dump()
        lan["route"] = _route(custody.RouteClass.CUSTOMER_LAN, custody.ProviderType.OLLAMA, "10.0.0.5").model_dump()
        jobs.JobDefinition(**{**_v2_jobs()[2], "selection": lan})


def test_v2_jobs_freeze_the_taxonomy_and_never_window():
    asr, enrich, cat, sub, ext, merge = _v2_jobs()
    rules = jobs.JOB_TYPE_RULES
    assert {jt for jt, r in rules.items() if r.needs_signal_taxonomy} == jobs.SIGNAL_V2_JOB_TYPES
    assert rules[jobs.JobType.CONTACT_SIGNALS_CATEGORIZE].purpose is catalog.ModelPurpose.SIGNAL_CATEGORY and rules[jobs.JobType.CONTACT_SIGNALS_CATEGORIZE].outputs == {"categories": artifacts.ArtifactKind.SIGNAL_CATEGORIES}
    assert rules[jobs.JobType.CONTACT_SIGNALS_SUBCATEGORIZE].purpose is catalog.ModelPurpose.SIGNAL_SUBCATEGORY and rules[jobs.JobType.CONTACT_SIGNALS_SUBCATEGORIZE].outputs == {"subcategories": artifacts.ArtifactKind.SIGNAL_SUBCATEGORIES}
    assert rules[jobs.JobType.CONTACT_SIGNALS_EXTRACT].purpose is catalog.ModelPurpose.SIGNAL_EXTRACTION and rules[jobs.JobType.CONTACT_SIGNALS_EXTRACT].outputs == {"extraction": artifacts.ArtifactKind.SIGNAL_EXTRACTION, "prompt_input": artifacts.ArtifactKind.PROMPT_INPUT}
    for bad, match in (
        ({**cat, "parameters": {}}, "freezes parameters.signals"),
        ({**cat, "inputs": [i for i in cat["inputs"] if i["role"] != "taxonomy"]}, "pins one signal_taxonomy_snapshot"),
        ({**cat, "inputs": cat["inputs"][:2] + [{"role": "taxonomy", "upstream": {"ref": "asr", "output_role": "transcript"}}]}, "pins one signal_taxonomy_snapshot"),
        ({**cat, "parameters": {"signals": {"taxonomy_digest": D1}, "window": {"turn_start": 0, "turn_end": 3}}}, "never a windowed"),
        ({**cat, "parameters": {"signals": {"taxonomy_digest": D1, "span_keys": ["intent.t1b0"]}}}, "span_keys"),
        ({**sub, "parameters": {"signals": {"taxonomy_digest": D1, "fallback_entry_id": "call1-bundled"}}}, "fallback"),
        ({**sub, "parameters": {"signals": {"taxonomy_digest": D1, "stage1_mode": "rederive"}}}, "re-derives"),
        ({**cat, "parameters": {"signals": {"taxonomy_digest": D1, "stage1_mode": "rederive"}}}, "code stage"),
    ):
        with pytest.raises(ValidationError, match=match):
            jobs.JobDefinition(**bad)
    with pytest.raises(ValidationError, match="only Contact Signals v2 jobs"):
        jobs.JobDefinition(**_job_def("card", "qa_scorecard", parameters={"rubric": RUBRIC_REF, "signals": {"taxonomy_digest": D1}}))
    rederive = jobs.JobDefinition(**{k: v for k, v in cat.items() if k != "selection"} | {"parameters": {"signals": {"taxonomy_digest": D1, "stage1_mode": "rederive"}}})
    assert not rederive.model_backed  # a threshold-only edit re-derives spans with no model
    narrowed = jobs.JobDefinition(**{**sub, "parameters": {"signals": {"taxonomy_digest": D1, "span_keys": ["intent.t1b0", "issue.t1b0"]}}})
    assert narrowed.parameters.signals.span_keys == ["intent.t1b0", "issue.t1b0"]
    with pytest.raises(ValidationError):
        jobs.SignalJobParameters(taxonomy_digest=D1, span_keys=["intent.t1b0", "intent.t1b0"])
    with pytest.raises(ValidationError):
        jobs.SignalJobParameters(taxonomy_digest=D1, span_keys=["Intent spans"])
    assert "Never set on a v2 job" in jobs.JobParameters.model_fields["window"].description


def test_one_publisher_per_group_is_unchanged_by_v2():
    rules = jobs.JOB_TYPE_RULES
    publishers = [r.job_type for r in rules.values() if r.publishes is calls.ResultKind.CONTACT_SIGNALS]
    assert publishers == [jobs.JobType.CONTACT_SIGNALS_MERGE]
    for jt in jobs.SIGNAL_V2_JOB_TYPES:
        assert rules[jt].group is calls.ResultKind.CONTACT_SIGNALS and rules[jt].publishes is None
    for group in calls.ResultKind:
        assert len([r for r in rules.values() if r.publishes is group]) == 1, group
    assert set(calls.ResultKind) == {calls.ResultKind(v) for v in ("transcript", "tone", "text_sentiment", "qa", "summary", "contact_signals")}  # no new ResultKind


def test_signal_previews_are_draft_tests_and_claims_filter_by_kind():
    K = jobs.ReanalysisKind
    assert jobs.DRAFT_TEST_KINDS == {K.QA_DRAFT_TEST, K.CONTACT_SIGNALS_PREVIEW}
    assert jobs.REANALYSIS_KIND_AFFECTS[K.CONTACT_SIGNALS_PREVIEW] == frozenset()
    for kind in (K.CONTACT_SIGNALS, K.SPEAKER_CORRECTION, K.FULL):
        assert calls.ResultKind.CONTACT_SIGNALS in jobs.REANALYSIS_KIND_AFFECTS[kind]
    assert api.contract_extensions()["draft_test_kinds"] == ["contact_signals_preview", "qa_draft_test"]
    with pytest.raises(ValidationError, match="signal previews"):
        jobs.ReanalysisRequestCreate(kind="contact_signals_preview")
    jobs.ReanalysisRequestCreate(kind="contact_signals", rescore_signals=True)
    with pytest.raises(ValidationError, match="rescore_signals"):
        jobs.ReanalysisRequestCreate(kind="qa", rescore_signals=True)
    base = dict(id="rq_1", call_id="call_1", conversation_id="conv_1", status="pending", requested_at=NOW, idempotency_key="click-0001")
    preview = jobs.ReanalysisRequest(**base, kind="contact_signals_preview", priority=5, signal_preview_id="prv_1", signal_pipeline="v2", signal_taxonomy_version=3, signal_taxonomy_snapshot_artifact_id="art_snap", preview_result_artifact_id="art_cs")
    assert preview.model_dump()["priority"] == 5
    for bad in (dict(kind="contact_signals_preview"), dict(kind="contact_signals", signal_preview_id="prv_1"), dict(kind="contact_signals", preview_result_artifact_id="art_cs"),
                dict(kind="contact_signals_preview", signal_preview_id="prv_1", signal_pipeline="v1"), dict(kind="qa", rescore_signals=True), dict(kind="qa", signal_backfill_id="bf_1"),
                dict(kind="qa", signal_taxonomy_version=3), dict(kind="embeddings", signal_pipeline="v1")):
        with pytest.raises(ValidationError):
            jobs.ReanalysisRequest(**base, **bad)
    jobs.ReanalysisRequest(**base, kind="contact_signals", priority=-10, rescore_signals=True, signal_backfill_id="bf_1", signal_taxonomy_version=3, signal_pipeline="v2")
    # a 1.2 request record (no new fields) still validates, and priority is always present
    old = jobs.ReanalysisRequest.model_validate({**base, "kind": "qa", "requested_at": NOW.isoformat()})
    assert old.priority == 0 and "priority" in old.model_dump()
    reqs = [jobs.ReanalysisRequest(**{**base, "id": i, "requested_at": NOW + timedelta(minutes=m)}, kind="contact_signals", priority=p) for i, m, p in (("rq_a", 0, 0), ("rq_b", 5, 5), ("rq_c", 1, -10), ("rq_d", 1, 0), ("rq_e", 0, 0))]
    assert [r.id for r in sorted(reqs, key=jobs.reanalysis_claim_order_key)] == ["rq_b", "rq_a", "rq_e", "rq_d", "rq_c"]
    # kinds filters a claim; absent means every kind
    every = jobs.ReanalysisClaimRequest(worker_id="w1")
    assert every.kinds is None and all(every.accepts(k) for k in K)
    previews_only = jobs.ReanalysisClaimRequest(worker_id="w1", kinds=["contact_signals_preview"])
    assert [k for k in K if previews_only.accepts(k)] == [K.CONTACT_SIGNALS_PREVIEW]
    with pytest.raises(ValidationError):
        jobs.ReanalysisClaimRequest(worker_id="w1", kinds=[])
    assert "kinds" in api.routes_by_operation()["claimReanalysisRequests"].description
    assert "draft-test" in calls.GroupGraphStake.model_fields["draft_test"].description and "contact_signals_preview" in calls.GroupGraphStake.model_fields["draft_test"].description


def test_contract_1_3_0_readme_says_process_and_store_upgrade_together():
    assert contracts.CONTRACT_VERSION >= "1.3.0" and contracts.STORE_API_PREFIX == "/store/v1"
    readme = (REPO / "call1" / "contracts" / "README.md").read_text()
    assert readme.startswith(f"# The Store contract (v{contracts.CONTRACT_VERSION})")
    note = readme[readme.index("Version 1.3.0"):readme.index("| File | What it holds |")]
    assert "Process and Store upgrade together" in note and "extra=\"forbid\"" in note and "`priority`" in note
    assert "`signals.py`" in readme and "Stage 2" in note


# F1: permissions, change feed, errors and routes ----------------------------------------------


def test_manage_signals_is_admin_only_and_new_change_kinds_reach_the_listed_principals():
    roles = auth.ROLE_PERMISSIONS
    assert P.MANAGE_SIGNALS in roles[common.ReviewerRole.ADMIN]
    assert P.MANAGE_SIGNALS not in roles[common.ReviewerRole.SUPERVISOR] and P.MANAGE_SIGNALS not in roles[common.ReviewerRole.REVIEWER]
    with pytest.raises(ValueError):
        api.session(common.ReviewerRole.SUPERVISOR, P.MANAGE_SIGNALS)
    byp = events.CHANGE_KINDS_BY_PRINCIPAL
    C = events.ChangeKind
    new = {C.SIGNAL_TAXONOMY, C.SIGNAL_ALERT_RULE, C.SIGNAL_ALERT}
    assert set(byp["process_service_key"]) & new == {C.SIGNAL_TAXONOMY}
    assert new <= set(byp["reviewer"]) and new <= set(byp["supervisor"]) and set(byp["admin"]) == set(C)
    assert events.signal_taxonomy_saved_status(7) == "saved:v7" and events.signal_alert_fired_status("cancel_alert") == "fired:cancel_alert"
    assert events.SIGNAL_SETTINGS_STATUS == "settings" and events.SIGNAL_ALERT_RULE_STATUSES == ("saved", "enabled", "disabled") and events.SIGNAL_FEEDBACK_STATUS == "signal_feedback"
    assert {a.value for a in events.AuditAction} >= {"signal_taxonomy_saved", "signal_settings_changed", "signal_taxonomy_redacted", "signal_alert_rule_saved", "signal_backfill_requested", "signal_preview_requested", "signal_hit_reviewed"}
    assert errors.ERROR_HTTP_STATUS[errors.ErrorCode.SIGNAL_TAXONOMY_CONFLICT] == 409


SIGNAL_ROUTES = {
    # operationId: (method, path, principals, idempotency, audited)
    "getSignalTaxonomy": ("GET", "/signals/taxonomy", {("reviewer_session", "read_calls"), ("process_service_key", "jobs:write")}, "none", False),
    "listSignalTaxonomyVersions": ("GET", "/signals/taxonomy/versions", {("reviewer_session", "read_calls"), ("process_service_key", "jobs:write")}, "none", False),
    "getSignalTaxonomyVersion": ("GET", "/signals/taxonomy/versions/{version}", {("reviewer_session", "read_calls"), ("process_service_key", "jobs:write")}, "none", False),
    "saveSignalTaxonomy": ("PUT", "/signals/taxonomy", {("reviewer_session", "manage_signals")}, "expected_version", True),
    "saveSignalSettings": ("PUT", "/signals/settings", {("reviewer_session", "manage_signals")}, "expected_version", True),
    "redactSignalTaxonomyText": ("POST", "/signals/taxonomy/versions/{version}/redaction", {("reviewer_session", "manage_signals")}, "natural", True),
    "listSignalAlertRules": ("GET", "/signals/alert-rules", {("reviewer_session", "read_calls")}, "none", False),
    "saveSignalAlertRule": ("PUT", "/signals/alert-rules/{rule_id}", {("reviewer_session", "manage_signals")}, "expected_version", True),
    "saveSignalHitFeedback": ("PUT", "/calls/{call_id}/signal-hits/{hit_id}/feedback", {("reviewer_session", "override_verdict")}, "expected_version", True),
    "getSignalMetrics": ("GET", "/metrics/signals", {("reviewer_session", "read_metrics")}, "none", False),
    "mintSignalTaxonomySnapshot": ("POST", "/conversations/{conversation_id}/signal-taxonomy-snapshots", {("process_service_key", "jobs:write")}, "natural", False),
    "createSignalPreview": ("POST", "/signals/previews", {("reviewer_session", "manage_signals")}, "header", True),
    "getSignalPreview": ("GET", "/signals/previews/{preview_id}", {("reviewer_session", "manage_signals")}, "none", False),
    "createSignalBackfill": ("POST", "/signals/backfills", {("reviewer_session", "manage_signals")}, "header", True),
}


def test_signal_routes_match_the_route_table_and_ship_in_stage_2():
    ops = api.routes_by_operation()
    doc = api.build_openapi()
    assert len(SIGNAL_ROUTES) == 14
    for op, (method, path, principals, idem, audited) in SIGNAL_ROUTES.items():
        route = ops[op]
        assert (route.method, route.path) == (method, contracts.STORE_API_PREFIX + path), op
        got = {(p.kind.value, (p.permission.value if p.permission else p.scope.value)) for p in route.principals}
        assert got == principals, op
        for p in route.principals:
            if p.permission is P.MANAGE_SIGNALS:
                assert p.min_role is common.ReviewerRole.ADMIN, op
        assert (route.idempotency.value, route.audited, route.stage) == (idem, audited, 2), op
        assert doc["paths"][route.path][method.lower()]["x-call1-stage"] == 2, op
    for op in ("saveSignalTaxonomy", "saveSignalSettings"):
        assert errors.ErrorCode.SIGNAL_TAXONOMY_CONFLICT in ops[op].errors
    assert ops["saveSignalTaxonomy"].request is signals.SignalTaxonomySave and ops["saveSignalTaxonomy"].response is signals.SignalTaxonomyRecord
    assert ops["getSignalMetrics"].query is metrics.SignalMetricsQuery and ops["getSignalMetrics"].response is metrics.SignalMetrics
    assert "signal_taxonomy_cap_violations" in ops["saveSignalTaxonomy"].object_rule and "never the value" in ops["saveSignalTaxonomy"].object_rule
    assert "redact_current" in ops["redactSignalTaxonomyText"].description
    for op in ("getContactSignals", "requestReanalysis", "claimReanalysisRequests"):
        assert "1.3.0" in ops[op].description, op
    x = api.contract_extensions()
    assert set(x["builtin_signal_categories"]) == set(signals.BUILTIN_SIGNAL_CATEGORIES) and x["forbidden_field_pii_classes"] == sorted(c.value for c in signals.FORBIDDEN_FIELD_PII_CLASSES)
    assert x["signal_stages"] == ["categorize", "subcategorize", "extract"] and x["reserved_signal_subcategory_ids"] == ["not", "other"]
    assert x["job_type_rules"]["contact_signals_categorize"]["needs_signal_taxonomy"] is True


# F1: alerts, feedback, calls, queue, metrics, previews -------------------------------------------


def test_signal_alert_conditions_compare_enum_or_boolean_only():
    t = _taxonomy(UPSELL, _custom("retired", name="Retired", active=False), intent={"subcategories": [CANCEL, {**CANCEL, "subcategory_id": "old_sub", "name": "Old", "active": False, "fields": []}]})
    check = signals.signal_alert_condition_problem
    C = signals.SignalAlertCondition
    assert check(C(category_id="intent"), t) is None
    assert check(C(category_id="intent", subcategory_id="other"), t) is None
    assert check(C(category_id="intent", subcategory_id="cancel_account", field_id="reason", field_equals="price"), t) is None
    assert check(C(category_id="intent", field_id="reason"), t) is None  # a subcategory's field, named at category level
    assert check(C(category_id="upsell_attempt", field_id="accepted", field_equals=True), t) is None
    assert check(C(category_id="nope"), t) == "unknown_category"
    assert check(C(category_id="intent", subcategory_id="nope"), t) == "unknown_subcategory"
    assert check(C(category_id="intent", subcategory_id="cancel_account", field_id="nope"), t) == "unknown_field"
    assert check(C(category_id="intent", subcategory_id="cancel_account", field_id="reason", field_equals="cheaper"), t) == "field_equals_type"
    assert check(C(category_id="intent", subcategory_id="cancel_account", field_id="reason", field_equals=True), t) == "field_equals_type"
    assert check(C(category_id="upsell_attempt", field_id="accepted", field_equals="true"), t) == "field_equals_type"
    typed = C.model_validate_json('{"category_id": "upsell_attempt", "field_id": "accepted", "field_equals": true}')
    assert typed.field_equals is True and C.model_validate_json('{"category_id": "intent", "field_id": "reason", "field_equals": "price"}').field_equals == "price"
    with pytest.raises(ValidationError, match="field_id"):
        C(category_id="intent", field_equals="price")
    with pytest.raises(ValidationError, match="not"):
        C(category_id="intent", subcategory_id="not")
    with pytest.raises(ValidationError):
        C(category_id="intent", min_confidence=0.99)
    assert signals.signal_alert_node_active(C(category_id="intent", subcategory_id="cancel_account"), t)
    assert not signals.signal_alert_node_active(C(category_id="retired"), t) and not signals.signal_alert_node_active(C(category_id="intent", subcategory_id="old_sub"), t)
    assert not signals.signal_alert_node_active(C(category_id="gone"), t)
    rule = signals.SignalAlertRule(rule_id="cancel_alert", name="Cancellations", condition=C(category_id="intent", subcategory_id="cancel_account"))
    signals.SignalAlertRuleRecord(**rule.model_dump(), record_version=1, node_active=True, created_at=NOW, updated_at=NOW, updated_by_account_id="acct_admin")
    signals.SignalAlertRuleSave(rule=rule, expected_record_version=0)
    with pytest.raises(ValidationError):
        signals.SignalAlertMatch(rule_id="cancel_alert", name="Cancellations", hit_ids=["h"] * 21)


def test_signal_feedback_has_two_levels():
    base = dict(call_id="call_1", hit_id="intent.0123456789ab.abababab.t1b0", account_id="acct_1", feedback_version=1, updated_at=NOW)
    signals.SignalHitFeedback(**base, category_verdict="confirmed", subcategory_id="cancel_account", subcategory_digest="0123456789ab", subcategory_verdict="corrected", corrected_subcategory_id="other")
    signals.SignalHitFeedback(**base, category_verdict="dismissed")
    for bad in (dict(subcategory_verdict="corrected", subcategory_id="cancel_account"), dict(subcategory_verdict="confirmed"), dict(subcategory_id="cancel_account", subcategory_verdict="confirmed", corrected_subcategory_id="other"),
                dict(subcategory_id="cancel_account", subcategory_verdict="corrected", corrected_subcategory_id="not"), dict(subcategory_digest="0123")):
        with pytest.raises(ValidationError):
            signals.SignalHitFeedback(**base, **bad)
    signals.SignalHitFeedbackSave(category_verdict="confirmed", expected_feedback_version=0)
    with pytest.raises(ValidationError):
        signals.SignalHitFeedbackSave(subcategory_verdict="corrected", expected_feedback_version=0)
    assert "judged an earlier subcategory" in signals.SignalHitFeedback.__doc__


def test_calls_queue_and_metrics_carry_signals_additively():
    item = calls.CallListItem(call_id="call_1", conversation_id="conv_1", agent_id="A1", created_at=NOW, transcript_state="available", qa_state="available", summary_state="available", review_version=0)
    assert (item.contact_signals_state, item.signal_categories, item.caller_needs, item.signal_alerts) == (calls.ResultState.DISABLED, [], [], [])
    assert {"signal_category", "signal_subcategory", "signal_alert"} <= set(calls.CallListQuery.model_fields)
    schemas = api.build_openapi()["components"]["schemas"]
    assert {"contact_signals_state", "signal_categories", "caller_needs", "signal_alerts"} <= set(schemas["CallListItem"]["required"])  # always sent
    view = calls.ContactSignalsView(**_v2_content(), call_id="call_1", artifact_id="art_cs", version=3, taxonomy_status={"scored_version": 2, "current_version": 3, "outdated_stages": ["subcategorize"], "thresholds_changed": False},
                                    alerts=[{"rule_id": "cancel_alert", "name": "Cancellations", "hit_ids": [_v2_hit()["id"]]}], comparison_preview_id=None)
    assert view.text_withheld is False and view.feedback == []
    # queue: a SIGNAL rule names at least one alert rule
    rule = dict(id="signals-cancel", name="Cancellations", stream="SIGNAL")
    with pytest.raises(ValidationError, match="SIGNAL rule"):
        reviews.ReviewQueueRule(**rule)
    reviews.ReviewQueueRule(**rule, target_signal_alerts=["cancel_alert"])
    reviews.ReviewQueueRule(id="audit-cancel", name="20% of cancellations", stream="AUDIT_SAMPLE", sampling_rate=0.2, target_signal_alerts=["cancel_alert"])
    with pytest.raises(ValidationError):
        reviews.ReviewQueueRule(**rule, target_signal_alerts=["cancel_alert", "cancel_alert"])
    queued = reviews.ReviewQueueItem(id="rvw_1", call_id="call_1", conversation_id="conv_1", rule_id="signals-cancel", rule_name="Cancellations", stream="SIGNAL", reason="Signal: Cancel account (caller, 0:42)", urgency_score=50, evaluation_version=2, stale=False, status="PENDING", item_version=1, created_at=NOW, signals_version=3, trigger_alert_rule_ids=["cancel_alert"])
    assert queued.trigger_alert_rule_ids == ["cancel_alert"]
    # metrics: precision is null under five judged hits
    assert metrics.signal_precision_pct(4, 0) is None and metrics.signal_precision_pct(4, 1) == 80.0
    count = dict(id="cancel_account", name="Cancel account", calls_with_hit=3, hits=4, share_pct=30.0, confirmed=3, dismissed=1)
    metrics.SignalCount(**count)
    with pytest.raises(ValidationError, match="precision"):
        metrics.SignalCount(**count, precision_pct=75.0)
    metrics.SignalCount(**{**count, "confirmed": 4}, precision_pct=80.0)
    with pytest.raises(ValidationError):
        metrics.SignalMetrics(calls_scored=1, calls_with_signal=1, top_caller_needs=[], categories=[], alerts=[], calls_by_pipeline={"v3": 1})


def test_signal_previews_and_backfills():
    t = _taxonomy(intent={"subcategories": [CANCEL]})
    signals.SignalPreviewCreate(taxonomy=t, call_ids=["call_1", "call_2"])
    signals.SignalPreviewCreate(call_ids=["call_1"])
    for bad in (dict(call_ids=[]), dict(call_ids=[f"c{i}" for i in range(11)]), dict(call_ids=["c1", "c1"])):
        with pytest.raises(ValidationError):
            signals.SignalPreviewCreate(**bad)
    ref = {"version": None, "digest": signals.taxonomy_digest(t)}
    ok = dict(call_id="call_1", request_id="rq_1", state="available", result=_v2_content(), diff={"added": ["h1"], "removed": [], "relabelled": [], "fields_changed": []})
    signals.SignalPreviewCall(**ok)
    signals.SignalPreviewCall(call_id="call_2", request_id="rq_2", state="failed", failure_code="model_unavailable")
    for bad in (dict(state="pending"), dict(result=None), dict(state="available", failure_code="model_unavailable")):
        with pytest.raises(ValidationError):
            signals.SignalPreviewCall(**{**ok, **bad})
    signals.SignalPreview(id="prv_1", source="preview", taxonomy_ref=ref, calls=[ok], created_at=NOW, created_by_account_id="acct_admin")
    signals.SignalPreview(id="prv_2", source="compare", taxonomy_ref={"version": 3, "digest": D1}, calls=[], created_at=NOW, created_by_account_id=None)
    with pytest.raises(ValidationError, match="10 calls"):
        signals.SignalPreview(id="prv_3", source="preview", taxonomy_ref=ref, calls=[{**ok, "request_id": f"rq_{i}"} for i in range(11)], created_at=NOW, created_by_account_id="acct_admin")
    signals.SignalBackfillCreate(mode="rescore", created_after=NOW - timedelta(days=7))
    for bad in (dict(mode="compare", rescore_signals=True), dict(mode="rescore", max_calls=501), dict(mode="rescore", created_before=NOW - timedelta(days=8))):
        with pytest.raises(ValidationError):
            signals.SignalBackfillCreate(**{"created_after": NOW - timedelta(days=7), **bad})
    backfill = dict(id="bf_1", taxonomy_version=3, calls_matched=12, requests_created=10, calls_skipped=2, created_at=NOW, created_by_account_id="acct_admin")
    signals.SignalBackfill(**backfill, mode="rescore")
    signals.SignalBackfill(**backfill, mode="compare", preview_id="prv_2")
    with pytest.raises(ValidationError):
        signals.SignalBackfill(**backfill, mode="compare")


# F1: the retail seed (F2 owns the file) ------------------------------------------------------------


def _seed_needs_1_3_0_shapes(raw) -> bool:
    cats = raw["taxonomy"]["categories"]
    nodes = cats + [s for c in cats for s in c.get("subcategories", [])]
    return any(isinstance(c.get("speaker"), str) and c["speaker"] != c["speaker"].upper() for c in cats) or any(isinstance(e, dict) for n in nodes for e in n.get("examples", []))


def test_signal_retail_seed_validates_against_the_save_model():
    """call1/store/seeds/signals_retail_v1.json is a SignalTaxonomySave. Two shapes predate the F1
    contract: speakers must use the SpeakerRole vocabulary (``CALLER``/``AGENT``) and examples are
    plain strings. With exactly those normalized it validates and passes every section 9.6 cap; the
    strict check runs as soon as F2 updates the file."""
    import json

    if not SEED_PATH.exists():
        pytest.skip("the retail seed is not in this tree")
    raw = json.loads(SEED_PATH.read_text())
    pending = _seed_needs_1_3_0_shapes(raw)
    if pending:
        for c in raw["taxonomy"]["categories"]:
            if c.get("speaker"):
                c["speaker"] = c["speaker"].upper()
            for node in [c] + c.get("subcategories", []):
                node["examples"] = [e["text"] if isinstance(e, dict) else e for e in node.get("examples", [])]
    save = signals.SignalTaxonomySave.model_validate(raw)
    t = save.taxonomy
    assert signals.signal_taxonomy_cap_violations(t) == []
    assert {c.category_id for c in t.categories if c.builtin} == set(signals.BUILTIN_SIGNAL_CATEGORIES)
    assert all(c.description is None for c in t.categories if c.builtin)
    fields = [f for c in t.categories for f in signals.path_fields(c) + [f for s in c.subcategories for f in s.fields]]
    assert fields and not any(f.pii_class in signals.FORBIDDEN_FIELD_PII_CLASSES for f in fields)
    order_numbers = [f for f in fields if "order" in f.field_id and f.type is contents.SignalFieldType.STRING]
    assert order_numbers and all(f.pii_class is signals.FieldPiiClass.NONE for f in order_numbers)  # decision 22: the order number is business data
    if pending:
        pytest.xfail("F2: signals_retail_v1.json needs upper-case speakers and plain-string examples (contract 1.3.0)")


# F1: round-trips of the new models ----------------------------------------------------------------


def _signal_samples():
    t = _taxonomy(UPSELL, intent={"subcategories": [CANCEL], "threshold": 0.3})
    v2 = _version(t, 2)
    yield v2
    yield signals.SignalTaxonomyRecord(current=v2, settings=signals.SignalSettings(pipeline="shadow"), record_version=4, updated_at=NOW, updated_by_account_id="acct_admin")
    yield signals.SignalTaxonomySave(taxonomy=t, expected_record_version=4, notes="add cancel")
    yield signals.SignalSettingsSave(settings=signals.SignalSettings(pipeline="v2", v1_fallback=False, fallback_extraction_entry_id="call1-bundled"), expected_record_version=4)
    yield signals.SignalTaxonomySnapshotContent(source="published", taxonomy_ref=v2.ref, taxonomy=t, settings=signals.SignalSettings())
    yield contents.SignalCategoriesContent(**_categories_content())
    yield contents.ContactSignalsContent(**_v2_content())
    yield calls.ContactSignalsView(**_v2_content(), call_id="call_1", artifact_id="art_cs", version=3, feedback=[{"call_id": "call_1", "hit_id": "h1", "category_verdict": "confirmed", "account_id": "acct_1", "feedback_version": 1, "updated_at": NOW}])
    yield signals.SignalAlertRuleRecord(rule_id="cancel_alert", name="Cancellations", condition={"category_id": "intent", "subcategory_id": "cancel_account", "field_id": "reason", "field_equals": "price"}, record_version=2, node_active=True, created_at=NOW, updated_at=NOW, updated_by_account_id="acct_admin")
    yield signals.SignalAlertRuleRecord(rule_id="accepted", name="Upsell accepted", condition={"category_id": "upsell_attempt", "field_id": "accepted", "field_equals": True}, record_version=1, node_active=True, created_at=NOW, updated_at=NOW, updated_by_account_id="acct_admin")
    yield signals.SignalPreview(id="prv_1", source="preview", taxonomy_ref={"version": None, "digest": signals.taxonomy_digest(t)}, calls=[{"call_id": "call_1", "request_id": "rq_1", "state": "available", "result": _v2_content(), "diff": {"added": ["h1"], "removed": [], "relabelled": [], "fields_changed": [], "builtin_changed": ["h0"]}}], options_trimmed=["upsell_attempt"], created_at=NOW, created_by_account_id="acct_admin")
    yield signals.SignalBackfill(id="bf_1", mode="compare", taxonomy_version=3, calls_matched=3, requests_created=3, calls_skipped=0, preview_id="prv_2", created_at=NOW, created_by_account_id="acct_admin")
    yield metrics.SignalMetrics(start=NOW, calls_scored=10, calls_with_signal=6, top_caller_needs=[{"id": "cancel_account", "name": "Cancel account", "calls_with_hit": 3, "hits": 3, "share_pct": 30.0, "confirmed": 5, "dismissed": 0, "precision_pct": 100.0}],
                                categories=[{"category_id": "intent", "name": "Caller objective", "builtin": True, "active": True, "calls_scored": 10, "calls_with_hit": 6, "hit_rate_pct": 60.0, "hits_total": 7, "confirmed": 1, "dismissed": 0, "subcategories": [], "by_day": [{"day": "2026-09-24", "calls_scored": 10, "calls_with_hit": 6}], "top_agents": [{"agent_id": "A1", "agent_display_name": "Samantha Reyes", "calls_with_hit": 2}], "fields": [{"node_id": "cancel_account", "field_id": "reason", "name": "Reason", "values": [{"value": "price", "count": 2}, {"value": False, "count": 1}]}]}],
                                alerts=[{"rule_id": "cancel_alert", "name": "Cancellations", "enabled": True, "calls_matched": 3, "match_rate_pct": 30.0, "by_day": []}], calls_by_pipeline={"v1": 4, "v2": 6})
    yield jobs.ReanalysisRequest(id="rq_1", call_id="call_1", conversation_id="conv_1", kind="contact_signals_preview", status="claimed", requested_at=NOW, idempotency_key="prv-0001", priority=5, signal_preview_id="prv_1", signal_pipeline="v2", signal_taxonomy_version=3, signal_taxonomy_snapshot_artifact_id="art_snap", claimed_by_installation_id="inst_1", claim_expires_at=NOW)
    yield jobs.JobGraphRequest(idempotency_key="conv1:signals-v2:v1", reason="ingest", jobs=list(_v2_jobs()))


@pytest.mark.parametrize("sample", list(_signal_samples()), ids=lambda s: type(s).__name__)
def test_signal_models_round_trip_through_json(sample):
    model = type(sample)
    restored = model.model_validate_json(sample.model_dump_json())
    assert restored == sample
    assert model.model_validate(sample.model_dump(mode="json")) == sample
    assert common.canonical_digest(restored) == common.canonical_digest(sample)


# --- on-device training labels (1.3.0 addition, decision 28) ------------------------------------

from call1.contracts import training  # noqa: E402

HIT = "intent.0123456789ab.abababab.t4b1"
_SRC = {
    "transcript": artifacts.ArtifactKind.TRANSCRIPT, "speaker_attribution": artifacts.ArtifactKind.SPEAKER_ATTRIBUTION,
    "pii_findings": artifacts.ArtifactKind.PII_FINDINGS, "rubric": artifacts.ArtifactKind.RUBRIC_SNAPSHOT,
    "assessment": artifacts.ArtifactKind.QA_ASSESSMENT, "prompt_input": artifacts.ArtifactKind.PROMPT_INPUT,
    "escalation_assessment": artifacts.ArtifactKind.QA_ASSESSMENT, "contact_signals": artifacts.ArtifactKind.CONTACT_SIGNALS,
    "taxonomy": artifacts.ArtifactKind.SIGNAL_TAXONOMY_SNAPSHOT, "stage:categorize": artifacts.ArtifactKind.SIGNAL_CATEGORIES,
    "stage:subcategorize": artifacts.ArtifactKind.SIGNAL_SUBCATEGORIES, "enrichment": artifacts.ArtifactKind.ENRICHMENT,
}


def _sources(*roles):
    return [{"role": r, "artifact_id": f"art_{i}", "checksum": D1, "kind": _SRC[r]} for i, r in enumerate(roles)]


def _qa_label(**over):
    base = dict(seq=1, kind="qa_verdict", subject="qa:call_1:greeting", call_id="call_1", conversation_id="conv_1", recorded_at=NOW,
                source_job_id="job_qa", sources=_sources("transcript", "speaker_attribution", "rubric", "assessment", "prompt_input", "pii_findings"),
                qa={"override_id": "ovr_1", "criterion_id": "greeting", "evaluation_version": 2, "original_status": "FAIL", "status": "PASS", "reason_code": "model_misread_evidence"})
    return {**base, **over}


def _signal_label(**over):
    base = dict(seq=2, kind="signal_hit", subject=f"signal:call_1:{HIT}", call_id="call_1", conversation_id="conv_1", recorded_at=NOW,
                source_job_id="job_merge", sources=_sources("transcript", "speaker_attribution", "pii_findings", "taxonomy", "stage:categorize", "stage:subcategorize", "contact_signals"),
                signal={"hit_id": HIT, "category_id": "intent", "signals_version": 3, "spans": [{"turn_id": 4, "block": 1}, {"turn_id": 6, "block": 0}],
                        "category_verdict": "confirmed", "subcategory_id": "cancel_account", "subcategory_digest": "0123456789ab",
                        "subcategory_verdict": "corrected", "corrected_subcategory_id": "other", "feedback_version": 2})
    return {**base, **over}


def _speaker_label(**over):
    base = dict(seq=3, kind="speaker_role", subject="speaker:call_1:SPEAKER_01", call_id="call_1", conversation_id="conv_1", recorded_at=NOW,
                source_job_id="job_spk", sources=_sources("transcript", "speaker_attribution", "pii_findings"),
                speaker={"turn_id": 5, "speaker": "AGENT", "apply_to_cluster": True, "speaker_cluster": "SPEAKER_01", "reanalysis_request_id": "rq_9"})
    return {**base, **over}


def test_training_read_scope_audit_action_and_route():
    assert common.ServiceScope.TRAINING_READ.value == "training:read"
    assert events.AuditAction.TRAINING_LABELS_READ.value == "training_labels_read"
    route = api.routes_by_operation()["listTrainingLabels"]
    assert (route.method, route.path, route.tag, route.stage) == ("GET", contracts.STORE_API_PREFIX + "/training/labels", "training", 2)
    assert route.principals == [api.process(common.ServiceScope.TRAINING_READ)]  # Process key only; no reviewer session
    assert route.audited and route.idempotency is api.Idempotency.NONE
    assert route.query is training.TrainingLabelQuery and route.response is training.TrainingLabelPage
    assert E.INSUFFICIENT_SCOPE in route.all_errors()
    for phrase in ("limit 0", "not audited", "training_labels_read", "never content", "artifacts:read", "getJob", "no transcript text"):
        assert phrase in route.description, phrase
    doc = api.build_openapi()
    op = doc["paths"][route.path]["get"]
    assert {p["name"] for p in op["parameters"]} == {"after", "limit", "kinds"}
    assert op["x-call1-principals"] == [{"kind": "process_service_key", "scope": "training:read"}] and op["x-call1-audited"] is True
    assert "training:read" in doc["x-call1"]["service_scopes"]
    x = doc["x-call1"]
    assert x["training_judged_artifact_kinds"] == {"qa_verdict": "qa_scorecard", "signal_hit": "contact_signals", "speaker_role": "speaker_attribution"}
    assert x["excluded_qa_reason_codes"] == ["policy_exception", "speaker_misattributed", "transcription_error"]
    assert x["training_source_roles"]["speaker_role"]["required"] == ["speaker_attribution", "transcript"]
    from call1.store.routing import OWNER_BY_OPERATION, Area
    assert OWNER_BY_OPERATION["listTrainingLabels"] is Area.RESULTS


def test_training_label_query_bounds():
    q = training.TrainingLabelQuery()
    assert (q.after, q.limit, q.kinds) == (0, 200, None)
    assert training.TrainingLabelQuery(limit=0).limit == 0  # counts only
    training.TrainingLabelQuery(after=40, limit=500, kinds=["qa_verdict", "signal_hit"])
    for bad in (dict(limit=501), dict(limit=-1), dict(after=-1), dict(kinds=["summary"])):
        with pytest.raises(ValidationError):
            training.TrainingLabelQuery(**bad)


def test_job_list_query_filters_by_memory_slot():
    q = jobs.JobListQuery(memory_slot="local_memory", status="QUEUED", limit=1)
    assert q.memory_slot is jobs.MemorySlot.LOCAL_MEMORY
    assert jobs.JobListQuery().memory_slot is None
    with pytest.raises(ValidationError):
        jobs.JobListQuery(memory_slot="gpu")
    assert "memory_slot" in {p["name"] for p in api.build_openapi()["paths"][contracts.STORE_API_PREFIX + "/jobs"]["get"]["parameters"]}


def test_training_labels_set_exactly_the_field_of_their_kind_and_a_normative_subject():
    training.TrainingLabel(**_qa_label())
    training.TrainingLabel(**_signal_label())
    training.TrainingLabel(**_speaker_label())
    assert training.training_label_subject("qa_verdict", "call_1", "greeting") == "qa:call_1:greeting"
    assert training.speaker_subject_key(5, True, "SPEAKER_01") == "SPEAKER_01"
    assert training.speaker_subject_key(5, False, "SPEAKER_01") == "t5" and training.speaker_subject_key(5, True, None) == "t5"
    training.TrainingLabel(**_speaker_label(subject="speaker:call_1:t5", speaker={**_speaker_label()["speaker"], "apply_to_cluster": False}))
    with pytest.raises(ValidationError, match="exactly"):
        training.TrainingLabel(**_qa_label(signal=_signal_label()["signal"]))
    with pytest.raises(ValidationError, match="exactly"):
        training.TrainingLabel(**_qa_label(kind="speaker_role"))
    with pytest.raises(ValidationError, match="exactly"):
        training.TrainingLabel(**_qa_label(qa=None))
    for bad in ("qa:call_2:greeting", "signal:call_1:greeting", "qa:call_1:other"):
        with pytest.raises(ValidationError, match="subject"):
            training.TrainingLabel(**_qa_label(subject=bad))
    with pytest.raises(ValidationError, match="subject"):
        training.TrainingLabel(**_speaker_label(subject="speaker:call_1:t5"))
    long_subject = training.training_label_subject("qa_verdict", "c" * 128, "k" * 128)
    assert len(long_subject) > 200
    training.TrainingLabel(**_qa_label(subject=long_subject, call_id="c" * 128, qa={**_qa_label()["qa"], "criterion_id": "k" * 128}))


def test_training_labels_carry_ids_enums_and_versions_only():
    forbidden = re.compile(r"(^|_)(text|quote|note|notes|reviewer_notes|account_id|reviewer|name|value|evidence|prompt|transcript)$")
    for model in (training.TrainingLabel, training.QaVerdictLabel, training.SignalHitLabel, training.SignalSpanAt, training.SpeakerRoleLabel, training.TrainingSourceRef, training.TrainingLabelPage):
        offenders = [f for f in model.model_fields if forbidden.search(f)]
        assert not offenders, (model.__name__, offenders)
    with pytest.raises(ValidationError):
        training.QaVerdictLabel(**_qa_label()["qa"], reviewer_notes="the agent said hi")
    with pytest.raises(ValidationError):
        training.SignalHitLabel(**_signal_label()["signal"], note="cancel")
    with pytest.raises(ValidationError):
        training.SpeakerRoleLabel(**_speaker_label()["speaker"], notes="agent")
    assert "IDs, enums and versions only" in training.__doc__ and "never leave" in training.__doc__
    assert training.EXCLUDED_QA_REASON_CODES == {reviews.OverrideReasonCode.TRANSCRIPTION_ERROR, reviews.OverrideReasonCode.SPEAKER_MISATTRIBUTED, reviews.OverrideReasonCode.POLICY_EXCEPTION}


def test_signal_hit_labels_mirror_the_feedback_rules():
    sig = _signal_label()["signal"]
    training.SignalHitLabel(**{**sig, "subcategory_verdict": "confirmed", "corrected_subcategory_id": None})
    training.SignalHitLabel(**{**sig, "category_verdict": "dismissed", "subcategory_id": None, "subcategory_digest": None, "subcategory_verdict": None, "corrected_subcategory_id": None})
    for bad in (dict(subcategory_verdict="corrected", corrected_subcategory_id=None), dict(subcategory_verdict="confirmed"), dict(subcategory_id=None),
                dict(corrected_subcategory_id="not"), dict(subcategory_id="not", subcategory_verdict="confirmed", corrected_subcategory_id=None),
                dict(spans=[]), dict(spans=[{"turn_id": 4, "block": 1}, {"turn_id": 4, "block": 1}]), dict(spans=[{"turn_id": 4, "block": 1}] * 65),
                dict(category_id="issue"), dict(signals_version=0), dict(feedback_version=0), dict(subcategory_digest="0123"), dict(category_verdict="maybe")):
        with pytest.raises(ValidationError):
            training.SignalHitLabel(**{**sig, **bad})


def test_withdrawn_labels_are_cleared_signal_hits_that_resolve_nothing():
    cleared = {**_signal_label()["signal"], "category_verdict": None, "subcategory_verdict": None, "corrected_subcategory_id": None}
    w = training.TrainingLabel(**_signal_label(withdrawn=True, source_job_id=None, sources=[], signal=cleared))
    assert w.withdrawn and w.signal.cleared
    with pytest.raises(ValidationError, match="withdrawn"):
        training.TrainingLabel(**_signal_label(withdrawn=True, source_job_id=None, sources=[]))  # verdicts still set
    with pytest.raises(ValidationError, match="withdrawn"):
        training.TrainingLabel(**_signal_label(signal=cleared))  # cleared but live
    with pytest.raises(ValidationError, match="resolves no source"):
        training.TrainingLabel(**_signal_label(withdrawn=True, signal=cleared))
    with pytest.raises(ValidationError, match="only a signal_hit"):
        training.TrainingLabel(**_qa_label(withdrawn=True, source_job_id=None, sources=[]))


def test_training_label_sources_follow_the_roles_of_their_kind():
    # an unresolvable source (orphaned artifact) leaves sources empty; Process skips it as source_unavailable
    training.TrainingLabel(**_qa_label(source_job_id=None, sources=[]))
    training.TrainingLabel(**_qa_label(sources=_sources("transcript", "assessment")))  # no pii_findings: Process skips, never trains unmasked
    training.TrainingLabel(**_qa_label(sources=_sources("transcript", "assessment", "escalation_assessment", "enrichment")))
    with pytest.raises(ValidationError, match="carries the roles"):
        training.TrainingLabel(**_qa_label(sources=_sources("transcript", "rubric")))
    with pytest.raises(ValidationError, match="unknown source roles"):
        training.TrainingLabel(**_qa_label(sources=_sources("transcript", "assessment", "contact_signals")))
    with pytest.raises(ValidationError, match="once"):
        training.TrainingLabel(**_qa_label(sources=_sources("transcript", "assessment", "transcript")))
    with pytest.raises(ValidationError, match="must be a"):
        training.TrainingLabel(**_qa_label(sources=[*_sources("transcript"), {"role": "assessment", "artifact_id": "art_x", "checksum": D1, "kind": "qa_scorecard"}]))
    with pytest.raises(ValidationError, match="source job"):
        training.TrainingLabel(**_speaker_label(source_job_id=None))
    with pytest.raises(ValidationError):
        training.TrainingSourceRef(role="transcript", artifact_id="art_1", checksum="md5:abc", kind="transcript")
    for kind, (required, optional) in training.TRAINING_SOURCE_ROLES.items():
        assert "transcript" in required and not required & optional
        assert (required | optional) <= set(training.TRAINING_SOURCE_ROLE_KINDS), kind
        assert training.TRAINING_JUDGED_ARTIFACT_KIND[kind] in {k for r in required for k in training.TRAINING_SOURCE_ROLE_KINDS[r]} | {artifacts.ArtifactKind.QA_SCORECARD}
    assert "pii_findings" in training.TRAINING_SOURCE_ROLES[training.TrainingLabelKind.SIGNAL_HIT][1]


def test_training_label_page_is_seq_ordered():
    items = [_qa_label(), _signal_label(), _speaker_label()]
    page = training.TrainingLabelPage(items=items, next_after=3, count_after=3, high_water=9)
    assert [i.seq for i in page.items] == [1, 2, 3]
    training.TrainingLabelPage(items=[], next_after=7, count_after=12, high_water=19)  # limit 0: counts only
    training.TrainingLabelPage(items=[], next_after=0, count_after=0, high_water=0)  # empty log
    for bad in (dict(items=list(reversed(items)), next_after=1), dict(items=[items[0], items[0]], next_after=1), dict(items=items, next_after=2),
                dict(items=items, next_after=3, count_after=2), dict(items=items, next_after=3, high_water=2)):
        with pytest.raises(ValidationError):
            training.TrainingLabelPage(**{"items": items, "next_after": 3, "count_after": 3, "high_water": 9, **bad})


def test_contract_1_3_0_readme_notes_the_training_addition():
    readme = (REPO / "call1" / "contracts" / "README.md").read_text()
    note = readme[readme.index("**1.3.0 addition: on-device training labels**"):readme.index("| File | What it holds |")]
    for phrase in ("`training.py`", "`listTrainingLabels`", "`training:read`", "`training_labels_read`", "`JobListQuery.memory_slot`", "IDs, enums and versions only", "version is unchanged"):
        assert phrase in note, phrase
    assert "| `training.py` |" in readme and contracts.CONTRACT_VERSION >= "1.3.0"


def _training_samples():
    yield training.TrainingLabel(**_qa_label())
    yield training.TrainingLabel(**_signal_label())
    yield training.TrainingLabel(**_speaker_label())
    yield training.TrainingLabelPage(items=[_qa_label(), _signal_label(), _speaker_label()], next_after=3, count_after=40, high_water=40)
    yield training.TrainingLabelQuery(after=3, limit=0, kinds=["speaker_role"])


@pytest.mark.parametrize("sample", list(_training_samples()), ids=lambda s: type(s).__name__)
def test_training_models_round_trip_through_json(sample):
    model = type(sample)
    restored = model.model_validate_json(sample.model_dump_json())
    assert restored == sample
    assert model.model_validate(sample.model_dump(mode="json")) == sample
    assert common.canonical_digest(restored) == common.canonical_digest(sample)


# --- 1.3.0 addition: dual transcription vocabulary (decision 33, docs/DualAsr.md) -------------

from call1.contracts import vocabulary  # noqa: E402

VOCAB_SEED_PATH = REPO / "call1" / "store" / "seeds" / "asr_vocabulary_retail_v1.json"


@pytest.mark.parametrize("term,problem", [
    ("Chadstone", None), ("Wi-Fi", None), ("Stouffer's", None), ("click and collect", None), ("Pokémon", None),
    ("AT&T", None), ("St. Stephen's Green", None), ("DHL", None),
    ("Galaxy S24", "digit"), ("4K TV", "digit"), ("Win٣", "digit"), ("Ⅷ Series", "digit"),
    ("me@example.com", "character"), ("example.com/returns", "character"), ("C++", "character"), ("#promo", "character"),
    ("  Nike", "not_normalized"), ("Best  Buy", "not_normalized"), ("", "empty"), ("x" * 61, "too_long"),
    ("one two three four five six seven", "too_many_words"), ("-Nike", "must_start_with_letter"), ("X", "too_few_letters"),
])
def test_vocabulary_terms_are_business_terms_without_digits(term, problem):
    assert vocabulary.vocabulary_term_problem(term) == problem
    if problem is None:
        vocabulary.AsrVocabularyTerm(term=term, source="customer")
    else:
        with pytest.raises(ValidationError):
            vocabulary.AsrVocabularyTerm(term=term, source="customer")


def test_vocabulary_term_key_and_uniqueness():
    assert vocabulary.vocabulary_term_key("Wi-Fi") == vocabulary.vocabulary_term_key("wifi") == "wifi"
    assert vocabulary.vocabulary_term_key("Pokémon") == "pokemon"
    assert vocabulary.normalize_vocabulary_term("  Best \t Buy ") == "Best Buy"
    with pytest.raises(ValidationError, match="twice"):
        vocabulary.AsrVocabularySettings(customer_terms=["Wi-Fi", "wifi"])
    with pytest.raises(ValidationError, match="twice"):
        vocabulary.AsrVocabularyPack(pack_id="retail", version=1, industry="Retail", title="Retail", terms=["Visa", "VISA"])


def _vocab_pack(**kw):
    return vocabulary.AsrVocabularyPack(**{"pack_id": "retail", "version": 1, "industry": "Retail", "title": "Retail starter", "terms": ["Visa", "Wi-Fi", "Chadstone"], **kw})


def test_effective_vocabulary_orders_pack_then_customer_and_honours_switches():
    pack = _vocab_pack()
    settings = vocabulary.AsrVocabularySettings(customer_terms=["Stanley cup", "wifi"], disabled_pack_terms=["Chadstone"])
    eff = vocabulary.effective_vocabulary(pack, settings)
    assert [(t.term, t.source.value) for t in eff] == [("Visa", "industry_pack"), ("Wi-Fi", "industry_pack"), ("Stanley cup", "customer")]
    assert vocabulary.vocabulary_active(settings, eff)
    assert not vocabulary.vocabulary_active(settings.model_copy(update={"enabled": False}), eff)
    assert not vocabulary.vocabulary_active(vocabulary.AsrVocabularySettings(), vocabulary.effective_vocabulary(None, vocabulary.AsrVocabularySettings()))
    assert vocabulary.AsrVocabularySettings().enabled is True  # on by default once the vocabulary is non-empty
    digest = vocabulary.vocabulary_digest(eff)
    assert digest == common.canonical_digest([t.model_dump(mode="json") for t in eff]) and digest != vocabulary.vocabulary_digest(eff[::-1])
    record = vocabulary.AsrVocabularyRecord(record_version=3, settings=settings, pack=pack, effective_terms=eff, effective_digest=digest, active=True)
    assert vocabulary.AsrVocabularyRecord.model_validate_json(record.model_dump_json()) == record
    for bad in (dict(effective_terms=eff[:2]), dict(effective_digest=None), dict(active=False)):
        with pytest.raises(ValidationError):
            vocabulary.AsrVocabularyRecord(**{**record.model_dump(), **bad})
    empty = vocabulary.AsrVocabularyRecord(record_version=0, settings=vocabulary.AsrVocabularySettings(), effective_terms=[], active=False)
    assert empty.effective_digest is None


def _vocab_parameters(**kw):
    terms = [vocabulary.AsrVocabularyTerm(term="Chadstone", source="industry_pack"), vocabulary.AsrVocabularyTerm(term="Stanley cup", source="customer")]
    return {"digest": vocabulary.vocabulary_digest(terms), "terms": [t.model_dump(mode="json") for t in terms],
            "candidate_entry": {"entry_id": "whisper-small-vocab", "entry_version": 1}, **kw}


def test_asr_vocabulary_parameters_are_frozen_on_asr_jobs_only():
    params = vocabulary.AsrVocabularyParameters(**_vocab_parameters())
    assert params.rule == vocabulary.VocabularyMergeRule() and params.rule.phonetic_min == 0.70 and params.rule.character_min == 0.60
    assert params.rule.max_word_delta == 1 and params.rule.time_slack_seconds == 0.3 and params.glossary_prompt_limit == 120
    with pytest.raises(ValidationError, match="digest"):
        vocabulary.AsrVocabularyParameters(**_vocab_parameters(digest=D0))
    with pytest.raises(ValidationError):
        vocabulary.AsrVocabularyParameters(**_vocab_parameters(terms=[]))
    asr, crit, _card = _ingest_jobs()
    job = jobs.JobDefinition(**{**asr, "parameters": {"asr_vocabulary": _vocab_parameters()}})
    assert job.parameters.asr_vocabulary.digest == params.digest
    with pytest.raises(ValidationError, match="only an asr job"):
        jobs.JobDefinition(**{**crit, "parameters": {**crit["parameters"], "asr_vocabulary": _vocab_parameters()}})


def test_asr_declares_optional_raw_transcript_outputs():
    rule = jobs.JOB_TYPE_RULES[jobs.JobType.ASR]
    assert rule.outputs == {"transcript": artifacts.ArtifactKind.TRANSCRIPT} and rule.publishes is calls.ResultKind.TRANSCRIPT
    assert rule.optional_outputs == {jobs.ASR_BASE_TRANSCRIPT_ROLE: artifacts.ArtifactKind.ASR_BASE_TRANSCRIPT,
                                     jobs.ASR_VOCABULARY_PASS_ROLE: artifacts.ArtifactKind.ASR_VOCABULARY_PASS}
    assert all(not r.optional_outputs for t, r in jobs.JOB_TYPE_RULES.items() if t is not jobs.JobType.ASR)
    with pytest.raises(ValidationError, match="required or optional"):
        jobs.JobTypeRule(**{**rule.model_dump(), "optional_outputs": {"transcript": "transcript"}})
    assert artifacts.ARTIFACT_CONTENT_CONTRACTS[artifacts.ArtifactKind.ASR_BASE_TRANSCRIPT] == "transcript.v1"
    assert artifacts.content_model_for(artifacts.ArtifactKind.ASR_VOCABULARY_PASS) is contents.AsrVocabularyPassContent
    for kind in (artifacts.ArtifactKind.ASR_BASE_TRANSCRIPT, artifacts.ArtifactKind.ASR_VOCABULARY_PASS):
        assert artifacts.ALLOWED_SENSITIVITY[kind] == {artifacts.Sensitivity.RAW}
    # a dependent can bind only a required output, never a raw pass
    asr, crit, card = _ingest_jobs()
    with pytest.raises(ValidationError, match="no output role"):
        jobs.JobGraphRequest(idempotency_key="conv1:ingest:raw", reason="ingest", jobs=[asr, {**crit, "inputs": [{"role": "transcript", "upstream": {"ref": "asr", "output_role": "vocabulary_pass"}}]}, card])
    assert catalog.ModelPurpose.ASR_VOCABULARY.value == "asr_vocabulary"


def _replacement(**kw):
    return {"turn_id": 2, "word_start": 3, "word_end": 5, "char_start": 14, "char_end": 25, "term": "Stanley cup", "source": "customer",
            "heard": "standing cup", "candidate_text": "Stanley cup", "start_time": 41.2, "end_time": 41.9,
            "candidate_start_time": 41.1, "candidate_end_time": 41.95, "phonetic_similarity": 0.62, "character_similarity": 0.78, **kw}


def _correction(**kw):
    return {"status": "applied", "vocabulary_digest": D0, "term_count": 73, "glossary_term_count": 37, "base_engine": "parakeet-tdt-0.6b-v3",
            "candidate_engine": "whisper-small", "rule": vocabulary.VocabularyMergeRule().model_dump(mode="json"), "candidates": 4,
            "replacements": [_replacement()], **kw}


def test_transcript_vocabulary_correction_is_additive_and_consistent():
    plain = contents.TranscriptContent(duration_seconds=60, is_redacted=False, turns=[])
    assert plain.vocabulary_correction is None and plain.model_dump(mode="json")["vocabulary_correction"] is None
    merged = contents.TranscriptContent(duration_seconds=60, is_redacted=False, turns=[], vocabulary_correction=_correction())
    assert merged.vocabulary_correction.replacements[0].heard == "standing cup"
    artifacts.canonical_content("transcript.v1", merged.model_dump(mode="json"))
    contents.VocabularyCorrection(**_correction(status="base_only", replacements=[], candidate_engine=None, note="Whisper Small is not installed.", failure_code="model_unavailable"))
    for bad in (dict(status="base_only"), dict(status="base_only", replacements=[], note=None), dict(failure_code="model_unavailable"),
                dict(replacements=[_replacement(), _replacement(word_start=4, word_end=6, char_start=20, char_end=30)]),
                dict(replacements=[_replacement(turn_id=5), _replacement()])):
        with pytest.raises(ValidationError):
            contents.VocabularyCorrection(**_correction(**bad))
    for bad in (dict(word_end=3), dict(char_end=14), dict(end_time=40.0), dict(term="Galaxy S24")):
        with pytest.raises(ValidationError):
            contents.TranscriptReplacement(**_replacement(**bad))
    contents.AsrVocabularyPassContent(engine="whisper-small", duration_seconds=60, channels=1, glossary_terms=["Stanley cup"],
                                      words=[{"word": "Stanley", "start_time": 41.1, "end_time": 41.5, "probability": 0.9, "segment": 0}],
                                      segments=[{"start_time": 30.0, "end_time": 60.0, "avg_logprob": -0.2, "no_speech_prob": 0.01}])


def test_transcript_view_carries_masked_replacements():
    view = calls.VocabularyCorrectionView(status="applied", replacement_count=2, withheld_count=1, replacements=[
        {"turn_id": 2, "word_start": 3, "word_end": 5, "char_start": 14, "char_end": 25, "term": "Stanley cup", "source": "customer", "heard": None, "start_time": 41.2, "end_time": 41.9}])
    assert view.replacements[0].heard is None
    with pytest.raises(ValidationError):
        calls.TranscriptReplacementView(turn_id=2, word_start=3, word_end=5, char_start=14, char_end=None, term="Nike", source="customer", start_time=1, end_time=2)
    assert calls.TranscriptView.model_fields["vocabulary_correction"].default is None


def test_vocabulary_routes_permission_feed_and_caps():
    by_op = {r.operation_id: r for r in api.ROUTES}
    get, save = by_op["getAsrVocabulary"], by_op["saveAsrVocabulary"]
    assert (get.method, get.path, get.tag, get.stage) == ("GET", contracts.STORE_API_PREFIX + "/vocabulary", "vocabulary", 2)
    assert (save.method, save.path, save.audited, save.idempotency, save.stage) == ("PUT", contracts.STORE_API_PREFIX + "/vocabulary", True, api.Idempotency.EXPECTED_VERSION, 2)
    assert {(p.kind.value, p.min_role, p.scope) for p in get.principals} == {("reviewer_session", common.ReviewerRole.ADMIN, None), ("process_service_key", None, common.ServiceScope.JOBS_WRITE)}
    assert [p.kind.value for p in save.principals] == ["reviewer_session"] and save.principals[0].permission is auth.Permission.MANAGE_VOCABULARY
    assert auth.Permission.MANAGE_VOCABULARY in auth.ROLE_PERMISSIONS[common.ReviewerRole.ADMIN]
    assert auth.Permission.MANAGE_VOCABULARY not in auth.ROLE_PERMISSIONS[common.ReviewerRole.SUPERVISOR]
    assert events.ChangeKind.ASR_VOCABULARY in events.CHANGE_KINDS_BY_PRINCIPAL["process_service_key"]
    assert events.ChangeKind.ASR_VOCABULARY not in events.CHANGE_KINDS_BY_PRINCIPAL["supervisor"]
    assert events.AuditAction.ASR_VOCABULARY_SAVED.value == "asr_vocabulary_saved"
    assert (common.CONTRACT_PARAMETERS.max_vocabulary_terms, common.CONTRACT_PARAMETERS.max_vocabulary_pack_terms) == (500, 2000)
    ext = api.contract_extensions()["asr_vocabulary"]
    assert ext["asr_optional_output_roles"] == ["base_transcript", "vocabulary_pass"] and ext["merge_rule"]["rule_id"] == "vocab-merge-rule-v1"


def test_retail_vocabulary_seed_is_a_pack_of_business_terms():
    import json

    pack = vocabulary.AsrVocabularyPack.model_validate(json.loads(VOCAB_SEED_PATH.read_text()))
    assert (pack.pack_id, pack.version) == ("retail", 1) and len(pack.terms) <= common.CONTRACT_PARAMETERS.max_vocabulary_pack_terms
    assert not any(ch.isdigit() for t in pack.terms for ch in t)
    # Terms from the research vocabulary that look like fictional or call-specific shop names,
    # products or promos, or people's names, are left out (docs/DualAsr.md, "Retail seed").
    removed = {"Suit Haberdashery", "Style House", "Trend Star Retail", "Trend Line Apparel", "Luxy Wear", "Tops and Bottoms For You",
               "Lucinda's Boutique", "Brightmark", "Radio City", "Stop Shop Retail", "Luna and Co", "Pennies", "Michael's", "Plus Rewards",
               "Winter Wonderland Bonanza", "Aurora windbreaker", "Jack Frost", "Rio collection", "Mountain Equipment", "Harry Potter",
               "Coldplay", "Duke Blue Devils", "Herald Sun", "News Corp"}
    assert not removed & set(pack.terms)
    assert {"Visa", "PayPal", "DHL", "Australia Post", "Best Buy", "click and collect", "Stanley cup", "Chadstone"} <= set(pack.terms)


def test_contract_1_3_0_readme_notes_the_vocabulary_addition():
    readme = (REPO / "call1" / "contracts" / "README.md").read_text()
    note = readme[readme.index("**1.3.0 addition: dual transcription vocabulary**"):readme.index("| File | What it holds |")]
    for phrase in ("`vocabulary.py`", "`getAsrVocabulary`", "`saveAsrVocabulary`", "`manage_vocabulary`", "`asr_vocabulary_saved`",
                   "`optional_outputs`", "`asr_base_transcript`", "`asr_vocabulary_pass`", "`TranscriptContent.vocabulary_correction`", "version is unchanged"):
        assert phrase in note, phrase
    assert "| `vocabulary.py` |" in readme
