"""Who implements which contract route, and how an area registers its handlers.

Every operation in ``call1.contracts.api.ROUTES`` is either **owned** by exactly one area
(``OWNER_BY_OPERATION``) or **deferred** (``DEFERRED``: 501 ``not_implemented`` with the envelope,
per docs/SplitBuild.md). ``tests/store/test_store_routes.py`` checks that the two tables partition
the route table.

An area registers handlers on its ``AreaRouter`` (``call1/store/<area>/routes.py``)::

    router = AreaRouter(Area.QUEUE)

    @router.operation("getJob")
    def get_job(job_id: str, conn=Depends(get_conn), principal=Depends(current_principal)) -> jobs.Job:
        ...

A handler is an ordinary FastAPI endpoint. ``create_app`` adds the contract's method, path, status
code, response model and principal guard, and refuses to start when the handler's path
parameters, body model, query model or ``Idempotency-Key`` header differ from the contract route.
Return a contract model (FastAPI validates it against the route's response model), ``None`` for a
204, or a ``Response`` for binary/CSV bodies and for ``devmode.respond``. An in-scope operation with
no handler yet answers 501 with ``details.pending = true``.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, Dict

from call1.contracts.api import ROUTES, routes_by_operation


class Area(str, Enum):
    CORE = "core"
    QUEUE = "queue"
    AUTH = "auth"
    RESULTS = "results"


_CORE = ["getStatus", "getStatusDetail", "getContract", "listChanges", "listAuditEvents"]

_AUTH = [
    # passkey ceremonies and the signed-in reviewer's own sessions and authenticators
    "enrollBegin", "enrollFinish", "signInBegin", "signInFinish", "getSession", "signOut",
    "addAuthenticatorBegin", "addAuthenticatorFinish", "listOwnAuthenticators", "renameOwnAuthenticator",
    "removeOwnAuthenticator", "listOwnSessions", "revokeOwnSession",
    # admin identity: accounts, invitations, setup codes, break-glass, installations, service keys
    "listAccounts", "getAccount", "updateAccount", "listAccountAuthenticators", "revokeAccountAuthenticator",
    "revokeAccountSessions", "createInvitation", "listInvitations", "revokeInvitation", "listSetupCodes",
    "listBreakGlass", "registerInstallation", "listInstallations", "retireInstallation", "createServiceKey",
    "listServiceKeys", "rotateServiceKey", "revokeServiceKey",
]

_QUEUE = [
    # conversations and artifacts (uploads, grants, Store-minted rubric snapshots)
    "registerConversation", "getConversation", "createInlineArtifact", "mintRubricSnapshot", "createUploadGrant",
    "commitUpload", "listArtifacts", "getArtifact", "getContentGrant", "getArtifactContent",
    # the processing queue (never the human review queue)
    "createJobGraph", "getJobGraph", "claimJobs", "listJobs", "getJob", "listAttempts", "heartbeat", "completeJob",
    "failJob", "releaseJob", "attachLateUsage", "retryJob", "cancelJob", "getGroupProgress", "listConversationUsage",
    # reanalysis requests and draft tests (Evaluate -> Store -> Process)
    "requestReanalysis", "testRubricDraft", "getDraftTestResult", "listReanalysisRequests", "getReanalysisRequest",
    "claimReanalysisRequests", "rejectReanalysisRequest",
    # contact signals v2 (1.3.0): Store-minted taxonomy snapshots, previews and backfills
    "mintSignalTaxonomySnapshot", "createSignalPreview", "getSignalPreview", "createSignalBackfill",
    # usage rows and their report, hardware profiles, catalog snapshots
    "usageReport", "listUsageRecords", "exportUsageRecordsCsv", "getUsageMedians", "upsertHardwareProfile",
    "listHardwareProfiles", "publishCatalogSnapshot", "listCatalogSnapshots",
]

_RESULTS = [
    # calls and result projections, audio, semantic search (Nemotron-3-Embed-1B)
    "listCalls", "getCall", "getTranscript", "getEvaluation", "getEvaluationVersion", "getSummary",
    "getContactSignals", "getCallAudio", "semanticSearch",
    # reviews (human decisions)
    "getReviewState", "listReviewHistory", "overrideVerdict", "resolveEscalation", "retainReview", "correctSpeaker",
    "listEscalations",
    # the human review queue and its rules (never the processing queue), reviewer profiles
    "listReviewQueue", "getReviewQueueStats", "claimNextReview", "getReviewQueueItem", "assignReviewQueueItem",
    "startReview", "releaseReview", "resolveReview", "listReviewQueueRules", "saveReviewQueueRule",
    "deleteReviewQueueRule", "listReviewerProfiles", "updateReviewerProfile",
    # rubrics and metrics
    "listRubrics", "getRubric", "listRubricVersions", "getRubricVersion", "getRubricDraft", "saveRubricDraft",
    "discardRubricDraft", "publishRubric", "retireRubric", "getExecutiveMetrics", "getRubricMetrics",
    "getReviewAgreement",
    # contact signals v2 (1.3.0): taxonomy, settings, alert rules, hit feedback and metrics
    "getSignalTaxonomy", "listSignalTaxonomyVersions", "getSignalTaxonomyVersion", "saveSignalTaxonomy", "saveSignalSettings",
    "redactSignalTaxonomyText", "listSignalAlertRules", "saveSignalAlertRule", "saveSignalHitFeedback", "getSignalMetrics",
    # on-device training (1.3.0, decision 28): the reviewer-label log
    "listTrainingLabels",
    # ASR vocabulary for dual transcription (1.3.0, decision 33, docs/DualAsr.md)
    "getAsrVocabulary", "saveAsrVocabulary",
]

OWNER_BY_OPERATION: Dict[str, Area] = {
    **{op: Area.CORE for op in _CORE},
    **{op: Area.AUTH for op in _AUTH},
    **{op: Area.QUEUE for op in _QUEUE},
    **{op: Area.RESULTS for op in _RESULTS},
}

_CUSTODY = "Pro1 custody (key releases and attestation evidence) is deferred (docs/SplitBuild.md)"
_TRUST = "Release trust is deferred (docs/SplitBuild.md); Pro1 stays closed"
_STAGE4 = "Ships in Stage 4 (TLS/CA and backup/restore)"
_STAGE5 = "Ships in Stage 5 (updater and egress allowlist)"
_ADMIN_STATE = ("Admin state (route opt-ins, masking, attestation policy) is not in the Stage 2 split scope "
                "(docs/SplitBuild.md); Store applies the contract defaults: appliance route only, masking on")

DEFERRED: Dict[str, str] = {
    **{op: _CUSTODY for op in ["recordKeyRelease", "closeKeyRelease", "listKeyReleases", "uploadAttestationEvidence",
                                "getAttestationEvidence", "getAttestationEvidenceContent"]},
    **{op: _TRUST for op in ["getPro1Connection", "clearPro1Block", "reportPro1Verification", "listPendingReleases",
                              "declineRelease", "listReleaseApprovals", "approveRelease", "withdrawReleaseApproval",
                              "listLogScanStates", "putLogScanState", "uploadTrustAnchorBundle", "submitTrustAnchorBundle",
                              "getTrustAnchorBundle"]},
    **{op: _STAGE5 for op in ["stageUpdatePackage", "listUpdatePackages", "getUpdatePackage", "commitUpdatePackage",
                               "installUpdatePackage", "listUpdaterVerifications", "getEgressAllowlist"]},
    **{op: _STAGE4 for op in ["getTlsState", "getBackupManifest", "restorePreflight"]},
    "getPriceTable": "The usage price table is deferred (docs/SplitBuild.md)",
    "savePriceTable": "The usage price table is deferred (docs/SplitBuild.md)",
    "getAdminState": _ADMIN_STATE,
    "changeAdminState": _ADMIN_STATE,
}


def owner_of(operation_id: str) -> Area:
    return OWNER_BY_OPERATION[operation_id]


class AreaRouter:
    """The handlers one area implements, keyed by contract ``operation_id``."""

    def __init__(self, area: Area) -> None:
        self.area = Area(area)
        self.handlers: Dict[str, Callable] = {}

    def operation(self, operation_id: str) -> Callable[[Callable], Callable]:
        routes = routes_by_operation()
        if operation_id not in routes:
            raise KeyError(f"{operation_id} is not a contract operation")
        if operation_id in DEFERRED:
            raise ValueError(f"{operation_id} is deferred: {DEFERRED[operation_id]}")
        owner = OWNER_BY_OPERATION.get(operation_id)
        if owner is not self.area:
            raise ValueError(f"{operation_id} belongs to the {owner.value if owner else '?'} area, not {self.area.value}")

        def register(handler: Callable) -> Callable:
            if operation_id in self.handlers:
                raise ValueError(f"{operation_id} already has a handler")
            self.handlers[operation_id] = handler
            return handler

        return register


def operations_for(area: Area):
    return [route for route in ROUTES if OWNER_BY_OPERATION.get(route.operation_id) is area]
