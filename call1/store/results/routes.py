"""Results-area handlers, registered on ``router`` by the modules imported at the bottom.

Owned operations (``call1.store.routing._RESULTS``): listCalls, getCall, getTranscript,
getEvaluation, getEvaluationVersion, getSummary, getContactSignals, getCallAudio, semanticSearch
(``http_calls``); getReviewState, listReviewHistory, overrideVerdict, resolveEscalation,
retainReview, correctSpeaker, listEscalations (``http_reviews``); listReviewQueue,
getReviewQueueStats, claimNextReview, getReviewQueueItem, assignReviewQueueItem, startReview,
releaseReview, resolveReview, listReviewQueueRules, saveReviewQueueRule, deleteReviewQueueRule,
listReviewerProfiles, updateReviewerProfile (``http_queue``); listRubrics, getRubric,
listRubricVersions, getRubricVersion, getRubricDraft, saveRubricDraft, discardRubricDraft,
publishRubric, retireRubric, getExecutiveMetrics, getRubricMetrics, getReviewAgreement
(``http_rubrics``); getSignalTaxonomy, listSignalTaxonomyVersions, getSignalTaxonomyVersion,
saveSignalTaxonomy, saveSignalSettings, redactSignalTaxonomyText, listSignalAlertRules,
saveSignalAlertRule, saveSignalHitFeedback, getSignalMetrics (``http_signals``, contract 1.3.0); listTrainingLabels
(``http_training``, contract 1.3.0, on-device training); getAsrVocabulary, saveAsrVocabulary (``http_vocabulary``,
contract 1.3.0, dual transcription).
"""

from call1.store.routing import Area, AreaRouter

router = AreaRouter(Area.RESULTS)

from . import http_calls, http_queue, http_reviews, http_rubrics, http_signals, http_training, http_vocabulary  # noqa: E402,F401  (registers the handlers)
