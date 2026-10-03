// Named contract shapes Evaluate uses. Every one is a generated contract type; nothing here is a
// hand-written mirror. Add names as views need them.

import type { Output, Input, Schema } from '@/contracts';

export type ContractInfo = Output<'ContractInfo'>;
export type ContractParameters = Output<'ContractParameters'>;
export type StoreHealth = Output<'StoreHealth'>;

export type SessionInfo = Output<'SessionInfo'>;
export type ReviewerRole = Schema<'ReviewerRole'>;
export type Permission = Schema<'Permission'>;
export type SessionListItem = Output<'SessionListItem'>;
export type WebAuthnCredentialRecord = Output<'WebAuthnCredentialRecord'>;

export type ReviewerAccount = Output<'ReviewerAccount'>;
export type AccountStatus = Schema<'AccountStatus'>;
export type AccountUpdate = Input<'AccountUpdate'>;
export type Invitation = Output<'Invitation'>;
export type InvitationCreate = Input<'InvitationCreate'>;
export type InvitationIssued = Output<'InvitationIssued'>;
export type InvitationStatus = Schema<'InvitationStatus'>;
export type ProcessInstallation = Output<'ProcessInstallation'>;
export type ServiceKeyRecord = Output<'ServiceKeyRecord'>;
export type ServiceKeyIssued = Output<'ServiceKeyIssued'>;
export type ServiceScope = Schema<'ServiceScope'>;

export type CallListItem = Output<'CallListItem'>;
export type CallDetail = Output<'CallDetail'>;
export type ResultState = Schema<'ResultState'>;
export type ResultKind = Schema<'ResultKind'>;
export type ResultGroup = Output<'ResultGroup'>;
export type PendingWorkIndicator = Output<'PendingWorkIndicator'>;

export type ChangeFeed = Output<'ChangeFeed'>;
export type ChangeEvent = Output<'ChangeEvent'>;
export type ChangeKind = Schema<'ChangeKind'>;

// --- call detail: transcript, evaluation, summary, contact signals, review -----------------------

export type TranscriptView = Output<'TranscriptView'>;
export type TranscriptTurnView = Output<'TranscriptTurnView'>;
export type ToneBlockView = Output<'ToneBlockView'>;
export type SpeakerRole = Schema<'SpeakerRole'>;
export type TextSentimentLabel = Schema<'TextSentimentLabel'>;

export type EvaluationView = Output<'EvaluationView'>;
export type VerdictView = Output<'VerdictView'>;
export type VerdictStatus = Schema<'VerdictStatus'>;
export type OverrideReasonCode = Schema<'OverrideReasonCode'>;
export type ScorecardRubricRef = Output<'ScorecardRubricRef'>;

export type SummaryView = Output<'SummaryView'>;
export type RubricHighlightView = Output<'RubricHighlightView'>;
export type SummaryCitation = Output<'SummaryCitation'>;

export type ContactSignalsView = Output<'ContactSignalsView'>;
export type ContactSignalView = Output<'ContactSignalView'>;
export type ContactSignalKind = Schema<'ContactSignalKind'>;
export type ContactSignalsPassOutcome = Output<'ContactSignalsPassOutcome'>;

export type CallReviewState = Output<'CallReviewState'>;
export type ReviewStaleness = Schema<'ReviewStaleness'>;
export type EscalationStatus = Schema<'EscalationStatus'>;
export type VerdictOverrideRecord = Output<'VerdictOverrideRecord'>;
export type ReviewHistoryEntry = Output<'ReviewHistoryEntry'>;
export type ReviewHistoryKind = Schema<'ReviewHistoryKind'>;
export type ReviewWriteResult = Output<'ReviewWriteResult'>;
export type VerdictOverride = Input<'VerdictOverride'>;
export type EscalationResolution = Input<'EscalationResolution'>;
export type RetainReviewRequest = Input<'RetainReviewRequest'>;
export type SpeakerCorrectionRequest = Input<'SpeakerCorrectionRequest'>;
export type SpeakerCorrection = Input<'SpeakerCorrection'>;

export type ReanalysisKind = Schema<'ReanalysisKind'>;
export type ReanalysisStatus = Schema<'ReanalysisStatus'>;
export type ReanalysisRequest = Output<'ReanalysisRequest'>;
export type ReanalysisRequestCreate = Input<'ReanalysisRequestCreate'>;
export type RubricVersionRef = Output<'RubricVersionRef'>;
export type DraftRubricRef = Output<'DraftRubricRef'>;
export type DraftTestResult = Output<'DraftTestResult'>;

// --- contact signals v2 (contract 1.3.0) -------------------------------------------------------

export type SignalTaxonomy = Output<'SignalTaxonomy'>;
export type SignalTaxonomyInput = Input<'SignalTaxonomy'>;
export type SignalCategory = Output<'SignalCategory'>;
export type SignalSubcategory = Output<'SignalSubcategory'>;
export type SignalField = Output<'SignalField'>;
export type SignalFieldType = Schema<'SignalFieldType'>;
export type FieldPiiClass = Schema<'FieldPiiClass'>;
export type SignalSettings = Output<'SignalSettings'>;
export type SignalPipeline = SignalSettings['pipeline'];
// Rules engine (contract 1.4.0, docs/SignalsEmbeddings.md).
export type SignalRecipe = Output<'SignalRecipe'>;
export type SignalRuleExpr = Output<'SignalRuleExpr'>;
export type SignalRule = Output<'SignalRule'>;
export type SignalLexicon = Output<'SignalLexicon'>;
export type SignalRulesConfig = Output<'SignalRulesConfig'>;
export type SignalHitWhy = Output<'SignalHitWhy'>;
export type SignalRuleDecision = Output<'SignalRuleDecision'>;
export type SignalNeighbour = Output<'SignalNeighbour'>;
export type SignalRuleOutcome = Output<'SignalRuleOutcome'>;
export type SignalTaxonomyRecord = Output<'SignalTaxonomyRecord'>;
export type SignalTaxonomyVersion = Output<'SignalTaxonomyVersion'>;
export type SignalTaxonomySave = Input<'SignalTaxonomySave'>;
export type SignalTaxonomyRef = Output<'SignalTaxonomyRef'>;
export type SignalTaxonomyStatus = Output<'SignalTaxonomyStatus'>;
export type SignalAlertCondition = Output<'SignalAlertCondition'>;
export type SignalAlertConditionInput = Input<'SignalAlertCondition'>;
export type SignalAlertRule = Input<'SignalAlertRule'>;
export type SignalAlertRuleRecord = Output<'SignalAlertRuleRecord'>;
export type SignalAlertMatch = Output<'SignalAlertMatch'>;
export type SignalHitFeedback = Output<'SignalHitFeedback'>;
export type SignalHitFeedbackSave = Input<'SignalHitFeedbackSave'>;
export type SignalMetrics = Output<'SignalMetrics'>;
export type SignalCount = Output<'SignalCount'>;
export type SignalCategoryMetric = Output<'SignalCategoryMetric'>;
export type SignalAlertMetric = Output<'SignalAlertMetric'>;
export type SignalDayCount = Output<'SignalDayCount'>;
export type SignalFieldDistribution = Output<'SignalFieldDistribution'>;
export type SignalPreview = Output<'SignalPreview'>;
export type SignalPreviewCall = Output<'SignalPreviewCall'>;
export type SignalPreviewDiff = Output<'SignalPreviewDiff'>;
export type SignalBackfill = Output<'SignalBackfill'>;
export type SignalStageOutcome = Output<'SignalStageOutcome'>;
export type SegmentationSummary = Output<'SegmentationSummary'>;
export type ExtractedFieldView = Output<'ExtractedFieldView'>;
export type SignalSpanView = Output<'SignalSpanView'>;
export type SignalHitPart = Schema<'SignalHitPart'>;
export type ContactSignalsContent = Output<'ContactSignalsContent'>;
export type CatalogSnapshot = Output<'CatalogSnapshot'>;

// --- ASR vocabulary / dual transcription (contract 1.3.0, decision 33, docs/DualAsr.md) ---------

export type AsrVocabularyRecord = Output<'AsrVocabularyRecord'>;
export type AsrVocabularySave = Input<'AsrVocabularySave'>;
export type AsrVocabularySettings = Output<'AsrVocabularySettings'>;
export type AsrVocabularySettingsInput = Input<'AsrVocabularySettings'>;
export type AsrVocabularyPack = Output<'AsrVocabularyPack'>;
export type AsrVocabularyTerm = Output<'AsrVocabularyTerm'>;
export type VocabularyTermSource = Schema<'VocabularyTermSource'>;
export type VocabularyCorrection = Output<'VocabularyCorrection'>;
export type VocabularyCorrectionView = Output<'VocabularyCorrectionView'>;
export type VocabularyCorrectionStatus = Schema<'VocabularyCorrectionStatus'>;
export type TranscriptReplacement = Output<'TranscriptReplacement'>;
export type TranscriptReplacementView = Output<'TranscriptReplacementView'>;

// --- escalations ------------------------------------------------------------------------------

export type EscalationListItem = Output<'EscalationListItem'>;

// --- human review queue -------------------------------------------------------------------------

export type ReviewQueueItem = Output<'ReviewQueueItem'>;
export type ReviewQueueStatus = Schema<'ReviewQueueStatus'>;
export type ReviewStream = Schema<'ReviewStream'>;
export type ClaimNextResponse = Output<'ClaimNextResponse'>;
export type AssignRequest = Input<'AssignRequest'>;
export type ReleaseRequest = Input<'ReleaseRequest'>;
export type ResolveRequest = Input<'ResolveRequest'>;
export type StartReviewRequest = Input<'StartReviewRequest'>;
export type ReviewQueueRuleRecord = Output<'ReviewQueueRuleRecord'>;
export type ReviewQueueRule = Input<'ReviewQueueRule'>;
export type ReviewQueueRuleSave = Input<'ReviewQueueRuleSave'>;
export type ReviewQueueStats = Output<'ReviewQueueStats'>;
export type DistributionStrategy = Schema<'DistributionStrategy'>;

// --- rubrics --------------------------------------------------------------------------------------

export type RubricSummary = Output<'RubricSummary'>;
export type RubricVersion = Output<'RubricVersion'>;
export type RubricVersionStatus = Schema<'RubricVersionStatus'>;
export type RubricDraft = Output<'RubricDraft'>;
export type RubricDraftSave = Input<'RubricDraftSave'>;
export type RubricDefinition = Input<'RubricDefinition'>;
export type RubricCriterion = Input<'RubricCriterion'>;
export type RubricCheck = Input<'RubricCheck'>;
export type RubricCategory = Schema<'RubricCategory'>;
export type CheckType = Schema<'CheckType'>;
export type EscalationTrigger = Schema<'EscalationTrigger'>;
export type DraftTestRequest = Input<'DraftTestRequest'>;
export type RubricPublishRequest = Input<'RubricPublishRequest'>;
export type RubricRetireRequest = Input<'RubricRetireRequest'>;
export type QaScorecardContent = Output<'QaScorecardContent'>;
export type JobErrorCode = Schema<'JobErrorCode'>;

// --- metrics ----------------------------------------------------------------------------------

export type ExecutiveMetrics = Output<'ExecutiveMetrics'>;
export type ReviewAgreementMetrics = Output<'ReviewAgreementMetrics'>;
export type CriterionAgreement = Output<'CriterionAgreement'>;
export type RubricMetrics = Output<'RubricMetrics'>;
export type CriterionMetric = Output<'CriterionMetric'>;
export type CriterionCounts = Output<'CriterionCounts'>;
export type DailyMetric = Output<'DailyMetric'>;

/** Every list route returns `Page[T] {items, next_page_token}`. */
export interface Page<T> {
  items: T[];
  next_page_token: string | null;
}
