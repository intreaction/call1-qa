// Typed client for the Process operator API (call1/process/app.py, documented in
// call1/process/README.md "Operator API"). This app talks to `/process/api/*` only — never to
// Store directly and never to `/api/v1` (docs/SplitBuild.md "Architecture rules" §3 applies to
// Evaluate; the Process console keeps the same discipline for its own backend). It is loopback
// only; the console token is a Process-only credential, never sent to Store.
//
// There is no generated contract for this API (it is Process's own loopback surface, not part of
// the frozen `call1/contracts/` Store contract), so the shapes below are hand-typed from
// call1/process/app.py, runtime.py, worker.py and catalog.py.

const API = '/process/api';

// --- errors --------------------------------------------------------------------------------

export interface ErrorBody {
  code?: string;
  message?: string;
  details?: Record<string, unknown>;
}

export class ProcessApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly details: Record<string, unknown>;

  constructor(status: number, body: ErrorBody | null) {
    super(body?.message ?? `Process answered HTTP ${status}`);
    this.name = 'ProcessApiError';
    this.status = status;
    this.code = body?.code ?? 'unknown_error';
    this.details = body?.details ?? {};
  }
}

/** The request never reached Process: it isn't running, or this machine can't reach loopback. */
export class ProcessUnreachableError extends Error {
  constructor() {
    super("Can't reach Process. Check that `python -m call1.process serve` (or `python -m call1.launch`) is running on this machine.");
    this.name = 'ProcessUnreachableError';
  }
}

export function isInsufficientScope(err: unknown): boolean {
  return err instanceof ProcessApiError && err.code === 'insufficient_scope';
}

export function isCredentialError(err: unknown): boolean {
  return err instanceof ProcessApiError && (err.status === 401 || err.code === 'console_credential_missing');
}

export function isNotConfigured(err: unknown): boolean {
  return err instanceof ProcessApiError && (err.status === 503 || err.code === 'not_configured');
}

/** One safe, human sentence for any error this client can throw. */
export function describeError(err: unknown): string {
  if (err instanceof ProcessUnreachableError) return err.message;
  if (err instanceof ProcessApiError) return err.message;
  if (err instanceof Error) return err.message;
  return 'Something went wrong.';
}

async function request<T>(path: string, token: string | null, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = { Accept: 'application/json', ...((init?.headers as Record<string, string>) ?? {}) };
  if (token) headers['X-Call1-Console-Token'] = token;
  let res: Response;
  try {
    res = await fetch(`${API}${path}`, { ...init, headers });
  } catch {
    throw new ProcessUnreachableError();
  }
  if (!res.ok) {
    let body: ErrorBody | null = null;
    try {
      body = await res.json();
    } catch {
      // a non-JSON error body (e.g. a dev proxy failure) — the status still tells the story
    }
    throw new ProcessApiError(res.status, body);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

// --- shapes ----------------------------------------------------------------------------------

export interface SessionInfo {
  loopback: boolean;
  console_credential_configured: boolean;
  token_valid: boolean;
  write_header: string;
  demo?: boolean;
}

export interface StoreParameters {
  lease_duration_seconds: number;
  heartbeat_interval_seconds: number;
  max_claim_batch: number;
  default_max_attempts: number;
  inline_artifact_max_bytes: number;
}

export interface StoreConnection {
  url: string | null;
  dev_mode: boolean;
  state: string;
  error: { code: string; message: string } | null;
  process_contract_version: string;
  contract_version: string | null;
  compatible: boolean | null;
  parameters?: StoreParameters;
}

export interface SlotPoolDescribe {
  pool: string;
  size: number;
  in_use: number;
  memory_slots: string[];
  outbound_connection_ref: string | null;
  job_types: string[];
}

export interface RunningJobDescribe {
  job_id: string;
  job_type: string;
  conversation_id: string;
  pool: string;
  attempt_number: number;
  running_seconds: number;
  cancel_requested: boolean;
  progress: number | null;
  /** What a finished attempt still waits to deliver while Store is away, kept in the spool:
   * "publish" (its outputs are not uploaded yet), "complete" or "fail". Null while running. */
  awaiting_delivery: string | null;
}

export interface WorkerStats {
  claimed: number;
  succeeded: number;
  failed: number;
  released: number;
  lost: number;
  last_claim_at: string | null;
  last_error: string | null;
}

export interface WorkerDescribe {
  state: string;
  worker_id: string;
  primary_host: boolean;
  handlers: string;
  stats: WorkerStats;
  pools: SlotPoolDescribe[];
  running: RunningJobDescribe[];
  spooled: number;
  /** Finished attempts whose result is in the spool, their claims kept alive by heartbeats. */
  awaiting_delivery: RunningJobDescribe[];
}

export interface Overview {
  app: string;
  version: string;
  state: string;
  started_at: string | null;
  installation_id: string | null;
  worker_id: string;
  bind: string;
  store: StoreConnection;
  handlers: { mode: string; notes: string[]; missing_job_types: string[] };
  worker: WorkerDescribe | null;
  slots: Array<{ pool: string; size: number }>;
  reanalysis: { handled: number; rejected: number; last_error: string | null } | null;
  catalog: { version: string; published: { catalog_version: string; published_at: string } | null };
  admin_state: string;
  hardware_profile_id: string | null;
  scratch_bytes: number;
  conversations: number;
  evaluate_url: string | null;
  /** On-device training's line on the Overview (the Settings tab has the rest). */
  training?: { enabled: boolean; active_version: string | null; claims_paused: ClaimsPaused | null } | null;
}

export interface GroupProgress {
  kind: string;
  state: string;
  total: number;
  succeeded: number;
  running: number;
  queued: number;
  blocked: number;
  dead_blocked: number;
  failed: number;
  cancelled: number;
  waiting_reason: string | null;
}

export interface JobGroupProgress {
  conversation_id: string;
  groups: GroupProgress[];
  supporting_jobs_total: number;
  settled: boolean;
  updated_at: string;
}

export interface GraphRef {
  graph_id: string;
  reason: string | null;
  at: string | null;
}

export interface ConversationListItem {
  conversation_id: string;
  call_id: string | null;
  label: string | null;
  graphs: GraphRef[];
  first_seen_at: string | null;
  last_seen_at: string | null;
  evaluate_url: string | null;
  progress: JobGroupProgress | null;
  progress_line: string | null;
  progress_error: string | null;
  /** True when `progress` is the last one seen, served because Store could not be asked now. */
  progress_stale: boolean;
}

export interface ConversationsResponse {
  items: ConversationListItem[];
  total: number;
  /** Store was unreachable while listing: progress is the last seen (or absent). */
  store_unavailable: boolean;
}

export interface ConversationJob {
  id: string;
  graph_id: string;
  job_type: string;
  status: string;
  priority: number;
  attempt_count: number;
  max_attempts: number;
  retry_generation: number;
  waiting_reason: string | null;
  error_code: string | null;
  cancel_requested: boolean;
  blocking: Array<Record<string, unknown>>;
  catalog_entry: string | null;
  route_class: string | null;
  destination_host: string | null;
  criterion_id: string | null;
  result_version: number | null;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
  /** Upstream jobs that must succeed (`requires`) or finish (`after`) before this one runs. */
  requires_job_ids?: string[];
  after_job_ids?: string[];
  /** The latest counted attempt: when it started and ended (`ended_at` null while running). */
  started_at?: string | null;
  ended_at?: string | null;
}

export interface ConversationDetail {
  conversation: Record<string, unknown>;
  label: string | null;
  graphs: GraphRef[];
  evaluate_url: string | null;
  jobs: ConversationJob[];
  progress: JobGroupProgress | null;
  progress_line: string | null;
  progress_error: string | null;
  progress_stale: boolean;
}

export interface AttemptProvenance {
  worker_id?: string;
  installation_id?: string;
  adapter_id?: string;
  adapter_version?: string;
  model_revision?: string | null;
  provider_reported_model_id?: string | null;
  route?: { route_class?: string; destination_host?: string | null } | null;
  [key: string]: unknown;
}

export interface Attempt {
  job_id: string;
  attempt_number: number;
  status: string;
  counts_as_attempt: boolean;
  started_at: string;
  ended_at: string | null;
  worker_id: string;
  error_code: string | null;
  error_detail: string | null;
  provenance: AttemptProvenance | null;
  usage_record_id: string | null;
}

export interface JobDetail {
  job: Record<string, unknown> & {
    id: string;
    status: string;
    job_type: string;
    attempt_count: number;
    max_attempts: number;
  };
  attempts: Attempt[];
  evaluate_url: string | null;
}

export interface CatalogEntryDescribe {
  entry_id: string;
  entry_version: number;
  display_name: string;
  purposes: string[];
  default_for: string[];
  status: string;
  qualified_for: string[];
  runtime: string;
  provider_type: string;
  route_class: string;
  destination_host: string | null;
  model_family: string;
  model_revision: string | null;
  adapter_id: string;
  adapter_version: string;
  license_notice: string | null;
  legacy_question_model_id: string | null;
  /** Why an entry is not available (missing files, not qualified), when it is not. */
  detail?: string | null;
}

export interface CatalogResponse {
  version: string;
  published: { catalog_version: string; published_at: string } | null;
  defaults: Record<string, string>;
  escalation_entry_id: string | null;
  entries: CatalogEntryDescribe[];
  handlers: {
    mode: string;
    registered: Array<{ job_type: string; adapter_id: string; adapter_version: string }>;
    missing_job_types: string[];
    notes: string[];
  };
  admin_state: string;
  masking: Record<string, unknown>;
}

export interface RecordingUploadResult {
  conversation_id: string;
  call_id: string;
  graph_id: string;
  conversation_created: boolean;
  graph_created: boolean;
  jobs: number;
  evaluate_url: string;
  /** Contract 1.1.0: the same recording with different call metadata updated the call's metadata
   * (audited by Store). Nothing was reprocessed. */
  metadata_updated: boolean;
  updated_fields: string[];
  /** The call's agent as every client shows it (contract `agent_label`), after any update. */
  agent_label: string | null;
}

/** The optional call metadata an upload carries. Only the fields given are sent, so a re-upload
 * that leaves a field out keeps the call's stored value (contract 1.1.0 re-registration). */
export interface RecordingFields {
  agent_id?: string;
  agent_display_name?: string;
  agent_extension?: string;
  agent_channel?: number;
  external_call_ref?: string;
}

// --- on-device training (docs/OnDeviceTraining.md §6.1) -------------------------------------
//
// A private LoRA for the included model, trained on the device from reviewers' own corrections.
// Labels, datasets and the adapter never leave this Process host — Store sees only the version
// string in provenance.

export type TrainingFrequency = 'daily' | 'weekly';

export interface TrainingSchedule {
  frequency: TrainingFrequency;
  /** 0 (Monday) – 6 (Sunday), Python's `date.weekday()`. Always sent; only used for `weekly`. */
  weekday: number;
  /** Local time of day, `HH:MM`, interpreted in `timezone`. */
  time: string;
}

/** The console-editable settings (`PUT /process/api/training/settings`). */
export interface TrainingSettings {
  enabled: boolean;
  schedule: TrainingSchedule;
  min_new_labels: number;
  max_duration_minutes: number;
  only_when_idle: boolean;
}

/** Every setting as `GET /process/api/training` returns it (the config-only knobs are read-only). */
export interface TrainingSettingsView extends TrainingSettings {
  max_seq_length: number;
  keep_versions: number;
  trainer: string;
  min_labeled_calls: number;
  min_train_examples: number;
  min_eval_items: number;
  max_train_examples: number;
}

export interface TrainingNotice {
  code: string;
  message: string;
}

export interface TrainingLabelsSummary {
  /** Null when the count read failed (`error` says why). */
  total: number | null;
  new_since_last_run: number | null;
  error: TrainingNotice | null;
}

/** The last scheduled occurrence's label check (`state.json`). */
export interface TrainingLastCheck {
  at: string;
  occurrence: string;
  new_labels?: number | null;
  min_new_labels?: number;
  started?: boolean;
  base_changed?: boolean;
  /** Set instead of the counts when the count read failed. */
  error?: string;
}

export interface TrainingProgress {
  iteration: number | null;
  iterations: number | null;
  train_loss: number | null;
  val_loss: number | null;
}

export interface ClaimsPaused {
  run_id: string;
  since: string;
  until: string | null;
}

export interface TrainingStatus {
  phase: string;
  run_id: string | null;
  trigger: string | null;
  detail: string | null;
  progress: TrainingProgress | null;
  claims_paused: ClaimsPaused | null;
}

export interface TrainingEvalTask {
  n: number;
  active: number | null;
  candidate: number | null;
  invalid_active: number;
  invalid_candidate: number;
}

export interface TrainingEval {
  tasks: Record<string, TrainingEvalTask>;
  overall: { n: number; active: number | null; candidate: number | null };
  /** Absent for the trainer's own runs (held-out answer accuracy). An installed adapter's offline
   * evaluation names its metric, what `n` counts and where the numbers come from. */
  metric?: string;
  unit?: string;
  source?: string;
  note?: string;
  extra?: Record<string, { active: number | null; candidate: number | null }>;
}

/** How a kept version got here: `installed` (copied in with an offline evaluation) or absent for a
 * version this Process trained itself. */
export interface TrainingProvenance {
  kind: string;
  summary?: string;
  teacher?: string;
  training_data?: string;
  trainer?: string;
  checkpoint?: string;
  source?: string;
}

/** `active.json` plus its version's evaluation (`GET /process/api/training` → `active`). */
export interface TrainingActiveModel {
  version: string;
  base: string;
  base_fingerprint: string | null;
  tasks: string[];
  activated_at: string;
  previous: string | null;
  eval: TrainingEval | null;
  decision: string | null;
}

/** A kept (promoted) version's `manifest.json`, newest first. */
export interface TrainingVersion {
  version: string;
  created_at: string;
  base: string;
  base_fingerprint: string | null;
  trainer: string;
  tasks: string[];
  label_cursor: number | null;
  eval: TrainingEval | null;
  decision: string | null;
  run_id: string | null;
  provenance?: TrainingProvenance;
}

export interface TrainingRun {
  run_id: string;
  trigger: string;
  requested_at: string;
  started_at: string | null;
  ended_at: string | null;
  status: string;
  reason: string | null;
  label_cursor: { from: number | null; to: number | null };
  labels: { qa_verdict: number; signal_hit: number; speaker_role: number; withdrawn: number };
  examples: { train: number; valid: number; eval_items: number; by_task: Record<string, number> };
  skipped: Record<string, number>;
  trainer: {
    iters: number | null;
    it_per_s: number | null;
    train_loss: number | null;
    val_loss: number | null;
    peak_memory_gb: number | null;
    seconds: number | null;
    name?: string;
  } | null;
  eval: TrainingEval | null;
  candidate_version: string | null;
  active_before: string | null;
  active_after: string | null;
  notes?: Record<string, number>;
  /** Only on the run in progress. */
  phase?: string;
  detail?: string;
}

export interface TrainingState {
  available: boolean;
  unavailable_reason: string | null;
  settings: TrainingSettingsView;
  trainer: string;
  timezone: string;
  next_run_at: string | null;
  labels: TrainingLabelsSummary;
  status: TrainingStatus;
  last_check: TrainingLastCheck | null;
  active: TrainingActiveModel | null;
  versions: TrainingVersion[];
  /** The run in progress (if any), then the last finished runs, newest first (20 in all). */
  runs: TrainingRun[];
  /** training_unavailable, insufficient_scope, qa_too_long, base_changed. */
  notices: TrainingNotice[];
}

export function isTrainingUnavailable(err: unknown): boolean {
  return err instanceof ProcessApiError && err.code === 'training_unavailable';
}

export function isTrainingBusy(err: unknown): boolean {
  return err instanceof ProcessApiError && err.code === 'training_busy';
}

// --- reads -------------------------------------------------------------------------------------

export function getHealth(): Promise<{ ok: boolean; state: string }> {
  return request('/health', null);
}

export function getSession(token: string | null): Promise<SessionInfo> {
  return request('/session', token);
}

export function getOverview(): Promise<Overview> {
  return request('/overview', null);
}

export function listConversations(limit = 50): Promise<ConversationsResponse> {
  return request(`/conversations?limit=${encodeURIComponent(String(limit))}`, null);
}

export function getConversation(conversationId: string): Promise<ConversationDetail> {
  return request(`/conversations/${encodeURIComponent(conversationId)}`, null);
}

export function getJob(jobId: string): Promise<JobDetail> {
  return request(`/jobs/${encodeURIComponent(jobId)}`, null);
}

export function getCatalog(): Promise<CatalogResponse> {
  return request('/catalog', null);
}

export interface SignalFirstPass {
  engine: 'semantic-laya-gemma' | 'fake';
  available: boolean;
  reason: string | null;
  model: string;
  endpoint: string;
  experimental: boolean;
}

export function getSignalFirstPass(): Promise<SignalFirstPass> {
  return request('/signals/first-pass', null);
}

export function getTraining(): Promise<TrainingState> {
  return request('/training', null);
}

export function listTrainingRuns(limit = 50): Promise<{ items: TrainingRun[] }> {
  return request(`/training/runs?limit=${encodeURIComponent(String(limit))}`, null);
}

// --- writes --------------------------------------------------------------------------------

export function retryJob(jobId: string, reason: string, token: string | null): Promise<{ job: Record<string, unknown> }> {
  return request(`/jobs/${encodeURIComponent(jobId)}/retry`, token, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ reason }),
  });
}

export function cancelJob(
  jobId: string,
  reason: string,
  cascade: boolean,
  token: string | null,
): Promise<{ job: Record<string, unknown>; cancelled_job_ids: string[] }> {
  return request(`/jobs/${encodeURIComponent(jobId)}/cancel`, token, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ reason, cascade }),
  });
}

/** Multipart upload with progress, via XHR (fetch has no portable upload-progress event). */
export function uploadRecording(
  file: File,
  fields: RecordingFields,
  token: string | null,
  onProgress?: (fraction: number) => void,
): Promise<RecordingUploadResult> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append('file', file, file.name);
    if (fields.agent_id) form.append('agent_id', fields.agent_id);
    if (fields.agent_display_name) form.append('agent_display_name', fields.agent_display_name);
    if (fields.agent_extension) form.append('agent_extension', fields.agent_extension);
    if (fields.agent_channel !== undefined) form.append('agent_channel', String(fields.agent_channel));
    if (fields.external_call_ref) form.append('external_call_ref', fields.external_call_ref);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API}/recordings`);
    xhr.responseType = 'json';
    if (token) xhr.setRequestHeader('X-Call1-Console-Token', token);
    xhr.upload.onprogress = (event) => {
      if (onProgress && event.lengthComputable) onProgress(event.loaded / event.total);
    };
    xhr.onerror = () => reject(new ProcessUnreachableError());
    xhr.onload = () => {
      let body: unknown = xhr.response;
      if (body == null && xhr.responseText) {
        try {
          body = JSON.parse(xhr.responseText);
        } catch {
          body = null;
        }
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(body as RecordingUploadResult);
      } else {
        reject(new ProcessApiError(xhr.status, body as ErrorBody | null));
      }
    };
    xhr.send(form);
  });
}

export function updateTrainingSettings(settings: TrainingSettings, token: string | null): Promise<{ settings: TrainingSettings }> {
  return request('/training/settings', token, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(settings),
  });
}

export function startTrainingRun(token: string | null): Promise<{ run: TrainingRun }> {
  return request('/training/runs', token, { method: 'POST' });
}

export function cancelTrainingRun(runId: string, token: string | null): Promise<{ run: TrainingRun }> {
  return request(`/training/runs/${encodeURIComponent(runId)}/cancel`, token, { method: 'POST' });
}

export function setActiveAdapter(version: string | null, token: string | null): Promise<{ active: Record<string, unknown> }> {
  return request('/training/active', token, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ version }),
  });
}

// --- query keys ----------------------------------------------------------------------------

export const queryKeys = {
  session: (token: string | null) => ['process-session', token] as const,
  overview: ['process-overview'] as const,
  conversations: (limit: number) => ['process-conversations', limit] as const,
  conversation: (id: string) => ['process-conversation', id] as const,
  job: (id: string) => ['process-job', id] as const,
  catalog: ['process-catalog'] as const,
  signalFirstPass: ['process-signal-first-pass'] as const,
  training: ['process-training'] as const,
  trainingRuns: (limit: number) => ['process-training-runs', limit] as const,
};

export function processDemoCall(token: string | null): Promise<RecordingUploadResult> {
  return request("/demo/recordings", token, { method: "POST" });
}
