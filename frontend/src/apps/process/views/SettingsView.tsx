// Settings → On-device training (docs/OnDeviceTraining.md §6). Trains a private LoRA for the
// included model from reviewers' own corrections in Evaluate, on a schedule the customer sets
// here. Labels, the dataset and the adapter never leave this Process host — Store sees only the
// version string in provenance (§7.2). The pipeline status uses `/signals/first-pass`;
// training uses `/process/api/training*` (§6.1). Neither reaches Store or the trainer directly.

import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { ChevronDown, ChevronRight, History, Play, RefreshCw, Settings2, SlidersHorizontal } from 'lucide-react';
import {
  cancelTrainingRun,
  getTraining,
  queryKeys,
  setActiveAdapter,
  startTrainingRun,
  updateTrainingSettings,
  type TrainingEval,
  type TrainingFrequency,
  type TrainingRun,
  type TrainingSchedule,
  type TrainingSettings,
  type TrainingState,
  type TrainingVersion,
} from '../api';
import { ModelPicker } from '../components/ModelPicker';
import { SignalPipelineStatus } from '../components/SignalPipelineStatus';
import { FineTuneExperience } from '../components/FineTuneExperience';
import { ConsoleTokenNotice } from '../components/ConsoleTokenNotice';
import { ConfirmAction } from '../components/ConfirmAction';
import {
  Button,
  Card,
  ErrorNotice,
  Field,
  formatDateTime,
  formatDuration,
  formatRelative,
  humanize,
  Loading,
  Notice,
  PageHeader,
  ProgressBar,
  SelectInput,
  StatusPill,
  Switch,
  TextInput,
  type Tone,
} from '../components/ui';
import { taskLabel } from '../labels';

/** Index = the backend's weekday: 0 = Monday … 6 = Sunday (Python's `date.weekday()`, §3.1). */
const WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

const RUN_STATUS_TONE: Record<string, Tone> = {
  promoted: 'green',
  rejected: 'yellow',
  skipped: 'neutral',
  cancelled: 'neutral',
  timed_out: 'yellow',
  interrupted: 'yellow',
  failed: 'red',
  running: 'blue',
  queued: 'neutral',
};

/** A run is in flight when it hasn't reached a settled status yet — used to speed up polling and
 * to decide whether "Train now" is disabled in favor of "Cancel run". Every phase the scheduler
 * and the runner report before a terminal status (§4.1), including the drain (`pausing`, up to 10
 * minutes) and `deciding`. */
const ACTIVE_PHASES = new Set(['waiting', 'pausing', 'collecting', 'building', 'training', 'evaluating', 'deciding']);

function pct(n: number | null | undefined): string {
  return n === null || n === undefined ? '—' : `${Math.round(n * 100)}%`;
}

/** Held-out accuracy of a promoted version: its candidate score in the evaluation that promoted it. */
function heldOutAccuracy(evaluation: TrainingEval | null | undefined): number | null {
  return evaluation?.overall?.candidate ?? null;
}

/** What the version's held-out score measures: answer accuracy for this Process's own runs, or the
 * metric an installed adapter's offline evaluation names (signal precision for the demo adapter). */
function metricName(evaluation: TrainingEval | null | undefined): string {
  return evaluation?.metric ? `held-out ${evaluation.metric}` : 'held-out accuracy';
}

function tasksText(tasks: string[]): string {
  return tasks.map(taskLabel).join(', ');
}

const SHORT_WEEKDAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

/** "Mon 02:00" in the Process host's own wall-clock time: `next_run_at` carries the host's offset,
 * so its date and time are read from the string rather than converted to the browser's zone. */
function hostWallClock(iso: string): string | null {
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/.exec(iso);
  if (!m) return null;
  const day = new Date(Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3]))).getUTCDay();
  return `${SHORT_WEEKDAYS[day]} ${m[4]}:${m[5]}`;
}

function scheduleSummary(next_run_at: string | null, timezone: string): string {
  if (!next_run_at) return 'No run scheduled';
  const when = hostWallClock(next_run_at);
  return `Next run: ${when ?? next_run_at} (${timezone})`;
}

function lastCheckSummary(state: TrainingState): string | null {
  const check = state.last_check;
  if (!check) return null;
  const at = formatDateTime(check.at);
  if (check.error) return `${at}: the label count could not be read (${humanize(check.error)})`;
  if (check.new_labels === null || check.new_labels === undefined) {
    return check.base_changed ? `${at}: the base model changed; retraining` : `${at}: run started`;
  }
  return `${at}: ${check.new_labels} of ${check.min_new_labels ?? state.settings.min_new_labels} new labels`;
}

function labelsSubtitle(state: TrainingState): string | undefined {
  const { total, new_since_last_run } = state.labels;
  if (total === null || total === undefined) return undefined;
  return `${total} label(s) logged, ${new_since_last_run ?? 0} new since the last run`;
}

// --- schedule form -----------------------------------------------------------------------------

function ScheduleForm({
  state,
  token,
  canWrite,
}: {
  state: TrainingState;
  token: string | null;
  canWrite: boolean;
}) {
  const queryClient = useQueryClient();
  const initialized = useRef(false);
  const [enabled, setEnabled] = useState(state.settings.enabled);
  const [frequency, setFrequency] = useState<TrainingFrequency>(state.settings.schedule.frequency);
  const [weekday, setWeekday] = useState(state.settings.schedule.weekday ?? 6);
  const [time, setTime] = useState(state.settings.schedule.time);
  const [minNewLabels, setMinNewLabels] = useState(String(state.settings.min_new_labels));
  const [maxDuration, setMaxDuration] = useState(String(state.settings.max_duration_minutes));
  const [onlyWhenIdle, setOnlyWhenIdle] = useState(state.settings.only_when_idle);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [saved, setSaved] = useState(false);

  // Seeded once from the server's settings, so a background poll never clobbers an edit in
  // progress; a fresh load of this view (a reload, or the settings identity changing after a
  // successful save elsewhere) re-seeds it.
  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;
    setEnabled(state.settings.enabled);
    setFrequency(state.settings.schedule.frequency);
    setWeekday(state.settings.schedule.weekday ?? 6);
    setTime(state.settings.schedule.time);
    setMinNewLabels(String(state.settings.min_new_labels));
    setMaxDuration(String(state.settings.max_duration_minutes));
    setOnlyWhenIdle(state.settings.only_when_idle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    setSaved(false);
    // The weekday is always sent (the backend wants a whole number); a daily schedule ignores it.
    const schedule: TrainingSchedule = { frequency, weekday, time };
    const settings: TrainingSettings = {
      enabled,
      schedule,
      min_new_labels: Number(minNewLabels),
      max_duration_minutes: Number(maxDuration),
      only_when_idle: onlyWhenIdle,
    };
    try {
      await updateTrainingSettings(settings, token);
      await queryClient.invalidateQueries({ queryKey: queryKeys.training });
      setSaved(true);
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  };

  const disabled = !canWrite || busy;

  return (
    <form onSubmit={submit} className="flex flex-col gap-4">
      <div className="flex items-center gap-3">
        <Switch checked={enabled} onChange={setEnabled} disabled={disabled} label="Scheduled training" />
        <span className="text-sm font-medium text-fg">Scheduled training</span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <Field label="Frequency">
          {(id) => (
            <SelectInput
              id={id}
              value={frequency}
              disabled={disabled}
              onChange={(e) => setFrequency(e.target.value as TrainingFrequency)}
            >
              <option value="daily">Daily</option>
              <option value="weekly">Weekly</option>
            </SelectInput>
          )}
        </Field>
        {frequency === 'weekly' && (
          <Field label="Weekday">
            {(id) => (
              <SelectInput id={id} value={weekday} disabled={disabled} onChange={(e) => setWeekday(Number(e.target.value))}>
                {WEEKDAYS.map((day, i) => (
                  <option key={day} value={i}>
                    {day}
                  </option>
                ))}
              </SelectInput>
            )}
          </Field>
        )}
        <Field label="Time" hint={state.timezone}>
          {(id) => <TextInput id={id} type="time" value={time} disabled={disabled} onChange={(e) => setTime(e.target.value)} />}
        </Field>
        <Field label="Minimum new labels" hint="Skip a scheduled run below this many new labels since the last one.">
          {(id) => (
            <TextInput
              id={id}
              type="number"
              inputMode="numeric"
              min={1}
              max={10000}
              value={minNewLabels}
              disabled={disabled}
              onChange={(e) => setMinNewLabels(e.target.value)}
            />
          )}
        </Field>
        <Field label="Maximum duration (minutes)" hint="The trainer is stopped after this long.">
          {(id) => (
            <TextInput
              id={id}
              type="number"
              inputMode="numeric"
              min={15}
              max={720}
              value={maxDuration}
              disabled={disabled}
              onChange={(e) => setMaxDuration(e.target.value)}
            />
          )}
        </Field>
      </div>

      <label className="flex items-center gap-2 text-sm text-fg-muted">
        <input
          type="checkbox"
          checked={onlyWhenIdle}
          disabled={disabled}
          onChange={(e) => setOnlyWhenIdle(e.target.checked)}
          className="rounded border-border-control focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
        />
        Only when idle (no calls processing)
      </label>

      <ErrorNotice error={error} />
      {saved && !error && <Notice tone="green">Saved.</Notice>}

      <div className="flex items-center gap-2">
        <Button type="submit" variant="primary" busy={busy} disabled={disabled} title={!canWrite ? 'Connect the console token to change settings' : undefined}>
          Save
        </Button>
      </div>

      <div className="text-xs text-fg-muted space-y-0.5">
        <p>{scheduleSummary(state.next_run_at, state.timezone)}</p>
        {lastCheckSummary(state) && <p>Last check: {lastCheckSummary(state)}</p>}
      </div>
    </form>
  );
}

// --- status --------------------------------------------------------------------------------

function StatusCard({ state, token, canWrite }: { state: TrainingState; token: string | null; canWrite: boolean }) {
  const queryClient = useQueryClient();
  const invalidate = () => queryClient.invalidateQueries({ queryKey: queryKeys.training });
  const [error, setError] = useState<unknown>(null);
  const { status } = state;
  const running = ACTIVE_PHASES.has(status.phase);
  const { total, new_since_last_run: fresh } = state.labels;
  const minNew = state.settings.min_new_labels;
  const baseChanged = (state.notices ?? []).some((n) => n.code === 'base_changed');
  // Nothing to learn from: no reviewer correction has ever been logged (and the base model has not
  // changed, which is the one reason to retrain on the same labels).
  const noLabels = total === 0 && !baseChanged;
  const fewLabels = typeof fresh === 'number' && fresh < minNew;
  const trainNowReason = !canWrite
    ? 'Connect the console token to start a run'
    : running
      ? 'A run is already in progress'
      : noLabels
        ? 'No reviewer corrections logged yet; there is nothing to train on'
        : undefined;
  const progress = status.progress;
  const showProgress = progress !== null && progress.iterations !== null && progress.iteration !== null;
  const pausedUntil = status.claims_paused?.until ? formatDateTime(status.claims_paused.until) : null;

  return (
    <Card title="Status" icon={Play}>
      <div className="flex flex-col gap-3">
        <div className="flex items-center gap-2 flex-wrap">
          <StatusPill tone={running ? 'blue' : status.phase === 'idle' ? 'neutral' : (RUN_STATUS_TONE[status.phase] ?? 'neutral')}>
            {humanize(status.phase)}
          </StatusPill>
          {status.detail && <span className="text-sm text-fg-muted">{status.detail}</span>}
        </div>
        {showProgress && progress && (
          <ProgressBar
            fraction={progress.iterations! > 0 ? progress.iteration! / progress.iterations! : 0}
            label={`Iteration ${progress.iteration} of ${progress.iterations}${
              progress.train_loss !== null ? `, loss ${progress.train_loss.toFixed(2)}` : ''
            }`}
          />
        )}
        {status.claims_paused && (
          <Notice tone="yellow">
            Processing paused{pausedUntil ? ` until ${pausedUntil}` : ''} while this run uses the GPU; new jobs wait for it.
          </Notice>
        )}
        <ErrorNotice error={error} />
        {noLabels && canWrite && !running && (
          <p className="text-xs text-fg-muted">Train now unlocks once reviewers have corrected at least one call in Evaluate.</p>
        )}
        <div className="flex items-center gap-2 flex-wrap">
          <ConfirmAction
            label="Train now"
            confirmLabel="Start training"
            dismissLabel="Not now"
            variant="primary"
            icon={Play}
            disabled={trainNowReason !== undefined}
            disabledReason={trainNowReason}
            warning={
              <div className="space-y-1.5">
                <p>
                  Train now pauses call processing on this computer: Process stops taking new jobs, lets running ones finish, then trains
                  for up to {state.settings.max_duration_minutes} minutes (the maximum duration above). Queued calls wait and resume
                  when the run ends. You can cancel at any time.
                </p>
                <p className="text-fg-muted">
                  The new adapter replaces the active one only if it scores at least as well on held-out calls.
                </p>
                {fewLabels && (
                  <p className="text-primer-yellowFg">
                    Only {fresh} new label{fresh === 1 ? '' : 's'} since the last run (a scheduled run waits for {minNew}). Train now runs anyway.
                  </p>
                )}
              </div>
            }
            onConfirm={async () => {
              setError(null);
              try {
                await startTrainingRun(token);
              } finally {
                void invalidate();
              }
            }}
          />
          {running && status.run_id && (
            <ConfirmAction
              label="Cancel run"
              confirmLabel="Cancel run"
              dismissLabel="Keep training"
              variant="danger"
              disabled={!canWrite}
              disabledReason="Connect the console token to cancel the run"
              warning="Cancel the run in progress? The active adapter is unchanged."
              onConfirm={async () => {
                setError(null);
                try {
                  await cancelTrainingRun(status.run_id!, token);
                } finally {
                  void invalidate();
                }
              }}
            />
          )}
        </div>
      </div>
    </Card>
  );
}

// --- active model ----------------------------------------------------------------------------

/** Where the active version came from and how it scored on held-out calls, from its manifest: the
 * evaluation that promoted it (this Process's own run), or the offline evaluation it was installed
 * with (an adapter trained elsewhere, e.g. the class-demo retail LoRA). */
function VersionDetails({ manifest, evaluation, decision }: { manifest: TrainingVersion | undefined; evaluation: TrainingEval | null | undefined; decision: string | null | undefined }) {
  const provenance = manifest?.provenance;
  const installed = provenance?.kind === 'installed';
  const overall = evaluation?.overall;
  const hasOverall = overall !== undefined && overall.candidate !== null && overall.n > 0;
  const taskRows = Object.entries(evaluation?.tasks ?? {}).filter(([, e]) => e.n > 0);
  const order = (name: string) => ['precision', 'recall', 'f1'].indexOf(name.toLowerCase()) + 1 || 99;
  const extra = Object.entries(evaluation?.extra ?? {}).sort(([a], [b]) => order(a) - order(b));
  if (!installed && !hasOverall && !decision) return null;
  const unit = evaluation?.unit ?? 'items';
  const baseName = installed ? 'base model' : 'previous';
  return (
    <div className="border-t border-border-muted pt-2 mt-1 space-y-1.5 text-xs">
      <div className="flex items-center gap-2 flex-wrap">
        <StatusPill tone={installed ? 'neutral' : 'green'}>{installed ? 'Installed' : 'Trained on this appliance'}</StatusPill>
        {installed && provenance?.summary && <span className="text-fg-muted">{provenance.summary}</span>}
      </div>
      {/* Stacked label-over-value on a phone, two columns from sm up. */}
      <dl className="grid grid-cols-1 sm:grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 [&>dt]:mt-1 sm:[&>dt]:mt-0 [&>div>dt]:mt-1 sm:[&>div>dt]:mt-0">
        {hasOverall && (
          <>
            <dt className="text-fg-muted">{installed ? 'Held-out ' + (evaluation?.metric ?? 'accuracy') : 'Held-out accuracy'}</dt>
            <dd className="text-fg tabular-nums">
              {pct(overall.active)} {baseName} → <span className="font-medium">{pct(overall.candidate)}</span> with adapter · n = {overall.n} {unit}
            </dd>
          </>
        )}
        {extra.map(([name, e]) => (
          <div key={name} className="contents">
            <dt className="text-fg-muted">{name.charAt(0).toUpperCase() + name.slice(1)}</dt>
            <dd className="text-fg tabular-nums">
              {pct(e.active)} → {pct(e.candidate)}
            </dd>
          </div>
        ))}
        {taskRows.length > 0 && (
          <>
            <dt className="text-fg-muted">Per task</dt>
            <dd className="text-fg">{taskRows.map(([task, e]) => `${taskLabel(task)}: ${pct(e.active)} → ${pct(e.candidate)} (n=${e.n})`).join(' · ')}</dd>
          </>
        )}
        {installed && provenance?.training_data && (
          <>
            <dt className="text-fg-muted">Trained on</dt>
            <dd className="text-fg">{provenance.training_data}</dd>
          </>
        )}
        {installed && provenance?.trainer && (
          <>
            <dt className="text-fg-muted">Training</dt>
            <dd className="text-fg">{provenance.trainer}</dd>
          </>
        )}
        {(evaluation?.source ?? provenance?.source) && (
          <>
            <dt className="text-fg-muted">Source</dt>
            <dd className="text-fg break-words">{evaluation?.source ?? provenance?.source}</dd>
          </>
        )}
        {!installed && decision && (
          <>
            <dt className="text-fg-muted">Decision</dt>
            <dd className="text-fg break-words">{decision}</dd>
          </>
        )}
      </dl>
      {evaluation?.note && <p className="text-fg-subtle">{evaluation.note}</p>}
    </div>
  );
}

function ActiveModelCard({ state, token, canWrite }: { state: TrainingState; token: string | null; canWrite: boolean }) {
  const queryClient = useQueryClient();
  const invalidate = () => queryClient.invalidateQueries({ queryKey: queryKeys.training });
  const { active, versions } = state;
  const busyReason = ACTIVE_PHASES.has(state.status.phase) ? 'Wait for the training run to finish, or cancel it' : null;
  const disabled = !canWrite || busyReason !== null;
  const disabledReason = !canWrite ? 'Connect the console token to change the active model' : (busyReason ?? undefined);
  // Every kept version but the active one, newest first; the rollback target is the version the
  // active one replaced (`previous`) when it is still kept, else the newest other kept version.
  const others = versions.filter((v) => v.version !== active?.version);
  const rollback = active ? (others.find((v) => v.version === active.previous) ?? others[0] ?? null) : null;
  const activeAccuracy = heldOutAccuracy(active?.eval);
  const activeManifest = active?.version ? versions.find((v) => v.version === active.version) : undefined;

  return (
    <Card title="Active model" icon={SlidersHorizontal}>
      <div className="flex flex-col gap-3">
        <div className="rounded-md border border-border-muted p-3">
          <div className="flex items-center justify-between gap-2 flex-wrap">
            <div className="min-w-0">
              <p className="text-sm font-medium text-fg">{active?.version ? active.version : 'Base model (no adapter)'}</p>
              {active?.version ? (
                <p className="text-xs text-fg-muted mt-0.5">
                  Active since {formatDateTime(active.activated_at)}
                  {active.tasks.length > 0 ? ` · used for ${tasksText(active.tasks)}` : ''}
                  {activeAccuracy !== null ? ` · ${pct(activeAccuracy)} ${metricName(active.eval)}` : ''}
                </p>
              ) : (
                <p className="text-xs text-fg-muted mt-0.5">Every task runs on the included model's shipped weights.</p>
              )}
            </div>
            {active?.version && (
              <div className="flex items-center gap-2 flex-wrap">
                {rollback && (
                  <ConfirmAction
                    label={`Roll back to ${rollback.version}`}
                    confirmLabel={`Roll back to ${rollback.version}`}
                    dismissLabel="Keep this version"
                    disabled={disabled}
                    disabledReason={disabledReason}
                    warning={`Stop using ${active.version} and run jobs on ${rollback.version} instead? No evaluation runs.`}
                    onConfirm={async () => {
                      await setActiveAdapter(rollback.version, token);
                      void invalidate();
                    }}
                  />
                )}
                <ConfirmAction
                  label="Use base model"
                  confirmLabel="Use base model"
                  dismissLabel="Keep this version"
                  variant="danger"
                  disabled={disabled}
                  disabledReason={disabledReason}
                  warning={`Stop using ${active.version} and run jobs on the base model instead?`}
                  onConfirm={async () => {
                    await setActiveAdapter(null, token);
                    void invalidate();
                  }}
                />
              </div>
            )}
          </div>
          {active?.version && <VersionDetails manifest={activeManifest} evaluation={active.eval} decision={active.decision} />}
        </div>

        {others.length > 0 && (
          <div className="space-y-1.5">
            <p className="text-xs font-medium text-fg-muted">Kept versions</p>
            {others.map((v) => {
              const accuracy = heldOutAccuracy(v.eval);
              return (
                <div key={v.version} className="flex items-center justify-between gap-2 flex-wrap rounded-md border border-border-muted p-2.5">
                  <div className="min-w-0 text-sm">
                    <span className="font-medium text-fg">{v.version}</span>
                    <span className="text-fg-muted">
                      {' '}
                      · {formatDateTime(v.created_at)}
                      {v.tasks.length > 0 ? ` · ${tasksText(v.tasks)}` : ''}
                      {accuracy !== null ? ` · ${pct(accuracy)} ${metricName(v.eval)}` : ''}
                    </span>
                  </div>
                  <ConfirmAction
                    label="Activate"
                    confirmLabel="Activate"
                    dismissLabel="Cancel"
                    disabled={disabled}
                    disabledReason={disabledReason === 'Connect the console token to change the active model' ? 'Connect the console token to activate a version' : disabledReason}
                    warning={active?.version ? `Switch from ${active.version} to ${v.version}? No evaluation runs.` : `Run jobs on ${v.version}? No evaluation runs.`}
                    onConfirm={async () => {
                      await setActiveAdapter(v.version, token);
                      void invalidate();
                    }}
                  />
                </div>
              );
            })}
          </div>
        )}
      </div>
    </Card>
  );
}

// --- run history ---------------------------------------------------------------------------

function RunRow({ run }: { run: TrainingRun }) {
  const [open, setOpen] = useState(false);
  const overall = run.eval?.overall;
  return (
    <li className="border-t border-border-muted first:border-t-0">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="w-full flex items-center gap-3 px-3 py-2.5 text-left hover:bg-canvas-inset rounded-md focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
      >
        {open ? <ChevronDown className="w-4 h-4 text-fg-subtle shrink-0" aria-hidden="true" /> : <ChevronRight className="w-4 h-4 text-fg-subtle shrink-0" aria-hidden="true" />}
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm font-medium text-fg" title={formatDateTime(run.started_at ?? run.requested_at)}>
              {formatRelative(run.started_at ?? run.requested_at)}
            </span>
            <StatusPill tone={RUN_STATUS_TONE[run.status] ?? 'neutral'}>{humanize(run.status)}</StatusPill>
            <StatusPill tone="neutral">{humanize(run.trigger)}</StatusPill>
          </div>
          <p className="text-xs text-fg-muted mt-0.5">
            {run.examples.train + run.examples.valid} example(s)
            {overall && overall.n > 0 ? ` · ${pct(overall.active)} → ${pct(overall.candidate)} held-out accuracy` : ''}
            {run.reason ? ` · ${run.reason}` : ''}
          </p>
        </div>
      </button>
      {open && (
        <div className="px-3 pb-3 pl-10 space-y-2 text-xs text-fg-muted">
          <dl className="grid grid-cols-[auto_1fr] gap-x-2 gap-y-0.5">
            <dt>Labels</dt>
            <dd>
              {run.labels.qa_verdict} QA verdict · {run.labels.signal_hit} signal hit · {run.labels.speaker_role} speaker role
              {run.labels.withdrawn ? ` · ${run.labels.withdrawn} withdrawn` : ''}
            </dd>
            <dt>Examples</dt>
            <dd>
              {run.examples.train} train · {run.examples.valid} valid · {run.examples.eval_items} held out
            </dd>
            {Object.keys(run.examples.by_task).length > 0 && (
              <>
                <dt>By task</dt>
                <dd>{Object.entries(run.examples.by_task).map(([t, n]) => `${taskLabel(t)}: ${n}`).join(' · ')}</dd>
              </>
            )}
            {Object.keys(run.skipped).length > 0 && (
              <>
                <dt>Skipped</dt>
                <dd>{Object.entries(run.skipped).map(([reason, n]) => `${humanize(reason)}: ${n}`).join(' · ')}</dd>
              </>
            )}
            {run.trainer && run.trainer.iters !== null && (
              <>
                <dt>Trainer</dt>
                <dd>
                  {run.trainer.iters ?? '—'} iterations
                  {run.trainer.it_per_s !== null ? ` · ${run.trainer.it_per_s.toFixed(2)} it/s` : ''}
                  {run.trainer.train_loss !== null ? ` · train loss ${run.trainer.train_loss.toFixed(3)}` : ''}
                  {run.trainer.val_loss !== null ? ` · val loss ${run.trainer.val_loss.toFixed(3)}` : ''}
                  {run.trainer.peak_memory_gb !== null ? ` · ${run.trainer.peak_memory_gb.toFixed(1)} GB peak` : ''}
                  {run.trainer.seconds !== null ? ` · ${formatDuration(run.trainer.seconds)}` : ''}
                </dd>
              </>
            )}
            {run.eval && Object.keys(run.eval.tasks).length > 0 && (
              <>
                <dt>Per task</dt>
                <dd>
                  {Object.entries(run.eval.tasks)
                    .map(([task, e]) => `${taskLabel(task)}: ${pct(e.active)} → ${pct(e.candidate)} (n=${e.n})`)
                    .join(' · ')}
                </dd>
              </>
            )}
          </dl>
        </div>
      )}
    </li>
  );
}

function RunHistoryCard({ runs, installedActive }: { runs: TrainingRun[]; installedActive: boolean }) {
  return (
    <Card title="Run history" icon={History} subtitle={runs.length === 0 ? undefined : `${runs.length} of the last 20`}>
      {runs.length === 0 ? (
        <p className="text-sm text-fg-muted">
          No training runs on this appliance yet.
          {installedActive ? ' The active adapter was installed, not trained here; its offline evaluation is under Active model.' : ''}
        </p>
      ) : (
        <ul>
          {runs.map((run) => (
            <RunRow key={run.run_id} run={run} />
          ))}
        </ul>
      )}
    </Card>
  );
}

// --- notices (§6.2 item 6) ------------------------------------------------------------------

const NOTICE_TONE: Record<string, Tone> = {
  training_unavailable: 'yellow',
  insufficient_scope: 'yellow',
  qa_too_long: 'neutral',
  base_changed: 'yellow',
};

function TrainingNotices({ state }: { state: TrainingState }) {
  const notices = [...(state.notices ?? [])];
  const codes = new Set(notices.map((n) => n.code));
  if (!state.available && !codes.has('training_unavailable')) {
    notices.unshift({ code: 'training_unavailable', message: state.unavailable_reason ?? 'Training is unavailable on this host.' });
  }
  // A label-count failure the backend did not already turn into a notice (Store unreachable, not
  // configured): say so without hiding the rest of the page.
  const labelError = state.labels.error;
  if (labelError && !codes.has(labelError.code) && !(labelError.code === 'not_configured' && !state.available)) {
    notices.push({ code: labelError.code, message: `Label counts unavailable: ${labelError.message}` });
  }
  if (notices.length === 0) return null;
  return (
    <div className="space-y-2">
      {notices.map((n) => (
        <Notice key={n.code} tone={NOTICE_TONE[n.code] ?? 'yellow'}>
          {n.code === 'training_unavailable' ? `Training unavailable: ${n.message}` : n.message}
        </Notice>
      ))}
    </div>
  );
}

// --- view ------------------------------------------------------------------------------------

export default function SettingsView({
  demo,
  token,
  tokenConfigured,
  onToken,
}: {
  demo: boolean;
  token: string | null;
  tokenConfigured: boolean | undefined;
  onToken(token: string): void;
}) {
  const canWrite = Boolean(token);
  const query = useQuery({
    queryKey: queryKeys.training,
    queryFn: getTraining,
    refetchInterval: (q) => (q.state.data && ACTIVE_PHASES.has(q.state.data.status.phase) ? 5_000 : 30_000),
  });

  return (
    <div className="max-w-4xl mx-auto space-y-3">
      <PageHeader
        title="Settings"
        description={demo ? 'Choose a Call1 fine-tune and add your team’s knowledge from your own calls.' : 'Choose your model and improve it from your own calls.'}
        right={
          <Button icon={RefreshCw} size="sm" busy={query.isFetching} onClick={() => void query.refetch()}>
            Refresh
          </Button>
        }
      />
      {!canWrite && <ConsoleTokenNotice configured={tokenConfigured} onToken={onToken} />}
      <SignalPipelineStatus />
      {query.isLoading && <Loading label="Loading training settings…" />}
      <ErrorNotice error={query.error} />

      {query.data && (
        <div aria-live="polite" className="space-y-3">
          {demo ? <FineTuneExperience canWrite={canWrite} /> : <ModelPicker state={query.data} token={token} />}
          {!demo && <StatusCard state={query.data} token={token} canWrite={canWrite && query.data.available} />}
          <details className="rounded-lg border border-border bg-canvas-subtle p-4" data-testid="live-training-tools">
            <summary className="cursor-pointer text-sm font-semibold text-fg">{demo ? 'Live model and training tools' : 'Training schedule and history'}</summary>
            <div className="space-y-3 mt-4">
              {demo && <ModelPicker state={query.data} token={token} />}
              <TrainingNotices state={query.data} />

          <Card title="On-device training" icon={Settings2} subtitle={labelsSubtitle(query.data)}>
            <ScheduleForm state={query.data} token={token} canWrite={canWrite && query.data.available} />
          </Card>

          {demo && <StatusCard state={query.data} token={token} canWrite={canWrite && query.data.available} />}
          <ActiveModelCard state={query.data} token={token} canWrite={canWrite} />
          <RunHistoryCard
            runs={query.data.runs}
            installedActive={query.data.versions.some((v) => v.version === query.data.active?.version && v.provenance?.kind === 'installed')}
          />
            </div>
          </details>
        </div>
      )}
    </div>
  );
}
