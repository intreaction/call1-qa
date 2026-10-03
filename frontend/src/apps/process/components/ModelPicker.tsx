import { useEffect, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { Brain } from 'lucide-react';
import { queryKeys, setActiveAdapter, type TrainingState, type TrainingVersion } from '../api';
import { taskLabel } from '../labels';
import { Button, Card, ErrorNotice, Field, formatDateTime, SelectInput, StatusPill } from './ui';

const BASE = 'base';
const RUNNING_PHASES = new Set(['waiting', 'pausing', 'collecting', 'building', 'training', 'evaluating', 'deciding']);
const BASE_LABEL = 'Gemma 4 E2B · Base';

function modelName(model: TrainingVersion): string {
  return `Gemma 4 E2B · ${model.version}`;
}

/** Choose a kept fine-tune or the shipped base through Process's existing activation API.
 * The registry checks adapter completeness and base compatibility before accepting a change. */
export function ModelPicker({ state, token }: { state: TrainingState; token: string | null }) {
  const queryClient = useQueryClient();
  const activeId = state.active?.version ?? BASE;
  const [choice, setChoice] = useState(activeId);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [saved, setSaved] = useState<string | null>(null);
  useEffect(() => {
    setChoice(activeId);
  }, [activeId]);

  const selected = state.versions.find((v) => v.version === choice);
  const active = state.versions.find((v) => v.version === activeId);
  const running = RUNNING_PHASES.has(state.status.phase);
  const unavailableChoice = choice !== BASE && !selected;
  const disabledReason = !token ? 'Connect the console credential to choose a model.'
    : running ? 'Model selection is paused while a training run is queued or running.' : null;
  const same = choice === activeId;
  const activeName = active ? modelName(active) : activeId === BASE ? BASE_LABEL : `Gemma 4 E2B · ${activeId}`;

  async function apply() {
    setBusy(true);
    setError(null);
    setSaved(null);
    try {
      await setActiveAdapter(choice === BASE ? null : choice, token);
      await queryClient.invalidateQueries({ queryKey: queryKeys.training });
      setSaved(choice);
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card title="Processing model" icon={Brain} subtitle="Choose the base model or a fine-tuned edition installed on this computer">
      <div className="space-y-4" data-testid="processing-model-picker">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border-muted pb-3">
          <div className="min-w-0">
            <p className="text-xs text-fg-muted mb-1">Currently in use</p>
            <p className="text-sm font-semibold text-fg break-words" data-testid="active-processing-model">{activeName}</p>
          </div>
          <StatusPill tone={activeId === BASE ? 'neutral' : 'green'}>{activeId === BASE ? 'Base model' : 'Fine-tuned'}</StatusPill>
        </div>

        <Field label="Model to use">
          {(id) => (
            <SelectInput id={id} value={choice} disabled={busy || !!disabledReason} onChange={(e) => { setChoice(e.target.value); setError(null); setSaved(null); }}>
              <optgroup label="Included base model">
                <option value={BASE}>{BASE_LABEL} — no Call1 fine-tuning</option>
              </optgroup>
              {state.versions.length > 0 && (
                <optgroup label="Fine-tuned models">
                  {state.versions.map((v) => <option key={v.version} value={v.version}>{modelName(v)} — {v.provenance?.kind === 'installed' ? 'installed' : 'trained on this computer'}</option>)}
                </optgroup>
              )}
              {unavailableChoice && <option value={choice} disabled>{activeName} — unavailable</option>}
            </SelectInput>
          )}
        </Field>

        <div className="rounded-md border border-border-muted bg-canvas-subtle p-3 space-y-2" data-testid="selected-model-details">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-medium text-fg">{selected ? modelName(selected) : choice === BASE ? BASE_LABEL : activeName}</span>
            {choice === BASE && <StatusPill tone="blue">Included default</StatusPill>}
          </div>
          {choice === BASE ? (
            <p className="text-sm text-fg-muted">The shipped Gemma model, with no Call1 or customer fine-tune applied. The included default and fallback model.</p>
          ) : selected ? (
            <>
              <p className="text-sm text-fg-muted">{selected.provenance?.summary ?? 'Fine-tuned on this computer using reviewer corrections.'}</p>
              <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
                <dt className="text-fg-muted">Used for</dt>
                <dd className="text-fg">{selected.tasks.map(taskLabel).join(', ') || 'No processing tasks assigned'}</dd>
                <dt className="text-fg-muted">Created</dt>
                <dd className="text-fg">{formatDateTime(selected.created_at)}</dd>
                {selected.eval?.overall.candidate != null && (
                  <>
                    <dt className="text-fg-muted">Evaluation</dt>
                    <dd className="text-fg">{Math.round(selected.eval.overall.candidate * 100)}% {selected.eval.metric ?? 'held-out answer accuracy'} · {selected.eval.overall.n} {selected.eval.unit ?? 'items'}</dd>
                  </>
                )}
              </dl>
              <p className="text-xs text-fg-muted">Other tasks use the base model. Rule-based signal detection continues to follow each category’s recipe.</p>
            </>
          ) : <p className="text-sm text-fg-muted">This fine-tune is no longer in the installed model list. Select another model.</p>}
        </div>

        {state.versions.length === 0 && <p className="text-sm text-fg-muted">No fine-tuned models installed yet. Train one from reviewer corrections in Settings, or install a prepared fine-tune on this computer.</p>}
        {disabledReason && <p className="text-sm text-fg-muted">{disabledReason}</p>}
        <ErrorNotice error={error} />
        <div className="flex flex-wrap items-center gap-3">
          <Button variant="primary" disabled={same || !!disabledReason || unavailableChoice} busy={busy} onClick={() => void apply()}>Use selected model</Button>
          {saved === activeId && <p role="status" className="text-sm text-primer-greenFg">Model selection saved.</p>}
        </div>
        <p className="text-xs text-fg-muted">The selection is saved on this computer and applies when processing jobs start. Existing call results change only after reanalysis.</p>
      </div>
    </Card>
  );
}
