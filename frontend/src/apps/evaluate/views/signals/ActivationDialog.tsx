// The activation dialog after a taxonomy save (docs/ContactSignalsV2.md §9.1): optionally update
// recent calls (a digest-driven `rescore` backfill), send matching calls to the review queue (a
// SIGNAL queue rule) and alert on the edited node. Requests go in the order the spec fixes —
// alert rule, then queue rule, then backfill — so queue items are created as the backfilled
// results publish. Each step's outcome is shown; the first failure stops the rest.

import { useState } from 'react';
import { CheckCircle2, XCircle } from 'lucide-react';
import {
  categoryName,
  daysAgoIso,
  describeError,
  sendIdempotent,
  slugId,
  subcategoryName,
  useIdempotencyKey,
  type ContractInfo,
  type SignalAlertRuleRecord,
  type SignalTaxonomy,
  type StoreClient,
} from '../../api';
import { Button, Checkbox, Dialog, Field, Notice, TextInput } from '../../components/ui';
import type { SignedInSession } from '../../state/app';

export interface EditedNode {
  categoryId: string;
  subcategoryId?: string;
}

interface StepResult {
  step: string;
  ok: boolean;
  text: string;
}

const RULE_ID_RESERVED = new Set<string>();

export function ActivationDialog({
  client,
  contract,
  session,
  version,
  taxonomy,
  node,
  recipeChanged = false,
  alertRules,
  onClose,
  onDone,
}: {
  client: StoreClient;
  contract: ContractInfo;
  session: SignedInSession;
  version: number;
  taxonomy: SignalTaxonomy;
  node: EditedNode | null;
  /** A detection recipe changed (contract 1.4.0): recipes do not mark results outdated, so the
   * update reruns every signal stage (`rescore_signals`). */
  recipeChanged?: boolean;
  alertRules: SignalAlertRuleRecord[];
  onClose(): void;
  onDone(): void;
}) {
  const canQueue = session.can('manage_queue_rules');
  const maxBackfill = Math.min(200, contract.parameters.signal_backfill_max_calls);
  const nodeLabel = node
    ? node.subcategoryId
      ? `${categoryName(taxonomy, node.categoryId)} › ${subcategoryName(taxonomy, node.categoryId, node.subcategoryId)}`
      : categoryName(taxonomy, node.categoryId)
    : null;
  const [backfill, setBackfill] = useState(false);
  const [days, setDays] = useState(7);
  const [queue, setQueue] = useState(false);
  const [alert, setAlert] = useState(false);
  const [alertName, setAlertName] = useState((nodeLabel ?? '').slice(0, 60));
  const [busy, setBusy] = useState(false);
  const [results, setResults] = useState<StepResult[]>([]);
  const backfillKeys = useIdempotencyKey();
  const alertCap = alertRules.length >= contract.parameters.max_signal_alert_rules;
  const wantAlert = (alert || queue) && node !== null;
  const done = results.length > 0 && !busy;

  async function apply() {
    setBusy(true);
    const out: StepResult[] = [];
    const push = (r: StepResult) => {
      out.push(r);
      setResults([...out]);
    };
    try {
      let alertRuleId: string | null = null;
      if (wantAlert && node) {
        const base = node.subcategoryId ? `${node.categoryId}-${node.subcategoryId}` : node.categoryId;
        const id = slugId(base.replace(/[^a-z0-9_-]/g, '_').slice(0, 34), new Set(alertRules.map((r) => r.rule_id)), RULE_ID_RESERVED);
        try {
          const rule = await client.put('/store/v1/signals/alert-rules/{rule_id}', {
            path: { rule_id: id },
            body: {
              expected_record_version: 0,
              rule: {
                rule_id: id,
                name: alertName.trim() || id,
                enabled: true,
                condition: { category_id: node.categoryId, subcategory_id: node.subcategoryId ?? null },
              },
            },
          });
          alertRuleId = rule.rule_id;
          push({ step: 'alert', ok: true, text: `Alert rule created: "${rule.name}".` });
        } catch (err) {
          push({ step: 'alert', ok: false, text: `Alert not created: ${describeError(err)}` });
          return;
        }
      }
      if (queue && alertRuleId) {
        const queueRuleId = `signal-${alertRuleId}`;
        try {
          await client.put('/store/v1/review-queue/rules/{rule_id}', {
            path: { rule_id: queueRuleId },
            body: {
              expected_rule_version: 0,
              rule: {
                id: queueRuleId,
                name: `Signal: ${nodeLabel ?? alertRuleId}`.slice(0, 200),
                description: `Calls where the signal alert "${alertName.trim() || alertRuleId}" matches.`,
                stream: 'SIGNAL',
                enabled: true,
                rank: 100,
                distribution_strategy: 'UNASSIGNED_CLAIM',
                sampling_rate: 0,
                critical_failure_only: false,
                low_confidence_only: false,
                target_agents: [],
                target_domains: [],
                target_skills: [],
                target_signal_alerts: [alertRuleId],
              },
            },
          });
          push({ step: 'queue', ok: true, text: 'Matching calls will go to the review queue.' });
        } catch (err) {
          push({ step: 'queue', ok: false, text: `Queue rule not created: ${describeError(err)}` });
          return;
        }
      }
      if (backfill) {
        const body = { mode: 'rescore' as const, rescore_signals: recipeChanged, created_after: daysAgoIso(days), created_before: null, max_calls: maxBackfill };
        try {
          const b = await sendIdempotent(backfillKeys, body, (key) => client.post('/store/v1/signals/backfills', { headers: { 'Idempotency-Key': key }, body }));
          push({
            step: 'backfill',
            ok: true,
            text: `Updating ${b.requests_created} call${b.requests_created === 1 ? '' : 's'} to taxonomy v${b.taxonomy_version}${b.calls_skipped ? ` (${b.calls_skipped} already up to date or queued)` : ''}. They read "Refreshing" until each finishes.`,
          });
        } catch (err) {
          push({ step: 'backfill', ok: false, text: `Update not started: ${describeError(err)}` });
          return;
        }
      }
      if (!out.length) push({ step: 'none', ok: true, text: 'Nothing else to do. Existing calls keep their signals until they are updated.' });
    } finally {
      setBusy(false);
      onDone();
    }
  }

  return (
    <Dialog
      title={`Saved as taxonomy v${version}`}
      onClose={onClose}
      footer={
        done ? (
          <Button variant="primary" onClick={onClose}>
            Close
          </Button>
        ) : (
          <>
            <Button variant="ghost" disabled={busy} onClick={onClose}>
              Not now
            </Button>
            <Button variant="primary" busy={busy} disabled={!backfill && !wantAlert} onClick={() => void apply()}>
              Confirm
            </Button>
          </>
        )
      }
    >
      <p className="text-sm text-fg-muted">
        New calls use v{version} from now on. Existing calls keep the signals they were scored with until they are updated.
      </p>
      <div className="space-y-2" aria-disabled={done}>
        <div className="flex flex-wrap items-center gap-2">
          <Checkbox
            checked={backfill}
            disabled={busy || done}
            onChange={(e) => setBackfill(e.target.checked)}
            label={`Update calls from the last ${days} days (up to ${maxBackfill})`}
            hint={
              recipeChanged
                ? 'A detection recipe changed, so every signal stage reruns on those calls. Updates run behind new calls.'
                : 'Reruns only the stages this change outdated. Updates run behind new calls.'
            }
          />
          <Field label="Days">
            {(id) => (
              <TextInput
                id={id}
                type="number"
                min={1}
                max={90}
                value={days}
                disabled={busy || done || !backfill}
                className="w-20"
                onChange={(e) => setDays(Math.max(1, Math.min(90, Number(e.target.value) || 1)))}
              />
            )}
          </Field>
        </div>
        {canQueue && (
          <Checkbox
            checked={queue}
            disabled={busy || done || node === null || alertCap}
            onChange={(e) => setQueue(e.target.checked)}
            label="Send matching calls to the review queue"
            hint="Adds a SIGNAL queue rule on the alert below. Needs a QA evaluation on the call."
          />
        )}
        <Checkbox
          checked={alert || queue}
          disabled={busy || done || node === null || queue || alertCap}
          onChange={(e) => setAlert(e.target.checked)}
          label={nodeLabel ? `Alert on this: ${nodeLabel}` : 'Alert on this'}
          hint={
            node === null
              ? 'No single category or subcategory was edited.'
              : alertCap
                ? `At most ${contract.parameters.max_signal_alert_rules} alert rules exist already.`
                : 'In-app only: alerts feed filters, metrics, the queue and live updates. Nothing is sent outside Call1.'
          }
        />
        {(alert || queue) && node && (
          <Field label="Alert name">
            {(id) => <TextInput id={id} value={alertName} maxLength={60} disabled={busy || done} onChange={(e) => setAlertName(e.target.value)} />}
          </Field>
        )}
      </div>
      {results.length > 0 && (
        <ul className="space-y-1" aria-live="polite">
          {results.map((r) => (
            <li key={r.step} className={`flex items-start gap-1.5 text-sm ${r.ok ? 'text-primer-greenFg' : 'text-primer-redFg'}`}>
              {r.ok ? <CheckCircle2 className="w-4 h-4 mt-0.5 shrink-0" aria-hidden="true" /> : <XCircle className="w-4 h-4 mt-0.5 shrink-0" aria-hidden="true" />}
              {r.text}
            </li>
          ))}
        </ul>
      )}
      {!canQueue && <Notice>Review-queue rules need a supervisor or admin role.</Notice>}
    </Dialog>
  );
}
