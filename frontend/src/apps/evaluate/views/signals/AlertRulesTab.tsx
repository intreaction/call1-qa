// Alert rules (docs/ContactSignalsV2.md §9.2, §10.1): named conditions at category, subcategory or
// field level, evaluated by Store at read time, so an edit applies at once with no model run. They
// feed review-queue rules, metrics, the calls filter and the change feed. Nothing is sent outside
// Call1 (decision 22 Q6).

import { useState } from 'react';
import { useQueryClient, type UseQueryResult } from '@tanstack/react-query';
import { Bell, Pencil, Plus } from 'lucide-react';
import {
  conditionText,
  fieldsForCondition,
  isVersionConflict,
  queryKeys,
  slugId,
  type Page,
  type SignalAlertRuleRecord,
  type SignalMetrics,
  type SignalTaxonomyRecord,
} from '../../api';
import { Button, Card, CharCount, EmptyState, ErrorNotice, Field, Loading, Notice, SelectInput, StatusPill, TextInput } from '../../components/ui';
import { usePollChanges } from '../../state/app';
import type { SignalsViewProps } from '../types';

interface Props extends SignalsViewProps {
  record: SignalTaxonomyRecord;
  rulesQuery: UseQueryResult<Page<SignalAlertRuleRecord>>;
  week: SignalMetrics | undefined;
  weekError: unknown;
}

export function AlertRulesTab({ client, contract, session, record, rulesQuery, week, weekError }: Props) {
  const canManage = session.can('manage_signals');
  const rules = rulesQuery.data?.items ?? [];
  const taxonomy = record.current.taxonomy;
  const [editing, setEditing] = useState<SignalAlertRuleRecord | 'new' | null>(null);
  const cap = contract.parameters.max_signal_alert_rules;
  const matched = new Map((week?.alerts ?? []).map((a) => [a.rule_id, a.calls_matched]));

  return (
    <Card
      title="Alert rules"
      icon={Bell}
      subtitle="In-app only: alerts feed the review queue, metrics, the calls filter and live updates. Editing a rule runs no model."
      right={
        canManage ? (
          <Button size="sm" icon={Plus} disabled={rules.length >= cap || editing !== null} onClick={() => setEditing('new')}>
            New alert rule
          </Button>
        ) : undefined
      }
    >
      {canManage && rules.length >= cap && (
        <p className="text-xs text-primer-yellowFg mb-2" role="status">
          At most {cap} alert rules: Store evaluates every rule on every read.
        </p>
      )}
      {editing !== null && (
        <AlertRuleEditor
          client={client}
          record={record}
          rule={editing === 'new' ? null : editing}
          takenIds={rules.map((r) => r.rule_id)}
          onClose={() => setEditing(null)}
        />
      )}
      {rulesQuery.isLoading && <Loading label="Loading alert rules…" />}
      <ErrorNotice error={rulesQuery.error} />
      {rulesQuery.isSuccess && rules.length === 0 && (
        <EmptyState icon={Bell} title="No alert rules yet">
          {canManage ? 'Add one here, or tick "Alert on this" when you save a taxonomy change.' : 'An admin adds alert rules.'}
        </EmptyState>
      )}
      {rules.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm" aria-label="Alert rules">
            <thead>
              <tr className="text-left text-xs text-fg-muted border-b border-border">
                <th scope="col" className="font-medium py-1.5 pr-3">Name</th>
                <th scope="col" className="font-medium py-1.5 pr-3">Condition</th>
                <th scope="col" className="font-medium py-1.5 pr-3">Status</th>
                <th scope="col" className="font-medium py-1.5 pr-3">Calls in 7 days</th>
                {canManage && <th scope="col" className="font-medium py-1.5"><span className="sr-only">Actions</span></th>}
              </tr>
            </thead>
            <tbody className="divide-y divide-border-muted">
              {rules.map((r) => (
                <tr key={r.rule_id} data-alert-rule={r.rule_id}>
                  <td className="py-2 pr-3 text-fg font-medium">{r.name}</td>
                  <td className="py-2 pr-3 text-fg-muted">
                    {conditionText(r.condition, taxonomy)}
                    {!r.node_active && (
                      <span className="block text-xs text-primer-yellowFg">Matches nothing: this category or subcategory is inactive in the current taxonomy.</span>
                    )}
                  </td>
                  <td className="py-2 pr-3">
                    <StatusPill tone={r.enabled ? (r.node_active ? 'green' : 'yellow') : 'neutral'}>
                      {r.enabled ? (r.node_active ? 'Enabled' : 'Enabled, inactive node') : 'Disabled'}
                    </StatusPill>
                  </td>
                  <td className="py-2 pr-3 tabular-nums text-fg-muted" title={weekError ? 'Signal metrics are unavailable' : undefined}>
                    {matched.get(r.rule_id) ?? '—'}
                  </td>
                  {canManage && (
                    <td className="py-2 text-right">
                      <Button size="sm" variant="ghost" icon={Pencil} disabled={editing !== null} onClick={() => setEditing(r)}>
                        Edit
                      </Button>
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

function AlertRuleEditor({
  client,
  record,
  rule,
  takenIds,
  onClose,
}: {
  client: SignalsViewProps['client'];
  record: SignalTaxonomyRecord;
  rule: SignalAlertRuleRecord | null;
  takenIds: string[];
  onClose(): void;
}) {
  const qc = useQueryClient();
  const pollNow = usePollChanges();
  const taxonomy = record.current.taxonomy;
  const [name, setName] = useState(rule?.name ?? '');
  const [categoryId, setCategoryId] = useState(rule?.condition.category_id ?? taxonomy.categories[0]?.category_id ?? '');
  const [subcategoryId, setSubcategoryId] = useState(rule?.condition.subcategory_id ?? '');
  const [fieldId, setFieldId] = useState(rule?.condition.field_id ?? '');
  const initialEquals = rule?.condition.field_equals;
  const [equals, setEquals] = useState(initialEquals === null || initialEquals === undefined ? '' : typeof initialEquals === 'boolean' ? `bool:${initialEquals}` : `enum:${initialEquals}`);
  const [minConfidence, setMinConfidence] = useState(rule?.condition.min_confidence != null ? String(rule.condition.min_confidence) : '');
  const [enabled, setEnabled] = useState(rule?.enabled ?? true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  const category = taxonomy.categories.find((c) => c.category_id === categoryId);
  const fields = fieldsForCondition(taxonomy, categoryId, subcategoryId || null);
  const field = fields.find((f) => f.field_id === fieldId);
  // "equals" is offered only for enum and boolean fields (§10.1); other types alert on presence.
  const equalsOptions: { value: string; label: string }[] =
    field?.type === 'enum'
      ? field.enum_values.map((v) => ({ value: `enum:${v}`, label: v }))
      : field?.type === 'boolean'
        ? [
            { value: 'bool:true', label: 'yes' },
            { value: 'bool:false', label: 'no' },
          ]
        : [];
  const conf = minConfidence === '' ? null : Number(minConfidence);
  const confInvalid = conf !== null && (!Number.isFinite(conf) || conf < 0.05 || conf > 0.95);
  const nameInvalid = !name.trim() || name.length > 60;

  async function save() {
    setBusy(true);
    setError(null);
    const id = rule?.rule_id ?? slugId(name, new Set(takenIds));
    const fieldEquals = equals.startsWith('bool:') ? equals === 'bool:true' : equals.startsWith('enum:') ? equals.slice(5) : null;
    try {
      await client.put('/store/v1/signals/alert-rules/{rule_id}', {
        path: { rule_id: id },
        body: {
          expected_record_version: rule?.record_version ?? 0,
          rule: {
            rule_id: id,
            name: name.trim(),
            enabled,
            condition: {
              category_id: categoryId,
              subcategory_id: subcategoryId || null,
              field_id: fieldId || null,
              field_equals: fieldId ? fieldEquals : null,
              min_confidence: conf,
            },
          },
        },
      });
      void qc.invalidateQueries({ queryKey: queryKeys.signalAlertRules });
      pollNow();
      onClose();
    } catch (err) {
      setError(err);
      if (isVersionConflict(err)) void qc.invalidateQueries({ queryKey: queryKeys.signalAlertRules });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mb-4 rounded-md border border-border-muted bg-canvas p-3 space-y-3" data-testid="signals-alert-editor">
      <p className="text-sm font-medium text-fg">{rule ? `Edit "${rule.name}"` : 'New alert rule'}</p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <Field label="Alert name" hint={<CharCount value={name} max={60} />}>
          {(id) => <TextInput id={id} value={name} maxLength={80} onChange={(e) => setName(e.target.value)} />}
        </Field>
        <Field label="Category">
          {(id) => (
            <SelectInput
              id={id}
              value={categoryId}
              onChange={(e) => {
                setCategoryId(e.target.value);
                setSubcategoryId('');
                setFieldId('');
                setEquals('');
              }}
            >
              {taxonomy.categories.map((c) => (
                <option key={c.category_id} value={c.category_id}>
                  {c.name}
                  {c.active ? '' : ' (retired)'}
                </option>
              ))}
            </SelectInput>
          )}
        </Field>
        <Field label="Subcategory">
          {(id) => (
            <SelectInput
              id={id}
              value={subcategoryId}
              onChange={(e) => {
                setSubcategoryId(e.target.value);
                setFieldId('');
                setEquals('');
              }}
            >
              <option value="">Any subcategory</option>
              {(category?.subcategories ?? []).map((s) => (
                <option key={s.subcategory_id} value={s.subcategory_id}>
                  {s.name}
                  {s.active ? '' : ' (inactive)'}
                </option>
              ))}
              <option value="other">Other</option>
            </SelectInput>
          )}
        </Field>
        <Field label="Field" hint={fields.length === 0 ? 'No fields on this path.' : 'Alone, the alert fires when the field was extracted.'}>
          {(id) => (
            <SelectInput
              id={id}
              value={fieldId}
              disabled={fields.length === 0}
              onChange={(e) => {
                setFieldId(e.target.value);
                setEquals('');
              }}
            >
              <option value="">No field condition</option>
              {fields.map((f) => (
                <option key={f.field_id} value={f.field_id}>
                  {f.name}
                </option>
              ))}
            </SelectInput>
          )}
        </Field>
        <Field label="Equals" hint={field && equalsOptions.length === 0 ? 'Only "one of a list" and yes/no fields can be compared.' : undefined}>
          {(id) => (
            <SelectInput id={id} value={equals} disabled={equalsOptions.length === 0} onChange={(e) => setEquals(e.target.value)}>
              <option value="">Any value</option>
              {equalsOptions.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </SelectInput>
          )}
        </Field>
        <Field label="Minimum confidence (optional)" hint={confInvalid ? <span className="text-primer-redFg">Use 0.05 to 0.95, or leave empty.</span> : 'A calibrated score from 0.05 to 0.95.'}>
          {(id) => <TextInput id={id} type="number" min={0.05} max={0.95} step={0.05} value={minConfidence} onChange={(e) => setMinConfidence(e.target.value)} />}
        </Field>
      </div>
      <label className="flex items-center gap-2 text-sm text-fg">
        <input type="checkbox" checked={enabled} onChange={(e) => setEnabled(e.target.checked)} />
        Enabled
      </label>
      <p className="text-xs text-fg-muted">
        Condition:{' '}
        <span className="text-fg">
          {conditionText(
            {
              category_id: categoryId,
              subcategory_id: subcategoryId || null,
              field_id: fieldId || null,
              field_equals: equals.startsWith('bool:') ? equals === 'bool:true' : equals.startsWith('enum:') ? equals.slice(5) : null,
              min_confidence: confInvalid ? null : conf,
            },
            taxonomy,
          )}
        </span>
      </p>
      {record.settings.pipeline === 'v1' && (subcategoryId || fieldId) && (
        <Notice>Subcategory and field conditions match once Contact Signals v2 is on; v1 results have categories only.</Notice>
      )}
      <ErrorNotice error={error} />
      <div className="flex gap-2">
        <Button variant="primary" busy={busy} disabled={nameInvalid || confInvalid || !categoryId} onClick={() => void save()}>
          {rule ? 'Save alert rule' : 'Create alert rule'}
        </Button>
        <Button variant="ghost" disabled={busy} onClick={onClose}>
          Cancel
        </Button>
      </div>
    </div>
  );
}
