// Taxonomy versions (docs/ContactSignalsV2.md §10.1, §9.4): notes, who and when, and a short
// digest. Taxonomy text is customer data, so an admin can tombstone a non-current version's custom
// text ("Redact text"); its digest is kept, so results scored with it keep their provenance.

import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { EyeOff, History } from 'lucide-react';
import { queryKeys, type SignalTaxonomyRecord, type SignalTaxonomyVersion } from '../../api';
import { Button, Card, EmptyState, ErrorNotice, Field, Loading, StatusPill, TextInput, formatDateTime } from '../../components/ui';
import { usePollChanges } from '../../state/app';
import type { SignalsViewProps } from '../types';

export function VersionsTab({ client, session, record }: SignalsViewProps & { record: SignalTaxonomyRecord }) {
  const versionsQuery = useQuery({
    queryKey: queryKeys.signalTaxonomyVersions,
    queryFn: ({ signal }) => client.get('/store/v1/signals/taxonomy/versions', { signal }),
  });
  const versions = [...(versionsQuery.data?.items ?? [])].sort((a, b) => b.version - a.version);
  const me = session.session.account_id;

  return (
    <Card title="Versions" icon={History} subtitle="Every save publishes a new immutable version. Results record the version they were scored with.">
      {versionsQuery.isLoading && <Loading label="Loading versions…" />}
      <ErrorNotice error={versionsQuery.error} />
      {versionsQuery.isSuccess && versions.length === 0 && <EmptyState title="No versions" />}
      {versions.length > 0 && (
        <ul className="divide-y divide-border-muted">
          {versions.map((v) => (
            <VersionRow key={v.version} v={v} current={v.version === record.current.version} me={me} canRedact={session.can('manage_signals')} client={client} />
          ))}
        </ul>
      )}
    </Card>
  );
}

function VersionRow({ v, current, me, canRedact, client }: { v: SignalTaxonomyVersion; current: boolean; me: string; canRedact: boolean; client: SignalsViewProps['client'] }) {
  const qc = useQueryClient();
  const pollNow = usePollChanges();
  const [open, setOpen] = useState(false);
  const [reason, setReason] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const custom = v.taxonomy.categories.filter((c) => !c.builtin).length;
  const subs = v.taxonomy.categories.reduce((n, c) => n + c.subcategories.length, 0);
  const who = v.published_by_account_id === null ? 'Installed with Call1' : v.published_by_account_id === me ? 'You' : `Account ${v.published_by_account_id.slice(0, 8)}`;

  async function redact() {
    setBusy(true);
    setError(null);
    try {
      await client.post('/store/v1/signals/taxonomy/versions/{version}/redaction', {
        path: { version: v.version },
        body: { digest: v.digest, reason: reason.trim() },
      });
      setOpen(false);
      void qc.invalidateQueries({ queryKey: queryKeys.signalTaxonomyVersions });
      pollNow();
    } catch (err) {
      setError(err);
      void qc.invalidateQueries({ queryKey: queryKeys.signalTaxonomyVersions });
    } finally {
      setBusy(false);
    }
  }

  return (
    <li className="py-3 space-y-1.5" data-taxonomy-version={v.version}>
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium text-fg">Taxonomy v{v.version}</span>
        {current && <StatusPill tone="green">Current</StatusPill>}
        {v.text_redacted && (
          <StatusPill tone="neutral" title={v.redacted_at ? `Redacted ${formatDateTime(v.redacted_at)}` : undefined}>
            Text redacted
          </StatusPill>
        )}
        <span className="text-xs text-fg-muted">
          {who} · {formatDateTime(v.published_at)} · digest <span title={v.digest}>{v.digest.slice(0, 12)}</span>
        </span>
      </div>
      <p className="text-xs text-fg-muted">
        {custom} custom categor{custom === 1 ? 'y' : 'ies'} · {subs} subcategor{subs === 1 ? 'y' : 'ies'}
        {v.notes ? ` · ${v.notes}` : ''}
      </p>
      {canRedact && !current && !v.text_redacted && (
        <div className="space-y-1.5">
          {open ? (
            <div className="flex flex-wrap items-end gap-2">
              <Field label="Why redact (kept in the audit log)">
                {(id) => <TextInput id={id} value={reason} maxLength={200} className="w-72" onChange={(e) => setReason(e.target.value)} />}
              </Field>
              <Button size="sm" variant="danger" icon={EyeOff} busy={busy} disabled={!reason.trim()} onClick={() => void redact()}>
                Redact text of v{v.version}
              </Button>
              <Button size="sm" variant="ghost" disabled={busy} onClick={() => setOpen(false)}>
                Cancel
              </Button>
              <p className="basis-full text-xs text-fg-subtle">
                Replaces this version's custom names, glosses, descriptions, examples and field text with numbered [REDACTED] placeholders. It cannot be undone, and the version can no longer be used for updates.
              </p>
            </div>
          ) : (
            <Button size="sm" variant="ghost" icon={EyeOff} onClick={() => setOpen(true)}>
              Redact text
            </Button>
          )}
          <ErrorNotice error={error} />
        </div>
      )}
    </li>
  );
}
