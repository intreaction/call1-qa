import { useState, type FormEvent } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { KeyRound, Plus, RotateCw, Server } from 'lucide-react';
import {
  queryKeys,
  type ProcessInstallation,
  type ServiceKeyIssued,
  type ServiceKeyRecord,
  type ServiceScope,
  type Tone,
} from '../api';
import { Button, Card, EmptyState, ErrorNotice, Field, Loading, Notice, OneTimeSecret, StatusPill, TextInput, formatDateTime, formatRelative } from '../components/ui';
import { useStore } from '../state/app';
import { ReasonAction } from './common';

/** What a Process worker calls (call1/store/__main__.py `PROCESS_DEFAULT_SCOPES`). */
const DEFAULT_SCOPES: ServiceScope[] = [
  'calls:write',
  'artifacts:read',
  'artifacts:write',
  'jobs:write',
  'jobs:claim',
  'reanalysis:claim',
  'changes:read',
  'hardware:write',
  'catalog:publish',
  'usage:read',
  'admin-state:read',
];
/** Granted only on purpose: retry/cancel from the Process console, Pro1 key release, release trust. */
const EXTRA_SCOPES: ServiceScope[] = ['jobs:control', 'key-release:write', 'release-trust:write'];

/** Plain-language names for service-key scopes (display only; the raw scope stays in the tooltip). */
const SCOPE_LABELS: Record<ServiceScope, string> = {
  'calls:write': 'Add calls',
  'artifacts:read': 'Read call files',
  'artifacts:write': 'Save call files',
  'jobs:write': 'Report job results',
  'jobs:claim': 'Pick up jobs',
  'jobs:control': 'Retry and cancel jobs',
  'reanalysis:claim': 'Pick up reanalysis requests',
  'changes:read': 'Follow changes',
  'hardware:write': 'Report hardware',
  'catalog:publish': 'Publish the model list',
  'usage:read': 'Read usage',
  'admin-state:read': 'Read admin settings',
  'key-release:write': 'Release Pro1 keys',
  'release-trust:write': 'Manage release trust',
  'training:read': 'Read training labels',
};
const scopeLabel = (s: ServiceScope) => SCOPE_LABELS[s] ?? s;

function keyStatus(k: ServiceKeyRecord, now = Date.now()): { label: string; tone: Tone } {
  if (k.revoked_at) return { label: 'Revoked', tone: 'red' };
  if (k.expires_at && Date.parse(k.expires_at) <= now) return { label: 'Expired', tone: 'neutral' };
  if (k.superseded_by_key_id) {
    if (k.grace_until && Date.parse(k.grace_until) > now) return { label: `Rotated · valid until ${formatDateTime(k.grace_until)}`, tone: 'yellow' };
    return { label: 'Rotated', tone: 'neutral' };
  }
  return { label: 'Active', tone: 'green' };
}

function isLive(k: ServiceKeyRecord): boolean {
  const s = keyStatus(k).label;
  return s === 'Active' || s.startsWith('Rotated ·');
}

export function InstallationsPanel() {
  const { client } = useStore();
  const installations = useQuery({
    queryKey: queryKeys.admin.installations,
    queryFn: () => client.get('/store/v1/admin/installations'),
  });
  const keys = useQuery({
    queryKey: queryKeys.admin.serviceKeys,
    queryFn: () => client.get('/store/v1/admin/service-keys'),
  });
  const [issued, setIssued] = useState<{ what: string; value: ServiceKeyIssued } | null>(null);
  const items = installations.data?.items ?? [];
  const allKeys = keys.data?.items ?? [];

  return (
    <div className="flex flex-col gap-4">
      <Card
        title="Process installations"
        icon={Server}
        subtitle="Each Process installation reaches Store with its own service keys. Store keeps only each key's hash."
      >
        <div className="flex flex-col gap-3">
          <RegisterInstallation hasPrimary={items.some((i) => i.primary_host && !i.retired_at)} />
          {issued && (
            <OneTimeSecret label={issued.what} value={issued.value.token} onDismiss={() => setIssued(null)}>
              Put it in the Process installation's config (<span className="font-medium text-fg">CALL1_PROCESS_CONFIG</span>, default{' '}
              <span className="font-medium text-fg">data/process/config.json</span>, key <span className="font-medium text-fg">service_key</span>) or its environment as <strong>{issued.value.recommended_env_name}</strong>. Store keeps only
              its hash, so it cannot be shown again; if it is lost, rotate the key.
            </OneTimeSecret>
          )}
          {(installations.isLoading || keys.isLoading) && <Loading />}
          <ErrorNotice error={installations.error ?? keys.error} />
          {installations.isSuccess && items.length === 0 && (
            <EmptyState icon={Server} title="No Process installations">
              The installer normally registers the first one with <span className="font-medium text-fg">python -m call1.store issue-service-key --installation &lt;name&gt;</span>.
            </EmptyState>
          )}
          {items.map((inst) => (
            <InstallationBlock
              key={inst.id}
              installation={inst}
              keys={allKeys.filter((k) => k.installation_id === inst.id)}
              onIssued={(what, value) => setIssued({ what, value })}
            />
          ))}
          {(installations.data?.next_page_token || keys.data?.next_page_token) && (
            <Notice tone="yellow">Store returned more rows than this page shows.</Notice>
          )}
        </div>
      </Card>
    </div>
  );
}

function RegisterInstallation({ hasPrimary }: { hasPrimary: boolean }) {
  const { client } = useStore();
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [label, setLabel] = useState('');
  const [primary, setPrimary] = useState(false);
  const m = useMutation({
    mutationFn: () => client.post('/store/v1/admin/installations', { body: { label: label.trim(), primary_host: primary } }),
    onSuccess: () => {
      setOpen(false);
      setLabel('');
      setPrimary(false);
      void queryClient.invalidateQueries({ queryKey: queryKeys.admin.installations });
    },
  });
  if (!open) {
    return (
      <div>
        <Button icon={Plus} onClick={() => setOpen(true)}>
          Register installation
        </Button>
      </div>
    );
  }
  return (
    <form
      className="rounded-md border border-border bg-canvas p-3 flex flex-col gap-3"
      onSubmit={(e: FormEvent) => {
        e.preventDefault();
        m.mutate();
      }}
    >
      <Field label="Label" hint="For example the computer's name, “mac-mini”.">
        {(id) => <TextInput id={id} required maxLength={120} value={label} onChange={(e) => setLabel(e.target.value)} autoFocus />}
      </Field>
      <label className="flex items-start gap-2 text-sm text-fg">
        <input type="checkbox" className="mt-1" checked={primary} disabled={hasPrimary} onChange={(e) => setPrimary(e.target.checked)} />
        <span>
          Primary host — runs the on-appliance ML stages (audio never leaves it).{' '}
          {hasPrimary && <span className="text-fg-muted">An active installation is already the primary host.</span>}
        </span>
      </label>
      <ErrorNotice error={m.error} />
      <div className="flex gap-2">
        <Button type="submit" variant="primary" busy={m.isPending} disabled={!label.trim()}>
          Register
        </Button>
        <Button variant="ghost" onClick={() => setOpen(false)}>
          Cancel
        </Button>
      </div>
    </form>
  );
}

function InstallationBlock({
  installation: inst,
  keys,
  onIssued,
}: {
  installation: ProcessInstallation;
  keys: ServiceKeyRecord[];
  onIssued(what: string, value: ServiceKeyIssued): void;
}) {
  const { client } = useStore();
  const queryClient = useQueryClient();
  const [showOld, setShowOld] = useState(false);
  const [issuing, setIssuing] = useState(false);
  const refresh = () => {
    void queryClient.invalidateQueries({ queryKey: queryKeys.admin.installations });
    void queryClient.invalidateQueries({ queryKey: queryKeys.admin.serviceKeys });
  };
  const live = keys.filter(isLive);
  const shown = showOld ? keys : live;

  return (
    <div className="rounded-md border border-border-muted">
      <div className="p-3 flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-border-muted">
        <Server className="w-4 h-4 text-fg-subtle" aria-hidden="true" />
        <span className="text-sm font-medium text-fg">{inst.label}</span>
        {inst.primary_host && <StatusPill tone="blue">Primary host</StatusPill>}
        {inst.retired_at ? <StatusPill tone="neutral">Retired {formatDateTime(inst.retired_at)}</StatusPill> : <StatusPill tone="green">Active</StatusPill>}
        <span className="text-xs text-fg-muted">Registered {formatDateTime(inst.created_at)}</span>
        <span className="flex-1" />
        {!inst.retired_at && (
          <>
            <Button size="sm" icon={KeyRound} onClick={() => setIssuing(true)}>
              Issue key
            </Button>
            <ReasonAction
              label="Retire"
              variant="danger"
              warning={`Retiring ${inst.label} revokes all of its service keys. Its Process stops reaching Store.`}
              onConfirm={(reason) =>
                client.post('/store/v1/admin/installations/{installation_id}/retire', { path: { installation_id: inst.id }, body: { reason } }).then(refresh)
              }
            />
          </>
        )}
      </div>
      {issuing && (
        <IssueKey
          installation={inst}
          onCancel={() => setIssuing(false)}
          onIssued={(res) => {
            setIssuing(false);
            refresh();
            onIssued(`Service key for ${inst.label}`, res);
          }}
        />
      )}
      <div className="p-3 flex flex-col gap-2">
        {shown.length === 0 && <p className="text-sm text-fg-muted">{keys.length ? 'No live keys.' : 'No service keys.'}</p>}
        {shown.map((k) => (
          <KeyRow key={k.id} k={k} onChanged={refresh} onIssued={onIssued} />
        ))}
        {keys.length > live.length && (
          <div>
            <Button size="sm" variant="ghost" onClick={() => setShowOld((s) => !s)}>
              {showOld ? 'Hide revoked and expired keys' : `Show ${keys.length - live.length} revoked or expired key${keys.length - live.length === 1 ? '' : 's'}`}
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}

function KeyRow({ k, onChanged, onIssued }: { k: ServiceKeyRecord; onChanged(): void; onIssued(what: string, value: ServiceKeyIssued): void }) {
  const { client, contract } = useStore();
  const [rotating, setRotating] = useState(false);
  const [grace, setGrace] = useState('');
  const status = keyStatus(k);
  const defaultGraceHours = Math.round(contract.parameters.service_key_rotation_grace_seconds / 3600);
  const rotate = useMutation({
    mutationFn: () =>
      client.post('/store/v1/admin/service-keys/{key_id}/rotate', {
        path: { key_id: k.id },
        body: { grace_seconds: grace.trim() === '' ? null : Math.max(0, Math.round(Number(grace) * 3600)) },
      }),
    onSuccess: (res) => {
      setRotating(false);
      setGrace('');
      onChanged();
      onIssued(`Replacement key for ${k.label}`, res);
    },
  });
  const active = status.label === 'Active';

  return (
    <div className="flex flex-col gap-2 rounded-md bg-canvas border border-border-muted p-2.5">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="text-sm text-fg">{k.label}</span>
        <span className="text-xs text-fg-muted">{k.key_prefix}…</span>
        <StatusPill tone={status.tone}>{status.label}</StatusPill>
        <span className="text-xs text-fg-muted">
          Created {formatDateTime(k.created_at)} · last used {formatRelative(k.last_used_at)}
          {k.expires_at && ` · expires ${formatDateTime(k.expires_at)}`}
        </span>
        <span className="flex-1" />
        {active && (
          <Button size="sm" icon={RotateCw} onClick={() => setRotating(true)}>
            Rotate
          </Button>
        )}
        {!k.revoked_at && (
          <ReasonAction
            label="Revoke"
            variant="danger"
            warning="The key stops working at once. Leases and job IDs are unaffected, but its Process cannot reach Store until it gets a new key."
            onConfirm={(reason) => client.post('/store/v1/admin/service-keys/{key_id}/revoke', { path: { key_id: k.id }, body: { reason } }).then(onChanged)}
          />
        )}
      </div>
      <p className="text-xs text-fg-subtle" title={k.scopes.join(', ')}>
        Can: {k.scopes.map(scopeLabel).join(', ')}
      </p>
      {rotating && (
        <form
          className="flex flex-wrap items-end gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            rotate.mutate();
          }}
        >
          <Field label="Old key stays valid for (hours)" hint={`Leave empty for Store's default (${defaultGraceHours} h); 0 revokes it now.`}>
            {(id) => <TextInput id={id} type="number" min={0} step="any" className="w-40" value={grace} onChange={(e) => setGrace(e.target.value)} />}
          </Field>
          <Button type="submit" size="md" variant="primary" busy={rotate.isPending}>
            Issue replacement
          </Button>
          <Button variant="ghost" onClick={() => setRotating(false)}>
            Cancel
          </Button>
        </form>
      )}
      <ErrorNotice error={rotate.error} />
    </div>
  );
}

function IssueKey({ installation, onIssued, onCancel }: { installation: ProcessInstallation; onIssued(res: ServiceKeyIssued): void; onCancel(): void }) {
  const { client } = useStore();
  const [label, setLabel] = useState('');
  const [scopes, setScopes] = useState<Set<ServiceScope>>(new Set(DEFAULT_SCOPES));
  const [expires, setExpires] = useState('');
  const m = useMutation({
    mutationFn: () =>
      client.post('/store/v1/admin/service-keys', {
        body: {
          installation_id: installation.id,
          label: label.trim(),
          scopes: [...DEFAULT_SCOPES, ...EXTRA_SCOPES].filter((s) => scopes.has(s)),
          expires_at: expires ? new Date(expires).toISOString() : null,
        },
      }),
    onSuccess: onIssued,
  });
  const toggle = (s: ServiceScope) =>
    setScopes((prev) => {
      const next = new Set(prev);
      if (next.has(s)) next.delete(s);
      else next.add(s);
      return next;
    });

  return (
    <form
      className="p-3 border-b border-border-muted flex flex-col gap-3 bg-canvas"
      onSubmit={(e) => {
        e.preventDefault();
        m.mutate();
      }}
    >
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <Field label="Key label">
          {(id) => <TextInput id={id} required maxLength={120} value={label} onChange={(e) => setLabel(e.target.value)} autoFocus />}
        </Field>
        <Field label="Expires (optional)">
          {(id) => <TextInput id={id} type="datetime-local" value={expires} onChange={(e) => setExpires(e.target.value)} />}
        </Field>
      </div>
      <fieldset className="flex flex-col gap-2">
        <legend className="text-xs font-medium text-fg-muted mb-1">What this key can do</legend>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-1">
          {DEFAULT_SCOPES.map((s) => (
            <label key={s} className="flex items-center gap-2 text-sm text-fg" title={s}>
              <input type="checkbox" checked={scopes.has(s)} onChange={() => toggle(s)} />
              {scopeLabel(s)}
            </label>
          ))}
        </div>
        <p className="text-xs text-fg-muted">Grant these only on purpose:</p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-1">
          {EXTRA_SCOPES.map((s) => (
            <label key={s} className="flex items-center gap-2 text-sm text-fg" title={s}>
              <input type="checkbox" checked={scopes.has(s)} onChange={() => toggle(s)} />
              {scopeLabel(s)}
            </label>
          ))}
        </div>
      </fieldset>
      <ErrorNotice error={m.error} />
      <div className="flex gap-2">
        <Button type="submit" variant="primary" busy={m.isPending} disabled={!label.trim() || scopes.size === 0}>
          Issue key
        </Button>
        <Button variant="ghost" onClick={onCancel}>
          Cancel
        </Button>
      </div>
    </form>
  );
}

