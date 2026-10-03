import { Activity } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { getStoreHealth } from '../api';
import Panel from './Panel';
import StatusPill from './StatusPill';

const TLS_TONE: Record<string, { tone: 'green' | 'yellow' | 'red' | 'neutral'; label: string }> = {
  ok: { tone: 'green', label: 'OK' },
  expiring: { tone: 'yellow', label: 'Expiring soon' },
  expired: { tone: 'red', label: 'Expired' },
  hostname_mismatch: { tone: 'red', label: 'Hostname mismatch' },
  untrusted: { tone: 'neutral', label: 'Untrusted (Stage 4 not built)' },
};

export default function HealthPanel() {
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['store-health'],
    queryFn: getStoreHealth,
    refetchInterval: 30_000,
  });

  return (
    <Panel
      title="Health"
      icon={Activity}
      subtitle="GET /store/v1/status"
      right={
        !isLoading &&
        !isError &&
        data && <StatusPill tone="green">Reachable</StatusPill>
      }
    >
      {isLoading && <p className="text-sm text-fg-muted">Checking…</p>}
      {isError && (
        <div className="space-y-1">
          <StatusPill tone="red">Unreachable</StatusPill>
          <p className="text-xs text-fg-muted">{error instanceof Error ? error.message : 'Request failed'}</p>
        </div>
      )}
      {data && (
        <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1.5 text-sm">
          <dt className="text-fg-muted">Contract version</dt>
          <dd className="text-fg">{data.contract.contract_version}</dd>

          <dt className="text-fg-muted">Store hostname</dt>
          <dd className="text-fg break-all">{data.store_hostname}</dd>

          <dt className="text-fg-muted">Relying party</dt>
          <dd className="text-fg break-all">
            {data.relying_party.rp_id}
            <span className="text-fg-subtle"> · {data.relying_party.allowed_origins.join(', ')}</span>
          </dd>

          <dt className="text-fg-muted">TLS</dt>
          <dd>
            {(() => {
              const t = TLS_TONE[data.tls_health] ?? { tone: 'neutral' as const, label: data.tls_health };
              return <StatusPill tone={t.tone}>{t.label}</StatusPill>;
            })()}
          </dd>

          <dt className="text-fg-muted">Server time</dt>
          <dd className="text-fg">{new Date(data.server_time).toLocaleString()}</dd>
        </dl>
      )}
    </Panel>
  );
}
