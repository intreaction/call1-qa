import { Search } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { getStoreStatus } from '../api';
import { useSession, isAdmin, roleOf } from '../useSession';
import Panel from './Panel';
import StatusPill from './StatusPill';
import { SignedOutState, InsufficientRoleState } from './SignedOutState';

// Contract 1.2.0: StoreStatus.search_embedder, the one model Store runs (the local search embedder).
const STATE: Record<string, { tone: 'green' | 'yellow' | 'red' | 'neutral' | 'blue'; label: string }> = {
  loaded: { tone: 'green', label: 'Loaded' },
  installed: { tone: 'blue', label: 'Installed (loads on first search)' },
  not_installed: { tone: 'red', label: 'Not installed' },
  failed: { tone: 'red', label: 'Failed to load' },
  fake: { tone: 'neutral', label: 'Fake (test vectors)' },
};

export default function SearchEmbedderPanel() {
  const session = useSession();
  const admin = session.kind === 'signed-in' && isAdmin(session.session);
  const status = useQuery({ queryKey: ['console-status-detail'], queryFn: getStoreStatus, enabled: admin, refetchInterval: 30_000 });
  const embedder = status.data?.search_embedder;

  return (
    <Panel title="Search embedder" icon={Search} subtitle="GET /store/v1/status/detail">
      {session.kind === 'loading' && <p className="text-sm text-fg-muted">Checking session…</p>}
      {session.kind === 'error' && <p className="text-sm text-primer-redFg">{session.message}</p>}
      {session.kind === 'signed-out' && <SignedOutState />}
      {session.kind === 'signed-in' && !admin && <InsufficientRoleState role={roleOf(session.session)} />}
      {admin && status.isLoading && <p className="text-sm text-fg-muted">Loading…</p>}
      {admin && status.isError && <p className="text-sm text-primer-redFg">{(status.error as Error).message}</p>}
      {admin && status.data && !embedder && <p className="text-sm text-fg-muted">Store did not report a search embedder.</p>}
      {admin && embedder && (
        <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1.5 text-sm">
          <dt className="text-fg-muted">State</dt>
          <dd>
            {(() => {
              const s = STATE[embedder.state] ?? { tone: 'neutral' as const, label: embedder.state };
              return <StatusPill tone={s.tone}>{s.label}</StatusPill>;
            })()}
          </dd>
          <dt className="text-fg-muted">Model</dt>
          <dd className="text-fg break-all">{embedder.model}</dd>
          <dt className="text-fg-muted">Scheme</dt>
          <dd className="text-fg break-all">
            {embedder.scheme}
            <span className="text-fg-subtle"> · {embedder.dimensions} dimensions</span>
          </dd>
          {embedder.detail && (
            <>
              <dt className="text-fg-muted">Detail</dt>
              <dd className="text-fg">{embedder.detail}</dd>
            </>
          )}
        </dl>
      )}
    </Panel>
  );
}
