import { Server } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { listInstallations, listServiceKeys } from '../api';
import { useSession, isAdmin, roleOf } from '../useSession';
import Panel from './Panel';
import StatusPill from './StatusPill';
import { SignedOutState, InsufficientRoleState } from './SignedOutState';

export default function InstallationsPanel() {
  const session = useSession();
  const admin = session.kind === 'signed-in' && isAdmin(session.session);

  const installations = useQuery({
    queryKey: ['console-installations'],
    queryFn: listInstallations,
    enabled: admin,
  });
  const serviceKeys = useQuery({
    queryKey: ['console-service-keys'],
    queryFn: listServiceKeys,
    enabled: admin,
  });

  return (
    <Panel title="Process installations & service keys" icon={Server} subtitle="GET /store/v1/admin/installations, /admin/service-keys">
      {session.kind === 'loading' && <p className="text-sm text-fg-muted">Checking session…</p>}
      {session.kind === 'error' && <p className="text-sm text-primer-redFg">{session.message}</p>}
      {session.kind === 'signed-out' && <SignedOutState />}
      {session.kind === 'signed-in' && !admin && <InsufficientRoleState role={roleOf(session.session)} />}

      {admin && (
        <div className="space-y-4">
          <div>
            <h3 className="text-xs font-semibold text-fg-muted uppercase tracking-wide mb-1.5">Installations</h3>
            {installations.isLoading && <p className="text-xs text-fg-muted">Loading…</p>}
            {installations.isError && <p className="text-xs text-primer-redFg">{(installations.error as Error).message}</p>}
            {installations.data && installations.data.items.length === 0 && (
              <p className="text-xs text-fg-muted">No Process installation is registered yet.</p>
            )}
            {installations.data && installations.data.items.length > 0 && (
              <ul className="space-y-1.5">
                {installations.data.items.map((inst) => (
                  <li key={inst.id} className="flex items-center justify-between gap-2 text-sm">
                    <span className="text-fg truncate">{inst.label}</span>
                    <span className="flex items-center gap-1.5 shrink-0">
                      {inst.primary_host && <StatusPill tone="blue">Primary host</StatusPill>}
                      {inst.retired_at ? (
                        <StatusPill tone="neutral">Retired</StatusPill>
                      ) : (
                        <StatusPill tone="green">Active</StatusPill>
                      )}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div>
            <h3 className="text-xs font-semibold text-fg-muted uppercase tracking-wide mb-1.5">Service keys</h3>
            {serviceKeys.isLoading && <p className="text-xs text-fg-muted">Loading…</p>}
            {serviceKeys.isError && <p className="text-xs text-primer-redFg">{(serviceKeys.error as Error).message}</p>}
            {serviceKeys.data && serviceKeys.data.items.length === 0 && (
              <p className="text-xs text-fg-muted">No service key has been issued yet.</p>
            )}
            {serviceKeys.data && serviceKeys.data.items.length > 0 && (
              <ul className="space-y-1.5">
                {serviceKeys.data.items.map((key) => (
                  <li key={key.id} className="flex items-center justify-between gap-2 text-sm">
                    <span className="text-fg-muted truncate">{key.id}</span>
                    <StatusPill tone={key.grace_until ? 'yellow' : 'green'}>
                      {key.grace_until ? 'Rotating out' : 'Active'}
                    </StatusPill>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      )}
    </Panel>
  );
}
