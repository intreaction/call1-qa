import { useState } from 'react';
import { Rss } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { listChanges } from '../api';
import { useSession, isAdmin, roleOf } from '../useSession';
import Panel from './Panel';
import { SignedOutState, InsufficientRoleState } from './SignedOutState';

export default function ChangeFeedPanel() {
  const session = useSession();
  const admin = session.kind === 'signed-in' && isAdmin(session.session);
  const [after, setAfter] = useState<string | undefined>(undefined);

  const feed = useQuery({
    queryKey: ['console-changes', after],
    queryFn: () => listChanges(after, 25),
    enabled: admin,
  });

  return (
    <Panel title="Change feed" icon={Rss} subtitle="GET /store/v1/changes">
      {session.kind === 'loading' && <p className="text-sm text-fg-muted">Checking session…</p>}
      {session.kind === 'error' && <p className="text-sm text-primer-redFg">{session.message}</p>}
      {session.kind === 'signed-out' && <SignedOutState />}
      {session.kind === 'signed-in' && !admin && <InsufficientRoleState role={roleOf(session.session)} />}

      {admin && (
        <div>
          {feed.isLoading && <p className="text-xs text-fg-muted">Loading…</p>}
          {feed.isError && <p className="text-xs text-primer-redFg">{(feed.error as Error).message}</p>}
          {feed.data && feed.data.events.length === 0 && (
            <p className="text-xs text-fg-muted">No change events yet.</p>
          )}
          {feed.data && feed.data.events.length > 0 && (
            <ul className="space-y-1.5 max-h-72 overflow-y-auto pr-1">
              {feed.data.events.map((ev) => (
                <li key={ev.cursor} className="text-xs border-b border-border-muted last:border-0 pb-1.5">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-fg font-medium">{ev.kind}</span>
                    <span className="text-fg-subtle">{new Date(ev.occurred_at).toLocaleTimeString()}</span>
                  </div>
                  <div className="text-fg-muted truncate">
                    {ev.resource_id}
                    {ev.status ? ` · ${ev.status}` : ''}
                    {ev.version != null ? ` · v${ev.version}` : ''}
                  </div>
                </li>
              ))}
            </ul>
          )}
          {feed.data?.next_cursor && feed.data.next_cursor !== after && (
            <button
              type="button"
              onClick={() => setAfter(feed.data!.next_cursor)}
              className="mt-2 px-2.5 py-1 text-xs font-medium rounded-md bg-canvas-inset hover:bg-canvas border border-border-control text-fg-muted hover:text-fg transition-colors"
            >
              Load more
            </button>
          )}
        </div>
      )}
    </Panel>
  );
}
