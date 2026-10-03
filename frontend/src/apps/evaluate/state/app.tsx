// App-wide state: the Store client, the contract Store reported, the reviewer session, and the
// change feed. Views read these through hooks (`useStore`, `useSession`, `useChangeEvents`).

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { useQueryClient, type QueryClient } from '@tanstack/react-query';
import {
  StoreClient,
  describeError,
  isSignedOutError,
  keysForChange,
  queryKeys,
  startChangePoller,
  type ChangeEvent,
  type ChangeKind,
  type ChangePoller,
  type ContractInfo,
  type Permission,
  type ReviewerRole,
  type SessionInfo,
} from '../api';
import { markStoreReachable } from './connectivity';

// --- Store client and contract ------------------------------------------------------------------

interface StoreContextValue {
  client: StoreClient;
  contract: ContractInfo;
}

const StoreContext = createContext<StoreContextValue | null>(null);

export function StoreProvider({ client, contract, children }: StoreContextValue & { children: ReactNode }) {
  const value = useMemo(() => ({ client, contract }), [client, contract]);
  return <StoreContext.Provider value={value}>{children}</StoreContext.Provider>;
}

/** The typed Store client and the contract info (`parameters` holds Store's effective timings). */
export function useStore(): StoreContextValue {
  const ctx = useContext(StoreContext);
  if (!ctx) throw new Error('useStore outside StoreProvider');
  return ctx;
}

// --- session ------------------------------------------------------------------------------------

/** Drop every cached read that belongs to the previous session (keeps the contract info). */
function clearSessionData(queryClient: QueryClient) {
  queryClient.removeQueries({ predicate: (q) => q.queryKey[0] !== queryKeys.contract[0] });
}

const ROLE_RANK: Record<ReviewerRole, number> = { reviewer: 0, supervisor: 1, admin: 2 };

export type SessionState =
  | { status: 'loading' }
  | { status: 'signed-out'; notice?: string }
  | { status: 'signed-in'; session: SessionInfo }
  | { status: 'error'; message: string };

interface SessionContextValue {
  state: SessionState;
  /** Called by the sign-in and enrollment screens with the new `SessionInfo`. */
  signedIn(session: SessionInfo): void;
  signOut(): Promise<void>;
  /** Re-read `GET /auth/session` (role and permissions are re-evaluated by Store on every request). */
  refresh(): Promise<void>;
}

const SessionContext = createContext<SessionContextValue | null>(null);

export function SessionProvider({ children }: { children: ReactNode }) {
  const { client } = useStore();
  const queryClient = useQueryClient();
  const [state, setState] = useState<SessionState>({ status: 'loading' });

  const load = useCallback(async () => {
    try {
      const session = await client.get('/store/v1/auth/session');
      client.setCsrfToken(session.csrf_token);
      setState({ status: 'signed-in', session });
    } catch (err) {
      if (isSignedOutError(err)) {
        client.setCsrfToken(null);
        setState({ status: 'signed-out', notice: err.code === 'unauthenticated' ? undefined : describeError(err) });
      } else {
        setState((prev) => (prev.status === 'signed-in' ? prev : { status: 'error', message: describeError(err) }));
      }
    }
  }, [client]);

  useEffect(() => {
    void load();
  }, [load]);

  // Any request that finds the session gone returns the whole app to sign-in.
  useEffect(
    () =>
      client.subscribe((event) => {
        if (event.type !== 'signed-out') return;
        setState((prev) =>
          prev.status === 'signed-out' ? prev : { status: 'signed-out', notice: describeError(event.error) },
        );
        clearSessionData(queryClient);
      }),
    [client, queryClient],
  );

  // Pick up role changes and expiry when the reviewer comes back to the tab.
  useEffect(() => {
    const onVisible = () => {
      if (document.visibilityState === 'visible' && state.status === 'signed-in') void load();
    };
    document.addEventListener('visibilitychange', onVisible);
    return () => document.removeEventListener('visibilitychange', onVisible);
  }, [load, state.status]);

  const signedIn = useCallback(
    (session: SessionInfo) => {
      client.setCsrfToken(session.csrf_token);
      clearSessionData(queryClient);
      setState({ status: 'signed-in', session });
    },
    [client, queryClient],
  );

  const signOut = useCallback(async () => {
    try {
      await client.post('/store/v1/auth/sign-out');
    } catch (err) {
      if (!isSignedOutError(err)) throw err;
    }
    client.setCsrfToken(null);
    clearSessionData(queryClient);
    setState({ status: 'signed-out', notice: 'You signed out.' });
  }, [client, queryClient]);

  const value = useMemo(() => ({ state, signedIn, signOut, refresh: load }), [state, signedIn, signOut, load]);
  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSessionState(): SessionContextValue {
  const ctx = useContext(SessionContext);
  if (!ctx) throw new Error('useSessionState outside SessionProvider');
  return ctx;
}

export interface SignedInSession {
  session: SessionInfo;
  role: ReviewerRole;
  /** `SessionInfo.permissions` holds the permission (Store's `ROLE_PERMISSIONS`). */
  can(permission: Permission): boolean;
  /** The role is at least `min` (reviewer < supervisor < admin). */
  atLeast(min: ReviewerRole): boolean;
  signOut(): Promise<void>;
  refresh(): Promise<void>;
}

/** The signed-in session. Only call it below the signed-in shell. */
export function useSession(): SignedInSession {
  const { state, signOut, refresh } = useSessionState();
  if (state.status !== 'signed-in') throw new Error('useSession called while signed out');
  const { session } = state;
  return {
    session,
    role: session.role,
    can: (p) => session.permissions.includes(p),
    atLeast: (min) => (ROLE_RANK[session.role] ?? -1) >= ROLE_RANK[min],
    signOut,
    refresh,
  };
}

// --- change feed --------------------------------------------------------------------------------

type ChangeListener = (events: ChangeEvent[]) => void;

interface ChangeFeedContextValue {
  subscribe(listener: ChangeListener): () => void;
  pollNow(): void;
}

const ChangeFeedContext = createContext<ChangeFeedContextValue | null>(null);

/**
 * Polls the change feed while signed in, invalidates the query keys each event names
 * (`keysForChange`), and fans events out to `useChangeEvents` subscribers. On a lost cursor it
 * invalidates everything so every view re-reads its snapshot.
 */
export function ChangeFeedProvider({ children }: { children: ReactNode }) {
  const { client } = useStore();
  const queryClient = useQueryClient();
  const listeners = useRef(new Set<ChangeListener>());
  const poller = useRef<ChangePoller | null>(null);

  useEffect(() => {
    const p = startChangePoller(client, {
      onEvents: (events) => {
        const seen = new Set<string>();
        for (const event of events) {
          for (const key of keysForChange(event)) {
            const id = JSON.stringify(key);
            if (seen.has(id)) continue;
            seen.add(id);
            void queryClient.invalidateQueries({ queryKey: key as unknown[] });
          }
        }
        listeners.current.forEach((l) => l(events));
      },
      onResync: () => {
        void queryClient.invalidateQueries({ predicate: (q) => q.queryKey[0] !== queryKeys.contract[0] });
      },
      onConnectivity: (online) => markStoreReachable(online),
    });
    poller.current = p;
    const onVisible = () => {
      if (document.visibilityState === 'visible') p.pollNow();
    };
    document.addEventListener('visibilitychange', onVisible);
    return () => {
      document.removeEventListener('visibilitychange', onVisible);
      p.stop();
      poller.current = null;
    };
  }, [client, queryClient]);

  const value = useMemo<ChangeFeedContextValue>(
    () => ({
      subscribe(listener) {
        listeners.current.add(listener);
        return () => listeners.current.delete(listener);
      },
      pollNow() {
        poller.current?.pollNow();
      },
    }),
    [],
  );
  return <ChangeFeedContext.Provider value={value}>{children}</ChangeFeedContext.Provider>;
}

/**
 * Receive change events (optionally only some kinds, or only one call's). Query invalidation
 * already happens centrally; use this for anything else a view must do on change.
 */
export function useChangeEvents(
  listener: (events: ChangeEvent[]) => void,
  filter?: { kinds?: ChangeKind[]; callId?: string },
) {
  const ctx = useContext(ChangeFeedContext);
  const latest = useRef(listener);
  latest.current = listener;
  const kinds = filter?.kinds?.join(',');
  const callId = filter?.callId;
  useEffect(() => {
    if (!ctx) return;
    const allowed = kinds ? new Set(kinds.split(',')) : null;
    return ctx.subscribe((events) => {
      const matching = events.filter(
        (e) => (!allowed || allowed.has(e.kind)) && (!callId || e.call_id === callId),
      );
      if (matching.length) latest.current(matching);
    });
  }, [ctx, kinds, callId]);
}

/** Ask the poller to read the feed now (e.g. right after a write). */
export function usePollChanges(): () => void {
  const ctx = useContext(ChangeFeedContext);
  return useCallback(() => ctx?.pollNow(), [ctx]);
}
