import { useQuery } from '@tanstack/react-query';
import { getSession, isAuthError, type SessionInfo } from './api';

export type SessionStatus =
  | { kind: 'loading' }
  | { kind: 'signed-out' }
  | { kind: 'signed-in'; session: SessionInfo }
  | { kind: 'error'; message: string };

/** GET /store/v1/auth/session once; 401/403 is a normal "signed out" outcome, not a failure to
 * surface as an error banner (docs/SplitBuild.md: "show a clear signed-out state"). */
export function useSession(): SessionStatus {
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['console-session'],
    queryFn: getSession,
    retry: false,
    refetchOnWindowFocus: true,
  });

  if (isLoading) return { kind: 'loading' };
  if (isError) {
    if (isAuthError(error)) return { kind: 'signed-out' };
    return { kind: 'error', message: error instanceof Error ? error.message : 'Session check failed' };
  }
  if (!data) return { kind: 'signed-out' };
  return { kind: 'signed-in', session: data };
}

export function isAdmin(session: SessionInfo): boolean {
  // auth.py: reviewer < supervisor < admin. Re-read on every /auth/session response per the
  // contract ("role ... re-read on every request"), never cached past this query's own staleness.
  return session.role === 'admin';
}

export function roleOf(session: SessionInfo): string {
  return session.role;
}
