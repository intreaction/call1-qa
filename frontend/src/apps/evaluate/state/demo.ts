// Demo mode status (`GET /demo/status`), shared by the sign-in screen (the buttons), the header
// (the badge) and the account page (persona switch). See api/demo.ts for the backend contract.

import { useQuery } from '@tanstack/react-query';
import { fetchDemoStatus, type DemoStatus } from '../api/demo';

export const DEMO_STATUS_QUERY_KEY = ['demo-status'] as const;

/** Reads before and after sign-in alike (it needs no session), so the sign-in screen, the header
 * badge and the account page all see the same answer without re-fetching it three times. Demo
 * mode is a server boot-time flag, not something that flips while the tab is open, so this is
 * cheap to keep a little stale rather than refetching on every focus. */
export function useDemoStatus() {
  return useQuery<DemoStatus>({
    queryKey: DEMO_STATUS_QUERY_KEY,
    queryFn: ({ signal }) => fetchDemoStatus(signal),
    staleTime: 30_000,
    refetchOnWindowFocus: false,
  });
}

/** True when `email` is one of the demo personas Store advertises (demo mode on). Used only to
 * hide passkey housekeeping prompts that make no sense for a keyless demo persona. */
export function useIsDemoPersona(email: string | undefined): boolean {
  const demo = useDemoStatus();
  if (!email || !demo.data?.demo) return false;
  const wanted = email.toLowerCase();
  return demo.data.personas.some((p) => p.email.toLowerCase() === wanted);
}
