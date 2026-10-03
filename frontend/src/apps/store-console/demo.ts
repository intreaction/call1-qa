// Demo mode badge (task item 1, 2026-09-25): the console shows a "Demo mode" badge whenever
// Store answers `GET /demo/status` with `demo: true` (localhost only, off by default —
// `CALL1_STORE_DEMO=1`; `call1/store/auth/demo.py`). Badge only here — persona sign-in and switch
// are Evaluate's (frontend/src/apps/evaluate/api/demo.ts); the console keeps its own real
// passkey-only admin sign-in untouched. `/demo/status` is outside `/store/v1`, so this calls
// `fetch()` directly instead of `api.ts`'s `request()`, which always prefixes `STORE_API_PREFIX`.

import { useQuery } from '@tanstack/react-query';

export interface DemoStatus {
  demo: boolean;
  label: string;
}

const DISABLED: DemoStatus = { demo: false, label: '' };

async function fetchDemoStatus(signal?: AbortSignal): Promise<DemoStatus> {
  try {
    const res = await fetch('/demo/status', { credentials: 'include', headers: { Accept: 'application/json' }, signal });
    if (!res.ok) return DISABLED;
    const body = (await res.json()) as Partial<DemoStatus>;
    return body.demo ? { demo: true, label: body.label ?? '' } : DISABLED;
  } catch {
    return DISABLED;
  }
}

export function useDemoStatus() {
  return useQuery<DemoStatus>({
    queryKey: ['demo-status'],
    queryFn: ({ signal }) => fetchDemoStatus(signal),
    staleTime: 30_000,
    refetchOnWindowFocus: false,
  });
}
