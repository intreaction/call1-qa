// Demo mode: canned persona sign-in for a class presentation, localhost only (John, 2026-09-25 —
// "auth does not need to be fully functional" for the demo pass; the real passkey flow in
// api/passkeys.ts is untouched and stays the only way to sign in for real).
//
// The backend is `call1/store/auth/demo.py` — off (404) unless Store was started with
// `CALL1_STORE_DEMO=1` in dev mode:
//   GET  /demo/status    -> { demo, label, personas: [{ persona, display_name, role, email }] }
//   POST /demo/sign-in   { persona } -> the same shape a passkey sign-in returns
//                                       ({ session: SessionInfo, ...}), plus { demo: true, persona }
//
// This is Evaluate's one documented exception to "talks only to /store/v1" (docs/SplitBuild.md
// rule 3; api/client.ts). `/demo/*` is not part of the frozen contract (no generated types), so
// this file calls `fetch()` directly instead of going through `StoreClient`. It is the ONLY other
// file under apps/evaluate allowed to do that — tests/test_split_boundaries.py checks it — and
// every request here targets literally `/demo/status` or `/demo/sign-in`, nothing else.

import type { SessionInfo } from './types';

export const DEMO_STATUS_PATH = '/demo/status';
export const DEMO_SIGN_IN_PATH = '/demo/sign-in';

export type DemoPersonaKey = 'admin' | 'supervisor' | 'reviewer';

export interface DemoPersona {
  persona: DemoPersonaKey;
  display_name: string;
  role: DemoPersonaKey;
  email: string;
}

export interface DemoStatus {
  demo: boolean;
  label: string;
  personas: DemoPersona[];
}

export interface DemoSignInResult {
  session: SessionInfo;
  demo: true;
  persona: DemoPersonaKey;
}

const DISABLED: DemoStatus = { demo: false, label: '', personas: [] };

/** `GET /demo/status`. Demo mode off, or Store unreachable, both read as disabled — never throws,
 * so a public sign-in screen can call this unconditionally without its own error handling. */
export async function fetchDemoStatus(signal?: AbortSignal): Promise<DemoStatus> {
  try {
    const res = await fetch(DEMO_STATUS_PATH, { credentials: 'include', headers: { Accept: 'application/json' }, signal });
    if (!res.ok) return DISABLED;
    const body = (await res.json()) as Partial<DemoStatus>;
    if (!body.demo) return DISABLED;
    return { demo: true, label: body.label ?? '', personas: body.personas ?? [] };
  } catch {
    return DISABLED;
  }
}

/** A Store-shaped error for a failed `/demo/*` call (mirrors `StoreError`'s envelope reading,
 * without importing it — that class's constructor assumes the full contract `ErrorResponse`). */
export class DemoError extends Error {
  readonly status: number;
  readonly code: string;
  constructor(status: number, body: { code?: string; message?: string } | null) {
    super(body?.message ?? `Demo sign-in failed (HTTP ${status})`);
    this.name = 'DemoError';
    this.status = status;
    this.code = body?.code ?? 'unknown_error';
  }
}

/** `POST /demo/sign-in`. Sets the session cookie itself, exactly like a passkey sign-in; the
 * caller passes `result.session` to `signedIn()` (state/app.tsx) the same way. */
export async function demoSignIn(persona: DemoPersonaKey): Promise<DemoSignInResult> {
  const res = await fetch(DEMO_SIGN_IN_PATH, {
    method: 'POST',
    credentials: 'include',
    headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
    body: JSON.stringify({ persona }),
  });
  if (!res.ok) {
    let body: { code?: string; message?: string } | null = null;
    try {
      body = (await res.json()) as { code?: string; message?: string };
    } catch {
      // not JSON: the status code still says what happened
    }
    throw new DemoError(res.status, body);
  }
  return (await res.json()) as DemoSignInResult;
}

/** True when `email` belongs to one of the fixed demo personas (`call1/store/auth/demo.py`
 * `PERSONAS`). `SessionInfo` carries no "this is a demo session" flag of its own — the contract is
 * frozen and demo mode is not part of it — so this is how the header badge and the account menu's
 * persona switch know a *restored* (page-reload) session is a demo one. */
export function isDemoPersonaEmail(email: string): boolean {
  return email.endsWith('@call1-demo.example') && email.startsWith('demo.');
}
