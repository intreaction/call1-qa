// Typed client for the Store operations console. Reads only — the console displays Store's
// state, it never writes it (docs/SplitBuild.md: "the plan says the console can display these
// but not change them"). Built on the generated contract types
// (frontend/src/contracts/store-v1.ts) and nothing else, same rule as Evaluate.

import { STORE_API_PREFIX, type Output } from '@/contracts';

export type StoreHealth = Output<'StoreHealth'>;
export type SessionInfo = Output<'SessionInfo'>;
export type ProcessInstallation = Output<'ProcessInstallation'>;
export type ServiceKeyRecord = Output<'ServiceKeyRecord'>;
export type ChangeFeed = Output<'ChangeFeed'>;
export type ErrorResponse = Output<'ErrorResponse'>;
export type ContractInfo = Output<'ContractInfo'>;
export type StoreStatus = Output<'StoreStatus'>;

export class StoreApiError extends Error {
  readonly status: number;
  readonly code?: string;
  readonly retryable: boolean;

  constructor(status: number, body: Partial<ErrorResponse> | null) {
    super(body?.message ?? `HTTP ${status}`);
    this.status = status;
    this.code = body?.code as string | undefined;
    this.retryable = body?.retryable ?? false;
  }
}

/** True for the two shapes the console treats as "not signed in", never a bug to report. */
export function isAuthError(err: unknown): boolean {
  return err instanceof StoreApiError && (err.status === 401 || err.status === 403);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${STORE_API_PREFIX}${path}`, {
    credentials: 'include',
    headers: { Accept: 'application/json', ...(init?.headers ?? {}) },
    ...init,
  });
  if (!res.ok) {
    let body: Partial<ErrorResponse> | null = null;
    try {
      body = await res.json();
    } catch {
      // non-JSON error body (e.g. a proxy/504) — the status code still tells the story
    }
    throw new StoreApiError(res.status, body);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

/** GET /store/v1/status — anonymous, always available. */
export function getStoreHealth(): Promise<StoreHealth> {
  return request<StoreHealth>('/status');
}

/** GET /store/v1/status/detail — admin session only (carries the search embedder's state). */
export function getStoreStatus(): Promise<StoreStatus> {
  return request<StoreStatus>('/status/detail');
}

/** GET /store/v1/contract — anonymous, always available. */
export function getStoreContract(): Promise<ContractInfo> {
  return request<ContractInfo>('/contract');
}

/** GET /store/v1/auth/session — 401/403 means signed out; the console treats that as data. */
export function getSession(): Promise<SessionInfo> {
  return request<SessionInfo>('/auth/session');
}

/** GET /store/v1/admin/installations — admin session only. */
export function listInstallations(): Promise<{ items: ProcessInstallation[] }> {
  return request('/admin/installations');
}

/** GET /store/v1/admin/service-keys — admin session only. */
export function listServiceKeys(): Promise<{ items: ServiceKeyRecord[] }> {
  return request('/admin/service-keys');
}

/** GET /store/v1/changes — admin session only in this console (reviewer or Process would also
 * qualify per the contract, but the console gates all three admin panels on `admin` alike, per
 * docs/SplitBuild.md). */
export function listChanges(after?: string, limit = 25): Promise<ChangeFeed> {
  const qs = new URLSearchParams();
  if (after) qs.set('after', after);
  qs.set('limit', String(limit));
  return request(`/changes?${qs.toString()}`);
}
