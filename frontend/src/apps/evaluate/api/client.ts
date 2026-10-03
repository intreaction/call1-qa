// The typed Store client Evaluate uses for every request. It is built on the generated contract
// (frontend/src/contracts/store-v1.ts) and talks only to `/store/v1` (docs/SplitBuild.md rule 3):
// never `/api/v1` (legacy), never Process. `api/demo.ts` is the one documented exception — demo
// mode's `/demo/status` and `/demo/sign-in` are not part of the frozen contract, so that file
// calls `fetch()` directly instead of going through this client; nowhere else does.
//
//   const page = await client.get('/store/v1/calls', { query: { limit: 50 } });
//   const call = await client.get('/store/v1/calls/{call_id}', { path: { call_id } });
//   await client.post('/store/v1/calls/{call_id}/reanalysis-requests', {
//     path: { call_id },
//     headers: { 'Idempotency-Key': newIdempotencyKey() },
//     body: { ... },
//   });
//
// Paths, path parameters, query parameters, required headers, request bodies and response bodies
// are all checked against the contract at compile time.
//
// Rules it implements:
// - `credentials: 'include'` on every request (the session cookie).
// - `X-Call1-CSRF` on every state-changing request. The token is `SessionInfo.csrf_token`, kept in
//   memory only (never web storage). If it is missing the client reads `GET /auth/session` first,
//   and on 403 `csrf_failed` it re-reads the session once and retries.
// - The contract error envelope becomes `StoreError`; no HTTP answer becomes
//   `StoreUnreachableError` and flips the connectivity signal to offline.
// - 401 `unauthenticated` / `session_expired` and 403 `account_disabled` notify the shell, which
//   returns to sign-in.

import { CSRF_HEADER, STORE_API_PREFIX, type paths } from '@/contracts';
import { StoreError, StoreUnreachableError, isSignedOutError } from './errors';

export type HttpMethod = 'get' | 'post' | 'put' | 'patch' | 'delete';

/** Every contract path that has `M`. */
export type PathsWith<M extends HttpMethod> = {
  [P in keyof paths]: paths[P] extends { [K in M]: unknown } ? P : never;
}[keyof paths];

/** The operation object of `M P`. */
export type OperationAt<P extends keyof paths, M extends HttpMethod> = paths[P] extends { [K in M]: infer O } ? O : never;

type ParamsOf<O> = O extends { parameters: infer X } ? X : Record<string, never>;
type PathParamsOf<O> = ParamsOf<O> extends { path: infer X } ? X : never;
type QueryOf<O> = ParamsOf<O> extends { query?: infer Q } ? Exclude<Q, undefined> : never;
type HeaderOf<O> = ParamsOf<O> extends { header: infer H } ? H : never;
type BodyOf<O> = O extends { requestBody: { content: { 'application/json': infer B } } }
  ? B
  : O extends { requestBody?: { content: { 'application/json': infer B } } }
    ? B | undefined
    : never;

/** The JSON body of the 200/201 response, or `void` for 204 and non-JSON responses. */
export type ResponseOf<O> = O extends { responses: { 200: { content: { 'application/json': infer R } } } }
  ? R
  : O extends { responses: { 201: { content: { 'application/json': infer R } } } }
    ? R
    : void;

type IsNever<T> = [T] extends [never] ? true : false;

export type RequestOptions<O> = (IsNever<PathParamsOf<O>> extends true ? { path?: undefined } : { path: PathParamsOf<O> }) &
  (IsNever<QueryOf<O>> extends true ? { query?: undefined } : { query?: QueryOf<O> }) &
  (IsNever<HeaderOf<O>> extends true ? { headers?: undefined } : { headers: HeaderOf<O> }) &
  (IsNever<BodyOf<O>> extends true
    ? { body?: undefined }
    : undefined extends BodyOf<O>
      ? { body?: BodyOf<O> }
      : { body: BodyOf<O> }) & {
    signal?: AbortSignal;
  };

// eslint-disable-next-line @typescript-eslint/ban-types
type Args<O> = {} extends RequestOptions<O> ? [opts?: RequestOptions<O>] : [opts: RequestOptions<O>];

/** Shape of the response for `client.get(P)`, e.g. `StoreResponse<'/store/v1/calls', 'get'>`. */
export type StoreResponse<P extends keyof paths, M extends HttpMethod> = ResponseOf<OperationAt<P, M>>;

export type ClientEvent =
  | { type: 'signed-out'; error: StoreError }
  | { type: 'reachable' }
  | { type: 'unreachable'; error: StoreUnreachableError };

type Listener = (event: ClientEvent) => void;

interface RawInit {
  method: string;
  path?: Record<string, unknown>;
  query?: Record<string, unknown>;
  headers?: Record<string, string>;
  body?: unknown;
  signal?: AbortSignal;
}

const SAFE_METHODS = new Set(['GET', 'HEAD', 'OPTIONS']);

/** A fresh `Idempotency-Key` for one logical request (reanalysis, draft tests). Reuse it on retry. */
export function newIdempotencyKey(): string {
  if (typeof crypto.randomUUID === 'function') return crypto.randomUUID();
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}

export function buildUrl(template: string, path?: Record<string, unknown>, query?: Record<string, unknown>): string {
  let url = template.replace(/\{([^}]+)\}/g, (_, name: string) => {
    const value = path?.[name];
    if (value === undefined || value === null) throw new Error(`Missing path parameter ${name} for ${template}`);
    return encodeURIComponent(String(value));
  });
  if (query) {
    const qs = new URLSearchParams();
    for (const [key, value] of Object.entries(query)) {
      if (value === undefined || value === null) continue;
      if (Array.isArray(value)) value.forEach((v) => qs.append(key, String(v)));
      else qs.append(key, String(value));
    }
    const s = qs.toString();
    if (s) url += `?${s}`;
  }
  return url;
}

export class StoreClient {
  private csrfToken: string | null = null;
  private csrfPromise: Promise<string | null> | null = null;
  private listeners = new Set<Listener>();
  private readonly fetchImpl: typeof fetch;

  constructor(fetchImpl: typeof fetch = (...args) => fetch(...args)) {
    this.fetchImpl = fetchImpl;
  }

  /** Subscribe to signed-out and connectivity events. Returns the unsubscribe function. */
  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private emit(event: ClientEvent) {
    this.listeners.forEach((l) => l(event));
  }

  /** Set from `SessionInfo.csrf_token` after sign-in, enrollment or `GET /auth/session`. */
  setCsrfToken(token: string | null) {
    this.csrfToken = token;
  }

  get hasCsrfToken(): boolean {
    return this.csrfToken !== null;
  }

  get<P extends PathsWith<'get'>>(path: P, ...args: Args<OperationAt<P, 'get'>>) {
    return this.call<ResponseOf<OperationAt<P, 'get'>>>('GET', path, args[0]);
  }

  post<P extends PathsWith<'post'>>(path: P, ...args: Args<OperationAt<P, 'post'>>) {
    return this.call<ResponseOf<OperationAt<P, 'post'>>>('POST', path, args[0]);
  }

  put<P extends PathsWith<'put'>>(path: P, ...args: Args<OperationAt<P, 'put'>>) {
    return this.call<ResponseOf<OperationAt<P, 'put'>>>('PUT', path, args[0]);
  }

  patch<P extends PathsWith<'patch'>>(path: P, ...args: Args<OperationAt<P, 'patch'>>) {
    return this.call<ResponseOf<OperationAt<P, 'patch'>>>('PATCH', path, args[0]);
  }

  delete<P extends PathsWith<'delete'>>(path: P, ...args: Args<OperationAt<P, 'delete'>>) {
    return this.call<ResponseOf<OperationAt<P, 'delete'>>>('DELETE', path, args[0]);
  }

  /**
   * A same-origin URL for a GET route whose response is not JSON (call audio, artifact content).
   * Use it as `<audio src>`: the browser sends the session cookie itself. Grant URLs Store returns
   * are used as given; in dev mode they are `http://localhost:…` (Store README deviation 3), so
   * never reject them for not being https.
   */
  url<P extends PathsWith<'get'>>(path: P, ...args: Args<OperationAt<P, 'get'>>): string {
    const opts = args[0] as { path?: Record<string, unknown>; query?: Record<string, unknown> } | undefined;
    return buildUrl(path, opts?.path, opts?.query);
  }

  /** The raw `Response` of a GET (binary bodies, CSV). Errors still throw `StoreError`. */
  async raw<P extends PathsWith<'get'>>(path: P, ...args: Args<OperationAt<P, 'get'>>): Promise<Response> {
    const opts = args[0] as { path?: Record<string, unknown>; query?: Record<string, unknown>; signal?: AbortSignal } | undefined;
    return this.send({ method: 'GET', path: opts?.path, query: opts?.query, signal: opts?.signal }, path, true);
  }

  private async call<R>(method: string, template: string, opts: unknown): Promise<R> {
    const o = (opts ?? {}) as Omit<RawInit, 'method'>;
    const res = await this.send({ method, ...o }, template, false);
    if (res.status === 204) return undefined as R;
    const type = res.headers.get('content-type') ?? '';
    if (!type.includes('json')) return undefined as R;
    return (await res.json()) as R;
  }

  private async send(init: RawInit, template: string, raw: boolean, retried = false): Promise<Response> {
    if (!template.startsWith(STORE_API_PREFIX)) throw new Error(`Evaluate only calls ${STORE_API_PREFIX}`);
    const url = buildUrl(template, init.path, init.query);
    const headers: Record<string, string> = { Accept: raw ? '*/*' : 'application/json', ...(init.headers ?? {}) };
    const unsafe = !SAFE_METHODS.has(init.method);
    if (unsafe) {
      const token = this.csrfToken ?? (await this.refreshCsrf());
      if (token) headers[CSRF_HEADER] = token;
    }
    let body: BodyInit | undefined;
    if (init.body !== undefined) {
      headers['Content-Type'] = 'application/json';
      body = JSON.stringify(init.body);
    }

    let res: Response;
    try {
      res = await this.fetchImpl(url, {
        method: init.method,
        credentials: 'include',
        headers,
        body,
        signal: init.signal,
      });
    } catch (cause) {
      if (cause instanceof DOMException && cause.name === 'AbortError') throw cause;
      const err = new StoreUnreachableError(cause);
      this.emit({ type: 'unreachable', error: err });
      throw err;
    }

    // A proxy in front of a stopped Store (the Vite dev server) answers 502/504 without the
    // envelope; that is "unreachable" too.
    if ((res.status === 502 || res.status === 504) && !(res.headers.get('content-type') ?? '').includes('json')) {
      const err = new StoreUnreachableError(new Error(`HTTP ${res.status}`));
      this.emit({ type: 'unreachable', error: err });
      throw err;
    }
    this.emit({ type: 'reachable' });

    if (res.ok) return res;

    let envelope: Record<string, unknown> | null = null;
    try {
      envelope = (await res.json()) as Record<string, unknown>;
    } catch {
      // not JSON: the status code still says what happened
    }
    const error = new StoreError(res.status, envelope, res.headers.get('Retry-After'));

    if (unsafe && error.code === 'csrf_failed' && !retried) {
      this.csrfToken = null;
      const token = await this.refreshCsrf();
      if (token) return this.send(init, template, raw, true);
    }
    if (isSignedOutError(error)) {
      this.csrfToken = null;
      this.emit({ type: 'signed-out', error });
    }
    throw error;
  }

  /** Re-read `GET /auth/session` for its CSRF token. Null when signed out. */
  private refreshCsrf(): Promise<string | null> {
    if (!this.csrfPromise) {
      this.csrfPromise = (async () => {
        try {
          const res = await this.fetchImpl(`${STORE_API_PREFIX}/auth/session`, {
            credentials: 'include',
            headers: { Accept: 'application/json' },
          });
          if (!res.ok) return null;
          const session = (await res.json()) as { csrf_token?: string };
          this.csrfToken = session.csrf_token ?? null;
          return this.csrfToken;
        } catch {
          return null;
        } finally {
          this.csrfPromise = null;
        }
      })();
    }
    return this.csrfPromise;
  }
}
