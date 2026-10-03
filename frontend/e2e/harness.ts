/**
 * Node-side helpers for the Playwright e2e suite: the running stack (started by global-setup.ts
 * through tests/e2e/serve_stack.py), the Store CLI, cross-worker locks, the virtual-authenticator
 * credential store, recordings, and small Store/Process API clients.
 *
 * Nothing here imports app code: Evaluate is exercised only through the browser, Store only over
 * HTTP and its CLI, Process only over its loopback API.
 */
import { execFile } from 'node:child_process';
import { randomBytes, randomInt, randomUUID } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { promisify } from 'node:util';
import type { APIRequestContext, APIResponse } from '@playwright/test';

const execFileAsync = promisify(execFile);

/** What tests/e2e/serve_stack.py prints (Stack.info()), plus the Playwright run directory. */
export interface StackInfo {
  ready: boolean;
  dir: string;
  handlers: 'fake' | 'real';
  real_models: boolean;
  /** The Contact Signals pipeline setting the stack started with (CALL1_SIGNALS_PIPELINE, default v1). */
  signals_pipeline: 'v1' | 'shadow' | 'v2';
  python: string;
  repo: string;
  /** `http://localhost:<port>`: Evaluate and /store/v1. The WebAuthn RP ID is `localhost`. */
  store_url: string;
  store_port: number;
  store_data: string;
  /** `http://127.0.0.1:<port>`: the Process console and /process/api. */
  process_url: string;
  process_port: number;
  process_config: string;
  /** The console with its one-time credential in the fragment. */
  process_console_url: string;
  console_token: string;
  installation_id: string;
  service_key_id: string;
  /** Process's service key (Bearer). The Stage 2 worker scopes plus jobs:control. */
  service_key: string;
  logs_dir: string;
  store_env: Record<string, string>;
  /** Set by global setup: /private/tmp/call1-e2e/pw-run-…; credentials, locks and state live here. */
  run_dir: string;
}

let cached: StackInfo | null = null;

export function stackInfo(): StackInfo {
  if (cached) return cached;
  const file = process.env.CALL1_E2E_STACK_FILE;
  if (!file || !fs.existsSync(file)) {
    throw new Error('No e2e stack: run the suite with `npm run test:e2e` (global-setup.ts starts Store and Process).');
  }
  cached = { ...(JSON.parse(fs.readFileSync(file, 'utf8')) as StackInfo), run_dir: process.env.CALL1_E2E_RUN_DIR ?? path.dirname(file) };
  return cached;
}

export const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

// --- Store CLI ----------------------------------------------------------------------------

function cliEnv(stack: StackInfo): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {};
  for (const [key, value] of Object.entries(process.env)) {
    if (!key.startsWith('CALL1_') || key === 'CALL1_REAL_MODELS') env[key] = value;
  }
  return { ...env, ...stack.store_env, PYTHONUNBUFFERED: '1', PYTHONDONTWRITEBYTECODE: '1' };
}

/** `python -m call1.store <args>` against the stack's data, as the Store host user would run it. */
export async function storeCli(args: string[]): Promise<{ stdout: string; stderr: string }> {
  const stack = stackInfo();
  try {
    return await execFileAsync(stack.python, ['-m', 'call1.store', ...args], { cwd: stack.dir, env: cliEnv(stack), timeout: 60_000 });
  } catch (err) {
    const e = err as { stdout?: string; stderr?: string; message: string };
    throw new Error(`python -m call1.store ${args.join(' ')} failed: ${e.message}\n${e.stdout ?? ''}\n${e.stderr ?? ''}`);
  }
}

export type SetupCodePurpose = 'first_admin' | 'break_glass';

/** A one-time enrollment code from `python -m call1.store setup-code` (the only way to get one). */
export async function setupCode(
  email: string,
  displayName: string,
  options: { purpose?: SetupCodePurpose; targetAccountId?: string } = {},
): Promise<string> {
  const args = ['setup-code', '--email', email, '--display-name', displayName, '--purpose', options.purpose ?? 'first_admin'];
  if (options.targetAccountId) args.push('--target-account-id', options.targetAccountId);
  const { stdout } = await storeCli(args);
  const lines = stdout.split('\n').map((l) => l.trim()).filter(Boolean);
  const code = lines[lines.length - 1];
  if (!code) throw new Error(`setup-code printed no code:\n${stdout}`);
  return code;
}

// --- cross-worker locks and run state ------------------------------------------------------

function runPath(...parts: string[]): string {
  return path.join(stackInfo().run_dir, ...parts);
}

const safeName = (name: string) => name.replace(/[^A-Za-z0-9_.@-]+/g, '_');

/** Runs `fn` while holding a lock shared by every worker of this run (a mkdir lock in the run dir). */
export async function withLock<T>(name: string, fn: () => Promise<T>, timeoutMs = 180_000): Promise<T> {
  const dir = runPath('locks', safeName(name));
  fs.mkdirSync(path.dirname(dir), { recursive: true });
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    try {
      fs.mkdirSync(dir);
      break;
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== 'EEXIST') throw err;
      try {
        if (Date.now() - fs.statSync(dir).mtimeMs > 150_000) fs.rmSync(dir, { recursive: true, force: true }); // holder died
      } catch {
        // raced with the holder's release
      }
      if (Date.now() > deadline) throw new Error(`timed out waiting for e2e lock ${name}`);
      await sleep(50);
    }
  }
  try {
    return await fn();
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

export function readState<T>(name: string): T | null {
  const file = runPath('state', `${safeName(name)}.json`);
  return fs.existsSync(file) ? (JSON.parse(fs.readFileSync(file, 'utf8')) as T) : null;
}

export function writeState(name: string, value: unknown): void {
  const file = runPath('state', `${safeName(name)}.json`);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, JSON.stringify(value, null, 2), { mode: 0o600 });
}

export function statePath(name: string): string {
  const file = runPath('state', safeName(name));
  fs.mkdirSync(path.dirname(file), { recursive: true });
  return file;
}

// --- virtual-authenticator credentials ------------------------------------------------------

/** A CDP `WebAuthn.Credential` (base64 credentialId and PKCS#8 privateKey). */
export interface StoredCredential {
  credentialId: string;
  isResidentCredential: boolean;
  rpId?: string;
  privateKey: string;
  userHandle?: string;
  signCount: number;
  [extra: string]: unknown;
}

/** The credentials enrolled for an account, so another page's authenticator can sign in as it. */
export function readCredentials(email: string): StoredCredential[] {
  const file = runPath('credentials', `${safeName(email.toLowerCase())}.json`);
  return fs.existsSync(file) ? (JSON.parse(fs.readFileSync(file, 'utf8')) as StoredCredential[]) : [];
}

/** Merge by credentialId, keeping the highest signature counter (Store rejects a counter that goes back). */
export function saveCredentials(email: string, credentials: StoredCredential[]): void {
  const file = runPath('credentials', `${safeName(email.toLowerCase())}.json`);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const merged = new Map(readCredentials(email).map((c) => [c.credentialId, c]));
  for (const c of credentials) {
    const prior = merged.get(c.credentialId);
    merged.set(c.credentialId, prior && prior.signCount > c.signCount ? { ...c, signCount: prior.signCount } : c);
  }
  fs.writeFileSync(file, JSON.stringify([...merged.values()], null, 2), { mode: 0o600 });
}

// --- recordings ------------------------------------------------------------------------------

export interface IngestReceipt {
  conversation_id: string;
  call_id: string;
  graph_id: string;
  conversation_created: boolean;
  graph_created: boolean;
  jobs: number | unknown;
  evaluate_url: string;
  [extra: string]: unknown;
}

export interface IngestOptions {
  /** Default true: a copy with a new content digest, so each ingest is a new call. */
  unique?: boolean;
  agentId?: string;
  agentChannel?: 0 | 1;
  externalCallRef?: string;
  filename?: string;
  contentType?: string;
  /** Default 201. Pass null to get the raw response body whatever the status. */
  expectStatus?: number | null;
}

export function samplePath(name: string): string {
  const stack = stackInfo();
  if (path.isAbsolute(name) && fs.existsSync(name)) return name;
  for (const option of [path.join(stack.repo, 'sample_audio', name), path.join(stack.repo, 'sample_audio', `${name}.wav`)]) {
    if (fs.existsSync(option)) return option;
  }
  throw new Error(`no sample ${name} in ${path.join(stack.repo, 'sample_audio')}`);
}

/** A PCM WAV copy whose first four samples are small random values: a new SHA-256, same audio. */
export function uniqueWav(bytes: Buffer): Buffer {
  const out = Buffer.from(bytes);
  if (out.toString('ascii', 0, 4) !== 'RIFF' || out.toString('ascii', 8, 12) !== 'WAVE') {
    throw new Error('unique: true needs a RIFF/WAVE file; pass unique: false for other formats');
  }
  let offset = 12;
  let bits = 16;
  while (offset + 8 <= out.length) {
    const id = out.toString('ascii', offset, offset + 4);
    const size = out.readUInt32LE(offset + 4);
    if (id === 'fmt ') bits = out.readUInt16LE(offset + 8 + 14);
    if (id === 'data') {
      const width = Math.max(1, bits / 8);
      for (let i = 0; i < 4 && (i + 1) * width <= size; i++) {
        const at = offset + 8 + i * width;
        if (width === 1) out.writeUInt8(128 + randomInt(-16, 17), at);
        else if (width === 2) out.writeInt16LE(randomInt(-255, 256), at);
        else randomBytes(width).copy(out, at);
      }
      return out;
    }
    offset += 8 + size + (size % 2);
  }
  throw new Error('WAV file has no data chunk');
}

/** Upload a recording through Process's API (`POST /process/api/recordings`), as the console's Import view does. */
export async function ingestSample(name = 'call_01_compliant', options: IngestOptions = {}): Promise<IngestReceipt> {
  const stack = stackInfo();
  const file = samplePath(name);
  let bytes: Buffer = fs.readFileSync(file);
  if (options.unique !== false) bytes = uniqueWav(bytes);
  const form = new FormData();
  form.append('file', new Blob([new Uint8Array(bytes)], { type: options.contentType ?? 'audio/wav' }), options.filename ?? path.basename(file));
  if (options.agentId !== undefined) form.append('agent_id', options.agentId);
  if (options.agentChannel !== undefined) form.append('agent_channel', String(options.agentChannel));
  if (options.externalCallRef !== undefined) form.append('external_call_ref', options.externalCallRef);
  const response = await fetch(`${stack.process_url}/process/api/recordings`, {
    method: 'POST',
    headers: { 'X-Call1-Console-Token': stack.console_token },
    body: form,
  });
  const body = (await response.json()) as IngestReceipt;
  const expected = options.expectStatus === undefined ? 201 : options.expectStatus;
  if (expected !== null && response.status !== expected) {
    throw new Error(`ingest of ${path.basename(file)} answered ${response.status}: ${JSON.stringify(body)}`);
  }
  return body;
}

// --- Store and Process over HTTP from Node ------------------------------------------------

export function storePath(p: string): string {
  if (/^https?:\/\//.test(p) || p.startsWith('/store/v1/') || p === '/store/v1') return p;
  return '/store/v1' + (p.startsWith('/') ? p : `/${p}`);
}

export function processPath(p: string): string {
  if (/^https?:\/\//.test(p) || p.startsWith('/process/api/')) return p;
  return '/process/api' + (p.startsWith('/') ? p : `/${p}`);
}

/** Store with Process's service key (`Authorization: Bearer c1sk_…`), over 127.0.0.1. */
export async function serviceRequest(method: string, p: string, body?: unknown, headers: Record<string, string> = {}): Promise<Response> {
  const stack = stackInfo();
  return fetch(`http://127.0.0.1:${stack.store_port}${storePath(p)}`, {
    method,
    headers: {
      Authorization: `Bearer ${stack.service_key}`,
      ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}),
      ...headers,
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
}

/** Process's loopback API; `token: false` omits the console credential (reads do not need it). */
export async function processRequest(
  method: string,
  p: string,
  options: { json?: unknown; token?: boolean | string; headers?: Record<string, string> } = {},
): Promise<Response> {
  const stack = stackInfo();
  const headers: Record<string, string> = { ...(options.headers ?? {}) };
  if (options.token !== false) headers['X-Call1-Console-Token'] = typeof options.token === 'string' ? options.token : stack.console_token;
  if (options.json !== undefined) headers['Content-Type'] = 'application/json';
  return fetch(`${stack.process_url}${processPath(p)}`, {
    method,
    headers,
    body: options.json !== undefined ? JSON.stringify(options.json) : undefined,
  });
}

export interface JobGroupProgress {
  conversation_id: string;
  groups: { kind: string; state: string; total: number; succeeded: number; failed: number; [k: string]: unknown }[];
  settled: boolean;
  [extra: string]: unknown;
}

/** Resolve a call ID to its conversation through Process's ledger (or pass a `conv_…` ID). */
export async function conversationOf(callOrConversationId: string): Promise<string> {
  if (callOrConversationId.startsWith('conv_')) return callOrConversationId;
  const response = await processRequest('GET', '/conversations?limit=200', { token: false });
  const body = (await response.json()) as { items: { call_id?: string; conversation_id: string }[] };
  const found = body.items.find((i) => i.call_id === callOrConversationId);
  if (!found) throw new Error(`${callOrConversationId} is not a call Process's ledger lists`);
  return found.conversation_id;
}

/** Poll Store's `JobGroupProgress` (service key) until `settled`: every job terminal or dead-blocked. */
export async function waitUntilSettled(callOrConversationId: string, timeoutMs?: number): Promise<JobGroupProgress> {
  const stack = stackInfo();
  const conversation = await conversationOf(callOrConversationId);
  const deadline = Date.now() + (timeoutMs ?? (stack.real_models ? 900_000 : 60_000));
  let last: JobGroupProgress | null = null;
  while (Date.now() < deadline) {
    const response = await serviceRequest('GET', `/conversations/${conversation}/progress`);
    if (!response.ok) throw new Error(`progress of ${conversation} answered ${response.status}: ${await response.text()}`);
    last = (await response.json()) as JobGroupProgress;
    if (last.settled) return last;
    await sleep(250);
  }
  throw new Error(`conversation ${conversation} not settled: ${JSON.stringify(last)}`);
}

// --- Store as a signed-in reviewer (cookie + CSRF) -----------------------------------------

export interface SessionInfo {
  session_id: string;
  account_id: string;
  email: string;
  display_name: string;
  role: 'reviewer' | 'supervisor' | 'admin';
  permissions: string[];
  csrf_token: string;
  prompt_second_authenticator: boolean;
  [extra: string]: unknown;
}

export interface StoreRequestOptions {
  data?: unknown;
  params?: Record<string, string | number | boolean>;
  headers?: Record<string, string>;
  /** true: a fresh `Idempotency-Key`; a string: that key (repeat a request under the same key). */
  idempotencyKey?: boolean | string;
  /** Default true on writes. */
  csrf?: boolean;
}

/**
 * `/store/v1` as one signed-in account, through a Playwright `APIRequestContext` that carries its
 * session cookie (a page's `page.request`, or a request context built from saved storage state).
 * Writes send `X-Call1-CSRF` and the Store `Origin`.
 */
export class StoreApi {
  constructor(
    readonly request: APIRequestContext,
    readonly storeURL: string,
    public session: SessionInfo,
  ) {}

  static async forRequest(request: APIRequestContext, storeURL: string): Promise<StoreApi> {
    const response = await request.get(`${storeURL}/store/v1/auth/session`);
    if (!response.ok()) throw new Error(`GET /auth/session answered ${response.status()}: ${await response.text()}`);
    const body = (await response.json()) as Record<string, unknown>;
    const session = (typeof body.session === 'object' && body.session !== null ? body.session : body) as SessionInfo;
    return new StoreApi(request, storeURL, session);
  }

  async refresh(): Promise<SessionInfo> {
    const fresh = await StoreApi.forRequest(this.request, this.storeURL);
    this.session = fresh.session;
    return this.session;
  }

  async fetch(method: string, p: string, options: StoreRequestOptions = {}): Promise<APIResponse> {
    const headers: Record<string, string> = { ...(options.headers ?? {}) };
    const write = !['GET', 'HEAD', 'OPTIONS'].includes(method.toUpperCase());
    if (write) {
      headers['Origin'] ??= this.storeURL;
      if (options.csrf !== false) headers['X-Call1-CSRF'] ??= this.session.csrf_token;
    }
    if (options.idempotencyKey) headers['Idempotency-Key'] = options.idempotencyKey === true ? `e2e-${randomUUID()}` : options.idempotencyKey;
    return this.request.fetch(`${this.storeURL}${storePath(p)}`, { method, headers, data: options.data, params: options.params });
  }

  get(p: string, options?: StoreRequestOptions) {
    return this.fetch('GET', p, options);
  }
  post(p: string, data?: unknown, options: StoreRequestOptions = {}) {
    return this.fetch('POST', p, { ...options, data });
  }
  put(p: string, data?: unknown, options: StoreRequestOptions = {}) {
    return this.fetch('PUT', p, { ...options, data });
  }
  patch(p: string, data?: unknown, options: StoreRequestOptions = {}) {
    return this.fetch('PATCH', p, { ...options, data });
  }
  delete(p: string, options?: StoreRequestOptions) {
    return this.fetch('DELETE', p, options);
  }

  /** GET (or any method) and return the parsed JSON, failing with the body on a non-2xx answer. */
  async json<T = unknown>(method: string, p: string, options?: StoreRequestOptions): Promise<T> {
    const response = await this.fetch(method, p, options);
    if (!response.ok()) throw new Error(`${method} ${storePath(p)} answered ${response.status()}: ${await response.text()}`);
    return (await response.json()) as T;
  }
}

let emailCounter = 0;

/** `<prefix>-<n>-<hex>@e2e.test`, unique across workers and projects. */
export function uniqueEmail(prefix = 'user'): string {
  emailCounter += 1;
  const safe = prefix.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'user';
  return `${safe}-w${process.env.TEST_WORKER_INDEX ?? '0'}-${emailCounter}-${randomBytes(3).toString('hex')}@e2e.test`;
}

/** A synthetic client address for one browser context's anonymous auth steps (see fixtures.ts). */
export function syntheticClientAddress(): string {
  return `10.${randomInt(20, 220)}.${randomInt(0, 256)}.${randomInt(1, 255)}`;
}
