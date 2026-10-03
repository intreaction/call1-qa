// The contract error envelope (`ErrorResponse {code, message, details, retryable, request_id}`)
// mapped to typed errors. Every non-2xx Store response becomes a `StoreError`; a request that never
// reached Store (DNS, connection refused, the laptop is offline) becomes a `StoreUnreachableError`.

import type { Schema } from '@/contracts';

export type ErrorCode = Schema<'ErrorCode'>;
export type ErrorResponse = Schema<'ErrorResponse'>;

/** `details` shapes for the conflicts a client must handle (contracts/README.md "Error model"). */
export interface ConflictDetails {
  review_version_conflict: { current_version: number };
  conflict: { current_evaluation_version?: number; current_version?: number; reason?: string };
  state_version_conflict: { current_state_version?: number; current_connection_version?: number };
  rubric_version_conflict: { current_version?: number; draft_revision?: number };
  signal_taxonomy_conflict: { current_version?: number; record_version?: number };
  idempotency_key_reused: { original_id?: string };
  cursor_expired: { oldest_cursor?: string };
  cursor_unknown: { feed_epoch?: string };
}

export class StoreError extends Error {
  readonly status: number;
  /** The contract `ErrorCode`, or `unknown_error` when Store (or a proxy) sent no envelope. */
  readonly code: ErrorCode | 'unknown_error';
  readonly details: Record<string, unknown>;
  readonly retryable: boolean;
  readonly requestId: string | null;
  /** Seconds from `Retry-After` on 429 `rate_limited`, when sent. */
  readonly retryAfterSeconds: number | null;

  constructor(status: number, body: Partial<ErrorResponse> | null, retryAfter: string | null = null) {
    super(body?.message ?? `Store answered HTTP ${status}`);
    this.name = 'StoreError';
    this.status = status;
    this.code = (body?.code as ErrorCode | undefined) ?? 'unknown_error';
    this.details = (body?.details as Record<string, unknown> | undefined) ?? {};
    this.retryable = body?.retryable ?? status >= 500;
    this.requestId = body?.request_id ?? null;
    const parsed = retryAfter ? Number(retryAfter) : NaN;
    this.retryAfterSeconds = Number.isFinite(parsed) ? parsed : null;
  }

  /** Narrow to one code and get its typed `details`. */
  is<C extends keyof ConflictDetails>(code: C): this is StoreError & { details: ConflictDetails[C] };
  is(code: ErrorCode): boolean;
  is(code: string): boolean {
    return this.code === code;
  }
}

/** The request never got an HTTP answer: offline, Store down, or the connection was refused. */
export class StoreUnreachableError extends Error {
  readonly cause?: unknown;
  constructor(cause?: unknown) {
    super("Can't reach Store. Check the network connection; Evaluate retries on its own.");
    this.name = 'StoreUnreachableError';
    this.cause = cause;
  }
}

/** Store speaks a different contract major than this build of Evaluate. */
export class ContractMismatchError extends Error {
  readonly storeVersion: string;
  readonly builtFor: string;
  constructor(storeVersion: string, builtFor: string) {
    super(`This Evaluate build speaks Store contract ${builtFor}; the Store reports ${storeVersion}.`);
    this.name = 'ContractMismatchError';
    this.storeVersion = storeVersion;
    this.builtFor = builtFor;
  }
}

/** Codes that mean "there is no usable session" — the shell returns to sign-in. */
export const SIGNED_OUT_CODES: ReadonlySet<string> = new Set([
  'unauthenticated',
  'session_expired',
  'account_disabled',
]);

export function isSignedOutError(err: unknown): err is StoreError {
  return err instanceof StoreError && SIGNED_OUT_CODES.has(err.code);
}

export function isStoreError(err: unknown, code?: ErrorCode): err is StoreError {
  return err instanceof StoreError && (code === undefined || err.code === code);
}

/** The expected-version conflicts a write can hit (contracts/README.md "Error model"). A view that
 * gets one re-reads the resource (invalidate its query), shows what changed and asks for a fresh
 * submit: it never retries the write on its own. */
export const VERSION_CONFLICT_CODES: ReadonlySet<string> = new Set([
  'review_version_conflict',
  'conflict',
  'rubric_version_conflict',
  'signal_taxonomy_conflict',
]);

export function isVersionConflict(err: unknown): err is StoreError {
  return err instanceof StoreError && VERSION_CONFLICT_CODES.has(err.code);
}

function versionSuffix(value: unknown, label: string): string {
  return typeof value === 'number' ? ` (now ${label} ${value})` : '';
}

/** Deferred routes answer 501 `not_implemented` (docs/SplitBuild.md): render "not built yet". */
export function isNotImplemented(err: unknown): boolean {
  return err instanceof StoreError && (err.code === 'not_implemented' || err.status === 501);
}

/** One safe, human sentence for any error the client can throw. */
export function describeError(err: unknown): string {
  if (err instanceof StoreUnreachableError || err instanceof ContractMismatchError) return err.message;
  if (err instanceof StoreError) {
    switch (err.code) {
      case 'session_expired':
        return 'Your session expired. Sign in again.';
      case 'unauthenticated':
        return 'You are signed out. Sign in again.';
      case 'account_disabled':
        return 'This account is disabled. Ask an admin to re-enable it.';
      case 'insufficient_role':
      case 'forbidden':
        return err.message || 'Your role does not allow this.';
      case 'csrf_failed':
        return 'Store rejected the request token. Reload the page and try again.';
      case 'rate_limited':
        return err.retryAfterSeconds
          ? `Too many attempts. Try again in ${Math.ceil(err.retryAfterSeconds)} seconds.`
          : 'Too many attempts. Wait a minute and try again.';
      case 'not_implemented':
        return 'Store has not built this yet.';
      case 'store_unavailable':
        return 'Store is temporarily unavailable. Try again shortly.';
      case 'search_unavailable':
        return err.details.reason === 'not_installed'
          ? 'Semantic search is unavailable: the search embedding model is not installed on Store. Ask an admin to install it.'
          : 'Semantic search is unavailable: the search embedding model could not be loaded on Store. Ask an admin to check it.';
      case 'review_version_conflict':
        return `Someone changed this review since you opened it${versionSuffix(err.details.current_version, 'version')}. The latest decisions are shown; check them and submit again.`;
      case 'conflict':
        // Signal routes (1.3.0) reuse `conflict`: taxonomy redaction names a reason, and alert-rule
        // and hit-feedback saves carry `current_version` rather than an evaluation version.
        if (err.details.reason === 'redact_current') return 'The current taxonomy version cannot be redacted. Publish a newer version first.';
        if (err.details.reason === 'digest_mismatch') return 'That taxonomy version changed since you opened it. Reload the versions and try again.';
        if (err.details.current_evaluation_version === undefined && typeof err.details.current_version === 'number') {
          return `Someone changed this since you opened it${versionSuffix(err.details.current_version, 'version')}. The latest is shown; check it and submit again.`;
        }
        return `This call changed since you opened it${versionSuffix(err.details.current_evaluation_version, 'evaluation version')}. The latest result is shown; check it and submit again.`;
      case 'signal_taxonomy_conflict':
        return `This taxonomy changed elsewhere since you opened it${typeof err.details.current_version === 'number' ? ` (now taxonomy v${err.details.current_version})` : ''}. The latest saved version is loaded; make your change again and save.`;
      case 'rubric_version_conflict':
        return `This rubric changed since you opened it${versionSuffix(err.details.current_version ?? err.details.draft_revision, 'revision')}. The latest saved version is loaded; check it and submit again.`;
      case 'idempotency_key_reused':
        return 'Store already has a different request under this key. Submit again to send it as a new request.';
      default:
        return err.message;
    }
  }
  if (err instanceof Error) return err.message;
  return 'Something went wrong.';
}
