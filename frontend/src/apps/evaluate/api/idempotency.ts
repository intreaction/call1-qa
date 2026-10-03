// `Idempotency-Key` handling for header-idempotent writes (reanalysis requests, rubric draft
// tests). Store scopes a key to the session and the route and digests the body, so one key must
// carry exactly one body: a retry of the same body reuses the key (Store answers with the original
// request), while any change to the body gets a fresh key. After a success the next submission is
// a new logical action and gets a new key.

import { useCallback, useRef } from 'react';
import { newIdempotencyKey } from './client';
import { StoreError } from './errors';

export interface IdempotencyKeys {
  /** The key for this body: the same one while the body is unchanged, a fresh one otherwise. */
  keyFor(body: unknown): string;
  /** Forget the current key (after a success, or when Store reports it reused). */
  reset(): void;
}

export function useIdempotencyKey(): IdempotencyKeys {
  const current = useRef<{ fingerprint: string; key: string } | null>(null);
  const keyFor = useCallback((body: unknown) => {
    const fingerprint = JSON.stringify(body ?? null);
    if (!current.current || current.current.fingerprint !== fingerprint) {
      current.current = { fingerprint, key: newIdempotencyKey() };
    }
    return current.current.key;
  }, []);
  const reset = useCallback(() => {
    current.current = null;
  }, []);
  return { keyFor, reset };
}

/**
 * Send one header-idempotent write. `send(key)` makes the request. If Store says the key was
 * already used for a different body (`idempotency_key_reused`), the key is replaced and the
 * request is sent once more under the new key; any other outcome is returned or thrown as is.
 */
export async function sendIdempotent<T>(keys: IdempotencyKeys, body: unknown, send: (key: string) => Promise<T>): Promise<T> {
  try {
    const result = await send(keys.keyFor(body));
    keys.reset();
    return result;
  } catch (err) {
    if (!(err instanceof StoreError && err.is('idempotency_key_reused'))) throw err;
    keys.reset();
    const result = await send(keys.keyFor(body));
    keys.reset();
    return result;
  }
}
