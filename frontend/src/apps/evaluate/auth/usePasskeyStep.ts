import { useCallback, useState } from 'react';
import { describeError, isStoreError } from '../api';
import { describeWebAuthnError, needsUserGesture } from '../api/passkeys';

export type StepStatus = 'idle' | 'working' | 'needs-gesture' | 'failed';

/**
 * Runs one browser passkey prompt (plus its Store request) and classifies the outcome, so a screen
 * can show "press the button to continue" when the browser demands a click, or a readable error.
 */
export function usePasskeyStep() {
  const [status, setStatus] = useState<StepStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  /** The thrown error behind `error`, for screens that special-case a Store code. */
  const [cause, setCause] = useState<unknown>(null);
  /** True when the failure consumed the Store ceremony, so a retry must begin a new one. */
  const [ceremonyUsed, setCeremonyUsed] = useState(false);

  const run = useCallback(async <T,>(fn: () => Promise<T>): Promise<T | undefined> => {
    setStatus('working');
    setError(null);
    setCause(null);
    try {
      const result = await fn();
      setStatus('idle');
      setCeremonyUsed(false);
      return result;
    } catch (err) {
      if (needsUserGesture(err) && !isStoreError(err)) {
        // Either the browser wanted a click, or the reviewer dismissed the prompt. Both are
        // retried with the same ceremony from a button.
        setStatus('needs-gesture');
        setError(describeWebAuthnError(err));
        setCause(err);
        setCeremonyUsed(false);
        return undefined;
      }
      setStatus('failed');
      setError(describeWebAuthnError(err) ?? describeError(err));
      setCause(err);
      setCeremonyUsed(isStoreError(err));
      return undefined;
    }
  }, []);

  const reset = useCallback(() => {
    setStatus('idle');
    setError(null);
    setCause(null);
    setCeremonyUsed(false);
  }, []);

  return { status, error, cause, ceremonyUsed, run, reset };
}
