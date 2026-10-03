// The Process console credential (call1/process/console.py). Process prints it once as
// `http://127.0.0.1:8020/#console_token=c1con_...`. This hook reads it from the URL fragment on
// first load, clears the fragment immediately (a hash is never sent to a server, but it still
// lingers in browser history otherwise), and keeps the token in sessionStorage — never
// localStorage, since it is a secret and the console README says "memory or sessionStorage".
//
// Reads (GET /process/api/*) work without it. Writes (import, retry, cancel) need it in
// `X-Call1-Console-Token`. A new tab that never saw the fragment (or a token rotated elsewhere)
// has no token; the app then offers a manual paste field instead of failing silently.

import { useCallback, useEffect, useState } from 'react';

const STORAGE_KEY = 'call1-process-console-token';

function extractFromHash(): string | null {
  const hash = window.location.hash.replace(/^#/, '');
  if (!hash.includes('console_token=')) return null;
  const params = new URLSearchParams(hash);
  const token = params.get('console_token');
  return token && token.startsWith('c1con_') ? token : null;
}

function readInitialToken(): string | null {
  try {
    const fromHash = extractFromHash();
    if (fromHash) {
      window.sessionStorage.setItem(STORAGE_KEY, fromHash);
      const clean = window.location.pathname + window.location.search;
      window.history.replaceState(null, '', clean);
      return fromHash;
    }
    return window.sessionStorage.getItem(STORAGE_KEY);
  } catch {
    // sessionStorage denied (private mode, blocked site data) — session-only, in memory
    return extractFromHash();
  }
}

export function useConsoleToken() {
  const [token, setTokenState] = useState<string | null>(readInitialToken);

  // A hash-only URL change (opening a freshly-printed console link in a tab that already has the
  // app loaded) does not remount React, so it never reaches the initializer above — only a
  // `hashchange` event does. Without this, pasting a new link into an open tab looks like it did
  // nothing.
  useEffect(() => {
    const onHashChange = () => {
      const fromHash = extractFromHash();
      if (fromHash) {
        try {
          window.sessionStorage.setItem(STORAGE_KEY, fromHash);
        } catch {
          // sessionStorage denied — still update in-memory state below
        }
        const clean = window.location.pathname + window.location.search;
        window.history.replaceState(null, '', clean);
        setTokenState(fromHash);
      }
    };
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  const setToken = useCallback((value: string | null) => {
    setTokenState(value);
    try {
      if (value) window.sessionStorage.setItem(STORAGE_KEY, value);
      else window.sessionStorage.removeItem(STORAGE_KEY);
    } catch {
      // sessionStorage denied — the token still lives in this render's state
    }
  }, []);

  return { token, setToken };
}
