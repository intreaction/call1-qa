// Is Store reachable? Combines the browser's own signal (`navigator.onLine`, online/offline
// events) with what the Store client and the change-feed poller observe: a request that gets no
// HTTP answer marks Store unreachable, and the next answered request marks it reachable again.

import { useSyncExternalStore } from 'react';
import type { StoreClient } from '../api';

export interface Connectivity {
  /** The browser reports a network connection. */
  browserOnline: boolean;
  /** The last Store request (or change-feed poll) got an HTTP answer. */
  storeReachable: boolean;
  /** When Store last became unreachable (ms since epoch), for "since …" copy. */
  unreachableSince: number | null;
}

let state: Connectivity = {
  browserOnline: typeof navigator === 'undefined' ? true : navigator.onLine,
  storeReachable: true,
  unreachableSince: null,
};
const listeners = new Set<() => void>();

function set(patch: Partial<Connectivity>) {
  const next = { ...state, ...patch };
  if (next.browserOnline === state.browserOnline && next.storeReachable === state.storeReachable) return;
  state = next;
  listeners.forEach((l) => l());
}

export function markStoreReachable(reachable: boolean) {
  if (reachable) set({ storeReachable: true, unreachableSince: null });
  else set({ storeReachable: false, unreachableSince: state.unreachableSince ?? Date.now() });
}

let wired = false;
/** Wire the browser events and the client's events once, at start. */
export function wireConnectivity(client: StoreClient) {
  if (wired) return;
  wired = true;
  window.addEventListener('online', () => set({ browserOnline: true }));
  window.addEventListener('offline', () => set({ browserOnline: false }));
  client.subscribe((event) => {
    if (event.type === 'reachable') markStoreReachable(true);
    if (event.type === 'unreachable') markStoreReachable(false);
  });
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useConnectivity(): Connectivity & { offline: boolean } {
  const s = useSyncExternalStore(subscribe, () => state);
  return { ...s, offline: !s.browserOnline || !s.storeReachable };
}
