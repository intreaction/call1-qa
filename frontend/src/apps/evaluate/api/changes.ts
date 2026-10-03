// Change-feed polling (`GET /store/v1/changes`), the v1 update transport. Events carry kinds, IDs,
// versions and statuses, never content: a view re-reads what an event names.
//
// - The first poll with no cursor records `latest_cursor` and delivers nothing (the caller has
//   just taken its own snapshot). Pass `after` (e.g. `CallDetail.change_cursor`) to resume from a
//   snapshot instead.
// - 410 `cursor_expired` / `cursor_unknown` (retention passed, or a restore started a new feed
//   epoch) calls `onResync`: re-snapshot everything, then polling continues from `latest_cursor`.
// - A failed poll reports offline and backs off; the first good poll reports online again.

import type { StoreClient } from './client';
import { StoreError, StoreUnreachableError } from './errors';
import type { ChangeEvent, ChangeKind } from './types';

export interface ChangePollerOptions {
  /** Poll interval while healthy. A client choice, not a contract timing. Default 5 s. */
  intervalMs?: number;
  /** Longest back-off after failures. Default 60 s. */
  maxBackoffMs?: number;
  /** Resume after this cursor; otherwise start at the feed's latest cursor. */
  after?: string;
  /** Only these kinds (Store still advances `next_cursor` past everything else). */
  kinds?: ChangeKind[];
  /** Page size, 1–200. Default 200. */
  limit?: number;
  onEvents: (events: ChangeEvent[]) => void;
  /** The cursor is no longer resumable; re-read every snapshot. */
  onResync?: (reason: 'cursor_expired' | 'cursor_unknown', feedEpoch: string | null) => void;
  /** Connectivity as the poller sees it. */
  onConnectivity?: (online: boolean, error?: unknown) => void;
}

export interface ChangePoller {
  stop(): void;
  /** Poll now (after a write, or when the tab becomes visible). */
  pollNow(): void;
  readonly cursor: string | null;
  readonly feedEpoch: string | null;
}

export function startChangePoller(client: StoreClient, options: ChangePollerOptions): ChangePoller {
  const interval = options.intervalMs ?? 5000;
  const maxBackoff = options.maxBackoffMs ?? 60_000;
  const limit = options.limit ?? 200;
  let cursor: string | null = options.after ?? null;
  let epoch: string | null = null;
  let stopped = false;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let failures = 0;
  let running = false;
  let online: boolean | null = null;

  const setOnline = (value: boolean, error?: unknown) => {
    if (online === value) return;
    online = value;
    options.onConnectivity?.(value, error);
  };

  const schedule = (ms: number) => {
    if (stopped) return;
    if (timer) clearTimeout(timer);
    timer = setTimeout(tick, ms);
  };

  async function tick() {
    if (stopped || running) return;
    running = true;
    try {
      // Drain: keep reading while a full page comes back.
      for (let page = 0; page < 20; page += 1) {
        const feed = await client.get('/store/v1/changes', {
          query: { after: cursor ?? undefined, limit: cursor ? limit : 1, kinds: options.kinds },
        });
        if (epoch !== null && feed.feed_epoch !== epoch) {
          epoch = feed.feed_epoch;
          cursor = feed.latest_cursor;
          options.onResync?.('cursor_unknown', epoch);
          break;
        }
        epoch = feed.feed_epoch;
        if (cursor === null) {
          cursor = feed.latest_cursor;
          break;
        }
        cursor = feed.next_cursor;
        if (feed.events.length) options.onEvents(feed.events);
        if (feed.events.length < limit || feed.next_cursor === feed.latest_cursor) break;
      }
      failures = 0;
      setOnline(true);
      schedule(document.visibilityState === 'hidden' ? interval * 4 : interval);
    } catch (err) {
      if (err instanceof StoreError && (err.code === 'cursor_expired' || err.code === 'cursor_unknown')) {
        const newEpoch = typeof err.details.feed_epoch === 'string' ? err.details.feed_epoch : null;
        cursor = null;
        epoch = null;
        options.onResync?.(err.code, newEpoch);
        schedule(0);
      } else if (err instanceof StoreError && err.status < 500 && err.code !== 'rate_limited') {
        // Signed out or forbidden: the shell handles it; stop quietly.
        stopped = true;
      } else {
        failures += 1;
        if (err instanceof StoreUnreachableError || (err instanceof StoreError && err.status >= 500)) setOnline(false, err);
        const retryAfter = err instanceof StoreError && err.retryAfterSeconds ? err.retryAfterSeconds * 1000 : 0;
        schedule(Math.max(retryAfter, Math.min(maxBackoff, interval * 2 ** failures)));
      }
    } finally {
      running = false;
    }
  }

  schedule(0);

  return {
    stop() {
      stopped = true;
      if (timer) clearTimeout(timer);
    },
    pollNow() {
      schedule(0);
    },
    get cursor() {
      return cursor;
    },
    get feedEpoch() {
      return epoch;
    },
  };
}
