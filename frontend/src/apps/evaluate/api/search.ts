// Semantic search over published transcripts (`POST /store/v1/search/semantic`, contract 1.2.0,
// decision 18). Store embeds the query with the same local model Process embeds turns with and
// ranks turn vectors; 503 `search_unavailable` when the embedder is not installed or failed to load.

import type { Input, Output } from '@/contracts';
import type { StoreClient } from './client';
import { isStoreError } from './errors';

export type SemanticSearchQuery = Input<'SemanticSearchQuery'>;
export type SemanticSearchHit = Output<'SemanticSearchHit'>;
export type SemanticSearchResponse = Output<'SemanticSearchResponse'>;

/** Queries shorter than this are not sent (a single letter ranks noise). */
export const SEARCH_MIN_CHARS = 2;
/** Contract maximum for `SemanticSearchQuery.query`. */
export const SEARCH_MAX_CHARS = 500;
export const SEARCH_TOP_K = 12;

export const searchKeys = {
  semantic: (query: string) => ['search', 'semantic', query] as const,
};

export function normalizeSearchQuery(raw: string): string {
  return raw.replace(/\s+/g, ' ').trim().slice(0, SEARCH_MAX_CHARS);
}

export function semanticSearch(client: StoreClient, query: string, signal?: AbortSignal): Promise<SemanticSearchResponse> {
  return client.post('/store/v1/search/semantic', {
    body: { query, top_k: SEARCH_TOP_K, min_score: 0 },
    signal,
  });
}

/** 503 `search_unavailable`: the local embedder is missing or failed to load (not retryable). */
export function isSearchUnavailable(err: unknown): boolean {
  return isStoreError(err, 'search_unavailable');
}

/** `m:ss` (or `h:mm:ss`) for a turn's start time in seconds. */
export function formatTimestamp(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = String(total % 60).padStart(2, '0');
  return h ? `${h}:${String(m).padStart(2, '0')}:${s}` : `${m}:${s}`;
}

/** The query's words (3+ letters, deduplicated, longest first) for highlighting in a snippet. */
export function highlightTerms(query: string): string[] {
  const words = query.toLowerCase().match(/[\p{L}\p{N}']{3,}/gu) ?? [];
  return Array.from(new Set(words)).sort((a, b) => b.length - a.length);
}

/** Split `text` into plain and matched parts on the terms (case-insensitive, whole-word starts). */
export function splitHighlights(text: string, terms: string[]): { text: string; match: boolean }[] {
  if (!terms.length) return [{ text, match: false }];
  const escaped = terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  const re = new RegExp(`\\b(${escaped.join('|')})`, 'giu');
  const parts: { text: string; match: boolean }[] = [];
  let last = 0;
  for (const m of text.matchAll(re)) {
    const start = m.index ?? 0;
    if (start > last) parts.push({ text: text.slice(last, start), match: false });
    parts.push({ text: m[0], match: true });
    last = start + m[0].length;
  }
  if (last < text.length) parts.push({ text: text.slice(last), match: false });
  return parts;
}
