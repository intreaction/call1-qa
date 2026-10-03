// Semantic call search (⌘K / Ctrl+K, or "Search calls" in the header). A natural-language query is
// sent to `POST /store/v1/search/semantic`; each ranked hit is one transcript turn, shown with the
// call's agent and date, the speaker, the matched text and its timestamp, and links to the
// Workbench at that turn (`#/calls/<id>?turn=<n>`).
//
// Keyboard: the input is a combobox over the result listbox. ArrowDown/ArrowUp move the active hit,
// Home/End jump, Enter opens it, Escape closes the dialog, Tab stays inside it.

import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { useQueries, useQuery } from '@tanstack/react-query';
import { Loader2, Search, X } from 'lucide-react';
import { agentLabel, describeError, queryKeys, type CallDetail } from '../api';
import {
  SEARCH_MAX_CHARS,
  SEARCH_MIN_CHARS,
  formatTimestamp,
  highlightTerms,
  isSearchUnavailable,
  normalizeSearchQuery,
  searchKeys,
  semanticSearch,
  splitHighlights,
  type SemanticSearchHit,
} from '../api/search';
import { useStore } from '../state/app';
import { href, navigate } from '../state/router';
import { formatDateTime } from './ui';

const EXAMPLES = ['Stanley cup in stock', 'customer wants a refund', 'agent asks to verify identity'];
const DEBOUNCE_MS = 250;

function useDebounced<T>(value: T, ms: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setDebounced(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return debounced;
}

const SPEAKER_LABEL: Record<string, string> = { AGENT: 'Agent', CALLER: 'Caller', UNKNOWN: 'Speaker' };

export function SearchDialog({ onClose }: { onClose(): void }) {
  const { client } = useStore();
  const titleId = useId();
  const listId = useId();
  const panel = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const [text, setText] = useState('');
  const [active, setActive] = useState(0);
  const query = normalizeSearchQuery(useDebounced(text, DEBOUNCE_MS));
  const enabled = query.length >= SEARCH_MIN_CHARS;
  const typing = normalizeSearchQuery(text) !== query;

  const search = useQuery({
    queryKey: searchKeys.semantic(query),
    queryFn: ({ signal }) => semanticSearch(client, query, signal),
    enabled,
    staleTime: 30_000,
    retry: (count, err) => !isSearchUnavailable(err) && count < 1,
  });
  const hits: SemanticSearchHit[] = useMemo(() => (enabled ? (search.data?.results ?? []) : []), [enabled, search.data]);

  // The call behind each hit, for its agent and date (one read per distinct call, cached).
  const callIds = useMemo(() => Array.from(new Set(hits.map((h) => h.call_id))), [hits]);
  const calls = useQueries({
    queries: callIds.map((callId) => ({
      queryKey: [...queryKeys.call(callId), 'search-label'] as const,
      queryFn: ({ signal }: { signal: AbortSignal }) => client.get('/store/v1/calls/{call_id}', { path: { call_id: callId }, signal }),
      staleTime: 60_000,
      retry: false,
    })),
  });
  const callById = useMemo(() => {
    const map = new Map<string, CallDetail>();
    calls.forEach((q, i) => q.data && map.set(callIds[i], q.data));
    return map;
  }, [calls, callIds]);

  const terms = useMemo(() => highlightTerms(query), [query]);

  useEffect(() => setActive(0), [query]);
  useEffect(() => {
    listRef.current?.querySelector<HTMLElement>(`[data-index="${active}"]`)?.scrollIntoView({ block: 'nearest' });
  }, [active]);

  // Focus the input on open; return focus to the opener on close.
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    input.current?.focus();
    return () => previous?.focus?.();
  }, []);

  const open = (hit: SemanticSearchHit) => {
    onClose();
    navigate({ name: 'workbench', callId: hit.call_id, turn: hit.turn_id });
  };

  const onInputKey = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (!hits.length) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActive((i) => (i + 1) % hits.length);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActive((i) => (i - 1 + hits.length) % hits.length);
    } else if (e.key === 'Home' && e.ctrlKey) {
      e.preventDefault();
      setActive(0);
    } else if (e.key === 'End' && e.ctrlKey) {
      e.preventDefault();
      setActive(hits.length - 1);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      const hit = hits[active];
      if (hit) open(hit);
    }
  };

  const onPanelKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      e.stopPropagation();
      onClose();
      return;
    }
    if (e.key !== 'Tab' || !panel.current) return;
    const focusable = Array.from(panel.current.querySelectorAll<HTMLElement>('input, button, a[href]')).filter(
      (el) => !el.hasAttribute('disabled') && el.tabIndex >= 0,
    );
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  };

  const activeId = hits.length ? `${listId}-hit-${active}` : undefined;
  const loading = enabled && (search.isFetching || typing) && !search.isError;
  const reembed = search.data?.calls_needing_reembedding ?? 0;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/40 px-4 pt-[10vh] pb-4 overflow-y-auto"
      onMouseDown={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        ref={panel}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onKeyDown={onPanelKey}
        data-testid="search-dialog"
        className="w-full max-w-2xl min-w-0 rounded-lg border border-border bg-canvas shadow-xl flex flex-col max-h-[80vh]"
      >
        <h2 id={titleId} className="sr-only">
          Search calls
        </h2>
        <div className="flex items-center gap-2 px-3 border-b border-border-muted">
          {loading ? (
            <Loader2 className="w-4 h-4 text-fg-muted animate-spin shrink-0" aria-hidden="true" />
          ) : (
            <Search className="w-4 h-4 text-fg-muted shrink-0" aria-hidden="true" />
          )}
          <input
            ref={input}
            type="search"
            role="combobox"
            aria-label="Search call transcripts"
            aria-expanded={hits.length > 0}
            aria-controls={listId}
            aria-activedescendant={activeId}
            aria-autocomplete="list"
            autoComplete="off"
            spellCheck={false}
            maxLength={SEARCH_MAX_CHARS}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={onInputKey}
            placeholder="Search what was said, e.g. “Stanley cup in stock”"
            className="flex-1 min-w-0 h-12 bg-transparent text-sm text-fg placeholder:text-fg-subtle focus:outline-none [&::-webkit-search-cancel-button]:hidden"
          />
          <button
            type="button"
            onClick={onClose}
            aria-label="Close search"
            className="w-7 h-7 rounded hover:bg-canvas-inset text-fg-muted hover:text-fg flex items-center justify-center shrink-0 focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
          >
            <X className="w-3.5 h-3.5" aria-hidden="true" />
          </button>
        </div>

        <div className="overflow-y-auto min-h-0" aria-live="polite" aria-busy={loading}>
          {!enabled && (
            <div className="px-4 py-5 text-sm text-fg-muted space-y-3">
              <p>Describe a moment in plain words. Search ranks transcript lines by meaning, not exact wording.</p>
              <div className="flex flex-wrap gap-2">
                {EXAMPLES.map((example) => (
                  <button
                    key={example}
                    type="button"
                    onClick={() => {
                      setText(example);
                      input.current?.focus();
                    }}
                    className="px-2.5 h-7 rounded-full border border-border bg-canvas-subtle text-xs text-fg hover:bg-canvas-inset focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>
          )}

          {enabled && search.isError && (
            <div className="px-4 py-5 text-sm" role="alert">
              {isSearchUnavailable(search.error) ? (
                <p className="text-fg">
                  Search is not available on this Store: its local search model is not installed or did not load.{' '}
                  <span className="text-fg-muted">An admin can install it; calls stay readable meanwhile.</span>
                </p>
              ) : (
                <p className="text-primer-redFg">{describeError(search.error)}</p>
              )}
            </div>
          )}

          {enabled && !search.isError && search.data && !typing && hits.length === 0 && !search.isFetching && (
            <p className="px-4 py-5 text-sm text-fg-muted" role="status">
              No matching moments for “{query}”. Try describing it differently.
            </p>
          )}

          {enabled && !search.data && loading && (
            <p className="px-4 py-5 text-sm text-fg-muted" role="status">
              Searching transcripts…
            </p>
          )}

          <div
            ref={listRef}
            id={listId}
            role="listbox"
            aria-label="Matching moments"
            className={hits.length ? 'py-1' : 'hidden'}
            data-testid="search-results"
          >
            {hits.map((hit, index) => {
              const detail = callById.get(hit.call_id);
              const agent = detail ? agentLabel(detail.call.agent_id, detail.call.agent_display_name, detail.call.agent_extension) : null;
              const selected = index === active;
              return (
                <a
                  key={`${hit.call_id}:${hit.turn_id}`}
                  id={`${listId}-hit-${index}`}
                  role="option"
                  aria-selected={selected}
                  data-index={index}
                  tabIndex={-1}
                  href={href({ name: 'workbench', callId: hit.call_id, turn: hit.turn_id })}
                  onClick={(e) => {
                    if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
                    e.preventDefault();
                    open(hit);
                  }}
                  onMouseMove={() => !selected && setActive(index)}
                  className={`block px-4 py-2.5 border-l-2 focus:outline-none ${
                    selected ? 'bg-canvas-inset border-primer-blue' : 'border-transparent hover:bg-canvas-subtle'
                  }`}
                >
                  <div className="flex items-center gap-2 min-w-0 text-xs">
                    <span className="font-medium text-fg truncate" data-testid="search-hit-call">
                      {agent ?? `Call ${hit.call_id}`}
                    </span>
                    {detail && <span className="text-fg-subtle whitespace-nowrap hidden sm:inline">{formatDateTime(detail.call.created_at)}</span>}
                    <span className="ml-auto flex items-center gap-2 shrink-0 text-fg-muted tabular-nums">
                      <span>{SPEAKER_LABEL[hit.speaker] ?? hit.speaker}</span>
                      <span aria-label={`at ${formatTimestamp(hit.start_time)}`}>{formatTimestamp(hit.start_time)}</span>
                      <span className="hidden sm:inline" title="Similarity">
                        {Math.round(hit.similarity_score * 100)}%
                      </span>
                    </span>
                  </div>
                  <p className="mt-1 text-sm text-fg-muted line-clamp-2 break-words" data-testid="search-hit-text">
                    {splitHighlights(hit.text, terms).map((part, i) =>
                      part.match ? (
                        <mark key={i} className="bg-primer-yellowSubtle text-fg rounded-sm px-0.5">
                          {part.text}
                        </mark>
                      ) : (
                        <span key={i}>{part.text}</span>
                      ),
                    )}
                  </p>
                </a>
              );
            })}
          </div>

          {enabled && reembed > 0 && (
            <p className="px-4 py-2 text-xs text-fg-subtle border-t border-border-muted">
              {reembed} {reembed === 1 ? 'call was' : 'calls were'} indexed with an older search model and not searched; reanalyze with
              “embeddings” to include {reembed === 1 ? 'it' : 'them'}.
            </p>
          )}
        </div>

        <footer className="hidden sm:flex items-center gap-3 px-4 py-2 border-t border-border-muted text-[11px] text-fg-subtle">
          <span>
            <kbd className="font-sans">↑</kbd> <kbd className="font-sans">↓</kbd> to move
          </span>
          <span>
            <kbd className="font-sans">Enter</kbd> to open at that moment
          </span>
          <span>
            <kbd className="font-sans">Esc</kbd> to close
          </span>
        </footer>
      </div>
    </div>
  );
}
