// Hash routing. Store serves Evaluate at `/` and falls back to index.html for unknown paths, so the
// app keeps its own routes in the hash (`/#/calls/abc`), and deep links survive reloads without any
// Store change.
//
// Invitation links: Store issues `<store>/enroll#<token>` (call1/store/auth/routes_admin.py). The
// app also accepts `/#/enroll?token=<token>`. Both normalize to the latter on load, and the token
// never leaves the browser (a hash is not sent to servers).

import { useEffect, useState } from 'react';

/**
 * Admin screens that are deferred in Evaluate. They are absent from the admin nav, and a deep link
 * renders a "not built yet" state rather than another screen or fake data.
 */
export type DeferredAdminSection = 'state' | 'audit' | 'release-trust' | 'pro1' | 'usage' | 'price-table';
export type AdminSection = 'accounts' | 'invitations' | 'installations' | 'vocabulary' | DeferredAdminSection;

export const DEFERRED_ADMIN_SECTIONS: Record<DeferredAdminSection, string> = {
  state: 'Admin state',
  audit: 'The audit log',
  'release-trust': 'Release trust',
  pro1: 'Pro1 key release',
  usage: 'Usage reporting',
  'price-table': 'The usage price table',
};

export function isDeferredAdminSection(section: AdminSection): section is DeferredAdminSection {
  return Object.prototype.hasOwnProperty.call(DEFERRED_ADMIN_SECTIONS, section);
}

/** Signal filters on `#/calls`, so a metrics row can link to its filtered call list (§10.3). */
export interface CallsFilters {
  signal_category?: string;
  signal_subcategory?: string;
  signal_alert?: string;
}

/** `#/signals` and `#/signals/:categoryId` (taxonomy), `#/signals/alerts`, `#/signals/versions`. */
export type SignalsTab = 'taxonomy' | 'alerts' | 'versions';

export type Route =
  | { name: 'calls'; filters?: CallsFilters }
  | { name: 'workbench'; callId: string; turn?: number }
  | { name: 'rubrics'; rubricId?: string }
  | { name: 'signals'; tab: SignalsTab; categoryId?: string }
  | { name: 'queue' }
  | { name: 'escalations' }
  | { name: 'metrics' }
  | { name: 'admin'; section: AdminSection }
  | { name: 'account' }
  | { name: 'enroll'; token?: string }
  | { name: 'sign-in' }
  | { name: 'not-found'; path: string };

const ADMIN_SECTIONS: AdminSection[] = ['accounts', 'invitations', 'installations', 'vocabulary', ...(Object.keys(DEFERRED_ADMIN_SECTIONS) as DeferredAdminSection[])];

export function parseHash(hash: string): Route {
  const raw = hash.replace(/^#/, '');
  const [pathPart, queryPart = ''] = raw.split('?');
  const segments = (pathPart ?? '').split('/').filter(Boolean).map(decodeURIComponent);
  const query = new URLSearchParams(queryPart);
  const [head, second] = segments;
  switch (head) {
    case undefined:
    case 'calls': {
      if (second) return workbenchRoute(second, query);
      const filters: CallsFilters = {};
      for (const key of ['signal_category', 'signal_subcategory', 'signal_alert'] as const) {
        const value = query.get(key);
        if (value) filters[key] = value;
      }
      return Object.keys(filters).length ? { name: 'calls', filters } : { name: 'calls' };
    }
    case 'workbench':
      return second ? workbenchRoute(second, query) : { name: 'calls' };
    case 'rubrics':
      return second ? { name: 'rubrics', rubricId: second } : { name: 'rubrics' };
    case 'signals':
      if (second === 'alerts' || second === 'versions') return { name: 'signals', tab: second };
      return second ? { name: 'signals', tab: 'taxonomy', categoryId: second } : { name: 'signals', tab: 'taxonomy' };
    case 'queue':
      return { name: 'queue' };
    case 'escalations':
      return { name: 'escalations' };
    case 'metrics':
      return { name: 'metrics' };
    case 'admin':
      return { name: 'admin', section: ADMIN_SECTIONS.includes(second as AdminSection) ? (second as AdminSection) : 'accounts' };
    case 'account':
      return { name: 'account' };
    case 'enroll':
      return { name: 'enroll', token: query.get('token') ?? undefined };
    case 'sign-in':
      return { name: 'sign-in' };
    default:
      return { name: 'not-found', path: `/${segments.join('/')}` };
  }
}

function workbenchRoute(callId: string, query: URLSearchParams): Route {
  const turn = Number.parseInt(query.get('turn') ?? '', 10);
  return Number.isInteger(turn) && turn >= 0 ? { name: 'workbench', callId, turn } : { name: 'workbench', callId };
}

export function href(route: Route): string {
  switch (route.name) {
    case 'calls': {
      const qs = new URLSearchParams();
      for (const [key, value] of Object.entries(route.filters ?? {})) if (value) qs.set(key, value);
      const q = qs.toString();
      return q ? `#/calls?${q}` : '#/calls';
    }
    case 'workbench':
      return `#/calls/${encodeURIComponent(route.callId)}${route.turn !== undefined ? `?turn=${route.turn}` : ''}`;
    case 'rubrics':
      return route.rubricId ? `#/rubrics/${encodeURIComponent(route.rubricId)}` : '#/rubrics';
    case 'signals':
      if (route.tab !== 'taxonomy') return `#/signals/${route.tab}`;
      return route.categoryId ? `#/signals/${encodeURIComponent(route.categoryId)}` : '#/signals';
    case 'queue':
      return '#/queue';
    case 'escalations':
      return '#/escalations';
    case 'metrics':
      return '#/metrics';
    case 'admin':
      return `#/admin/${route.section}`;
    case 'account':
      return '#/account';
    case 'enroll':
      return route.token ? `#/enroll?token=${encodeURIComponent(route.token)}` : '#/enroll';
    case 'sign-in':
      return '#/sign-in';
    case 'not-found':
      return `#${route.path}`;
  }
}

export function navigate(route: Route, options: { replace?: boolean } = {}) {
  const target = href(route);
  if (options.replace) {
    window.history.replaceState(null, '', `${window.location.pathname}${target}`);
    window.dispatchEvent(new HashChangeEvent('hashchange'));
  } else if (window.location.hash !== target) {
    window.location.hash = target;
  }
}

/**
 * Rewrite Store's invitation link form (`/enroll#<token>`) to `/#/enroll?token=<token>` before the
 * first render, so every other path goes through `parseHash`.
 */
export function normalizeInitialLocation() {
  const { pathname, hash } = window.location;
  const onEnrollPath = pathname.replace(/\/+$/, '').endsWith('/enroll');
  if (onEnrollPath) {
    const token = hash && !hash.startsWith('#/') ? decodeURIComponent(hash.slice(1)) : '';
    const target = token ? href({ name: 'enroll', token }) : hash.startsWith('#/') ? hash : '#/enroll';
    window.history.replaceState(null, '', `/${target}`);
  }
}

/** Remove a one-time secret from the address bar once it has been used. */
export function scrubEnrollmentToken() {
  if (window.location.hash.startsWith('#/enroll')) {
    window.history.replaceState(null, '', `${window.location.pathname}#/enroll`);
  }
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parseHash(window.location.hash));
  useEffect(() => {
    const onChange = () => setRoute(parseHash(window.location.hash));
    window.addEventListener('hashchange', onChange);
    return () => window.removeEventListener('hashchange', onChange);
  }, []);
  return route;
}
