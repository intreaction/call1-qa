import { BarChart3, ClipboardList, Inbox, LogOut, Moon, Phone, Radar, Search, Shield, Siren, Sun, UserRound } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { useEffect, useState } from 'react';
import type { Theme } from '@/hooks/useTheme';
import { useSession } from '../state/app';
import { useDemoStatus } from '../state/demo';
import { href, type Route } from '../state/router';
import { SearchDialog } from './SearchDialog';
import { StatusPill } from './ui';

interface NavItem {
  label: string;
  icon: LucideIcon;
  route: Route;
  match: Route['name'][];
}

const NAV: NavItem[] = [
  { label: 'Calls', icon: Phone, route: { name: 'calls' }, match: ['calls', 'workbench'] },
  { label: 'Rubrics', icon: ClipboardList, route: { name: 'rubrics' }, match: ['rubrics'] },
  { label: 'Signals', icon: Radar, route: { name: 'signals', tab: 'taxonomy' }, match: ['signals'] },
  { label: 'Queue', icon: Inbox, route: { name: 'queue' }, match: ['queue'] },
  { label: 'Escalations', icon: Siren, route: { name: 'escalations' }, match: ['escalations'] },
  { label: 'Metrics', icon: BarChart3, route: { name: 'metrics' }, match: ['metrics'] },
];
const ADMIN_NAV: NavItem = { label: 'Admin', icon: Shield, route: { name: 'admin', section: 'accounts' }, match: ['admin'] };

const ROLE_TONE = { reviewer: 'neutral', supervisor: 'blue', admin: 'magenta' } as const;

export function ThemeToggle({ theme, onToggle }: { theme: Theme; onToggle(): void }) {
  const label = theme === 'light' ? 'Switch to dark theme' : 'Switch to light theme';
  return (
    <button
      type="button"
      onClick={onToggle}
      className="w-7 h-7 rounded-md hover:bg-canvas border border-transparent hover:border-border text-fg-muted hover:text-fg transition-colors flex items-center justify-center focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
      title={label}
      aria-label={label}
    >
      {theme === 'light' ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
    </button>
  );
}

export function Brand({ theme }: { theme: Theme }) {
  return (
    <a href={href({ name: 'calls' })} className="flex items-center gap-2 min-w-0 rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue">
      <img src={theme === 'light' ? '/call1-light-32.png' : '/call1-dark-32.png'} className="w-4 h-4 rounded-sm shrink-0 shadow-sm" alt="" />
      <span className="font-semibold text-sm text-fg whitespace-nowrap">Call1 Evaluate</span>
    </a>
  );
}

/** Shown in every header whenever Store answers `GET /demo/status` with `demo: true` (localhost
 * only, off by default — `CALL1_STORE_DEMO=1`): a persistent, clearly-labelled reminder that
 * passkeys are bypassable on this Store, not just a one-time note on the sign-in screen. */
function DemoBadge() {
  const demo = useDemoStatus();
  if (!demo.data?.demo) return null;
  return (
    <span className="hidden sm:inline-flex">
      <StatusPill tone="magenta" title={demo.data.label}>
        Demo mode
      </StatusPill>
    </span>
  );
}

/** The header for signed-out screens: app name and theme toggle only. */
export function PublicHeader({ theme, onToggleTheme }: { theme: Theme; onToggleTheme(): void }) {
  return (
    <header className="border-b border-border bg-canvas-subtle px-4 shrink-0 flex items-center justify-between h-12">
      <div className="flex items-center gap-3 min-w-0">
        <Brand theme={theme} />
        <DemoBadge />
      </div>
      <ThemeToggle theme={theme} onToggle={onToggleTheme} />
    </header>
  );
}

/** True on Apple platforms, where the search shortcut is ⌘K rather than Ctrl+K. */
function isApple(): boolean {
  return typeof navigator !== 'undefined' && /Mac|iPhone|iPad|iPod/.test(navigator.platform || navigator.userAgent);
}

/**
 * "Search calls" plus the global ⌘K / Ctrl+K shortcut, which opens semantic call search from any
 * signed-in screen. Hidden without the `search` permission.
 */
function SearchButton() {
  const { can } = useSession();
  const [open, setOpen] = useState(false);
  const allowed = can('search');
  useEffect(() => {
    if (!allowed) return;
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && !e.altKey && !e.shiftKey && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setOpen((v) => !v);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [allowed]);
  if (!allowed) return null;
  const shortcut = isApple() ? '⌘K' : 'Ctrl+K';
  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        aria-haspopup="dialog"
        aria-label="Search calls"
        aria-keyshortcuts="Meta+K Control+K"
        title={`Search calls (${shortcut})`}
        className="h-7 w-7 md:w-auto md:px-2 rounded-md border border-border bg-canvas hover:bg-canvas-inset text-fg-muted hover:text-fg text-xs flex items-center justify-center gap-1.5 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
      >
        <Search className="w-3.5 h-3.5" aria-hidden="true" />
        <span className="hidden md:inline" aria-hidden="true">
          Search calls
        </span>
        <kbd className="hidden lg:inline font-sans text-[10px] text-fg-subtle border border-border-muted rounded px-1" aria-hidden="true">
          {shortcut}
        </kbd>
      </button>
      {open && <SearchDialog onClose={() => setOpen(false)} />}
    </>
  );
}

export function Header({ route, theme, onToggleTheme }: { route: Route; theme: Theme; onToggleTheme(): void }) {
  const { session, role, can, signOut } = useSession();
  const [signingOut, setSigningOut] = useState(false);
  const items = can('manage_accounts') ? [...NAV, ADMIN_NAV] : NAV;

  // Below `sm` the header wraps: brand and the account controls share the first row, and the nav
  // takes a full-width, horizontally scrollable second row (BUG D4: at 390px the nav collapsed to
  // zero width and pushed Sign out and the theme toggle off-screen). From `sm` up it is one row.
  return (
    <header className="border-b border-border bg-canvas-subtle shrink-0 min-w-0">
      <div className="px-4 flex flex-wrap sm:flex-nowrap items-center gap-x-4 min-h-12 sm:h-12">
        <div className="flex items-center gap-3 shrink-0 min-w-0 h-12 order-1">
          <Brand theme={theme} />
          <DemoBadge />
        </div>
        <nav
          aria-label="Evaluate"
          className="order-3 sm:order-2 basis-[calc(100%+2rem)] sm:basis-auto sm:flex-1 min-w-0 overflow-x-auto border-t border-border-muted sm:border-t-0 -mx-4 px-4 sm:mx-0 sm:px-0 py-1 sm:py-0"
        >
          <ul className="flex items-center gap-1 w-max sm:w-auto">
            {items.map((item) => {
              const active = item.match.includes(route.name);
              const Icon = item.icon;
              return (
                <li key={item.label} className="shrink-0">
                  <a
                    href={href(item.route)}
                    aria-current={active ? 'page' : undefined}
                    title={item.label}
                    className={`flex items-center gap-1.5 h-8 px-2.5 rounded-md text-sm whitespace-nowrap transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue ${
                      active ? 'bg-canvas text-fg border border-border' : 'text-fg-muted hover:text-fg hover:bg-canvas border border-transparent'
                    }`}
                  >
                    <Icon className="w-3.5 h-3.5" aria-hidden="true" />
                    <span className="sm:hidden lg:inline">{item.label}</span>
                    <span className="sr-only hidden sm:inline lg:hidden">{item.label}</span>
                  </a>
                </li>
              );
            })}
          </ul>
        </nav>
        <div className="order-2 sm:order-3 ml-auto flex items-center gap-2 shrink-0 h-12">
          <SearchButton />
          <a
            href={href({ name: 'account' })}
            aria-current={route.name === 'account' ? 'page' : undefined}
            className="flex items-center gap-2 h-8 px-2 rounded-md hover:bg-canvas text-sm text-fg focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
            title={`${session.email} — your authenticators and sessions`}
          >
            <UserRound className="w-3.5 h-3.5 text-fg-muted" aria-hidden="true" />
            <span className="hidden md:inline max-w-[16ch] truncate">{session.display_name}</span>
            <span className="sr-only md:hidden">Account</span>
            <span className="hidden sm:inline-flex">
              <StatusPill tone={ROLE_TONE[role] ?? 'neutral'}>{role}</StatusPill>
            </span>
          </a>
          <button
            type="button"
            onClick={async () => {
              setSigningOut(true);
              try {
                await signOut();
              } finally {
                setSigningOut(false);
              }
            }}
            disabled={signingOut}
            className="w-7 h-7 rounded-md hover:bg-canvas border border-transparent hover:border-border text-fg-muted hover:text-fg transition-colors flex items-center justify-center focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
            title="Sign out"
            aria-label="Sign out"
          >
            <LogOut className="w-3.5 h-3.5" />
          </button>
          <ThemeToggle theme={theme} onToggle={onToggleTheme} />
        </div>
      </div>
    </header>
  );
}
