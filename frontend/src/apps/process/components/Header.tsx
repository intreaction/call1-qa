import { Boxes, Gauge, Moon, SlidersHorizontal, Sun, UploadCloud, Workflow } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import type { Theme } from '@/hooks/useTheme';
import { StatusPill } from './ui';

export type Tab = 'overview' | 'import' | 'pipeline' | 'models' | 'settings';

const NAV: Array<{ tab: Tab; label: string; icon: LucideIcon }> = [
  { tab: 'overview', label: 'Overview', icon: Gauge },
  { tab: 'import', label: 'Import', icon: UploadCloud },
  { tab: 'pipeline', label: 'Pipeline', icon: Workflow },
  { tab: 'models', label: 'Models', icon: Boxes },
  { tab: 'settings', label: 'Settings', icon: SlidersHorizontal },
];

export function Header({
  active,
  onNavigate,
  theme,
  onToggleTheme,
  connection,
}: {
  active: Tab;
  onNavigate(tab: Tab): void;
  theme: Theme;
  onToggleTheme(): void;
  /** A one-glance read of whether this console can reach Process and write to it. */
  connection: { tone: 'green' | 'yellow' | 'red' | 'neutral'; label: string } | null;
}) {
  return (
    <header className="border-b border-border bg-canvas-subtle shrink-0">
      <div className="px-4 flex items-center gap-4 h-12">
        <div className="flex items-center gap-2 min-w-0">
          <img src={theme === 'light' ? '/call1-light-32.png' : '/call1-dark-32.png'} className="w-4 h-4 rounded-sm shrink-0 shadow-sm" alt="" />
          <span className="font-semibold text-sm text-fg whitespace-nowrap">Call1 Process</span>
        </div>
        <nav aria-label="Process" className="flex-1 min-w-0 overflow-x-auto">
          <ul className="flex items-center gap-1">
            {NAV.map((item) => {
              const active_ = item.tab === active;
              const Icon = item.icon;
              return (
                <li key={item.tab}>
                  <button
                    type="button"
                    onClick={() => onNavigate(item.tab)}
                    aria-current={active_ ? 'page' : undefined}
                    title={item.label}
                    className={`flex items-center gap-1.5 h-8 px-2.5 rounded-md text-sm whitespace-nowrap transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue ${
                      active_ ? 'bg-canvas text-fg border border-border' : 'text-fg-muted hover:text-fg hover:bg-canvas border border-transparent'
                    }`}
                  >
                    <Icon className="w-3.5 h-3.5" aria-hidden="true" />
                    <span className="hidden sm:inline">{item.label}</span>
                    <span className="sr-only sm:hidden">{item.label}</span>
                  </button>
                </li>
              );
            })}
          </ul>
        </nav>
        <div className="flex items-center gap-2 shrink-0">
          {connection && <StatusPill tone={connection.tone}>{connection.label}</StatusPill>}
          <button
            type="button"
            onClick={onToggleTheme}
            className="w-7 h-7 rounded-md hover:bg-canvas border border-transparent hover:border-border text-fg-muted hover:text-fg transition-colors flex items-center justify-center focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
            title={theme === 'light' ? 'Switch to dark theme' : 'Switch to light theme'}
            aria-label={theme === 'light' ? 'Switch to dark theme' : 'Switch to light theme'}
          >
            {theme === 'light' ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>
    </header>
  );
}
