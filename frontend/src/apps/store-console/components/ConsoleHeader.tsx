import { Sun, Moon } from 'lucide-react';
import type { Theme } from '@/hooks/useTheme';
import { useDemoStatus } from '../demo';
import StatusPill from './StatusPill';

interface ConsoleHeaderProps {
  theme: Theme;
  onToggleTheme: () => void;
}

/** Shown whenever Store answers `GET /demo/status` with `demo: true` (badge only — persona
 * sign-in and switch are Evaluate's; the console's own admin sign-in stays passkey-only). */
function DemoBadge() {
  const demo = useDemoStatus();
  if (!demo.data?.demo) return null;
  return (
    <StatusPill tone="magenta" title={demo.data.label}>
      Demo mode
    </StatusPill>
  );
}

export default function ConsoleHeader({ theme, onToggleTheme }: ConsoleHeaderProps) {
  return (
    <header className="border-b border-border bg-canvas-subtle px-4 py-2.5 shrink-0 flex items-center justify-between h-12">
      <div className="flex items-center gap-2 min-w-0">
        <img
          src={theme === 'light' ? '/call1-light-32.png' : '/call1-dark-32.png'}
          className="w-4 h-4 rounded-sm shrink-0 shadow-sm"
          alt=""
        />
        <span className="font-semibold text-sm text-fg">Call1 Store</span>
        <span className="text-xs text-fg-muted hidden sm:inline">operations console</span>
        <DemoBadge />
      </div>
      <button
        type="button"
        onClick={onToggleTheme}
        className="w-7 h-7 rounded-md hover:bg-canvas border border-transparent hover:border-border text-fg-muted hover:text-fg transition-colors flex items-center justify-center"
        title={theme === 'light' ? 'Switch to dark theme' : 'Switch to light theme'}
        aria-label={theme === 'light' ? 'Switch to dark theme' : 'Switch to light theme'}
      >
        {theme === 'light' ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
      </button>
    </header>
  );
}
