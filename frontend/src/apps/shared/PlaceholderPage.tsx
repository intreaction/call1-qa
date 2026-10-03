import { Sun, Moon, Hammer } from 'lucide-react';
import { useTheme } from '@/hooks/useTheme';

interface PlaceholderPageProps {
  appName: string;
  tagline: string;
  detail: string;
  logo: string; // '/call1-dark-32.png' / '/call1-light-32.png' basename, no leading path
}

/**
 * The "not built yet" page for an app whose real build hasn't landed. Used by the Evaluate and
 * Process placeholder entries so `vite build` succeeds for all three apps before their tracks
 * land (docs/SplitBuild.md: "Create placeholder evaluate.html/process.html entries with a
 * minimal 'being built' page in the design system"). Honest empty state, no fake data, same
 * tokens and IBM Plex Sans as every other Call1 UI.
 */
export default function PlaceholderPage({ appName, tagline, detail, logo }: PlaceholderPageProps) {
  const { theme, toggleTheme } = useTheme();

  return (
    <div className="flex-1 flex flex-col min-h-0 bg-canvas">
      <header className="border-b border-border bg-canvas-subtle px-4 py-2.5 shrink-0 flex items-center justify-between h-12">
        <div className="flex items-center gap-2 min-w-0">
          <img
            src={theme === 'light' ? `/call1-light-${logo}` : `/call1-dark-${logo}`}
            className="w-4 h-4 rounded-sm shrink-0 shadow-sm"
            alt=""
          />
          <span className="font-semibold text-sm text-fg">{appName}</span>
        </div>
        <button
          type="button"
          onClick={toggleTheme}
          className="w-7 h-7 rounded-md hover:bg-canvas border border-transparent hover:border-border text-fg-muted hover:text-fg transition-colors flex items-center justify-center"
          title={theme === 'light' ? 'Switch to dark theme' : 'Switch to light theme'}
          aria-label={theme === 'light' ? 'Switch to dark theme' : 'Switch to light theme'}
        >
          {theme === 'light' ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
        </button>
      </header>
      <main className="flex-1 flex items-center justify-center p-6">
        <div className="max-w-md text-center flex flex-col items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-canvas-inset border border-border flex items-center justify-center">
            <Hammer className="w-4.5 h-4.5 text-fg-subtle" aria-hidden="true" />
          </div>
          <h1 className="text-base font-semibold text-fg">{tagline}</h1>
          <p className="text-sm text-fg-muted">{detail}</p>
        </div>
      </main>
    </div>
  );
}
