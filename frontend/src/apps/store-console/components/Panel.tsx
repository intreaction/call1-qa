import type { LucideIcon } from 'lucide-react';
import type { ReactNode } from 'react';

interface PanelProps {
  title: string;
  icon: LucideIcon;
  subtitle?: string;
  right?: ReactNode;
  children: ReactNode;
}

/** The one card shell every console panel uses, so the grid reads as one system. */
export default function Panel({ title, icon: Icon, subtitle, right, children }: PanelProps) {
  return (
    <section className="rounded-lg border border-border bg-canvas-subtle flex flex-col min-w-0">
      <header className="flex items-center justify-between gap-2 px-4 py-3 border-b border-border-muted">
        <div className="flex items-center gap-2 min-w-0">
          <Icon className="w-4 h-4 text-fg-subtle shrink-0" aria-hidden="true" />
          <div className="min-w-0">
            <h2 className="text-sm font-semibold text-fg truncate">{title}</h2>
            {subtitle && <p className="text-xs text-fg-muted truncate">{subtitle}</p>}
          </div>
        </div>
        {right && <div className="shrink-0">{right}</div>}
      </header>
      <div className="p-4 flex-1 min-w-0">{children}</div>
    </section>
  );
}
