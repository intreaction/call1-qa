import type { ReactNode } from 'react';

type Tone = 'green' | 'yellow' | 'red' | 'blue' | 'magenta' | 'neutral';

const TONE_CLASSES: Record<Tone, string> = {
  green: 'bg-primer-greenSubtle text-primer-greenFg border-primer-greenBorder',
  yellow: 'bg-primer-yellowSubtle text-primer-yellowFg border-primer-yellowBorder',
  red: 'bg-primer-redSubtle text-primer-redFg border-primer-redBorder',
  blue: 'bg-primer-blueSubtle text-primer-blueFg border-primer-blueBorder',
  magenta: 'bg-primer-magentaSubtle text-primer-magentaFg border-primer-magentaBorder',
  neutral: 'bg-canvas-inset text-fg-muted border-border',
};

/** Color AND a text label, never color alone (docs/SplitBuild.md "Design principles"). */
export default function StatusPill({ tone, title, children }: { tone: Tone; title?: string; children: ReactNode }) {
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium border ${TONE_CLASSES[tone]}`}
    >
      <span className="w-1.5 h-1.5 rounded-full bg-current shrink-0" aria-hidden="true" />
      {children}
    </span>
  );
}
