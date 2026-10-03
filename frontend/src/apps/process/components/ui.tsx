// Small shared building blocks for the Process console, on the Ink & Signal / Frost & Ink tokens
// in src/index.css (same system as Evaluate and the Store console — docs/SplitBuild.md "Design
// principles"). Quiet by design: a few button styles, pills with text labels, cards, notices.

import { forwardRef, useId, type ButtonHTMLAttributes, type InputHTMLAttributes, type ReactNode, type SelectHTMLAttributes } from 'react';
import type { LucideIcon } from 'lucide-react';
import { AlertTriangle, Info, Loader2 } from 'lucide-react';
import { describeError } from '../api';

// --- buttons ------------------------------------------------------------------------------------

type ButtonVariant = 'primary' | 'secondary' | 'danger' | 'ghost';

const BUTTON: Record<ButtonVariant, string> = {
  primary: 'bg-primer-blue text-white border-transparent hover:bg-primer-blueHover',
  secondary: 'bg-canvas text-fg border-border hover:bg-canvas-inset',
  danger: 'bg-canvas text-primer-redFg border-primer-redBorder hover:bg-primer-redSubtle',
  ghost: 'bg-transparent text-fg-muted border-transparent hover:text-fg hover:bg-canvas-inset',
};

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
  size?: 'sm' | 'md';
  icon?: LucideIcon;
  busy?: boolean;
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = 'secondary', size = 'md', icon: Icon, busy, children, className = '', disabled, ...rest },
  ref,
) {
  const sizing = size === 'sm' ? 'h-7 px-2.5 text-xs gap-1.5' : 'h-8 px-3 text-sm gap-2';
  return (
    <button
      type="button"
      ref={ref}
      {...rest}
      disabled={disabled || busy}
      aria-busy={busy || undefined}
      className={`inline-flex items-center justify-center rounded-md border font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue disabled:opacity-50 disabled:cursor-not-allowed ${sizing} ${BUTTON[variant]} ${className}`}
    >
      {busy ? <Loader2 className="w-3.5 h-3.5 animate-spin" aria-hidden="true" /> : Icon ? <Icon className="w-3.5 h-3.5" aria-hidden="true" /> : null}
      {children}
    </button>
  );
});

/** A button-styled `<label>` for a file input. Put the input right before it with
 * `className="sr-only peer"` (visually hidden but still in the Tab order, so Space or Enter
 * opens the file picker); the label shows the input's keyboard focus ring. */
export function LabelButton({
  htmlFor,
  variant = 'secondary',
  size = 'md',
  icon: Icon,
  children,
}: {
  htmlFor: string;
  variant?: ButtonVariant;
  size?: 'sm' | 'md';
  icon?: LucideIcon;
  children: ReactNode;
}) {
  const sizing = size === 'sm' ? 'h-7 px-2.5 text-xs gap-1.5' : 'h-8 px-3 text-sm gap-2';
  return (
    <label
      htmlFor={htmlFor}
      className={`inline-flex items-center justify-center rounded-md border font-medium transition-colors cursor-pointer peer-focus-visible:ring-2 peer-focus-visible:ring-primer-blue ${sizing} ${BUTTON[variant]}`}
    >
      {Icon && <Icon className="w-3.5 h-3.5" aria-hidden="true" />}
      {children}
    </label>
  );
}

/** A native `role="switch"` toggle: Space and Enter both flip it (a plain `<button>` gives both for
 * free), `aria-checked` carries the state, and the track's color is never the only signal — the
 * knob's position moves too. */
export function Switch({
  checked,
  onChange,
  disabled,
  label,
}: {
  checked: boolean;
  onChange(checked: boolean): void;
  disabled?: boolean;
  label: string;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full border transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue disabled:opacity-50 disabled:cursor-not-allowed ${
        checked ? 'bg-primer-blue border-primer-blue' : 'bg-canvas-inset border-border-control'
      }`}
    >
      <span
        aria-hidden="true"
        className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow transition-transform ${checked ? 'translate-x-[18px]' : 'translate-x-0.5'}`}
      />
    </button>
  );
}

// --- status pills: color AND a text label, never color alone ------------------------------------

export type Tone = 'green' | 'yellow' | 'red' | 'blue' | 'magenta' | 'neutral';

const PILL: Record<Tone, string> = {
  green: 'bg-primer-greenSubtle text-primer-greenFg border-primer-greenBorder',
  yellow: 'bg-primer-yellowSubtle text-primer-yellowFg border-primer-yellowBorder',
  red: 'bg-primer-redSubtle text-primer-redFg border-primer-redBorder',
  blue: 'bg-primer-blueSubtle text-primer-blueFg border-primer-blueBorder',
  magenta: 'bg-primer-magentaSubtle text-primer-magentaFg border-primer-magentaBorder',
  neutral: 'bg-canvas-inset text-fg-muted border-border',
};

export function StatusPill({ tone, children, title }: { tone: Tone; children: ReactNode; title?: string }) {
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium border whitespace-nowrap ${PILL[tone]}`}
    >
      <span className="w-1.5 h-1.5 rounded-full bg-current shrink-0" aria-hidden="true" />
      {children}
    </span>
  );
}

// --- cards and page chrome ----------------------------------------------------------------------

export function Card({
  title,
  icon: Icon,
  subtitle,
  right,
  children,
  className = '',
}: {
  title?: string;
  icon?: LucideIcon;
  subtitle?: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`rounded-lg border border-border bg-canvas-subtle min-w-0 ${className}`}>
      {title && (
        <header className="flex items-center justify-between gap-2 px-4 py-3 border-b border-border-muted">
          <div className="flex items-center gap-2 min-w-0">
            {Icon && <Icon className="w-4 h-4 text-fg-subtle shrink-0" aria-hidden="true" />}
            <div className="min-w-0">
              <h2 className="text-sm font-semibold text-fg truncate">{title}</h2>
              {subtitle && <p className="text-xs text-fg-muted truncate">{subtitle}</p>}
            </div>
          </div>
          {right && <div className="shrink-0 flex items-center gap-2">{right}</div>}
        </header>
      )}
      <div className="p-4 min-w-0">{children}</div>
    </section>
  );
}

export function PageHeader({ title, description, right }: { title: string; description?: ReactNode; right?: ReactNode }) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-3 mb-4">
      <div className="min-w-0">
        <h1 className="text-base font-semibold text-fg">{title}</h1>
        {description && <p className="text-sm text-fg-muted mt-0.5">{description}</p>}
      </div>
      {right && <div className="flex items-center gap-2 shrink-0">{right}</div>}
    </div>
  );
}

export function EmptyState({ icon: Icon = Info, title, children }: { icon?: LucideIcon; title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center text-center gap-2 py-10 px-4">
      <div className="w-9 h-9 rounded-full bg-canvas-inset border border-border flex items-center justify-center">
        <Icon className="w-4 h-4 text-fg-subtle" aria-hidden="true" />
      </div>
      <p className="text-sm font-medium text-fg">{title}</p>
      {children && <div className="text-sm text-fg-muted max-w-md">{children}</div>}
    </div>
  );
}

export function Loading({ label = 'Loading…' }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-fg-muted py-6 justify-center" role="status">
      <Loader2 className="w-4 h-4 animate-spin" aria-hidden="true" />
      {label}
    </div>
  );
}

export function Notice({ tone = 'neutral', children, icon }: { tone?: Tone; children: ReactNode; icon?: LucideIcon }) {
  const Icon = icon ?? (tone === 'red' || tone === 'yellow' ? AlertTriangle : Info);
  return (
    <div className={`flex items-start gap-2 rounded-md border px-3 py-2 text-sm ${PILL[tone]}`} role={tone === 'red' ? 'alert' : 'status'}>
      <Icon className="w-4 h-4 mt-0.5 shrink-0" aria-hidden="true" />
      <div className="min-w-0">{children}</div>
    </div>
  );
}

export function ErrorNotice({ error, children }: { error: unknown; children?: ReactNode }) {
  if (!error) return null;
  return (
    <Notice tone="red">
      {describeError(error)}
      {children}
    </Notice>
  );
}

// --- form fields --------------------------------------------------------------------------------

const INPUT =
  'w-full h-8 rounded-md border border-border-control bg-canvas px-2.5 text-sm text-fg placeholder:text-fg-subtle focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue';

export function Field({ label, hint, children }: { label: string; hint?: ReactNode; children: (id: string) => ReactNode }) {
  const id = useId();
  return (
    <div className="flex flex-col gap-1 min-w-0">
      <label htmlFor={id} className="text-xs font-medium text-fg-muted">
        {label}
      </label>
      {children(id)}
      {hint && <p className="text-xs text-fg-subtle">{hint}</p>}
    </div>
  );
}

export const TextInput = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(function TextInput(props, ref) {
  return <input {...props} ref={ref} className={`${INPUT} ${props.className ?? ''}`} />;
});

export function SelectInput(props: SelectHTMLAttributes<HTMLSelectElement>) {
  return <select {...props} className={`${INPUT} pr-7 ${props.className ?? ''}`} />;
}

// --- progress -------------------------------------------------------------------------------

export function ProgressBar({ fraction, label }: { fraction: number; label?: string }) {
  const pct = Math.max(0, Math.min(100, Math.round(fraction * 100)));
  return (
    <div className="flex flex-col gap-1">
      <div className="h-1.5 rounded-full bg-canvas-inset border border-border-muted overflow-hidden" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
        <div className="h-full bg-primer-blue transition-[width]" style={{ width: `${pct}%` }} />
      </div>
      {label && <p className="text-xs text-fg-muted">{label}</p>}
    </div>
  );
}

// --- time and bytes -------------------------------------------------------------------------

export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
}

export function formatRelative(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return 'never';
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return iso;
  const s = Math.round((now - t) / 1000);
  const abs = Math.abs(s);
  const fmt = new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' });
  if (abs < 60) return fmt.format(-s, 'second');
  if (abs < 3600) return fmt.format(-Math.round(s / 60), 'minute');
  if (abs < 86400) return fmt.format(-Math.round(s / 3600), 'hour');
  return fmt.format(-Math.round(s / 86400), 'day');
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return '—';
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined) return '—';
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KB', 'MB', 'GB', 'TB'];
  let value = bytes / 1024;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return `${value.toFixed(value < 10 ? 1 : 0)} ${units[i]}`;
}

/** snake_case / SCREAMING_SNAKE contract identifiers rendered as words for the console UI. */
export function humanize(value: string | null | undefined): string {
  if (!value) return '—';
  return value
    .toLowerCase()
    .split(/[_-]/)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}
