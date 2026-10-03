// Small shared building blocks for Evaluate, on the Ink & Signal / Frost & Ink tokens in
// src/index.css. Quiet by design: a few button styles, pills with text labels, cards, notices.

import { forwardRef, useEffect, useId, useRef, useState, type ButtonHTMLAttributes, type InputHTMLAttributes, type ReactNode, type SelectHTMLAttributes, type TextareaHTMLAttributes } from 'react';
import type { LucideIcon } from 'lucide-react';
import { AlertTriangle, Check, Copy, Hammer, Info, Loader2, X } from 'lucide-react';
import { describeError, type Tone } from '../api';

// --- buttons ------------------------------------------------------------------------------------

type ButtonVariant = 'primary' | 'secondary' | 'danger' | 'ghost';

const BUTTON: Record<ButtonVariant, string> = {
  primary: 'bg-primer-blue text-white border-transparent hover:bg-primer-blueHover',
  secondary: 'bg-canvas text-fg border-border hover:bg-canvas-inset',
  danger: 'bg-canvas text-primer-redFg border-primer-redBorder hover:bg-primer-redSubtle',
  ghost: 'bg-transparent text-fg-muted border-transparent hover:text-fg hover:bg-canvas-inset',
};

export function Button({
  variant = 'secondary',
  size = 'md',
  icon: Icon,
  busy,
  children,
  className = '',
  disabled,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
  size?: 'sm' | 'md';
  icon?: LucideIcon;
  busy?: boolean;
}) {
  const sizing = size === 'sm' ? 'h-7 px-2.5 text-xs gap-1.5' : 'h-8 px-3 text-sm gap-2';
  return (
    <button
      type="button"
      {...rest}
      disabled={disabled || busy}
      aria-busy={busy || undefined}
      className={`inline-flex items-center justify-center rounded-md border font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue disabled:opacity-50 disabled:cursor-not-allowed ${sizing} ${BUTTON[variant]} ${className}`}
    >
      {busy ? <Loader2 className="w-3.5 h-3.5 animate-spin" aria-hidden="true" /> : Icon ? <Icon className="w-3.5 h-3.5" aria-hidden="true" /> : null}
      {children}
    </button>
  );
}

// --- status pills: color AND a text label, never color alone ------------------------------------

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
              {subtitle && <p className="text-xs text-fg-muted">{subtitle}</p>}
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
      {right && <div className="flex items-center gap-2">{right}</div>}
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

/** An honest "not built yet (Stage N)" state (docs/SplitBuild.md design principles). */
export function NotBuiltYet({ what, stage, children }: { what: string; stage?: number; children?: ReactNode }) {
  return (
    <EmptyState icon={Hammer} title={`${what} is not built yet${stage ? ` (Stage ${stage})` : ''}`}>
      {children}
    </EmptyState>
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

export const TextArea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(function TextArea(props, ref) {
  return (
    <textarea
      {...props}
      ref={ref}
      className={`w-full min-h-[3.5rem] rounded-md border border-border-control bg-canvas px-2.5 py-1.5 text-sm text-fg placeholder:text-fg-subtle focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue disabled:opacity-60 ${props.className ?? ''}`}
    />
  );
});

/** A labelled checkbox (the label is the accessible name). */
export function Checkbox({ label, hint, className = '', ...rest }: Omit<InputHTMLAttributes<HTMLInputElement>, 'type'> & { label: ReactNode; hint?: ReactNode }) {
  const id = useId();
  return (
    <div className={`flex items-start gap-2 ${className}`}>
      <input id={id} type="checkbox" {...rest} className="mt-0.5 accent-primer-blue focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue" />
      <label htmlFor={id} className={`text-sm text-fg ${rest.disabled ? 'opacity-60' : ''}`}>
        {label}
        {hint && <span className="block text-xs text-fg-subtle">{hint}</span>}
      </label>
    </div>
  );
}

/** "12/40" under a text input; over the limit it turns red and says so (text, not color alone). */
export function CharCount({ value, max }: { value: string | null | undefined; max: number }) {
  const n = (value ?? '').length;
  const over = n > max;
  return (
    <span className={`text-xs tabular-nums ${over ? 'text-primer-redFg font-medium' : 'text-fg-subtle'}`} aria-live="polite">
      {n}/{max}
      {over && ' — too long'}
    </span>
  );
}

// --- chips ---------------------------------------------------------------------------------------

/** A small labelled chip: a category, subcategory, field value or alert. Tone plus text. */
export function Chip({ tone = 'neutral', children, title, icon: Icon }: { tone?: Tone; children: ReactNode; title?: string; icon?: LucideIcon }) {
  return (
    <span title={title} className={`inline-flex items-center gap-1 max-w-full px-1.5 py-0.5 rounded border text-[11px] font-medium leading-tight ${PILL[tone]}`}>
      {Icon && <Icon className="w-3 h-3 shrink-0" aria-hidden="true" />}
      <span className="truncate">{children}</span>
    </span>
  );
}

// --- dialog ------------------------------------------------------------------------------------

/**
 * A modal dialog: `role="dialog"`, labelled by its title, focus moved inside on open and returned
 * on close, Escape closes it, and Tab stays inside while it is open.
 */
export function Dialog({ title, onClose, children, footer }: { title: string; onClose(): void; children: ReactNode; footer?: ReactNode }) {
  const titleId = useId();
  const panel = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    const first = panel.current?.querySelector<HTMLElement>('input, select, textarea, button, a[href]');
    (first ?? panel.current)?.focus();
    return () => previous?.focus?.();
  }, []);
  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      e.stopPropagation();
      onClose();
      return;
    }
    if (e.key !== 'Tab' || !panel.current) return;
    const focusable = Array.from(panel.current.querySelectorAll<HTMLElement>('input, select, textarea, button, a[href]')).filter((el) => !el.hasAttribute('disabled'));
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
  return (
    <div className="fixed inset-0 z-50 flex items-start sm:items-center justify-center bg-black/40 p-4 overflow-y-auto" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div
        ref={panel}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        onKeyDown={onKeyDown}
        className="w-full max-w-lg rounded-lg border border-border bg-canvas shadow-xl focus:outline-none"
      >
        <header className="flex items-center justify-between gap-2 px-4 py-3 border-b border-border-muted">
          <h2 id={titleId} className="text-sm font-semibold text-fg">
            {title}
          </h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="w-6 h-6 rounded hover:bg-canvas-inset text-fg-muted flex items-center justify-center focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
          >
            <X className="w-3.5 h-3.5" aria-hidden="true" />
          </button>
        </header>
        <div className="p-4 space-y-3">{children}</div>
        {footer && <footer className="flex flex-wrap justify-end gap-2 px-4 py-3 border-t border-border-muted">{footer}</footer>}
      </div>
    </div>
  );
}

// --- one-time secrets (invitation links, service-key tokens) ------------------------------------

/**
 * Shows a secret Store returns exactly once. It is held only in component state, never stored,
 * and disappears when dismissed or the page is left.
 */
export function OneTimeSecret({ label, value, children, onDismiss }: { label: string; value: string; children?: ReactNode; onDismiss(): void }) {
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    if (!copied) return;
    const t = setTimeout(() => setCopied(false), 2000);
    return () => clearTimeout(t);
  }, [copied]);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
    } catch {
      // clipboard denied: the value is selectable
    }
  };
  return (
    <div className="rounded-md border border-primer-yellowBorder bg-primer-yellowSubtle p-3 flex flex-col gap-2" role="status">
      <p className="text-sm font-medium text-primer-yellowFg">{label} — shown once</p>
      <div className="flex items-center gap-2">
        <input
          readOnly
          value={value}
          onFocus={(e) => e.currentTarget.select()}
          aria-label={label}
          className={`${INPUT} font-normal select-all`}
        />
        <Button size="md" icon={copied ? Check : Copy} onClick={copy}>
          {copied ? 'Copied' : 'Copy'}
        </Button>
      </div>
      {children && <div className="text-xs text-fg-muted">{children}</div>}
      <div>
        <Button size="sm" variant="ghost" onClick={onDismiss}>
          I have saved it — hide
        </Button>
      </div>
    </div>
  );
}

// --- inline tooltip -------------------------------------------------------------------------------

/**
 * A dotted-underline span with a tooltip on hover AND keyboard focus (Escape closes it), e.g. a
 * vocabulary-corrected word in the transcript (docs/DualAsr.md section 8): `Parakeet heard "…"`.
 * The underline keeps its contrast in both themes via the `primer-blueFg` token.
 */
export function InlineTooltip({ label, children, className = '' }: { label: string; children: ReactNode; className?: string }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  return (
    <span className="relative inline-block">
      <span
        tabIndex={0}
        aria-describedby={open ? id : undefined}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onKeyDown={(e) => {
          if (e.key === 'Escape') {
            e.stopPropagation();
            setOpen(false);
          }
        }}
        className={`underline decoration-dotted decoration-2 decoration-primer-blueFg underline-offset-4 cursor-help rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue ${className}`}
      >
        {children}
      </span>
      {open && (
        <span
          id={id}
          role="tooltip"
          className="absolute z-20 left-0 bottom-full mb-1.5 w-max max-w-xs whitespace-normal rounded-md border border-border bg-canvas px-2 py-1 text-xs text-fg shadow-lg"
        >
          {label}
        </span>
      )}
    </span>
  );
}

// --- time ---------------------------------------------------------------------------------------

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
