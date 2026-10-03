import { useEffect, useRef, useState, type ReactNode } from 'react';
import type { LucideIcon } from 'lucide-react';
import { Button, ErrorNotice } from './ui';

/**
 * A console write that needs a plain yes/no confirm — no reason field (the on-device training
 * "Roll back" and "Use base model" actions, docs/OnDeviceTraining.md §6.2: "the `ReasonAction`
 * pattern, without a reason field"). Opens inline, traps the two buttons, and surfaces the
 * server's own explanation on failure.
 *
 * The confirm and dismiss buttons always have distinct names, so a destructive action never shows
 * two buttons both called "Cancel". Escape dismisses and returns focus to the opener.
 */
export function ConfirmAction({
  label,
  confirmLabel,
  dismissLabel = 'Cancel',
  warning,
  variant = 'secondary',
  onConfirm,
  disabled,
  disabledReason,
  icon,
}: {
  label: string;
  confirmLabel: string;
  dismissLabel?: string;
  warning: ReactNode;
  /** `primary` for a main action that still needs a confirm (on-device training's "Train now"). */
  variant?: 'primary' | 'secondary' | 'danger';
  icon?: LucideIcon;
  onConfirm(): Promise<unknown>;
  disabled?: boolean;
  disabledReason?: string;
}) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const opener = useRef<HTMLButtonElement>(null);
  const confirmRef = useRef<HTMLButtonElement>(null);
  const reopenFocus = useRef(false);

  useEffect(() => {
    if (!open && reopenFocus.current) {
      reopenFocus.current = false;
      opener.current?.focus();
    }
  }, [open]);

  useEffect(() => {
    if (open) confirmRef.current?.focus();
  }, [open]);

  const close = () => {
    reopenFocus.current = true;
    setOpen(false);
    setError(null);
  };

  if (!open) {
    return (
      <Button
        ref={opener}
        size="sm"
        variant={variant}
        icon={icon}
        disabled={disabled}
        title={disabled ? disabledReason : undefined}
        aria-expanded={false}
        onClick={() => setOpen(true)}
      >
        {label}
      </Button>
    );
  }
  return (
    <div
      role="group"
      aria-label={label}
      className="flex w-full flex-col gap-2 rounded-md border border-border bg-canvas p-3"
      onKeyDown={(e) => {
        if (e.key === 'Escape' && !busy) {
          e.preventDefault();
          close();
        }
      }}
    >
      <div className="text-sm text-fg">{warning}</div>
      <ErrorNotice error={error} />
      <div className="flex gap-2">
        <Button
          ref={confirmRef}
          size="sm"
          variant={variant === 'danger' ? 'danger' : 'primary'}
          busy={busy}
          onClick={async () => {
            setBusy(true);
            setError(null);
            try {
              await onConfirm();
              setOpen(false);
            } catch (err) {
              setError(err);
            } finally {
              setBusy(false);
            }
          }}
        >
          {confirmLabel}
        </Button>
        <Button size="sm" variant="ghost" onClick={close}>
          {dismissLabel}
        </Button>
      </div>
    </div>
  );
}
