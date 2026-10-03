import { useEffect, useRef, useState, type ReactNode } from 'react';
import { Button, ErrorNotice, TextInput } from './ui';

/**
 * A console write that needs a reason (retry, cancel — both take `{reason}` in the contract).
 * Opens inline, asks for the reason, confirms, and surfaces the server's own explanation on
 * failure (Process turns 403 insufficient_scope into a plain-English fix-it message).
 *
 * The confirm and dismiss buttons always have distinct names (`confirmLabel`, `dismissLabel`), so
 * a "Cancel" action never shows two buttons both called "Cancel". Escape closes the form.
 */
export function ReasonAction({
  label,
  confirmLabel,
  dismissLabel = 'Close',
  warning,
  variant = 'secondary',
  onConfirm,
  disabled,
  disabledReason,
  extraFields,
}: {
  label: string;
  confirmLabel?: string;
  dismissLabel?: string;
  warning?: ReactNode;
  variant?: 'secondary' | 'danger';
  onConfirm(reason: string): Promise<unknown>;
  disabled?: boolean;
  disabledReason?: string;
  extraFields?: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const [reason, setReason] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const opener = useRef<HTMLButtonElement>(null);
  const reopenFocus = useRef(false);

  useEffect(() => {
    // Back on the opener after the form closes, so keyboard users keep their place.
    if (!open && reopenFocus.current) {
      reopenFocus.current = false;
      opener.current?.focus();
    }
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
    <form
      className="w-full flex flex-col gap-2 rounded-md border border-border bg-canvas p-3"
      aria-label={label}
      onKeyDown={(e) => {
        if (e.key === 'Escape' && !busy) {
          e.preventDefault();
          close();
        }
      }}
      onSubmit={async (e) => {
        e.preventDefault();
        setBusy(true);
        setError(null);
        try {
          await onConfirm(reason.trim());
          setOpen(false);
          setReason('');
        } catch (err) {
          setError(err);
        } finally {
          setBusy(false);
        }
      }}
    >
      {warning && <p className="text-sm text-fg">{warning}</p>}
      <TextInput
        aria-label="Reason"
        placeholder="Reason (shown on the job's attempt history)"
        required
        maxLength={2000}
        value={reason}
        onChange={(e) => setReason(e.target.value)}
        autoFocus
      />
      {extraFields}
      <ErrorNotice error={error} />
      <div className="flex gap-2">
        <Button type="submit" size="sm" variant={variant === 'danger' ? 'danger' : 'primary'} busy={busy} disabled={!reason.trim()}>
          {confirmLabel ?? label}
        </Button>
        <Button size="sm" variant="ghost" onClick={close}>
          {dismissLabel}
        </Button>
      </div>
    </form>
  );
}
