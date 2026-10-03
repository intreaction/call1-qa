import { useState, type ReactNode } from 'react';
import { Button, Notice, TextInput } from '../components/ui';
import { describeError } from '../api';

/**
 * An audited admin action that needs a reason (`AccountUpdate.reason`, `InstallationRetire.reason`,
 * `ServiceKeyRevoke.reason`). Opens inline, asks for the reason, confirms.
 */
export function ReasonAction({
  label,
  confirmLabel,
  warning,
  variant = 'secondary',
  onConfirm,
  disabled,
}: {
  label: string;
  confirmLabel?: string;
  warning?: ReactNode;
  variant?: 'secondary' | 'danger';
  onConfirm(reason: string): Promise<unknown>;
  disabled?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [reason, setReason] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  if (!open) {
    return (
      <Button size="sm" variant={variant} disabled={disabled} onClick={() => setOpen(true)}>
        {label}
      </Button>
    );
  }
  return (
    <form
      className="w-full flex flex-col gap-2 rounded-md border border-border bg-canvas p-3"
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
        aria-label="Reason (recorded in the audit log)"
        placeholder="Reason (recorded in the audit log)"
        required
        maxLength={500}
        value={reason}
        onChange={(e) => setReason(e.target.value)}
        autoFocus
      />
      {error != null && <Notice tone="red">{describeError(error)}</Notice>}
      <div className="flex gap-2">
        <Button type="submit" size="sm" variant={variant === 'danger' ? 'danger' : 'primary'} busy={busy} disabled={!reason.trim()}>
          {confirmLabel ?? label}
        </Button>
        <Button size="sm" variant="ghost" onClick={() => setOpen(false)}>
          Cancel
        </Button>
      </div>
    </form>
  );
}

export const ROLES = ['reviewer', 'supervisor', 'admin'] as const;

/** "reviewer" → "Reviewer" for display; the value sent to Store is unchanged. */
export function roleLabel(role: string): string {
  return role ? role.charAt(0).toUpperCase() + role.slice(1) : role;
}
