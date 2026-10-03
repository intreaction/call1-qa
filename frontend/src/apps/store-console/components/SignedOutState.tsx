import { Lock, ShieldAlert } from 'lucide-react';

/** Shown in place of a panel's body when no session cookie is present at all. */
export function SignedOutState() {
  return (
    <div className="flex flex-col items-center text-center gap-2 py-6 px-2">
      <Lock className="w-5 h-5 text-fg-subtle" aria-hidden="true" />
      <p className="text-sm text-fg">Signed out</p>
      <p className="text-xs text-fg-muted max-w-[28ch]">
        This panel needs an admin session on Store.
      </p>
      <a
        href="/"
        className="mt-1 px-3 py-1.5 text-xs font-medium rounded-md bg-primer-blueSubtle text-primer-blueFg border border-primer-blueBorder hover:bg-primer-blueSubtle/80 transition-colors"
      >
        Sign in with a passkey on Evaluate
      </a>
    </div>
  );
}

/** Shown when a session exists but its role is below what the panel needs. */
export function InsufficientRoleState({ role }: { role: string }) {
  return (
    <div className="flex flex-col items-center text-center gap-2 py-6 px-2">
      <ShieldAlert className="w-5 h-5 text-primer-yellowFg" aria-hidden="true" />
      <p className="text-sm text-fg">Admin role required</p>
      <p className="text-xs text-fg-muted max-w-[30ch]">
        Signed in as <span className="text-fg">{role}</span>. An admin account is needed for this
        panel.
      </p>
    </div>
  );
}
