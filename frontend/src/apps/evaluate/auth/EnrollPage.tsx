import { useState, type FormEvent } from 'react';
import { CheckCircle2, Fingerprint, Ticket } from 'lucide-react';
import { describeError, isStoreError } from '../api';
import {
  beginEnrollment,
  browserSupportsWebAuthn,
  ceremonyExpired,
  completeEnrollment,
  type EnrollmentSecret,
  type RegistrationBegin,
  type RegistrationFinish,
} from '../api/passkeys';
import { Button, Field, Notice, TextInput } from '../components/ui';
import { useSessionState, useStore } from '../state/app';
import { href, navigate, scrubEnrollmentToken } from '../state/router';
import { LostAuthenticatorsCopy } from './SignInPage';
import { usePasskeyStep } from './usePasskeyStep';

function enrollBeginErrorText(err: unknown, viaInvitation: boolean): string {
  // A malformed secret fails request validation before Store looks it up: same answer.
  if (isStoreError(err, 'validation_failed')) {
    return viaInvitation
      ? 'This invitation link is invalid or incomplete. Open the full link again, or ask an admin for a new invitation.'
      : 'That is not a valid setup code. Check it and try again.';
  }
  if (isStoreError(err, 'invitation_invalid')) {
    return 'This invitation link is invalid, expired, revoked or already used. Ask an admin for a new invitation.';
  }
  if (isStoreError(err, 'setup_code_invalid')) {
    return 'That setup code is invalid, expired or already used. Run the setup-code command on the Store host again.';
  }
  return describeError(err);
}

/**
 * Enrollment from an invitation link (`/#/enroll?token=…`, or Store's `/enroll#…` form) or a setup
 * code printed on the Store host. Enrolling registers the first authenticator and signs in.
 */
export function EnrollPage({ token }: { token?: string }) {
  const { client } = useStore();
  const { signedIn } = useSessionState();
  const [code, setCode] = useState('');
  const [nickname, setNickname] = useState('');
  const [begin, setBegin] = useState<RegistrationBegin | null>(null);
  const [beginError, setBeginError] = useState<unknown>(null);
  const [starting, setStarting] = useState(false);
  const [done, setDone] = useState<RegistrationFinish | null>(null);
  const step = usePasskeyStep();
  const supported = browserSupportsWebAuthn();
  const secret: EnrollmentSecret | null = token ? { invitation_token: token } : code.trim() ? { setup_code: code.trim() } : null;

  const prompt = async (ceremony: RegistrationBegin) => {
    const res = await step.run(() => completeEnrollment(client, ceremony, nickname));
    if (!res) return;
    scrubEnrollmentToken();
    setDone(res);
    // Store opened the session with the ceremony, so the app is signed in on every route from now
    // on. The shell renders the enroll route whatever the session state (App.tsx), so this
    // confirmation stays on screen until the reviewer leaves it.
    if (res.signed_in) signedIn(res.signed_in.session);
  };

  const enter = (route: 'account' | 'calls') => {
    navigate({ name: route }, { replace: true });
  };

  const start = async (e?: FormEvent) => {
    e?.preventDefault();
    if (!secret) return;
    setBeginError(null);
    setStarting(true);
    try {
      const ceremony = await beginEnrollment(client, secret);
      setBegin(ceremony);
      setStarting(false);
      await prompt(ceremony);
    } catch (err) {
      setStarting(false);
      setBeginError(err);
    }
  };

  const retry = async () => {
    // A finish failure consumed the ceremony and, for a failed finish, possibly the invitation.
    if (!begin || step.ceremonyUsed || ceremonyExpired(begin)) await start();
    else await prompt(begin);
  };

  if (done) {
    return (
      <div className="w-full max-w-sm flex flex-col gap-4">
        <div className="flex items-center gap-2">
          <CheckCircle2 className="w-5 h-5 text-primer-greenFg" aria-hidden="true" />
          <h1 className="text-base font-semibold text-fg">Authenticator enrolled</h1>
        </div>
        <p className="text-sm text-fg-muted">
          {done.account.display_name} ({done.account.email}) is enrolled as <strong className="text-fg">{done.account.role}</strong>.
        </p>
        <Notice tone="yellow">
          Add a second authenticator now, for example a backup security key. If you lose your only one, an admin has to
          re-invite you.
        </Notice>
        {done.signed_in ? (
          <div className="flex gap-2">
            <Button variant="primary" onClick={() => enter('account')}>
              Add a second authenticator
            </Button>
            <Button onClick={() => enter('calls')}>Go to calls</Button>
          </div>
        ) : (
          <Button variant="primary" onClick={() => navigate({ name: 'sign-in' }, { replace: true })}>
            Sign in
          </Button>
        )}
      </div>
    );
  }

  return (
    <div className="w-full max-w-sm flex flex-col gap-4">
      <div>
        <h1 className="text-base font-semibold text-fg">{token ? 'Accept your invitation' : 'Enroll with a setup code'}</h1>
        <p className="text-sm text-fg-muted mt-1">
          {token
            ? 'Register a passkey or security key for your Call1 account. It is the only way to sign in; there is no password.'
            : 'The first administrator (or a break-glass re-enrollment) uses the one-time code printed by `python -m call1.store setup-code` on the Store host.'}
        </p>
      </div>

      {!supported && <Notice tone="red">This browser does not support passkeys. Open Evaluate in a current browser.</Notice>}

      {!begin || step.ceremonyUsed ? (
        <form onSubmit={start} className="flex flex-col gap-3">
          {!token && (
            <Field label="Setup code">
              {(id) => (
                <TextInput
                  id={id}
                  required
                  autoFocus
                  autoComplete="one-time-code"
                  spellCheck={false}
                  value={code}
                  onChange={(e) => setCode(e.target.value)}
                />
              )}
            </Field>
          )}
          <Field label="Name this authenticator (optional)" hint="For example “YubiKey 5C” or “Work laptop”. You can rename it later.">
            {(id) => <TextInput id={id} maxLength={80} value={nickname} onChange={(e) => setNickname(e.target.value)} />}
          </Field>
          {beginError != null && <Notice tone="red">{enrollBeginErrorText(beginError, Boolean(token))}</Notice>}
          {step.error && step.ceremonyUsed && <Notice tone="red">{step.error}</Notice>}
          <Button type="submit" variant="primary" icon={token ? Fingerprint : Ticket} busy={starting} disabled={!supported || !secret}>
            Create passkey
          </Button>
        </form>
      ) : (
        <div className="flex flex-col gap-3">
          {step.status === 'working' && (
            <Notice icon={Fingerprint}>Follow your browser's prompt: touch your security key or create a passkey.</Notice>
          )}
          {step.error && <Notice tone="yellow">{step.error}</Notice>}
          {step.status !== 'working' && (
            <Button variant="primary" icon={Fingerprint} onClick={retry}>
              Register passkey or security key
            </Button>
          )}
        </div>
      )}

      <div className="border-t border-border-muted pt-3 flex flex-col gap-2">
        {token ? (
          <p className="text-xs text-fg-muted">
            Enrolling from an invitation that re-invites an existing account replaces that account's old authenticators.
          </p>
        ) : (
          <LostAuthenticatorsCopy />
        )}
        <p className="text-xs text-fg-muted">
          Already enrolled?{' '}
          <a className="text-primer-blueFg hover:underline" href={href({ name: 'sign-in' })}>
            Sign in
          </a>
        </p>
      </div>
    </div>
  );
}
