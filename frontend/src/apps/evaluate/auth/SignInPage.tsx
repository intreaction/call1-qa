import { useRef, useState, type FormEvent } from 'react';
import { ArrowLeft, Fingerprint, KeyRound, Sparkles } from 'lucide-react';
import { describeError, isStoreError, type SessionInfo } from '../api';
import { demoSignIn, type DemoPersonaKey } from '../api/demo';
import { beginSignIn, browserSupportsWebAuthn, ceremonyExpired, completeSignIn, type AuthenticationBegin } from '../api/passkeys';
import { Button, Field, Notice, TextInput } from '../components/ui';
import { useSessionState, useStore } from '../state/app';
import { useDemoStatus } from '../state/demo';
import { href } from '../state/router';
import { usePasskeyStep } from './usePasskeyStep';

/**
 * "Continue as Demo …" buttons, shown above the passkey form only when Store answers
 * `GET /demo/status` with `demo: true` (localhost only; off by default — `CALL1_STORE_DEMO=1`).
 * The real passkey form below is unchanged and always present; this is a shortcut on top of it,
 * not a replacement (John, 2026-09-25: demo-ready for a class presentation, auth need not be
 * fully functional, but the real passkey code stays intact).
 */
function DemoSignIn({ onSignedIn }: { onSignedIn(session: SessionInfo): void }) {
  const demo = useDemoStatus();
  const [pending, setPending] = useState<DemoPersonaKey | null>(null);
  const [error, setError] = useState<unknown>(null);

  if (!demo.data?.demo || demo.data.personas.length === 0) return null;

  const choose = async (persona: DemoPersonaKey) => {
    setError(null);
    setPending(persona);
    try {
      const result = await demoSignIn(persona);
      onSignedIn(result.session);
    } catch (err) {
      setError(err);
      setPending(null);
    }
  };

  return (
    <div
      className="flex flex-col gap-2 rounded-md border border-primer-magentaBorder bg-primer-magentaSubtle p-3"
      role="region"
      aria-label="Demo mode sign-in"
    >
      <div className="flex items-center gap-1.5">
        <Sparkles className="w-3.5 h-3.5 text-primer-magentaFg shrink-0" aria-hidden="true" />
        <p className="text-sm font-medium text-primer-magentaFg">Demo mode</p>
      </div>
      <p className="text-xs text-fg-muted">{demo.data.label}</p>
      <div className="flex flex-col gap-2">
        {demo.data.personas.map((p) => (
          <Button
            key={p.persona}
            variant="primary"
            busy={pending === p.persona}
            disabled={pending !== null && pending !== p.persona}
            onClick={() => void choose(p.persona)}
          >
            Continue as {p.display_name}
          </Button>
        ))}
      </div>
      {error != null && <Notice tone="red">{describeError(error)}</Notice>}
    </div>
  );
}

/** Shown on every signed-out screen. There is no password, and no reset. */
export function LostAuthenticatorsCopy() {
  return (
    <p className="text-xs text-fg-muted">
      <strong className="font-medium text-fg">Lost every authenticator?</strong> Evaluate has no password and no reset. Ask an
      admin for a re-invite; enrolling from it replaces your old authenticators.
    </p>
  );
}

/**
 * Account-first sign-in: the email, then the authenticator. Store answers any email with the same
 * shape (anti-enumeration), so a wrong address only shows up as the authenticator not matching.
 */
export function SignInPage({ notice }: { notice?: string }) {
  const { client } = useStore();
  const { signedIn } = useSessionState();
  const [email, setEmail] = useState('');
  const [begin, setBegin] = useState<AuthenticationBegin | null>(null);
  const [beginError, setBeginError] = useState<unknown>(null);
  const [starting, setStarting] = useState(false);
  const step = usePasskeyStep();
  const emailRef = useRef<HTMLInputElement>(null);
  const supported = browserSupportsWebAuthn();
  const demo = useDemoStatus();
  const demoOn = Boolean(demo.data?.demo && demo.data.personas.length > 0);

  const prompt = async (ceremony: AuthenticationBegin) => {
    const session = await step.run(() => completeSignIn(client, ceremony));
    if (session) signedIn(session);
  };

  const start = async (e?: FormEvent) => {
    e?.preventDefault();
    setBeginError(null);
    setStarting(true);
    try {
      const ceremony = await beginSignIn(client, email);
      setBegin(ceremony);
      setStarting(false);
      await prompt(ceremony);
    } catch (err) {
      setBeginError(err);
      setStarting(false);
    }
  };

  const retry = async () => {
    if (!begin || step.ceremonyUsed || ceremonyExpired(begin)) await start();
    else await prompt(begin);
  };

  const back = () => {
    setBegin(null);
    step.reset();
    setTimeout(() => emailRef.current?.focus(), 0);
  };

  const failedAtStore = step.status === 'failed' && step.ceremonyUsed;

  return (
    <div className="w-full max-w-sm flex flex-col gap-4">
      <div>
        <h1 className="text-base font-semibold text-fg">Sign in</h1>
        <p className="text-sm text-fg-muted mt-1">Use the passkey or security key registered to your account.</p>
      </div>

      {notice && <Notice>{notice}</Notice>}
      <DemoSignIn onSignedIn={signedIn} />
      {!supported && (
        <Notice tone="red">This browser does not support passkeys. Open Evaluate in a current browser.</Notice>
      )}

      {demoOn && (
        <div className="flex items-center gap-2 text-xs text-fg-subtle" aria-hidden="true">
          <div className="flex-1 h-px bg-border-muted" />
          or sign in with a passkey
          <div className="flex-1 h-px bg-border-muted" />
        </div>
      )}

      {!begin ? (
        <form onSubmit={start} className="flex flex-col gap-3">
          <Field label="Account email">
            {(id) => (
              <TextInput
                id={id}
                ref={emailRef}
                type="email"
                required
                autoFocus
                autoComplete="username webauthn"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
              />
            )}
          </Field>
          {beginError != null && <Notice tone="red">{describeError(beginError)}</Notice>}
          <Button type="submit" variant="primary" icon={KeyRound} busy={starting} disabled={!supported || !email}>
            Continue
          </Button>
        </form>
      ) : (
        <div className="flex flex-col gap-3">
          <div className="flex items-center justify-between gap-2 rounded-md border border-border bg-canvas px-3 py-2">
            <span className="text-sm text-fg truncate">{email}</span>
            <Button size="sm" variant="ghost" icon={ArrowLeft} onClick={back}>
              Change
            </Button>
          </div>
          {step.status === 'working' && (
            <Notice icon={Fingerprint}>Follow your browser's prompt: touch your security key or confirm your passkey.</Notice>
          )}
          {step.error && (
            <Notice tone={failedAtStore ? 'red' : 'yellow'}>
              {isStoreError(step.cause, 'webauthn_verification_failed')
                ? 'That authenticator is not registered to this account, or the account cannot sign in. Check the email and try again.'
                : step.error}
            </Notice>
          )}
          {step.status !== 'working' && (
            <Button variant="primary" icon={Fingerprint} onClick={retry}>
              Use passkey or security key
            </Button>
          )}
        </div>
      )}

      <div className="border-t border-border-muted pt-3 flex flex-col gap-2">
        <LostAuthenticatorsCopy />
        <p className="text-xs text-fg-muted">
          Have an invitation link or a setup code?{' '}
          <a className="text-primer-blueFg hover:underline" href={href({ name: 'enroll' })}>
            Enroll an authenticator
          </a>
        </p>
      </div>
    </div>
  );
}
