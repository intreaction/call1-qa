import { useState } from 'react';
import { Fingerprint, Plus, ShieldCheck } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { describeError, queryKeys } from '../api';
import { beginAddAuthenticator, ceremonyExpired, reauthenticate, registerAdditional, type AddAuthenticatorBegin } from '../api/passkeys';
import { Button, Field, Notice, TextInput } from '../components/ui';
import { useSession, useStore } from '../state/app';
import { usePasskeyStep } from './usePasskeyStep';

type Assertion = Awaited<ReturnType<typeof reauthenticate>>;

/**
 * Add another authenticator: a step-up ceremony. First a fresh, user-verified assertion from an
 * authenticator the account already has, then the new registration (contracts/README.md
 * "Identity details").
 */
export function AddAuthenticator({ onDone }: { onDone?(): void }) {
  const { client } = useStore();
  const { refresh } = useSession();
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [nickname, setNickname] = useState('');
  const [begin, setBegin] = useState<AddAuthenticatorBegin | null>(null);
  const [assertion, setAssertion] = useState<Assertion | null>(null);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<unknown>(null);
  const [added, setAdded] = useState<string | null>(null);
  const step = usePasskeyStep();

  const reset = () => {
    setBegin(null);
    setAssertion(null);
    step.reset();
  };

  const verify = async (ceremony: AddAuthenticatorBegin) => {
    const a = await step.run(() => reauthenticate(ceremony));
    if (!a) return;
    setAssertion(a);
    await register(ceremony, a);
  };

  const register = async (ceremony: AddAuthenticatorBegin, a: Assertion) => {
    const res = await step.run(() => registerAdditional(client, ceremony, a, nickname));
    if (!res) return;
    setAdded(res.credential.nickname ?? 'New authenticator');
    reset();
    setOpen(false);
    setNickname('');
    void queryClient.invalidateQueries({ queryKey: queryKeys.ownAuthenticators });
    void refresh();
    onDone?.();
  };

  const start = async () => {
    setStartError(null);
    setAdded(null);
    reset();
    setStarting(true);
    try {
      const ceremony = await beginAddAuthenticator(client, nickname);
      setBegin(ceremony);
      setStarting(false);
      await verify(ceremony);
    } catch (err) {
      setStarting(false);
      setStartError(err);
    }
  };

  const retry = async () => {
    if (!begin || step.ceremonyUsed || ceremonyExpired(begin)) return start();
    if (assertion) return register(begin, assertion);
    return verify(begin);
  };

  if (!open) {
    return (
      <div className="flex flex-col gap-2">
        {added && <Notice tone="green" icon={ShieldCheck}>Added “{added}”. You can now sign in with it.</Notice>}
        <div>
          <Button icon={Plus} onClick={() => setOpen(true)}>
            Add another authenticator
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-md border border-border bg-canvas p-3 flex flex-col gap-3">
      <p className="text-sm text-fg">
        First confirm it's you with an authenticator you already have, then register the new one.
      </p>
      {!begin && (
        <Field label="Name the new authenticator (optional)" hint="For example “Backup YubiKey”.">
          {(id) => <TextInput id={id} maxLength={80} value={nickname} onChange={(e) => setNickname(e.target.value)} autoFocus />}
        </Field>
      )}
      {begin && (
        <ol className="text-sm flex flex-col gap-1">
          <li className={assertion ? 'text-primer-greenFg' : 'text-fg'}>1. Confirm with an existing authenticator{assertion ? ' — done' : ''}</li>
          <li className="text-fg-muted">2. Register the new authenticator</li>
        </ol>
      )}
      {step.status === 'working' && <Notice icon={Fingerprint}>Follow your browser's prompt.</Notice>}
      {startError != null && <Notice tone="red">{describeError(startError)}</Notice>}
      {step.error && <Notice tone={step.ceremonyUsed ? 'red' : 'yellow'}>{step.error}</Notice>}
      <div className="flex gap-2">
        {!begin || step.ceremonyUsed ? (
          <Button variant="primary" icon={Fingerprint} busy={starting} onClick={start}>
            Start
          </Button>
        ) : (
          step.status !== 'working' && (
            <Button variant="primary" icon={Fingerprint} onClick={retry}>
              {assertion ? 'Register the new authenticator' : 'Confirm with an existing authenticator'}
            </Button>
          )
        )}
        <Button
          variant="ghost"
          onClick={() => {
            reset();
            setOpen(false);
          }}
        >
          Cancel
        </Button>
      </div>
    </div>
  );
}
