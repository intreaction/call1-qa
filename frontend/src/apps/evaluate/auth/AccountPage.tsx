import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { KeyRound, Laptop, LogOut, Pencil, Sparkles, Trash2, UserRound } from 'lucide-react';
import { describeError, queryKeys, type WebAuthnCredentialRecord } from '../api';
import { demoSignIn, isDemoPersonaEmail, type DemoPersonaKey } from '../api/demo';
import { Button, Card, EmptyState, ErrorNotice, Loading, Notice, PageHeader, StatusPill, TextInput, formatDateTime, formatRelative } from '../components/ui';
import { useSession, useSessionState, useStore } from '../state/app';
import { useDemoStatus, useIsDemoPersona } from '../state/demo';
import { AddAuthenticator } from './AddAuthenticator';

/** Your account: authenticators (add, rename, remove) and sessions (list, revoke, sign out). */
export function AccountPage() {
  const { session, role } = useSession();
  return (
    <div className="max-w-3xl mx-auto flex flex-col gap-4">
      <PageHeader title="Your account" description="Passkeys and security keys are the only way to sign in to Call1." />
      <DemoPersonaCard />
      <Card title="Profile" icon={UserRound}>
        <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1 text-sm">
          <dt className="text-fg-muted">Name</dt>
          <dd className="text-fg">{session.display_name}</dd>
          <dt className="text-fg-muted">Email</dt>
          <dd className="text-fg break-all">{session.email}</dd>
          <dt className="text-fg-muted">Role</dt>
          <dd className="text-fg">{role ? role.charAt(0).toUpperCase() + role.slice(1) : role}</dd>
        </dl>
      </Card>
      <AuthenticatorsCard />
      <SessionsCard />
    </div>
  );
}

/**
 * The account menu's persona switch (task item 1, 2026-09-25): shown only when Store answers
 * `GET /demo/status` with `demo: true`. Switching persona is just another demo sign-in — a fresh
 * `POST /demo/sign-in` for the chosen persona, handed to `signedIn()` exactly like the sign-in
 * screen's demo buttons, which drops every cached read for the old session.
 */
function DemoPersonaCard() {
  const demo = useDemoStatus();
  const { session } = useSession();
  const { signedIn } = useSessionState();
  const [pending, setPending] = useState<DemoPersonaKey | null>(null);
  const [error, setError] = useState<unknown>(null);

  if (!demo.data?.demo || demo.data.personas.length === 0) return null;

  const switchTo = async (persona: DemoPersonaKey) => {
    setError(null);
    setPending(persona);
    try {
      const result = await demoSignIn(persona);
      signedIn(result.session);
    } catch (err) {
      setError(err);
    } finally {
      setPending(null);
    }
  };

  return (
    <Card title="Demo mode" icon={Sparkles} subtitle={demo.data.label}>
      <div className="flex flex-col gap-2">
        <p className="text-xs text-fg-muted">Switch who you are signed in as, without a passkey. Only answers on localhost.</p>
        <div className="flex flex-wrap gap-2">
          {demo.data.personas.map((p) => {
            const current = isDemoPersonaEmail(session.email) && session.email === p.email;
            return (
              <Button
                key={p.persona}
                variant={current ? 'primary' : 'secondary'}
                busy={pending === p.persona}
                disabled={current || (pending !== null && pending !== p.persona)}
                onClick={() => void switchTo(p.persona)}
              >
                {current ? `Signed in as ${p.display_name}` : `Continue as ${p.display_name}`}
              </Button>
            );
          })}
        </div>
        <ErrorNotice error={error} />
      </div>
    </Card>
  );
}

function AuthenticatorsCard() {
  const { client } = useStore();
  const { session, refresh } = useSession();
  const queryClient = useQueryClient();
  const q = useQuery({
    queryKey: queryKeys.ownAuthenticators,
    queryFn: () => client.get('/store/v1/auth/authenticators'),
  });
  const items = q.data?.items ?? [];
  const demoPersona = useIsDemoPersona(session.email);
  const invalidate = () => {
    void queryClient.invalidateQueries({ queryKey: queryKeys.ownAuthenticators });
    void refresh();
  };

  return (
    <Card title="Authenticators" icon={KeyRound} subtitle="Security keys and passkeys that can sign in to this account.">
      <div className="flex flex-col gap-3">
        {session.prompt_second_authenticator && !demoPersona && (
          <Notice tone="yellow">
            You have one authenticator. Add a second one, such as a backup security key: if you lose your only one, an
            admin has to re-invite you.
          </Notice>
        )}
        {q.isLoading && <Loading />}
        <ErrorNotice error={q.error} />
        {q.isSuccess && items.length === 0 && <EmptyState icon={KeyRound} title="No authenticators" />}
        {items.length > 0 && (
          <ul className="divide-y divide-border-muted border border-border-muted rounded-md">
            {items.map((a) => (
              <AuthenticatorRow
                key={a.id}
                authenticator={a}
                usedNow={a.id === session.authenticator_id_used}
                demoPlaceholder={demoPersona && a.nickname === DEMO_PLACEHOLDER_NICKNAME}
                isLast={items.length === 1}
                onChanged={invalidate}
              />
            ))}
          </ul>
        )}
        <AddAuthenticator />
      </div>
    </Card>
  );
}

/** The stand-in credential demo sign-in creates (call1/store/auth/demo.py `PLACEHOLDER_NICKNAME`). */
const DEMO_PLACEHOLDER_NICKNAME = 'Demo mode (no passkey)';

function AuthenticatorRow({
  authenticator: a,
  usedNow,
  demoPlaceholder = false,
  isLast,
  onChanged,
}: {
  authenticator: WebAuthnCredentialRecord;
  usedNow: boolean;
  demoPlaceholder?: boolean;
  isLast: boolean;
  onChanged(): void;
}) {
  const { client } = useStore();
  const [renaming, setRenaming] = useState(false);
  const [name, setName] = useState(a.nickname ?? '');
  const [confirmRemove, setConfirmRemove] = useState(false);

  const rename = useMutation({
    mutationFn: () =>
      client.patch('/store/v1/auth/authenticators/{authenticator_id}', {
        path: { authenticator_id: a.id },
        body: { nickname: name.trim() },
      }),
    onSuccess: () => {
      setRenaming(false);
      onChanged();
    },
  });
  const remove = useMutation({
    mutationFn: () => client.delete('/store/v1/auth/authenticators/{authenticator_id}', { path: { authenticator_id: a.id } }),
    onSuccess: onChanged,
  });

  return (
    <li className="p-3 flex flex-col gap-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="min-w-0">
          {renaming ? (
            <form
              className="flex items-center gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                if (name.trim()) rename.mutate();
              }}
            >
              <TextInput aria-label="Authenticator name" value={name} maxLength={80} onChange={(e) => setName(e.target.value)} autoFocus />
              <Button type="submit" size="sm" variant="primary" busy={rename.isPending} disabled={!name.trim()}>
                Save
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setRenaming(false)}>
                Cancel
              </Button>
            </form>
          ) : (
            <p className="text-sm font-medium text-fg flex items-center gap-2">
              {demoPlaceholder ? 'Demo sign-in' : a.nickname || 'Unnamed authenticator'}
              {usedNow && <StatusPill tone="blue">Used for this session</StatusPill>}
              {a.backup_state && <StatusPill tone="neutral">Synced passkey</StatusPill>}
            </p>
          )}
          {demoPlaceholder ? (
            <p className="text-xs text-fg-muted mt-0.5">
              Added {formatDateTime(a.created_at)} · Demo mode signs you in without a passkey, so there is nothing to tap.
            </p>
          ) : (
            <p className="text-xs text-fg-muted mt-0.5">
              Added {formatDateTime(a.created_at)} · Last used {formatRelative(a.last_used_at)}
              {a.transports.length > 0 && ` · ${a.transports.join(', ')}`}
            </p>
          )}
        </div>
        {!renaming && !demoPlaceholder && (
          <div className="flex gap-1">
            <Button size="sm" variant="ghost" icon={Pencil} onClick={() => setRenaming(true)}>
              Rename
            </Button>
            <Button size="sm" variant="ghost" icon={Trash2} onClick={() => setConfirmRemove(true)}>
              Remove
            </Button>
          </div>
        )}
      </div>
      {confirmRemove && (
        <div className="flex flex-col gap-2 rounded-md border border-primer-redBorder bg-primer-redSubtle p-3">
          <p className="text-sm text-primer-redFg">
            {isLast
              ? 'This is your only authenticator. Removing it locks you out: you would need an admin to re-invite you.'
              : 'This authenticator will no longer sign in to your account.'}
          </p>
          <div className="flex gap-2">
            <Button size="sm" variant="danger" busy={remove.isPending} onClick={() => remove.mutate()}>
              Remove authenticator
            </Button>
            <Button size="sm" variant="ghost" onClick={() => setConfirmRemove(false)}>
              Keep it
            </Button>
          </div>
        </div>
      )}
      {(rename.error || remove.error) && <Notice tone="red">{describeError(rename.error ?? remove.error)}</Notice>}
    </li>
  );
}

function SessionsCard() {
  const { client } = useStore();
  const { signOut } = useSession();
  const queryClient = useQueryClient();
  const q = useQuery({
    queryKey: queryKeys.ownSessions,
    queryFn: () => client.get('/store/v1/auth/sessions'),
  });
  const revoke = useMutation({
    mutationFn: (sessionId: string) => client.delete('/store/v1/auth/sessions/{session_id}', { path: { session_id: sessionId } }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: queryKeys.ownSessions }),
  });
  const items = q.data?.items ?? [];

  return (
    <Card
      title="Sessions"
      icon={Laptop}
      subtitle="Browsers signed in to this account."
      right={
        <Button size="sm" icon={LogOut} onClick={() => void signOut()}>
          Sign out
        </Button>
      }
    >
      {q.isLoading && <Loading />}
      <ErrorNotice error={q.error} />
      <ErrorNotice error={revoke.error} />
      {items.length > 0 && (
        <ul className="divide-y divide-border-muted border border-border-muted rounded-md">
          {items.map((s) => (
            <li key={s.session_id} className="p-3 flex flex-wrap items-center justify-between gap-2">
              <div className="min-w-0">
                <p className="text-sm text-fg flex items-center gap-2">
                  Signed in {formatDateTime(s.created_at)}
                  {s.current && <StatusPill tone="green">This browser</StatusPill>}
                </p>
                <p className="text-xs text-fg-muted">
                  Last active {formatRelative(s.last_seen_at)} · Expires {formatDateTime(s.idle_expires_at)} if idle, at the
                  latest {formatDateTime(s.absolute_expires_at)}
                </p>
              </div>
              {!s.current && (
                <Button
                  size="sm"
                  variant="danger"
                  busy={revoke.isPending && revoke.variables === s.session_id}
                  onClick={() => revoke.mutate(s.session_id)}
                >
                  Revoke
                </Button>
              )}
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}
