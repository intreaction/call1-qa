import { useState } from 'react';
import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { ChevronDown, ChevronRight, Users } from 'lucide-react';
import { describeError, queryKeys, type AccountStatus, type InvitationIssued, type ReviewerAccount, type ReviewerRole } from '../api';
import { Button, Card, EmptyState, ErrorNotice, Field, Loading, Notice, OneTimeSecret, SelectInput, StatusPill, formatRelative } from '../components/ui';
import { useSession, useStore } from '../state/app';
import { ROLES, ReasonAction, roleLabel } from './common';

const STATUS_DISPLAY: Record<AccountStatus, { label: string; tone: 'green' | 'yellow' | 'red' | 'neutral' }> = {
  active: { label: 'Active', tone: 'green' },
  pending_enrollment: { label: 'Pending enrollment', tone: 'yellow' },
  reinvite_required: { label: 'Re-invite required', tone: 'yellow' },
  disabled: { label: 'Disabled', tone: 'red' },
};

export function AccountsPanel() {
  const { client } = useStore();
  const [status, setStatus] = useState<AccountStatus | ''>('');
  const q = useInfiniteQuery({
    queryKey: [...queryKeys.admin.accounts, status],
    initialPageParam: null as string | null,
    queryFn: ({ pageParam }) =>
      client.get('/store/v1/admin/accounts', { query: { limit: 100, page_token: pageParam, status: status || null } }),
    getNextPageParam: (last) => last.next_page_token,
  });
  const accounts = q.data?.pages.flatMap((p) => p.items) ?? [];

  return (
    <Card
      title="Accounts"
      icon={Users}
      subtitle="Reviewer accounts. Accounts are created when an invitation or setup code is redeemed."
      right={
        <SelectInput aria-label="Filter by status" value={status} onChange={(e) => setStatus(e.target.value as AccountStatus | '')} className="w-44">
          <option value="">All statuses</option>
          <option value="active">Active</option>
          <option value="reinvite_required">Re-invite required</option>
          <option value="disabled">Disabled</option>
        </SelectInput>
      }
    >
      {q.isLoading && <Loading />}
      <ErrorNotice error={q.error} />
      {q.isSuccess && accounts.length === 0 && (
        <EmptyState icon={Users} title="No accounts">
          Invite reviewers from the Invitations tab.
        </EmptyState>
      )}
      {accounts.length > 0 && (
        <ul className="divide-y divide-border-muted border border-border-muted rounded-md">
          {accounts.map((a) => (
            <AccountRow key={a.id} account={a} />
          ))}
        </ul>
      )}
      {q.hasNextPage && (
        <div className="mt-3">
          <Button size="sm" busy={q.isFetchingNextPage} onClick={() => void q.fetchNextPage()}>
            Load more
          </Button>
        </div>
      )}
    </Card>
  );
}

function AccountRow({ account: a }: { account: ReviewerAccount }) {
  const { client } = useStore();
  const { session } = useSession();
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [role, setRole] = useState<ReviewerRole>(a.role);
  const [reinvite, setReinvite] = useState<InvitationIssued | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const isSelf = a.id === session.account_id;
  const st = STATUS_DISPLAY[a.status] ?? { label: a.status, tone: 'neutral' as const };

  const refresh = () => {
    void queryClient.invalidateQueries({ queryKey: queryKeys.admin.accounts });
    void queryClient.invalidateQueries({ queryKey: queryKeys.admin.invitations });
  };

  const update = (body: { role?: ReviewerRole; status?: 'active' | 'disabled'; reason: string }) =>
    client.patch('/store/v1/admin/accounts/{account_id}', { path: { account_id: a.id }, body }).then((r) => {
      refresh();
      return r;
    });

  const revokeSessions = useMutation({
    mutationFn: () => client.delete('/store/v1/admin/accounts/{account_id}/sessions', { path: { account_id: a.id } }),
    onSuccess: () => setNotice('Every session of this account was revoked.'),
  });

  return (
    <li className="p-3 flex flex-col gap-3">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="w-full text-left flex flex-wrap items-center gap-x-3 gap-y-1 rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
      >
        {open ? <ChevronDown className="w-4 h-4 text-fg-subtle" aria-hidden="true" /> : <ChevronRight className="w-4 h-4 text-fg-subtle" aria-hidden="true" />}
        <span className="text-sm font-medium text-fg">{a.display_name}</span>
        {isSelf && <StatusPill tone="blue">You</StatusPill>}
        <span className="text-xs text-fg-muted break-all">{a.email}</span>
        <span className="flex-1" />
        <span className="text-xs text-fg-muted">{roleLabel(a.role)}</span>
        <StatusPill tone={st.tone}>{st.label}</StatusPill>
        <span className="text-xs text-fg-muted">
          {a.authenticator_count} authenticator{a.authenticator_count === 1 ? '' : 's'} · last sign-in {formatRelative(a.last_sign_in_at)}
        </span>
      </button>

      {open && (
        <div className="pl-6 flex flex-col gap-3">
          {notice && <Notice tone="green">{notice}</Notice>}
          <div className="flex flex-wrap items-end gap-2">
            <Field label="Role">
              {(id) => (
                <SelectInput id={id} value={role} onChange={(e) => setRole(e.target.value as ReviewerRole)} className="w-40">
                  {ROLES.map((r) => (
                    <option key={r} value={r}>
                      {roleLabel(r)}
                    </option>
                  ))}
                </SelectInput>
              )}
            </Field>
            {role !== a.role && (
              <ReasonAction
                label={`Change role to ${role}`}
                warning={`${a.display_name} becomes ${role} from their next request.`}
                onConfirm={(reason) => update({ role, reason })}
              />
            )}
          </div>
          <div className="flex flex-wrap gap-2 items-start">
            {a.status === 'disabled' ? (
              <ReasonAction label="Enable account" onConfirm={(reason) => update({ status: 'active', reason })} />
            ) : (
              <ReasonAction
                label="Disable account"
                variant="danger"
                warning="Disabling signs the account out everywhere and revokes its pending re-invites."
                onConfirm={(reason) => update({ status: 'disabled', reason })}
              />
            )}
            <Button size="sm" busy={revokeSessions.isPending} onClick={() => revokeSessions.mutate()}>
              Revoke all sessions
            </Button>
            <ReinviteButton account={a} onIssued={(issued) => { setReinvite(issued); refresh(); }} />
          </div>
          <ErrorNotice error={revokeSessions.error} />
          {reinvite?.invitation_url && (
            <OneTimeSecret label="Re-invitation link" value={reinvite.invitation_url} onDismiss={() => setReinvite(null)}>
              Send it to {reinvite.invitation.email} through a channel you trust (not this app). Their old authenticators and
              sessions are already revoked; enrolling from this link gives them new ones. It expires{' '}
              {new Date(reinvite.invitation.expires_at).toLocaleString()}.
            </OneTimeSecret>
          )}
          <AccountAuthenticators accountId={a.id} />
        </div>
      )}
    </li>
  );
}

function ReinviteButton({ account: a, onIssued }: { account: ReviewerAccount; onIssued(issued: InvitationIssued): void }) {
  const { client } = useStore();
  const [confirm, setConfirm] = useState(false);
  const m = useMutation({
    mutationFn: () =>
      client.post('/store/v1/admin/invitations', {
        body: {
          email: a.email,
          display_name: a.display_name,
          role: a.role,
          delivery: 'out_of_band',
          reinvite_of_account_id: a.id,
        },
      }),
    onSuccess: (issued) => {
      setConfirm(false);
      onIssued(issued);
    },
  });
  if (!confirm) {
    return (
      <Button size="sm" onClick={() => setConfirm(true)}>
        Re-invite…
      </Button>
    );
  }
  return (
    <div className="w-full flex flex-col gap-2 rounded-md border border-primer-yellowBorder bg-primer-yellowSubtle p-3">
      <p className="text-sm text-primer-yellowFg">
        A re-invite revokes every authenticator and session of {a.display_name} now. Use it when they lost their
        authenticators. The new link is shown once.
      </p>
      {m.error != null && <Notice tone="red">{describeError(m.error)}</Notice>}
      <div className="flex gap-2">
        <Button size="sm" variant="danger" busy={m.isPending} onClick={() => m.mutate()}>
          Revoke and issue re-invite
        </Button>
        <Button size="sm" variant="ghost" onClick={() => setConfirm(false)}>
          Cancel
        </Button>
      </div>
    </div>
  );
}

function AccountAuthenticators({ accountId }: { accountId: string }) {
  const { client } = useStore();
  const queryClient = useQueryClient();
  const q = useQuery({
    queryKey: queryKeys.admin.accountAuthenticators(accountId),
    queryFn: () => client.get('/store/v1/admin/accounts/{account_id}/authenticators', { path: { account_id: accountId } }),
  });
  const revoke = useMutation({
    mutationFn: (authenticatorId: string) =>
      client.delete('/store/v1/admin/accounts/{account_id}/authenticators/{authenticator_id}', {
        path: { account_id: accountId, authenticator_id: authenticatorId },
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.admin.accounts });
    },
  });
  const items = q.data?.items ?? [];
  return (
    <div className="flex flex-col gap-2">
      <h3 className="text-xs font-medium text-fg-muted">Authenticators</h3>
      {q.isLoading && <Loading />}
      <ErrorNotice error={q.error ?? revoke.error} />
      {q.isSuccess && items.length === 0 && <p className="text-sm text-fg-muted">None. The account needs a re-invite to sign in.</p>}
      {items.length > 0 && (
        <ul className="flex flex-col gap-1">
          {items.map((c) => (
            <li key={c.id} className="flex flex-wrap items-center justify-between gap-2 text-sm">
              <span className="text-fg">
                {c.nickname || 'Unnamed authenticator'}{' '}
                <span className="text-xs text-fg-muted">· last used {formatRelative(c.last_used_at)}</span>
              </span>
              <Button
                size="sm"
                variant="danger"
                busy={revoke.isPending && revoke.variables === c.id}
                onClick={() => revoke.mutate(c.id)}
                title={items.length === 1 ? 'Revoking the last authenticator makes the account need a re-invite' : undefined}
              >
                Revoke{items.length === 1 ? ' (last one)' : ''}
              </Button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
