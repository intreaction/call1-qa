import { useState, type FormEvent } from 'react';
import { useInfiniteQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { MailPlus, UserPlus } from 'lucide-react';
import { queryKeys, type Invitation, type InvitationIssued, type InvitationStatus, type ReviewerRole } from '../api';
import { Button, Card, EmptyState, ErrorNotice, Field, Loading, OneTimeSecret, SelectInput, StatusPill, TextInput, formatDateTime } from '../components/ui';
import { useStore } from '../state/app';
import { ROLES, roleLabel } from './common';

const STATUS_TONE: Record<InvitationStatus, 'green' | 'yellow' | 'red' | 'neutral'> = {
  pending: 'yellow',
  redeemed: 'green',
  revoked: 'red',
  expired: 'neutral',
};
const STATUS_LABEL: Record<InvitationStatus, string> = {
  pending: 'Pending',
  redeemed: 'Redeemed',
  revoked: 'Revoked',
  expired: 'Expired',
};

export function InvitationsPanel() {
  const [prefill, setPrefill] = useState<Invitation | null>(null);
  return (
    <div className="flex flex-col gap-4">
      <IssueInvitation prefill={prefill} key={prefill?.id ?? 'new'} />
      <InvitationList onIssueAgain={setPrefill} />
    </div>
  );
}

function IssueInvitation({ prefill }: { prefill: Invitation | null }) {
  const { client, contract } = useStore();
  const queryClient = useQueryClient();
  const [email, setEmail] = useState(prefill?.email ?? '');
  const [name, setName] = useState(prefill?.display_name ?? '');
  const [role, setRole] = useState<ReviewerRole>(prefill?.role ?? 'reviewer');
  const [issued, setIssued] = useState<InvitationIssued | null>(null);
  const days = Math.round(contract.parameters.invitation_lifetime_seconds / 86400);

  const m = useMutation({
    mutationFn: () =>
      client.post('/store/v1/admin/invitations', {
        body: { email: email.trim(), display_name: name.trim(), role, delivery: 'out_of_band', reinvite_of_account_id: null },
      }),
    onSuccess: (res) => {
      setIssued(res);
      setEmail('');
      setName('');
      setRole('reviewer');
      void queryClient.invalidateQueries({ queryKey: queryKeys.admin.invitations });
    },
  });

  const submit = (e: FormEvent) => {
    e.preventDefault();
    setIssued(null);
    m.mutate();
  };

  return (
    <Card
      title="Invite a reviewer"
      icon={UserPlus}
      subtitle={`The sign-up link is shown once. Send it to the reviewer yourself. Links expire after ${days} day${days === 1 ? '' : 's'}.`}
    >
      <form onSubmit={submit} className="grid grid-cols-1 sm:grid-cols-[1fr_1fr_10rem_auto] gap-3 items-end">
        <Field label="Email">
          {(id) => <TextInput id={id} type="email" required value={email} onChange={(e) => setEmail(e.target.value)} />}
        </Field>
        <Field label="Display name">
          {(id) => <TextInput id={id} required maxLength={120} value={name} onChange={(e) => setName(e.target.value)} />}
        </Field>
        <Field label="Role">
          {(id) => (
            <SelectInput id={id} value={role} onChange={(e) => setRole(e.target.value as ReviewerRole)}>
              {ROLES.map((r) => (
                <option key={r} value={r}>
                  {roleLabel(r)}
                </option>
              ))}
            </SelectInput>
          )}
        </Field>
        <Button type="submit" variant="primary" icon={MailPlus} busy={m.isPending} disabled={!email.trim() || !name.trim()}>
          Issue link
        </Button>
      </form>
      <p className="text-xs text-fg-muted mt-2">
        Call1 does not email invitations yet, so copy the link and send it yourself. To replace a lost passkey or security
        key, use Re-invite on the account instead.
      </p>
      <div className="mt-3 flex flex-col gap-2">
        <ErrorNotice error={m.error} />
        {issued?.invitation_url && (
          <OneTimeSecret label="Invitation link" value={issued.invitation_url} onDismiss={() => setIssued(null)}>
            Send it to {issued.invitation.email} through a channel you trust. Anyone with the link can enroll as{' '}
            {issued.invitation.display_name}. It expires {formatDateTime(issued.invitation.expires_at)}.
          </OneTimeSecret>
        )}
      </div>
    </Card>
  );
}

function InvitationList({ onIssueAgain }: { onIssueAgain(inv: Invitation): void }) {
  const { client } = useStore();
  const queryClient = useQueryClient();
  const [status, setStatus] = useState<InvitationStatus | ''>('pending');
  const q = useInfiniteQuery({
    queryKey: [...queryKeys.admin.invitations, status],
    initialPageParam: null as string | null,
    queryFn: ({ pageParam }) =>
      client.get('/store/v1/admin/invitations', { query: { limit: 100, page_token: pageParam, status: status || null } }),
    getNextPageParam: (last) => last.next_page_token,
  });
  const revoke = useMutation({
    mutationFn: (id: string) => client.delete('/store/v1/admin/invitations/{invitation_id}', { path: { invitation_id: id } }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: queryKeys.admin.invitations }),
  });
  const items = q.data?.pages.flatMap((p) => p.items) ?? [];

  return (
    <Card
      title="Invitations"
      icon={MailPlus}
      right={
        <SelectInput aria-label="Filter by status" value={status} onChange={(e) => setStatus(e.target.value as InvitationStatus | '')} className="w-36">
          <option value="pending">Pending</option>
          <option value="redeemed">Redeemed</option>
          <option value="revoked">Revoked</option>
          <option value="expired">Expired</option>
          <option value="">All</option>
        </SelectInput>
      }
    >
      {q.isLoading && <Loading />}
      <ErrorNotice error={q.error ?? revoke.error} />
      {q.isSuccess && items.length === 0 && <EmptyState icon={MailPlus} title={status ? `No ${STATUS_LABEL[status].toLowerCase()} invitations` : 'No invitations'} />}
      {items.length > 0 && (
        <ul className="divide-y divide-border-muted border border-border-muted rounded-md">
          {items.map((inv) => (
            <li key={inv.id} className="p-3 flex flex-wrap items-center gap-x-3 gap-y-1">
              <span className="text-sm font-medium text-fg">{inv.display_name}</span>
              <span className="text-xs text-fg-muted break-all">{inv.email}</span>
              <span className="text-xs text-fg-muted">{roleLabel(inv.role)}</span>
              {inv.reinvite_of_account_id && <StatusPill tone="neutral">Re-invite</StatusPill>}
              <StatusPill tone={STATUS_TONE[inv.status] ?? 'neutral'}>{STATUS_LABEL[inv.status] ?? inv.status}</StatusPill>
              <span className="flex-1" />
              <span className="text-xs text-fg-muted">
                Issued {formatDateTime(inv.issued_at)} ·{' '}
                {inv.redeemed_at ? `redeemed ${formatDateTime(inv.redeemed_at)}` : `expires ${formatDateTime(inv.expires_at)}`}
              </span>
              {inv.status === 'pending' && (
                <Button size="sm" variant="danger" busy={revoke.isPending && revoke.variables === inv.id} onClick={() => revoke.mutate(inv.id)}>
                  Revoke
                </Button>
              )}
              {(inv.status === 'expired' || inv.status === 'revoked') && !inv.reinvite_of_account_id && (
                <Button size="sm" onClick={() => onIssueAgain(inv)}>
                  Issue again
                </Button>
              )}
            </li>
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
