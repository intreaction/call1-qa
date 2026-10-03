import { Suspense, useEffect, useState, type ReactNode } from 'react';
import { useQuery } from '@tanstack/react-query';
import { KeyRound, RefreshCw, ServerCrash, X } from 'lucide-react';
import { useTheme } from '@/hooks/useTheme';
import { ContractMismatchError, checkContract, describeError, queryKeys, type StoreClient } from './api';
import { AdminPage } from './admin/AdminPage';
import { AccountPage } from './auth/AccountPage';
import { EnrollPage } from './auth/EnrollPage';
import { SignInPage } from './auth/SignInPage';
import { Header, PublicHeader } from './components/Header';
import { OfflineBanner } from './components/OfflineBanner';
import { Button, EmptyState, Loading, Notice } from './components/ui';
import { ChangeFeedProvider, SessionProvider, StoreProvider, useSession, useSessionState, useStore } from './state/app';
import { href, navigate, useRoute, type Route } from './state/router';
import { useIsDemoPersona } from './state/demo';
import { CallsView, EscalationsView, MetricsView, QueueView, RubricsView, SignalsView, WorkbenchView, type ViewProps } from './views';

/** Boot: check the contract major before anything else, then hand over to the session. */
export default function App({ client }: { client: StoreClient }) {
  const { theme, toggleTheme } = useTheme();
  const contract = useQuery({
    queryKey: queryKeys.contract,
    queryFn: () => checkContract(client),
    retry: (count, err) => !(err instanceof ContractMismatchError) && count < 2,
    staleTime: Infinity,
  });

  if (contract.isLoading) {
    return (
      <PublicLayout theme={theme} onToggleTheme={toggleTheme}>
        <Loading label="Connecting to Store…" />
      </PublicLayout>
    );
  }
  if (!contract.data) {
    const mismatch = contract.error instanceof ContractMismatchError;
    return (
      <PublicLayout theme={theme} onToggleTheme={toggleTheme}>
        <div className="max-w-md flex flex-col gap-3">
          <EmptyState icon={ServerCrash} title={mismatch ? 'Evaluate and Store do not match' : "Can't reach Store"}>
            {describeError(contract.error)}
            {mismatch && ' Update Evaluate and Store together; Evaluate will not run against a different contract major.'}
          </EmptyState>
          {!mismatch && (
            <div className="flex justify-center">
              <Button icon={RefreshCw} busy={contract.isFetching} onClick={() => void contract.refetch()}>
                Try again
              </Button>
            </div>
          )}
        </div>
      </PublicLayout>
    );
  }

  return (
    <StoreProvider client={client} contract={contract.data}>
      <SessionProvider>
        <Shell theme={theme} onToggleTheme={toggleTheme} />
      </SessionProvider>
    </StoreProvider>
  );
}

type ThemeProps = { theme: ReturnType<typeof useTheme>['theme']; onToggleTheme(): void };

function PublicLayout({ theme, onToggleTheme, children }: ThemeProps & { children: ReactNode }) {
  return (
    <div className="flex-1 flex flex-col min-h-0 bg-canvas">
      <PublicHeader theme={theme} onToggleTheme={onToggleTheme} />
      <OfflineBanner />
      <main className="flex-1 overflow-y-auto flex items-start justify-center px-4 py-10 sm:py-16">{children}</main>
    </div>
  );
}

function Shell(props: ThemeProps) {
  const route = useRoute();
  const { state, refresh } = useSessionState();

  // Enrollment works signed in or out: a new enrollment starts its own session.
  if (route.name === 'enroll') {
    return (
      <PublicLayout {...props}>
        <EnrollPage token={route.token} key={route.token ?? 'code'} />
      </PublicLayout>
    );
  }
  if (state.status === 'loading') {
    return (
      <PublicLayout {...props}>
        <Loading label="Checking your session…" />
      </PublicLayout>
    );
  }
  if (state.status === 'error') {
    return (
      <PublicLayout {...props}>
        <div className="max-w-md flex flex-col gap-3 items-center">
          <Notice tone="red">{state.message}</Notice>
          <Button icon={RefreshCw} onClick={() => void refresh()}>
            Try again
          </Button>
        </div>
      </PublicLayout>
    );
  }
  if (state.status === 'signed-out') {
    return (
      <PublicLayout {...props}>
        <SignInPage notice={state.notice} />
      </PublicLayout>
    );
  }
  return (
    <ChangeFeedProvider>
      <SignedInShell route={route} {...props} />
    </ChangeFeedProvider>
  );
}

function SignedInShell({ route, theme, onToggleTheme }: ThemeProps & { route: Route }) {
  const session = useSession();
  const { client, contract } = useStore();
  // A demo persona's only "authenticator" is Store's keyless placeholder, so the one-authenticator
  // nag would be noise in a class demo (and "Add a second one" is a real passkey ceremony).
  const demoPersona = useIsDemoPersona(session.session.email);

  useEffect(() => {
    if (route.name === 'sign-in') navigate({ name: 'calls' }, { replace: true });
  }, [route.name]);

  const viewProps: ViewProps = { client, contract, session, navigate };

  return (
    <div className="flex-1 flex flex-col min-h-0 bg-canvas">
      <Header route={route} theme={theme} onToggleTheme={onToggleTheme} />
      <OfflineBanner />
      {session.session.prompt_second_authenticator && !demoPersona && route.name !== 'account' && <SecondAuthenticatorBanner />}
      <main className="flex-1 overflow-y-auto p-4 sm:p-6">{renderRoute(route, viewProps)}</main>
    </div>
  );
}

function renderRoute(route: Route, props: ViewProps): ReactNode {
  switch (route.name) {
    case 'calls':
      return <CallsView {...props} filters={route.filters} key={JSON.stringify(route.filters ?? {})} />;
    case 'sign-in':
      return <CallsView {...props} />;
    case 'workbench':
      return <WorkbenchView {...props} callId={route.callId} turn={route.turn} key={route.callId} />;
    case 'rubrics':
      return <RubricsView {...props} rubricId={route.rubricId} />;
    case 'signals':
      return <SignalsView {...props} tab={route.tab} categoryId={route.categoryId} />;
    case 'queue':
      return <QueueView {...props} />;
    case 'escalations':
      return <EscalationsView {...props} />;
    case 'metrics':
      return <Suspense fallback={<Loading label="Loading metrics…" />}><MetricsView {...props} /></Suspense>;
    case 'account':
      return <AccountPage />;
    case 'admin':
      return <AdminPage section={route.section} />;
    case 'enroll':
      return null;
    case 'not-found':
      return (
        <EmptyState title="No such page">
          Nothing lives at <span className="break-all">{route.path}</span>.{' '}
          <a className="text-primer-blueFg hover:underline" href={href({ name: 'calls' })}>
            Go to calls
          </a>
        </EmptyState>
      );
  }
}

function SecondAuthenticatorBanner() {
  const [hidden, setHidden] = useState(false);
  if (hidden) return null;
  return (
    <div className="shrink-0 flex items-center gap-2 px-4 py-2 text-sm border-b border-border bg-canvas-subtle" role="status">
      <KeyRound className="w-4 h-4 text-primer-yellowFg shrink-0" aria-hidden="true" />
      <span className="text-fg-muted flex-1">
        You have one authenticator. If you lose it, an admin has to re-invite you.{' '}
        <a className="text-primer-blueFg hover:underline" href={href({ name: 'account' })}>
          Add a second one
        </a>
      </span>
      <button
        type="button"
        onClick={() => setHidden(true)}
        className="w-6 h-6 rounded hover:bg-canvas text-fg-muted flex items-center justify-center focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
        aria-label="Hide for now"
        title="Hide for now"
      >
        <X className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}
