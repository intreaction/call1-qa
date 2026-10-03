import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useTheme } from '@/hooks/useTheme';
import { getSession, queryKeys } from './api';
import { Header, type Tab } from './components/Header';
import OverviewView from './views/OverviewView';
import ImportView from './views/ImportView';
import PipelineView from './views/PipelineView';
import ModelsView from './views/ModelsView';
import SettingsView from './views/SettingsView';
import { useConsoleToken } from './useConsoleToken';
import { ViewBoundary } from './components/ViewBoundary';

const TAB_NAMES: Record<Tab, string> = { overview: 'Overview', import: 'Import', pipeline: 'Pipeline', models: 'Models', settings: 'Settings' };

export default function App() {
  const { theme, toggleTheme } = useTheme();
  const { token, setToken } = useConsoleToken();
  const [focusConversation, setFocusConversation] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>('overview');

  const session = useQuery({
    queryKey: queryKeys.session(token),
    queryFn: () => getSession(token),
    retry: false,
    refetchInterval: 15_000,
  });

  // Once the session check resolves, a token it reports as invalid (rotated elsewhere, expired)
  // stops being sent as if it worked — otherwise every write would fail with a confusing 401
  // after the button looked enabled. While the check is still in flight, send it optimistically.
  const writableToken = session.data && !session.data.token_valid ? null : token;

  const connection = session.data
    ? session.data.token_valid
      ? ({ tone: 'green', label: 'Console connected' } as const)
      : session.data.console_credential_configured
        ? ({ tone: 'yellow', label: 'Read only' } as const)
        : ({ tone: 'yellow', label: 'No token issued' } as const)
    : session.isError
      ? ({ tone: 'red', label: "Can't reach Process" } as const)
      : null;

  return (
    <div className="flex-1 flex flex-col min-h-0 bg-canvas">
      <Header active={tab} onNavigate={setTab} theme={theme} onToggleTheme={toggleTheme} connection={connection} />
      <main className="flex-1 overflow-y-auto p-4">
        <ViewBoundary key={tab} name={TAB_NAMES[tab]}>
          {tab === 'overview' && <OverviewView />}
          {tab === 'import' && (
            <ImportView onDemoStarted={(id) => { setFocusConversation(id); setTab('pipeline'); }} demo={session.data?.demo ?? false} token={writableToken} tokenConfigured={session.data?.console_credential_configured} onToken={setToken} />
          )}
          {tab === 'pipeline' && (
            <PipelineView focusConversation={focusConversation} token={writableToken} tokenConfigured={session.data?.console_credential_configured} onToken={setToken} />
          )}
          {tab === 'models' && <ModelsView demo={session.data?.demo ?? false} onSettings={() => setTab('settings')} />}
          {tab === 'settings' && (
            <SettingsView demo={session.data?.demo ?? false} token={writableToken} tokenConfigured={session.data?.console_credential_configured} onToken={setToken} />
          )}
        </ViewBoundary>
      </main>
    </div>
  );
}
