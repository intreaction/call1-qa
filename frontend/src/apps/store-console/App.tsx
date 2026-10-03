import { useTheme } from '@/hooks/useTheme';
import ConsoleHeader from './components/ConsoleHeader';
import HealthPanel from './components/HealthPanel';
import CoveragePanel from './components/CoveragePanel';
import InstallationsPanel from './components/InstallationsPanel';
import ChangeFeedPanel from './components/ChangeFeedPanel';
import SearchEmbedderPanel from './components/SearchEmbedderPanel';

export default function App() {
  const { theme, toggleTheme } = useTheme();

  return (
    <div className="flex-1 flex flex-col min-h-0 bg-canvas">
      <ConsoleHeader theme={theme} onToggleTheme={toggleTheme} />
      <main className="flex-1 overflow-y-auto p-4">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 max-w-5xl mx-auto">
          <HealthPanel />
          <CoveragePanel />
          <InstallationsPanel />
          <ChangeFeedPanel />
          <SearchEmbedderPanel />
        </div>
      </main>
    </div>
  );
}
