import { WifiOff } from 'lucide-react';
import { useConnectivity } from '../state/connectivity';

/** Shown under the header while the browser is offline or Store stops answering. */
export function OfflineBanner() {
  const c = useConnectivity();
  if (!c.offline) return null;
  const text = !c.browserOnline
    ? 'You are offline. What you see may be out of date, and changes cannot be saved until the connection returns.'
    : "Can't reach Store. What you see may be out of date; Evaluate keeps retrying.";
  return (
    <div
      role="status"
      aria-live="polite"
      className="shrink-0 flex items-center gap-2 px-4 py-2 text-sm border-b border-primer-yellowBorder bg-primer-yellowSubtle text-primer-yellowFg"
    >
      <WifiOff className="w-4 h-4 shrink-0" aria-hidden="true" />
      <span className="font-medium">Offline</span>
      <span className="text-fg-muted">{text}</span>
    </div>
  );
}
