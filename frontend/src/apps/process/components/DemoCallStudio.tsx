import { useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { Play } from 'lucide-react';
import { processDemoCall } from '../api';
import { Button, ErrorNotice } from './ui';

export function DemoCallStudio({ token, onStarted }: { token: string | null; onStarted(conversationId: string): void }) {
  const client = useQueryClient();
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<unknown>(null);
  async function start() {
    setSubmitting(true); setError(null);
    try {
      const result = await processDemoCall(token);
      await client.invalidateQueries({ queryKey: ['process-conversations'] });
      onStarted(result.conversation_id);
    } catch (err) { setError(err); }
    finally { setSubmitting(false); }
  }
  return <section className="rounded-lg border border-border-muted bg-canvas-subtle p-5 space-y-4" data-testid="demo-call-studio" aria-label="Demo call">
    <div className="flex flex-wrap items-center justify-between gap-4">
      <div><p className="text-xs uppercase tracking-widest text-primer-blueFg mb-1">Try it live</p><h2 className="text-xl font-semibold text-fg">Process a sample call</h2><p className="text-sm text-fg-muted mt-1">A 16-second AppTek stock inquiry. Follow real processing in Pipeline.</p></div>
      <Button icon={Play} variant="primary" disabled={!token || submitting} busy={submitting} onClick={() => void start()}>Process demo call</Button>
    </div>
    <div className="flex flex-wrap justify-between items-center gap-3">
      <p className="text-sm text-fg-muted">“Do you have another option if it’s out of stock?”</p>
      <audio controls preload="none" aria-label="Listen to the AppTek demo excerpt" className="h-9 max-w-full" src="/demo/apptek-retail-short.wav" />
    </div>
    <p className="text-xs text-fg-subtle">© AppTek · <a href="https://huggingface.co/datasets/apptek-com/apptek_callcenter_dialogues" target="_blank" rel="noreferrer" className="underline">Call-Center Dialogues</a> · <a href="https://creativecommons.org/licenses/by-sa/4.0/" target="_blank" rel="noreferrer" className="underline">CC BY-SA 4.0</a> · Role-played sample, shortened by Call1 (01:22–01:37.5). Every take processes afresh. First-run model loading can take longer.</p>
    <ErrorNotice error={error} />
  </section>;
}
