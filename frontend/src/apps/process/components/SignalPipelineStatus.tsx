import { useQuery } from '@tanstack/react-query';
import { Workflow } from 'lucide-react';
import { getSignalFirstPass, queryKeys } from '../api';
import { Card, ErrorNotice } from './ui';

export function SignalPipelineStatus() {
  const query = useQuery({ queryKey: queryKeys.signalFirstPass, queryFn: getSignalFirstPass, refetchInterval: 15_000 });
  return (
    <Card title="Contact Signals pipeline" icon={Workflow} subtitle="Semantic similarity → Laya → Gemma">
      <div className="space-y-3" data-testid="signal-pipeline-status">
        <p className="text-sm text-fg-muted">Semantic examples find candidates. Laya triages them. Strong agreement can keep a candidate directly; Gemma confirms every uncertain candidate and extracts evidence.</p>
        <p className="text-xs text-fg-muted">The shortcut requires a Laya score of at least 0.95, strong category and subcategory example votes, and a close semantic neighbour. Scores are experimental and are not calibrated accuracy probabilities.</p>
        {query.data?.reason && <p className="text-xs text-fg-muted">{query.data.reason}</p>}
        {query.data?.engine === 'fake' && <p className="text-xs text-fg-muted">Fake handlers are active; model inference is simulated.</p>}
        <ErrorNotice error={query.error} />
        <p className="text-xs text-fg-muted">New analyses use this process. Existing results require reanalysis.</p>
      </div>
    </Card>
  );
}
