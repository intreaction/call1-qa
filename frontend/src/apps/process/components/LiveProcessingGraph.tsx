import { useEffect, useId, useState, type ReactNode } from 'react';
import { Check, Clock3, Loader2, TriangleAlert, ChevronDown } from 'lucide-react';
import type { ConversationDetail, ConversationJob } from '../api';

export const PROCESSING_STAGES = [
  { id: 'audio', title: 'Recording', caption: 'Validate the audio', types: ['validation_vad'] },
  { id: 'transcript', title: 'Transcribe', caption: 'Words and speakers', types: ['asr', 'speaker_attribution', 'transcript_merge', 'asr_vocabulary_pass', 'asr_vocabulary_apply'] },
  { id: 'privacy', title: 'Protect', caption: 'Mask personal details', types: ['enrichment'] },
  { id: 'search', title: 'Index', caption: 'Make the call searchable', types: ['embeddings'] },
  { id: 'tone', title: 'Tone & sentiment', caption: 'Emotion and language', types: ['acoustic_tone', 'text_sentiment'] },
  { id: 'qa', title: 'Score the call', caption: 'Your rubric questions', types: ['qa_criterion', 'qa_deterministic', 'qa_escalation', 'qa_scorecard'] },
  { id: 'signals', title: 'Find signals', caption: 'Intent, issues, outcomes', types: ['contact_signals_categorize', 'contact_signals_subcategorize', 'contact_signals_extract', 'contact_signals_merge', 'contact_signals_lifecycle', 'contact_signals_resolution', 'contact_signals'] },
  { id: 'summary', title: 'Summarize', caption: 'The conversation at a glance', types: ['summary_segment', 'summary_synthesis', 'summary_assembly'] },
];

/** A compact view of conceptual stages. All counts and status come from actual jobs. */
export function LiveProcessingGraph({ detail, startedAt, active = false, renderJobs }: { detail?: ConversationDetail; startedAt: number | null; active?: boolean; compact?: boolean; renderJobs?(jobs: ConversationJob[]): ReactNode }) {
  const [now, setNow] = useState(Date.now());
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const regionId = useId();
  const settled = detail?.progress?.settled ?? false;
  useEffect(() => {
    if (!active || settled) return;
    const timer = window.setInterval(() => setNow(Date.now()), 500);
    return () => window.clearInterval(timer);
  }, [active, settled]);
  const jobs = detail?.jobs ?? [];
  const knownTypes = new Set(PROCESSING_STAGES.flatMap((stage) => stage.types));
  const remaining = [...new Set(jobs.filter((job) => !knownTypes.has(job.job_type)).map((job) => job.job_type))];
  const stages = remaining.length ? [...PROCESSING_STAGES, { id: 'other', title: 'Additional processing', caption: 'Supporting tasks', types: remaining }] : PROCESSING_STAGES;
  const allExpanded = stages.every((stage) => expanded.has(stage.id));
  const failed = jobs.some((j) => ['FAILED', 'CANCELLED'].includes(j.status));
  const done = jobs.filter((j) => j.status === 'SUCCEEDED').length;
  const ends = settled ? jobs.map((j) => Date.parse(j.ended_at ?? '')).filter(Number.isFinite) : [];
  const end = ends.length ? Math.max(...ends) : now;
  const elapsed = startedAt ? Math.max(0, Math.floor((end - startedAt) / 1000)) : 0;
  return <section className="rounded-lg border border-border-muted bg-canvas overflow-hidden" aria-label="Live processing" data-testid="live-processing-graph">
    <div className="px-5 py-4 flex justify-between items-center gap-4 border-b border-border-muted">
      <div><p className="text-[10px] font-medium uppercase tracking-[.16em] text-fg-subtle mb-1">Call processing</p>
        <h2 className="text-base font-semibold text-fg">{settled ? failed ? 'Review needed' : 'Your call is ready.' : 'Processing call'}</h2>
        <p className="text-xs text-fg-muted mt-1">{detail?.progress_stale ? 'Last known status · Store unavailable' : `${done} of ${jobs.length || '…'} jobs complete`}</p>
      </div>
      <div className="flex flex-wrap justify-end items-center gap-2">
      <button type="button" className="text-xs font-medium text-primer-blueFg px-2 py-2 rounded-md hover:bg-canvas-subtle focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue" onClick={() => setExpanded(allExpanded ? new Set() : new Set(stages.map((stage) => stage.id)))}>{allExpanded ? 'Hide all details' : 'Show all details'}</button>
      <div className="flex items-center gap-2 text-fg-muted rounded-md border border-border-muted px-3 py-2"><Clock3 size={14} aria-hidden="true" /><span className="text-sm tabular-nums font-medium text-fg">{String(Math.floor(elapsed / 60)).padStart(2, '0')}:{String(elapsed % 60).padStart(2, '0')}</span><span className="sr-only">elapsed</span></div></div>
    </div>
    <div className="h-1 bg-canvas-inset" role="progressbar" aria-label="Processing jobs complete" aria-valuemin={0} aria-valuemax={jobs.length || 1} aria-valuenow={done}><div className={`h-full motion-safe:transition-[width] duration-500 ${failed ? 'bg-primer-red' : 'bg-primer-green'}`} style={{ width: `${jobs.length ? done / jobs.length * 100 : 0}%` }} /></div>
    <ol className="px-5 py-3 max-h-[480px] overflow-y-auto" aria-label="Processing stages">
      {stages.map((stage, index) => {
        const group = jobs.filter((j) => stage.types.includes(j.job_type));
        const completed = group.filter((j) => j.status === 'SUCCEEDED').length;
        const error = group.some((j) => ['FAILED', 'CANCELLED'].includes(j.status));
        const ready = group.length > 0 && completed === group.length;
        const running = group.some((j) => ['RUNNING', 'WAITING_PROVIDER'].includes(j.status));
        const label = error ? 'Needs attention' : ready ? 'Complete' : running ? 'Processing' : settled && !group.length ? 'Not enabled' : 'Waiting';
        const Icon = error ? TriangleAlert : ready ? Check : running ? Loader2 : null;
        const color = error ? 'text-primer-redFg' : ready ? 'text-primer-greenFg' : running ? 'text-primer-blueFg' : 'text-fg-subtle';
        return <li key={stage.id} className="relative" data-testid="processing-stage">
          {index < stages.length - 1 && <span className={`absolute left-[15px] top-10 bottom-[-10px] w-px ${ready ? 'bg-primer-greenBorder' : 'bg-border-muted'}`} aria-hidden="true" />}
          <button type="button" aria-label={`Inspect ${stage.title} jobs`} aria-expanded={expanded.has(stage.id)} aria-controls={`${regionId}-${stage.id}`} onClick={() => setExpanded((current) => { const next = new Set(current); if (next.has(stage.id)) next.delete(stage.id); else next.add(stage.id); return next; })} className={`relative w-full flex items-center gap-3 py-2.5 text-left rounded-md focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue transition-colors hover:bg-canvas-subtle ${expanded.has(stage.id) ? 'bg-canvas-subtle' : ''}`}>
            <span className={`h-8 w-8 rounded-full border flex items-center justify-center shrink-0 bg-canvas ${running ? 'border-primer-blueBorder bg-primer-blueSubtle' : ready ? 'border-primer-greenBorder bg-primer-greenSubtle' : 'border-border-muted'} ${color}`}>
              {Icon ? <Icon size={14} className={running ? 'motion-safe:animate-spin' : ''} aria-hidden="true" /> : <span className="text-[10px] tabular-nums">{String(index + 1).padStart(2, '0')}</span>}
            </span>
            <span className="flex-1 min-w-0"><span className="block text-sm font-medium text-fg">{stage.title}</span><span className="block text-xs text-fg-subtle mt-0.5">{stage.caption}</span></span>
            <span className={`text-xs shrink-0 ${color}`}>{label}</span>
            <ChevronDown size={13} className={`text-fg-subtle mr-2 shrink-0 motion-safe:transition-transform ${expanded.has(stage.id) ? "" : "-rotate-90"}`} aria-hidden="true" />
          </button>
          {expanded.has(stage.id) && <div id={`${regionId}-${stage.id}`} className="relative ml-11 mb-3" role="region" aria-label={`${stage.title} details`}>
            {group.length ? renderJobs?.(group) : <p className="text-xs text-fg-subtle py-2">No jobs for this step.</p>}
          </div>}
        </li>;
      })}
    </ol>
    <p className="px-5 py-2.5 border-t border-border-muted text-[11px] text-fg-subtle">Live job status · analysis tasks can run in parallel</p>
    <div className="sr-only" aria-live="polite">{settled ? failed ? 'Processing finished with errors. Review needed.' : 'Processing complete. Your call is ready.' : `${done} jobs complete.`}</div>
  </section>;
}
