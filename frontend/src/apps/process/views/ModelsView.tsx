import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { ArrowDown, ArrowRight, AudioLines, Boxes, RefreshCw, Settings2 } from 'lucide-react';
import { getCatalog, getTraining, queryKeys, type CatalogEntryDescribe, type TrainingActiveModel } from '../api';
import { Button, Card, ErrorNotice, humanize, Loading, Notice, PageHeader, StatusPill, type Tone } from '../components/ui';
import { useDemoStack } from '../components/FineTuneExperience';
import { handlerModeLabel, jobTypeLabel, routeLabel, runtimeLabel, taskLabel } from '../labels';

/** The Models view's cards, in pipeline order. A card lists every entry tagged with any of its
 * purposes; `primary` is the purpose whose default earns the "Default" pill. Contact Signals v2's
 * three stage purposes share one card with the v1 purpose, because one model serves them all. */
const GROUPS: Array<{ key: string; label: string; purposes: string[]; primary: string; subtitle?: string }> = [
  { key: 'asr', label: 'Speech-to-text', purposes: ['asr'], primary: 'asr', subtitle: 'Writes the transcript' },
  {
    key: 'asr_vocabulary',
    label: 'Vocabulary pass (dual transcription)',
    purposes: ['asr_vocabulary'],
    primary: 'asr_vocabulary',
    subtitle: 'Catches your product and brand terms',
  },
  { key: 'speaker_diarization', label: 'Speaker diarization', purposes: ['speaker_diarization'], primary: 'speaker_diarization', subtitle: 'Who spoke when' },
  { key: 'acoustic_tone', label: 'Acoustic tone', purposes: ['acoustic_tone'], primary: 'acoustic_tone', subtitle: 'Emotion from the voice itself' },
  { key: 'text_sentiment', label: 'Text sentiment', purposes: ['text_sentiment'], primary: 'text_sentiment', subtitle: 'Sentiment of each turn' },
  { key: 'embeddings', label: 'Semantic search', purposes: ['embeddings'], primary: 'embeddings', subtitle: 'Search calls by meaning' },
  { key: 'semantic_qa', label: 'Rubric QA', purposes: ['semantic_qa'], primary: 'semantic_qa', subtitle: 'Scores each criterion, with a quote' },
  { key: 'summary', label: 'Call summary', purposes: ['summary'], primary: 'summary' },
  {
    key: 'contact_signals',
    label: 'Contact signals',
    purposes: ['contact_signals', 'signal_category', 'signal_subcategory', 'signal_extraction'],
    primary: 'signal_category',
    subtitle: 'Category, subcategory and fields',
  },
];
const GROUPED = new Set(GROUPS.flatMap((g) => g.purposes));

/** Which catalog purpose each adapter task serves (call1/process/training/registry.py `task_for`). */
const TASK_PURPOSE: Record<string, string> = {
  signal_stage1: 'signal_category',
  signal_stage2: 'signal_subcategory',
  qa_verdict: 'semantic_qa',
};

const STATUS_TONE: Record<string, Tone> = {
  available: 'green',
  not_installed: 'yellow',
  incompatible: 'red',
  unqualified: 'yellow',
};

/** The active on-device adapter's tasks that run on this entry: a task's jobs use the default entry
 * for its purpose (the included model), and the adapter is applied over it. */
function adapterTasks(entry: CatalogEntryDescribe, adapter: TrainingActiveModel | null | undefined, purposes: string[]): string[] {
  if (!adapter?.version) return [];
  return adapter.tasks.filter((t) => {
    const purpose = TASK_PURPOSE[t];
    return purpose !== undefined && purposes.includes(purpose) && entry.default_for.includes(purpose);
  });
}

function EntryCard({
  entry,
  isDefault,
  adapter,
  purposes,
}: {
  entry: CatalogEntryDescribe;
  isDefault: boolean;
  adapter: TrainingActiveModel | null | undefined;
  /** The card's purposes: the adapter row shows only where its tasks run. */
  purposes: string[];
}) {
  const tasks = adapterTasks(entry, adapter, purposes);
  const route = routeLabel(entry.route_class, entry.destination_host);
  return (
    <div className="rounded-md border border-border-muted p-3 space-y-1.5 min-w-0">
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <span className="text-sm font-medium text-fg">{entry.display_name}</span>
        <div className="flex items-center gap-1.5">
          {isDefault && <StatusPill tone="blue">Default</StatusPill>}
          <StatusPill tone={STATUS_TONE[entry.status] ?? 'neutral'}>{humanize(entry.status)}</StatusPill>
        </div>
      </div>
      <p className="text-xs text-fg-subtle break-all">
        {entry.entry_id} v{entry.entry_version}
      </p>
      <dl className="grid grid-cols-[auto_1fr] gap-x-2 gap-y-0.5 text-xs">
        <dt className="text-fg-muted">Runs</dt>
        <dd className="text-fg">
          {route ?? '—'}, {runtimeLabel(entry.runtime)}
        </dd>
        <dt className="text-fg-muted">Model family</dt>
        <dd className="text-fg">{entry.model_family}</dd>
        {entry.model_revision && (
          <>
            <dt className="text-fg-muted">Revision</dt>
            <dd className="text-fg break-all">{entry.model_revision}</dd>
          </>
        )}
        {tasks.length > 0 && (
          <>
            <dt className="text-fg-muted">LoRA adapter</dt>
            <dd className="text-fg">
              <span className="font-medium">{adapter!.version}</span> for {tasks.map(taskLabel).join(', ')}
            </dd>
          </>
        )}
      </dl>
      {entry.status !== 'available' && entry.detail && <p className="text-xs text-primer-yellowFg">{entry.detail}</p>}
      {entry.license_notice && <p className="text-xs text-fg-subtle border-t border-border-muted pt-1.5 mt-1.5">{entry.license_notice}</p>}
      {tasks.length > 0 && (
        <p className="text-xs text-fg-muted">
          The shipped weights are unmodified. Your on-device LoRA adapter is applied over them for the tasks listed; everything else
          uses the base weights. Manage fine-tuning in Settings.
        </p>
      )}
    </div>
  );
}

export default function ModelsView({ demo, onSettings }: { demo: boolean; onSettings(): void }) {
  const query = useQuery({ queryKey: queryKeys.catalog, queryFn: getCatalog, refetchInterval: 30_000 });
  const training = useQuery({ queryKey: queryKeys.training, queryFn: getTraining, refetchInterval: 30_000 });
  const adapter = training.data?.active ?? null;
  const { industryName, privateActive } = useDemoStack();
  const [selected, setSelected] = useState<string | null>(null);
  const catalog = query.data;
  const stages = [...GROUPS, { key: 'privacy', label: 'Protect the transcript', purposes: [], primary: '', subtitle: 'Mask personal details before text analysis' }];
  const entriesFor = (key: string) => {
    const group = stages.find((g) => g.key === key)!;
    return key === 'privacy' ? catalog?.entries.filter((e) => !e.purposes.some((p) => GROUPED.has(p))) ?? []
      : catalog?.entries.filter((e) => e.purposes.some((p) => group.purposes.includes(p))) ?? [];
  };
  function stage(key: string, number?: string) {
    const group = stages.find((g) => g.key === key)!;
    const entries = entriesFor(key);
    const entry = entries.find((e) => e.entry_id === catalog?.defaults[group.primary]) ?? entries[0];
    const tasks = entry ? adapterTasks(entry, adapter, group.purposes) : [];
    const gemma = entry?.model_family === 'gemma-4';
    return (
      <button type="button" key={key} aria-label={`Inspect ${group.label} models`} aria-expanded={selected === key} aria-controls="pipeline-model-details"
        onClick={() => setSelected(selected === key ? null : key)}
        className={`text-left w-full min-w-0 rounded-lg border p-3 bg-canvas-subtle focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue transition-colors hover:border-primer-blue ${selected === key ? 'border-primer-blue' : 'border-border'}`}>
        <span className="flex items-center gap-2 mb-2 text-sm font-semibold text-fg">{number && <span className="text-primer-blue text-xs">{number}</span>}{group.label}</span>
        <span className="block text-xs text-fg-muted mb-2">{group.subtitle ?? 'Turn the transcript into a concise summary'}</span>
        <span className="block text-sm text-fg break-words">{entry?.display_name ?? 'No model configured'}</span>
        {entry && <span className="block mt-2"><StatusPill tone={STATUS_TONE[entry.status] ?? 'neutral'}>{humanize(entry.status)}</StatusPill></span>}
        {tasks.length > 0 && <span className="block text-xs text-primer-blue mt-2">Fine-tune: {adapter!.version}</span>}
        {gemma && demo && (industryName || privateActive) && <span className="block text-xs text-fg-muted mt-2">Demo stack: {industryName ? `Call1 ${industryName}` : 'Gemma base'}{privateActive ? ' + your private layer' : ''}</span>}
      </button>
    );
  }
  const selectedGroup = stages.find((g) => g.key === selected);
  return (
    <div className="max-w-6xl mx-auto space-y-4">
      <PageHeader title="Models" description="Follow a call through the pipeline. Select a stage to inspect its models."
        right={<div className="flex gap-2"><Button size="sm" icon={Settings2} onClick={onSettings}>Fine-tuning settings</Button><Button icon={RefreshCw} size="sm" busy={query.isFetching} onClick={() => { void query.refetch(); void training.refetch(); }}>Refresh</Button></div>} />
      {query.isLoading && <Loading label="Loading pipeline models…" />}
      <ErrorNotice error={query.error} />
      {catalog && <>
        <section aria-label="Model pipeline" data-testid="model-pipeline" className="rounded-xl border border-border bg-canvas p-4 sm:p-6">
          <div className="mx-auto w-fit rounded-full border border-border px-4 py-2 flex gap-2 items-center text-sm font-semibold text-fg"><AudioLines className="h-4 w-4 text-primer-blue" aria-hidden="true" />Call recording</div>
          <ArrowDown className="h-5 w-5 mx-auto my-3 text-fg-subtle" aria-hidden="true" />
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="space-y-2 relative">{stage('asr', '01')}<ArrowRight aria-hidden="true" className="hidden sm:block absolute -right-4 top-12 w-5 h-5 text-primer-blue z-10 bg-canvas" />{stage('asr_vocabulary')}<p className="text-xs text-fg-muted text-center">Two transcription passes → one transcript</p></div>
            <div className="flex flex-col gap-2 relative"><ArrowDown aria-hidden="true" className="sm:hidden mx-auto w-5 h-5 text-primer-blue" /><ArrowRight aria-hidden="true" className="hidden sm:block absolute -right-4 top-12 w-5 h-5 text-primer-blue z-10 bg-canvas" /><div className="flex-1">{stage('speaker_diarization', '02')}</div><p className="text-xs text-fg-muted text-center">Identify caller and agent</p></div>
            <div className="flex flex-col gap-2"><ArrowDown aria-hidden="true" className="sm:hidden mx-auto w-5 h-5 text-primer-blue" /><div className="flex-1">{stage('privacy', '03')}</div><p className="text-xs text-fg-muted text-center">Prepare masked text</p></div>
          </div>
          <ArrowDown className="h-5 w-5 mx-auto my-3 text-fg-subtle" aria-hidden="true" />
          <div className="border-t border-primer-blue pt-4">
            <p className="text-sm font-semibold text-fg mb-1">04 · Analyze the call</p>
            <p className="text-xs text-fg-muted mb-3">These branches produce different insights. Tone reads the audio; text models read the masked transcript.</p>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">{['acoustic_tone', 'text_sentiment', 'embeddings', 'semantic_qa', 'summary', 'contact_signals'].map((key) => stage(key))}</div>
          </div>
          <ArrowDown className="h-5 w-5 mx-auto my-3 text-fg-subtle" aria-hidden="true" />
          <div className="rounded-lg border border-primer-greenBorder bg-primer-greenSubtle p-3 text-center"><p className="text-sm font-semibold text-primer-greenFg">Ready to review in Evaluate</p><p className="text-xs text-fg-muted mt-1">Transcript, scorecard, summary, signals, and supporting evidence</p></div>
        </section>
        {selectedGroup && <div id="pipeline-model-details" data-testid="pipeline-model-details"><Card title={selectedGroup.label} icon={Boxes} subtitle="Installed models and alternatives for this stage"><div className="grid grid-cols-1 sm:grid-cols-2 gap-3">{entriesFor(selectedGroup.key).map((entry) => <EntryCard key={entry.entry_id} entry={entry} isDefault={catalog.defaults[selectedGroup.primary] === entry.entry_id} adapter={adapter} purposes={selectedGroup.purposes} />)}{entriesFor(selectedGroup.key).length === 0 && <p className="text-sm text-fg-muted">No catalog entry for this stage.</p>}</div></Card></div>}
        {catalog.escalation_entry_id && <Notice>Second-opinion model for escalated QA criteria: {catalog.escalation_entry_id}</Notice>}
        <details className="rounded-lg border border-border p-3"><summary className="text-sm text-fg-muted cursor-pointer">Processing details · {handlerModeLabel(catalog.handlers.mode)}</summary><div className="mt-3 text-sm text-fg-muted">{catalog.handlers.missing_job_types.length === 0 ? <StatusPill tone="green">Every processing step has code to run it</StatusPill> : catalog.handlers.missing_job_types.map(jobTypeLabel).join(', ')}{catalog.handlers.notes.map((n, i) => <p key={i} className="text-xs mt-2">{n}</p>)}</div></details>
      </>}
    </div>
  );
}
