import { useCallback, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { CheckCircle2, ExternalLink, FileAudio, Loader2, UploadCloud, X } from 'lucide-react';
import {
  getConversation,
  queryKeys,
  uploadRecording,
  type ConversationDetail,
  type RecordingUploadResult,
} from '../api';
import { DemoCallStudio } from '../components/DemoCallStudio';
import { ConsoleTokenNotice } from '../components/ConsoleTokenNotice';
import {
  Button,
  Card,
  ErrorNotice,
  Field,
  LabelButton,
  Notice,
  PageHeader,
  ProgressBar,
  SelectInput,
  StatusPill,
  TextInput,
  type Tone,
} from '../components/ui';

interface ImportSession {
  key: string;
  file: File;
  fraction: number;
  status: 'uploading' | 'uploaded' | 'error';
  error?: unknown;
  result?: RecordingUploadResult;
}

/** The transcript group's state, read off JobGroupProgress, for the "receipt → transcription →
 * transcript available" stepper the task asks for. */
function transcriptState(detail: ConversationDetail | undefined): { tone: Tone; label: string } {
  const group = detail?.progress?.groups.find((g) => g.kind === 'transcript');
  if (!group) return { tone: 'neutral', label: 'Queued' };
  switch (group.state) {
    case 'available':
      return { tone: 'green', label: 'Transcript ready' };
    case 'failed':
      return { tone: 'red', label: 'Transcription failed' };
    case 'pending':
      return { tone: 'blue', label: 'Transcribing…' };
    default:
      return { tone: 'neutral', label: group.state };
  }
}

const FIELD_LABELS: Record<string, string> = {
  agent_id: 'agent ID',
  agent_display_name: 'agent name',
  agent_extension: 'agent extension',
  agent_channel: 'agent channel',
  external_call_ref: 'call reference',
  recorded_at: 'recorded at',
  caller_reference: 'caller reference',
};

function fieldLabel(name: string): string {
  return FIELD_LABELS[name] ?? name.replace(/_/g, ' ');
}

function ImportProgress({ result, onDismiss }: { result: RecordingUploadResult; onDismiss(): void }) {
  const detail = useQuery({
    queryKey: queryKeys.conversation(result.conversation_id),
    queryFn: () => getConversation(result.conversation_id),
    refetchInterval: (query) => (query.state.data?.progress?.settled ? false : 2_000),
  });
  const transcript = transcriptState(detail.data);

  return (
    <Card
      title={detail.data?.label ?? result.conversation_id}
      icon={FileAudio}
      right={
        <button
          type="button"
          onClick={onDismiss}
          className="w-6 h-6 rounded-md hover:bg-canvas-inset text-fg-muted hover:text-fg flex items-center justify-center"
          title="Dismiss"
          aria-label="Dismiss"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      }
    >
      <div className="space-y-3">
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <StatusPill tone="green">Receipt</StatusPill>
          <span className="text-fg-muted">
            conversation {result.conversation_created ? 'created' : 'reused'} · graph {result.graph_created ? 'created' : 'reused'} · {result.jobs} jobs planned
          </span>
        </div>
        {result.agent_label && (
          <p className="text-sm text-fg-muted">
            Agent: <span className="text-fg">{result.agent_label}</span>
          </p>
        )}
        {result.metadata_updated && (
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <StatusPill tone="blue">Metadata updated</StatusPill>
            <span className="text-fg-muted">
              {(result.updated_fields ?? []).map(fieldLabel).join(', ')} · nothing was reprocessed
            </span>
          </div>
        )}
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <StatusPill tone={transcript.tone}>{transcript.label}</StatusPill>
          {detail.data?.progress_line && <span className="text-fg-muted">{detail.data.progress_line}</span>}
        </div>
        {detail.data?.progress && !detail.data.progress.settled && (
          <ProgressBar
            fraction={
              detail.data.progress.groups.reduce((s, g) => s + g.succeeded, 0) /
              Math.max(1, detail.data.progress.groups.reduce((s, g) => s + g.total, 0) || 1)
            }
            label="Analysis in progress…"
          />
        )}
        {detail.data?.progress?.settled && <StatusPill tone="green">Analysis settled</StatusPill>}
        <a
          href={result.evaluate_url}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1.5 text-sm text-primer-blueFg hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue rounded"
        >
          Open in Evaluate <ExternalLink className="w-3.5 h-3.5" aria-hidden="true" />
        </a>
        <ErrorNotice error={detail.error} />
      </div>
    </Card>
  );
}

export default function ImportView({
  demo,
  onDemoStarted,
  token,
  tokenConfigured,
  onToken,
}: {
  demo: boolean;
  onDemoStarted(conversationId: string): void;
  token: string | null;
  tokenConfigured: boolean | undefined;
  onToken(token: string): void;
}) {
  const queryClient = useQueryClient();
  const [file, setFile] = useState<File | null>(null);
  const [agentId, setAgentId] = useState('');
  const [agentName, setAgentName] = useState('');
  const [agentExtension, setAgentExtension] = useState('');
  const [agentChannel, setAgentChannel] = useState('');
  const [externalRef, setExternalRef] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const [sessions, setSessions] = useState<ImportSession[]>([]);
  const fileInput = useRef<HTMLInputElement>(null);

  const onFiles = useCallback((files: FileList | null) => {
    const picked = files?.[0];
    if (picked) setFile(picked);
  }, []);

  const submit = async () => {
    if (!file) return;
    const key = `${file.name}-${Date.now()}`;
    setSessions((s) => [{ key, file, fraction: 0, status: 'uploading' }, ...s]);
    setFile(null);
    setAgentId('');
    setAgentName('');
    setAgentExtension('');
    setAgentChannel('');
    setExternalRef('');
    if (fileInput.current) fileInput.current.value = '';
    try {
      const result = await uploadRecording(
        file,
        {
          agent_id: agentId.trim() || undefined,
          agent_display_name: agentName.trim() || undefined,
          agent_extension: agentExtension.trim() || undefined,
          agent_channel: agentChannel === '' ? undefined : Number(agentChannel),
          external_call_ref: externalRef || undefined,
        },
        token,
        (fraction) => setSessions((s) => s.map((it) => (it.key === key ? { ...it, fraction } : it))),
      );
      setSessions((s) => s.map((it) => (it.key === key ? { ...it, status: 'uploaded', result, fraction: 1 } : it)));
      void queryClient.invalidateQueries({ queryKey: ['process-conversations'] });
    } catch (err) {
      setSessions((s) => s.map((it) => (it.key === key ? { ...it, status: 'error', error: err } : it)));
    }
  };

  const canWrite = Boolean(token);

  return (
    <div className={`${demo ? "max-w-6xl" : "max-w-3xl"} mx-auto space-y-6`}>
      <PageHeader title="Import" description="Upload a call recording. Process registers it with Store, stores the audio and plans its processing steps." />

      {!canWrite && <ConsoleTokenNotice configured={tokenConfigured} onToken={onToken} />}

      {demo && <DemoCallStudio token={token} onStarted={onDemoStarted} />}

      <Card title="Upload a recording" icon={UploadCloud} subtitle="WAV, MP3, FLAC, OGG or M4A, mono or stereo">
        <div className="space-y-3">
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              onFiles(e.dataTransfer.files);
            }}
            className={`rounded-md border-2 border-dashed p-6 text-center transition-colors ${
              dragOver ? 'border-primer-blueBorder bg-primer-blueSubtle' : 'border-border-muted'
            }`}
          >
            <FileAudio className="w-6 h-6 mx-auto text-fg-subtle mb-2" aria-hidden="true" />
            {file ? (
              <p className="text-sm text-fg">{file.name} · {(file.size / (1024 * 1024)).toFixed(1)} MB</p>
            ) : (
              <p className="text-sm text-fg-muted">Drag a recording here, or choose a file.</p>
            )}
            <div className="mt-2">
              <input
                ref={fileInput}
                type="file"
                accept="audio/*"
                className="sr-only peer"
                id="process-import-file"
                aria-label="Choose a recording"
                onChange={(e) => onFiles(e.target.files)}
              />
              <LabelButton htmlFor="process-import-file" variant="secondary" size="sm">
                Choose file
              </LabelButton>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <Field label="Agent ID" hint="Optional. Filters and metrics key on it">
              {(id) => <TextInput id={id} value={agentId} onChange={(e) => setAgentId(e.target.value)} placeholder="agent-9" maxLength={200} />}
            </Field>
            <Field label="Agent name" hint="Optional. Evaluate shows “Name (ext)”">
              {(id) => (
                <TextInput id={id} value={agentName} onChange={(e) => setAgentName(e.target.value)} placeholder="Samantha" maxLength={100} />
              )}
            </Field>
            <Field label="Agent extension" hint="Optional. Digits, letters, * # + . _ -">
              {(id) => (
                <TextInput
                  id={id}
                  value={agentExtension}
                  onChange={(e) => setAgentExtension(e.target.value)}
                  placeholder="104"
                  maxLength={20}
                />
              )}
            </Field>
            <Field label="Agent channel" hint="Stereo only">
              {(id) => (
                <SelectInput id={id} value={agentChannel} onChange={(e) => setAgentChannel(e.target.value)}>
                  <option value="">Unspecified</option>
                  <option value="0">Channel 0</option>
                  <option value="1">Channel 1</option>
                </SelectInput>
              )}
            </Field>
            <Field label="Call reference" hint="Optional. Your recorder's call ID or a short title; Evaluate's call search finds it">
              {(id) => (
                <TextInput id={id} value={externalRef} onChange={(e) => setExternalRef(e.target.value)} placeholder="Retail demo, ticket-1234" maxLength={200} />
              )}
            </Field>
          </div>
          <p className="text-xs text-fg-subtle">
            Uploading the same recording again with different details updates the call’s details in Store. Fields left blank keep
            their stored values, and nothing is reprocessed.
          </p>

          <div className="flex justify-end">
            <Button variant="primary" icon={UploadCloud} disabled={!file || !canWrite} onClick={() => void submit()}>
              Upload and ingest
            </Button>
          </div>
        </div>
      </Card>

      {sessions.map((s) => (
        <div key={s.key}>
          {s.status === 'uploading' && (
            <Card title={s.file.name} icon={Loader2}>
              <ProgressBar fraction={s.fraction} label={`Uploading… ${Math.round(s.fraction * 100)}%`} />
            </Card>
          )}
          {s.status === 'error' && (
            <Card title={s.file.name} icon={FileAudio}>
              <ErrorNotice error={s.error}>
                <div className="mt-1">
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => setSessions((cur) => cur.filter((it) => it.key !== s.key))}
                  >
                    Dismiss
                  </Button>
                </div>
              </ErrorNotice>
            </Card>
          )}
          {s.status === 'uploaded' && s.result && (
            <ImportProgress result={s.result} onDismiss={() => setSessions((cur) => cur.filter((it) => it.key !== s.key))} />
          )}
        </div>
      ))}

      {sessions.length === 0 && (
        <Notice tone="neutral" icon={CheckCircle2}>
          Uploads you make in this browser session appear here, following their transcription and analysis progress.
        </Notice>
      )}
    </div>
  );
}
