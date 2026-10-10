// `#/calls/:callId` — the Workbench. Call detail, audio, transcript (speakers/tone/sentiment),
// scorecard (quoted evidence, verdict override), summary, contact signals, escalation resolution
// and reanalysis requests. See ../README.md "View contract".
//
// Playback: a hidden `<audio>` element drives the transport, and `components/ThreadWaveform` draws
// the legacy three.js thread waveform as the seek control (a data-agnostic copy of the legacy
// WaveformDeck, fed from `/store/v1` only; it falls back to a native seek bar without WebGL).
// Nothing here imports or modifies the legacy components, so the legacy app keeps working
// unchanged.

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import {
  ArrowLeft,
  CheckCircle2,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Pause,
  Play,
  RefreshCw,
  ShieldAlert,
} from 'lucide-react';
import {
  agentLabel,
  callQaBadge,
  describeError,
  formatScore,
  isVersionConflict,
  sendIdempotent,
  useIdempotencyKey,
  queryKeys,
  resultStateDisplay,
  escalationStatusDisplay,
  verdictStatusDisplay,
  FAMILY_DISPLAY,
  fieldChipText,
  hitCategoryId,
  hitCategoryName,
  hitSegmentCount,
  hitSpanEnd,
  hitSubcategoryName,
  signalFamily,
  type CallDetail,
  type EvaluationView,
  type ContactSignalView,
  type CallReviewState,
  type OverrideReasonCode,
  type ReanalysisKind,
  type ResultGroup,
  type ResultKind,
  type SpeakerRole,
  type TranscriptReplacementView,
  type TranscriptTurnView,
  type VerdictStatus,
  type VocabularyCorrectionView,
} from '../api';
import {
  Button,
  Card,
  Dialog,
  EmptyState,
  ErrorNotice,
  Field,
  InlineTooltip,
  Loading,
  Notice,
  PageHeader,
  SelectInput,
  StatusPill,
  TextInput,
  formatDateTime,
  formatDuration,
} from '../components/ui';
import ThreadWaveform, { type WaveformMarker, type WaveformTurn } from '../components/ThreadWaveform';
import { failureReason, isRecordingRejected } from '../components/failureText';
import { usePollChanges } from '../state/app';
import type { WorkbenchViewProps } from './types';
import { ContactSignalsSection } from './workbench/ContactSignalsSection';
import { ScorecardPopout } from './workbench/ScorecardPopout';

function clock(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return '0:00';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

const SPEAKER_LABEL: Record<SpeakerRole, string> = { AGENT: 'Agent', CALLER: 'Caller', SYSTEM: 'System', UNKNOWN: 'Unknown' };


const OVERRIDE_REASONS: OverrideReasonCode[] = [
  'model_misread_evidence',
  'evidence_not_in_transcript',
  'transcription_error',
  'speaker_misattributed',
  'policy_exception',
  'criterion_not_applicable',
  'rubric_ambiguous',
  'other',
];

function resultOf(call: CallDetail, kind: ResultKind): ResultGroup | undefined {
  return call.results.find((r) => r.kind === kind);
}

/** The escalation banner in words: who has to act, not the status enum. */
function escalationBannerText(status: string): string {
  if (status === 'PENDING') return 'Escalated: waiting for a supervisor to approve or override';
  if (status === 'APPROVED') return 'Escalation resolved: a supervisor approved the result';
  if (status === 'OVERRIDDEN') return 'Escalation resolved: a supervisor overrode the result';
  return escalationStatusDisplay(status).label;
}

/** The first caller-objective subcategory on the call ("Check stock / availability"), or null when
 * there is none, it is "Other", or the signals have not loaded. */
function callObjective(signals: ContactSignalView[] | undefined): string | null {
  for (const sig of signals ?? []) {
    if (hitCategoryId(sig) !== 'intent' || !sig.subcategory_id || sig.subcategory_id === 'other') continue;
    return hitSubcategoryName(sig);
  }
  return null;
}

/**
 * Model-written turn references ("… refund (turn 21)", "… (turns 4-6)", a bare trailing "(60)")
 * stripped from summary text: the "jump to transcript" link beside a key point already does that.
 */
function stripTurnRefs(text: string): string {
  return text
    .replace(/\s*\((?:turns?|t)\s*\d+(?:\s*(?:[-–,]|and)\s*(?:turns?\s*)?\d+)*\)/gi, '')
    .replace(/\s*\(\d+(?:\s*[-–,]\s*\d+)*\)(?=\s*[.;:!?]?\s*$)/, '')
    .trim();
}

function SectionStatePill({ group }: { group: ResultGroup | undefined }) {
  const d = resultStateDisplay(group?.state ?? 'disabled');
  const reason = failureReason(group?.failure_code, group?.kind === 'transcript');
  const detail = reason ? `: ${reason}.` : group?.partial_reason ? ` — ${group.partial_reason}` : '';
  return (
    <StatusPill tone={d.tone} title={`${d.description}${detail}`}>
      {d.label}
    </StatusPill>
  );
}

/**
 * `#/calls/:callId` — the Workbench. Remounted per `callId` (the shell keys it), so every hook
 * below is scoped to one call for its whole lifetime.
 */
export default function WorkbenchView({ callId, turn: initialTurn, client, session, navigate }: WorkbenchViewProps) {
  const qc = useQueryClient();
  const pollNow = usePollChanges();
  const audioRef = useRef<HTMLAudioElement>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [audioError, setAudioError] = useState(false);
  const [activeTurnId, setActiveTurnId] = useState<number | null>(null);
  const [jumpTurnId, setJumpTurnId] = useState<number | null>(null);

  useEffect(() => {
    if (jumpTurnId === null) return;
    const el = document.getElementById(`workbench-turn-${jumpTurnId}`);
    el?.focus({ preventScroll: true });
    el?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    setJumpTurnId(null);
  }, [jumpTurnId]);

  const callQuery = useQuery({
    queryKey: queryKeys.call(callId),
    queryFn: ({ signal }) => client.get('/store/v1/calls/{call_id}', { path: { call_id: callId }, signal }),
  });
  const call = callQuery.data;

  const reviewQuery = useQuery({
    queryKey: [...queryKeys.call(callId), 'review'],
    queryFn: ({ signal }) => client.get('/store/v1/calls/{call_id}/review', { path: { call_id: callId }, signal }),
  });
  const review = reviewQuery.data;

  const transcriptGroup = call ? resultOf(call, 'transcript') : undefined;
  const transcriptFetchable = transcriptGroup ? ['available', 'partial', 'stale'].includes(transcriptGroup.state) : false;
  const transcriptQuery = useQuery({
    queryKey: [...queryKeys.call(callId), 'transcript'],
    queryFn: ({ signal }) => client.get('/store/v1/calls/{call_id}/transcript', { path: { call_id: callId }, signal }),
    enabled: transcriptFetchable,
  });

  const summaryGroup = call ? resultOf(call, 'summary') : undefined;
  const summaryFetchable = summaryGroup ? ['available', 'partial', 'stale'].includes(summaryGroup.state) : false;
  const summaryQuery = useQuery({
    queryKey: [...queryKeys.call(callId), 'summary'],
    queryFn: ({ signal }) => client.get('/store/v1/calls/{call_id}/summary', { path: { call_id: callId }, signal }),
    enabled: summaryFetchable,
  });

  const contactGroup = call ? resultOf(call, 'contact_signals') : undefined;
  const contactFetchable = contactGroup ? ['available', 'partial', 'stale'].includes(contactGroup.state) : false;
  const contactQuery = useQuery({
    queryKey: [...queryKeys.call(callId), 'contact-signals'],
    queryFn: ({ signal }) => client.get('/store/v1/calls/{call_id}/contact-signals', { path: { call_id: callId }, signal }),
    enabled: contactFetchable,
  });

  // `#/calls/:callId?turn=N` (a signal preview's "Open in Workbench at this turn"): once the
  // transcript is here, seek to the turn and scroll it into view, once.
  const jumpedToInitialTurn = useRef(false);

  // Audio source. A fresh call swaps the source and resets transport state.
  useEffect(() => {
    setCurrentTime(0);
    setDuration(0);
    setPlaying(false);
    setAudioError(false);
  }, [callId]);

  const audioSrc = client.url('/store/v1/calls/{call_id}/audio', { path: { call_id: callId } });

  // Contract 1.2.0: until the PII findings for the current transcript exist, Store withholds the
  // transcript text and refuses the audio (503 pii_findings_pending). Once the text is served,
  // retry an audio load that failed while it was withheld.
  const transcriptWithheld = transcriptQuery.data?.text_withheld === true;
  const transcriptReady = transcriptQuery.data !== undefined && !transcriptWithheld;
  const maskingPending = transcriptWithheld || transcriptGroup?.state === 'pending';
  useEffect(() => {
    if (transcriptReady && audioError) {
      setAudioError(false);
      audioRef.current?.load();
    }
    // Only a change in readiness triggers the retry.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [transcriptReady]);

  // Waveform inputs (ThreadWaveform): the same audio through the typed client, and markers built
  // only from what this view already read from Store — turns, contact signals, verdict evidence.
  const [rate, setRate] = useState(1);
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.defaultPlaybackRate = rate;
    audio.playbackRate = rate;
  }, [rate, callId]);
  const loadWaveformAudio = useCallback(
    (signal: AbortSignal) =>
      client.raw('/store/v1/calls/{call_id}/audio', { path: { call_id: callId }, signal }).then((res) => res.arrayBuffer()),
    [client, callId],
  );
  const waveformTurns = useMemo<WaveformTurn[]>(
    () =>
      transcriptWithheld
        ? []
        : (transcriptQuery.data?.turns ?? []).map((t) => ({
            id: t.turn_id,
            start: t.start_time,
            end: t.end_time,
            speaker: t.speaker,
            channel: t.channel,
            text: t.text,
          })),
    [transcriptQuery.data, transcriptWithheld],
  );
  const waveformMarkers = useMemo<WaveformMarker[]>(() => {
    const out: WaveformMarker[] = [];
    const turns = transcriptQuery.data?.turns ?? [];
    for (const sig of contactQuery.data?.signals ?? []) {
      // Color family (lifecycle, resolution, custom) plus its name in words: never color alone.
      const family = FAMILY_DISPLAY[signalFamily(hitCategoryId(sig))];
      out.push({
        id: `signal-${sig.id}`,
        hitId: sig.id,
        time: sig.start,
        // A multi-segment signal (decision 25) bands from its first span's start to its last part's end.
        end: hitSpanEnd(sig),
        segments: hitSegmentCount(sig),
        kind: 'signal',
        label: hitCategoryName(sig),
        detail: hitSubcategoryName(sig),
        chips: (sig.fields ?? []).map(fieldChipText).filter((t): t is string => t !== null),
        family: family.label,
        tone: family.tone,
        quote: sig.quote,
        speaker: sig.speaker,
        turnId: sig.turn_id,
      });
    }
    for (const v of call?.evaluation?.verdicts ?? []) {
      if (!v.quoted_evidence) continue;
      const time = v.timestamp_range?.[0] ?? turns.find((t) => t.turn_id === v.quote_turn_id)?.start_time;
      if (time == null) continue;
      const override = review?.overrides.filter((o) => o.criterion_id === v.criterion_id && o.evaluation_version === call?.evaluation?.version).slice(-1)[0];
      const d = verdictStatusDisplay(override?.status ?? v.status);
      out.push({
        id: `evidence-${v.criterion_id}`,
        time,
        kind: 'evidence',
        label: v.criterion_name,
        status: override ? `${d.label} (overridden)` : d.label,
        tone: d.tone,
        quote: v.quoted_evidence,
        speaker: v.speaker,
        turnId: v.quote_turn_id,
      });
    }
    return out;
  }, [contactQuery.data, call?.evaluation, review?.overrides, transcriptQuery.data]);

  // A rejected play() is not always a broken recording: AbortError means a pause or a new seek
  // interrupted it, NotAllowedError means the browser wants another user gesture. Only the rest
  // (NotSupportedError, a failed load) makes the audio unavailable; the element's own `error` event
  // covers a load failure.
  const playRejected = (err: unknown) => {
    const name = err instanceof DOMException ? err.name : '';
    if (name === 'AbortError' || name === 'NotAllowedError') return;
    setAudioError(true);
  };

  const seekTo = (seconds: number, play = false) => {
    const audio = audioRef.current;
    if (!audio) return;
    // Before the metadata arrives the duration is unknown; the browser still takes the position as
    // the default start position, so a jump from the transcript is never silently dropped.
    const known = Number.isFinite(audio.duration) && audio.duration > 0;
    const target = Math.max(0, known ? Math.min(audio.duration, seconds) : seconds);
    audio.currentTime = target;
    setCurrentTime(target);
    if (play) {
      audio.play().catch(playRejected);
    }
  };

  const jumpToTurn = (turnId: number | null | undefined, timeHint?: number | null) => {
    const turns = transcriptQuery.data?.turns ?? [];
    const turn = turnId != null ? turns.find((t) => t.turn_id === turnId) : undefined;
    if (turn) seekTo(turn.start_time);
    else if (timeHint != null) seekTo(timeHint);
    if (turn) {
      setJumpTurnId(turn.turn_id);
    }
  };

  useEffect(() => {
    if (initialTurn === undefined || jumpedToInitialTurn.current || !transcriptReady) return;
    jumpedToInitialTurn.current = true;
    jumpToTurn(initialTurn);
    // jumpToTurn reads the loaded transcript; only readiness triggers this.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [transcriptReady, initialTurn]);

  if (callQuery.isLoading) {
    return (
      <div className="max-w-6xl mx-auto">
        <PageHeader title="Workbench" description={<span className="break-all">Call {callId}</span>} />
        <Loading label="Loading call…" />
      </div>
    );
  }

  if (callQuery.isError || !call) {
    return (
      <div className="max-w-6xl mx-auto">
        <PageHeader
          title="Workbench"
          right={
            <Button icon={ArrowLeft} onClick={() => navigate({ name: 'calls' })}>
              All calls
            </Button>
          }
        />
        <ErrorNotice error={callQuery.error ?? new Error('Call not found.')} />
      </div>
    );
  }

  // The call's title: what the caller wanted (the first caller-objective subcategory), which tells
  // one agent's calls apart; the agent label is the fallback and otherwise the secondary line.
  const objective = callObjective(contactQuery.data?.signals);
  const agent = agentLabel(call.call.agent_id, call.call.agent_display_name, call.call.agent_extension);
  const currentReviewScore = review?.reviewed_score?.evaluation_version === call.evaluation?.version ? review?.reviewed_score : undefined;

  const badge = callQaBadge({
    qa_state: call.results.find((r) => r.kind === 'qa')?.state ?? 'disabled',
    overall_score: currentReviewScore?.overall_score ?? call.evaluation?.overall_score ?? null,
    passed: currentReviewScore?.passed ?? call.evaluation?.passed ?? null,
    critical_failure: currentReviewScore?.critical_failure ?? call.evaluation?.critical_failure ?? null,
    requires_human_review: currentReviewScore?.requires_human_review ?? call.evaluation?.requires_human_review ?? false,
  });

  return (
    <div className="max-w-6xl mx-auto space-y-4">
      {/* PageHeader's markup, with the agent label addressable (data-testid="call-agent") whether it
          is the title or the secondary line. */}
      <div className="flex flex-wrap items-end justify-between gap-3 mb-4" data-testid="workbench-header">
        <div className="min-w-0">
          <h1 className="text-base font-semibold text-fg">{objective ?? <span data-testid="call-agent">{agent}</span>}</h1>
          <p className="text-sm text-fg-muted mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-1">
            {objective && (
              <span data-testid="call-agent" className="text-fg">
                {agent}
              </span>
            )}
            <span>{formatDateTime(call.call.created_at)}</span>
            {call.call.duration_seconds != null && <span>{formatDuration(call.call.duration_seconds)}</span>}
            <span className="break-all text-xs text-fg-subtle" title="Call ID">
              {call.call.call_id}
            </span>
          </p>
        </div>
        <div className="ml-auto flex flex-wrap items-center justify-end gap-2">
          <StatusPill tone={badge.tone} title={badge.description}>
            {badge.label}
            {badge.score !== null && ` · ${formatScore(badge.score)}`}
          </StatusPill>
          <Button icon={ArrowLeft} onClick={() => navigate({ name: 'calls' })}>
            All calls
          </Button>
          <ReanalysisMenu client={client} callId={callId} session={session} onWrote={() => pollNow()} />
        </div>
      </div>

      {!call.pending_work.settled && (
        <Notice tone="blue">
          Processing is still under way: {call.pending_work.jobs_succeeded + call.pending_work.jobs_failed} of{' '}
          {call.pending_work.jobs_total} jobs settled.
        </Notice>
      )}

      <audio
        ref={audioRef}
        src={audioSrc}
        preload="metadata"
        onTimeUpdate={(e) => {
          const t = e.currentTarget.currentTime;
          setCurrentTime(t);
          const turn = (transcriptQuery.data?.turns ?? []).find((tn) => t >= tn.start_time && t <= tn.end_time);
          setActiveTurnId(turn?.turn_id ?? null);
        }}
        onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        onError={() => setAudioError(true)}
        className="hidden"
      />

      <Card>
        {/* No transport when the audio cannot load (the recording was rejected, or Store refused it):
            the waveform below says why. */}
        <div className={`${audioError ? 'hidden' : 'flex'} items-center gap-3`}>
          <Button
            variant="primary"
            icon={playing ? Pause : Play}
            disabled={audioError}
            onClick={() => {
              const audio = audioRef.current;
              if (!audio) return;
              if (audio.paused) audio.play().catch(playRejected);
              else audio.pause();
            }}
          >
            {playing ? 'Pause' : 'Play'}
          </Button>
          <span className="text-xs font-medium text-fg tabular-nums w-24">
            {clock(currentTime)} / {duration > 0 ? clock(duration) : clock(call.call.duration_seconds)}
          </span>
          <label className="ml-auto inline-flex items-center gap-1.5 text-xs text-fg-muted">
            Speed
            <select
              value={rate}
              onChange={(e) => setRate(Number(e.target.value))}
              className="h-7 rounded-md border border-border-control bg-canvas px-1.5 text-xs text-fg focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
            >
              {[1, 1.25, 1.5, 2].map((r) => (
                <option key={r} value={r}>
                  {r.toFixed(r % 1 ? 2 : 1)}x
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className={audioError ? '' : 'mt-3'}>
          <ThreadWaveform
            sourceKey={maskingPending ? `${audioSrc}#masking-pending` : audioSrc}
            loadAudio={loadWaveformAudio}
            describeLoadError={(err) =>
              maskingPending ? 'the audio is withheld until PII masking finishes' : describeError(err).replace(/\.$/, '')
            }
            audioRef={audioRef}
            currentTime={currentTime}
            duration={duration}
            fallbackDuration={call.call.duration_seconds}
            playing={playing}
            audioError={audioError}
            unavailableText={
              maskingPending
                ? 'Audio is withheld until PII masking finishes for this call.'
                : 'Audio unavailable for this call. Use the transcript to review it.'
            }
            onSeek={(t) => seekTo(t)}
            onTogglePlay={() => {
              const audio = audioRef.current;
              if (!audio) return;
              if (audio.paused) audio.play().catch(playRejected);
              else audio.pause();
            }}
            turns={waveformTurns}
            markers={waveformMarkers}
            onMarkerJump={(m) => jumpToTurn(m.turnId, m.time)}
          />
        </div>
        {!session.can('play_audio') && <p className="text-xs text-fg-muted mt-2">Your role does not include audio playback.</p>}
      </Card>

      <div data-testid="workbench-dashboard" className="grid grid-cols-1 lg:grid-cols-2 gap-4 lg:h-[calc(100dvh-28rem)] lg:min-h-[24rem] lg:grid-rows-2 [&>div]:min-h-0 [&>div]:min-w-0 [&>div>section]:h-full [&>div>section]:flex [&>div>section]:flex-col [&>div>section>header]:shrink-0 [&>div>section>div]:min-h-0 [&>div>section>div]:overflow-y-auto [&>div>section>div]:overscroll-contain [&>div>section>div]:flex-1">
        <div className="h-80 lg:h-auto">
          <SummarySection group={summaryGroup} transcriptGroup={transcriptGroup} query={summaryQuery} onJump={jumpToTurn} />
        </div>
        <div className="h-80 lg:h-auto">
          <TranscriptSection
            group={transcriptGroup}
            query={transcriptQuery}
            activeTurnId={activeTurnId}
            callId={callId}
            client={client}
            session={session}
            reviewVersion={review?.review_version ?? call.review_version}
            onSeek={(t) => seekTo(t, true)}
            onWrote={() => {
              void qc.invalidateQueries({ queryKey: queryKeys.call(callId) });
              void qc.invalidateQueries({ queryKey: [...queryKeys.call(callId), 'review'] });
              pollNow();
            }}
          />
        </div>
        <div className="h-80 lg:h-auto">
          <ContactSignalsSection
            client={client}
            session={session}
            callId={callId}
            group={contactGroup}
            query={contactQuery}
            transcriptWithheld={transcriptWithheld}
            transcriptFailed={transcriptGroup?.state === 'failed'}
            onJump={jumpToTurn}
            onWrote={() => {
              void qc.invalidateQueries({ queryKey: queryKeys.call(callId) });
              pollNow();
            }}
          />
        </div>
        <div className="h-80 lg:h-auto">
          <ScorecardSection
            call={call}
            review={review}
            reviewLoading={reviewQuery.isLoading}
            reviewError={reviewQuery.error}
            client={client}
            callId={callId}
            session={session}
            onJump={jumpToTurn}
            onWrote={() => {
              void qc.invalidateQueries({ queryKey: queryKeys.call(callId) });
              void qc.invalidateQueries({ queryKey: [...queryKeys.call(callId), 'review'] });
              pollNow();
            }}
          />
        </div>
      </div>
    </div>
  );
}

// --- transcript -----------------------------------------------------------------------------------

function TranscriptSection({
  group,
  query,
  activeTurnId,
  onSeek,
}: {
  group: ResultGroup | undefined;
  query: ReturnType<
    typeof useQuery<{
      turns: TranscriptTurnView[];
      tone_blocks: { turn_ids: number[]; emotion: string | null; valence: number | null }[];
      is_redacted: boolean;
      text_withheld?: boolean;
      vocabulary_correction?: VocabularyCorrectionView | null;
    }>
  >;
  activeTurnId: number | null;
  callId: string;
  client: WorkbenchViewProps['client'];
  session: WorkbenchViewProps['session'];
  reviewVersion: number;
  onSeek: (seconds: number) => void;
  onWrote: () => void;
}) {
  const turns = query.data?.turns ?? [];
  const toneByTurn = useMemo(() => {
    const map = new Map<number, { emotion: string | null; valence: number | null }>();
    for (const block of query.data?.tone_blocks ?? []) {
      for (const turnId of block.turn_ids) map.set(turnId, { emotion: block.emotion, valence: block.valence });
    }
    return map;
  }, [query.data]);

  const vocabularyCorrection = query.data?.vocabulary_correction ?? null;
  const replacementsByTurn = useMemo(() => {
    const map = new Map<number, TranscriptReplacementView[]>();
    for (const r of vocabularyCorrection?.replacements ?? []) {
      const list = map.get(r.turn_id) ?? [];
      list.push(r);
      map.set(r.turn_id, list);
    }
    return map;
  }, [vocabularyCorrection]);

  return (
    <Card title="Transcript" subtitle={query.data?.is_redacted ? 'Personal details are masked' : undefined} right={<SectionStatePill group={group} />}>
      {!group || group.state === 'disabled' ? (
        <EmptyState title="No transcript requested">Transcription has not run for this call.</EmptyState>
      ) : group.state === 'pending' ? (
        <EmptyState title="Analyzing">The transcript appears here once transcription and PII masking finish.</EmptyState>
      ) : group.state === 'failed' ? (
        isRecordingRejected(group.failure_code) ? (
          <EmptyState title="Recording rejected">
            The recording is too short, has no speech we could detect, or could not be read, so there is no transcript to review.
          </EmptyState>
        ) : (
          <EmptyState title="Transcript unavailable">
            Transcription stopped{group.failure_code ? `: ${failureReason(group.failure_code, true)}` : ''}. An admin can retry it.
          </EmptyState>
        )
      ) : query.isLoading ? (
        <Loading label="Loading transcript…" />
      ) : query.isError ? (
        <ErrorNotice error={query.error} />
      ) : query.data?.text_withheld ? (
        <EmptyState icon={ShieldAlert} title="Transcript text withheld">
          {group.partial_reason ?? 'PII masking has not finished for this transcript revision.'} The masked text appears here as
          soon as it does.
        </EmptyState>
      ) : turns.length === 0 ? (
        <EmptyState title="Transcript is empty" />
      ) : (
        <>
          <VocabularyCorrectionNotice correction={vocabularyCorrection} />
          <div className="space-y-2">
            {turns.map((turn) => {
              const tone = toneByTurn.get(turn.turn_id);
              const isAgent = turn.speaker === 'AGENT';
              return (
                <div
                  key={turn.turn_id}
                  id={`workbench-turn-${turn.turn_id}`}
                  tabIndex={-1}
                  className={`rounded-md border p-2.5 text-sm transition-colors ${
                    activeTurnId === turn.turn_id ? 'border-primer-blueBorder bg-primer-blueSubtle' : 'border-border-muted bg-canvas'
                  }`}
                >
                  <div className="flex items-center justify-between gap-2 mb-1">
                    <div className="flex items-center gap-2 text-xs text-fg-muted">
                      <span className={`font-semibold ${isAgent ? 'text-primer-blueFg' : 'text-fg'}`}>{SPEAKER_LABEL[turn.speaker]}</span>
                      <span className="tabular-nums">
                        {clock(turn.start_time)}–{clock(turn.end_time)}
                      </span>
                      {tone?.emotion && <span title="Voice tone">tone: {tone.emotion}</span>}
                      {turn.text_sentiment_label && (
                        <span title="Text sentiment">
                          sentiment: {turn.text_sentiment_label.toLowerCase()}
                          {turn.text_sentiment != null ? ` (${turn.text_sentiment > 0 ? '+' : ''}${turn.text_sentiment.toFixed(2)})` : ''}
                        </span>
                      )}
                    </div>
                    <button
                      type="button"
                      className="text-xs text-primer-blueFg hover:underline shrink-0 rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
                      aria-label={`Play from ${clock(turn.start_time)}`}
                      onClick={() => onSeek(turn.start_time)}
                    >
                      Play
                    </button>
                  </div>
                  <p className="text-fg leading-relaxed">{renderTurnText(turn.text, replacementsByTurn.get(turn.turn_id))}</p>
                </div>
              );
            })}
          </div>
        </>
      )}
    </Card>
  );
}

/** The transcript header's quiet correction line, or the `base_only` notice (docs/DualAsr.md
 * section 8): "Vocabulary correction didn't run for this call: …". Absent when no vocabulary ran. */
function VocabularyCorrectionNotice({ correction }: { correction: VocabularyCorrectionView | null }) {
  if (!correction) return null;
  if (correction.status === 'base_only') {
    return (
      <div className="mb-2">
        <Notice>Vocabulary correction didn't run for this call{correction.note ? `: ${correction.note}` : '.'}</Notice>
      </div>
    );
  }
  if (correction.replacement_count === 0) return null;
  const withheld = correction.withheld_count > 0 ? ` (${correction.withheld_count} not shown here, masked)` : '';
  return (
    <p className="text-xs text-fg-subtle mb-2" role="status">
      {correction.replacement_count} word{correction.replacement_count === 1 ? '' : 's'} corrected from the vocabulary{withheld}.
    </p>
  );
}

/**
 * `turn.text` with each corrected span marked: a dotted-underline, keyboard-focusable tooltip
 * saying what the base engine (Parakeet) heard, or that the original is masked when `heard` is
 * null (docs/DualAsr.md section 8). Replacements with no located offsets in this (masked) turn
 * text, or that overlap an earlier one, are skipped — never rendered against the wrong span.
 */
function renderTurnText(text: string, replacements: TranscriptReplacementView[] | undefined) {
  const marks = (replacements ?? [])
    .filter((r): r is TranscriptReplacementView & { char_start: number; char_end: number } => r.char_start !== null && r.char_end !== null && r.char_start < r.char_end && r.char_end <= text.length)
    .sort((a, b) => a.char_start - b.char_start);
  if (marks.length === 0) return text;

  const parts: ReactNode[] = [];
  let cursor = 0;
  marks.forEach((mark, i) => {
    if (mark.char_start < cursor) return; // overlaps the previous mark: skip rather than mis-render
    if (mark.char_start > cursor) parts.push(text.slice(cursor, mark.char_start));
    const label = mark.heard ? `Parakeet heard "${mark.heard}"` : 'Corrected from the vocabulary. The original words are masked.';
    parts.push(
      <InlineTooltip key={`${mark.turn_id}-${mark.word_start}-${i}`} label={label}>
        {text.slice(mark.char_start, mark.char_end)}
      </InlineTooltip>,
    );
    cursor = mark.char_end;
  });
  if (cursor < text.length) parts.push(text.slice(cursor));
  return parts;
}

// --- summary ----------------------------------------------------------------------------------

function SummarySection({
  group,
  transcriptGroup,
  query,
  onJump,
}: {
  group: ResultGroup | undefined;
  transcriptGroup: ResultGroup | undefined;
  query: ReturnType<typeof useQuery>;
  onJump: (turnId: number | null | undefined) => void;
}) {
  const summary = query.data as
    | { narrative: string; key_points: string[]; citations: { claim: string; index: number; turn_ids: number[] }[]; rubric_highlights: { criterion_id: string; criterion_name: string; status: VerdictStatus; note: string }[] }
    | undefined;

  return (
    <Card title="Summary" right={<SectionStatePill group={group} />}>
      {!group || group.state === 'disabled' ? (
        <EmptyState title="No summary requested">Summarization has not run for this call.</EmptyState>
      ) : group.state === 'pending' ? (
        <EmptyState title="Analyzing">The summary appears here once the transcript is summarized.</EmptyState>
      ) : group.state === 'failed' ? (
        <EmptyState title="Summary stopped">
          {transcriptGroup?.state === 'failed'
            ? 'There is no transcript to summarize.'
            : `Summarization stopped without a result${group.failure_code ? `: ${failureReason(group.failure_code)}` : ''}. An admin can retry it.`}
        </EmptyState>
      ) : query.isLoading ? (
        <Loading label="Loading summary…" />
      ) : query.isError ? (
        <ErrorNotice error={query.error} />
      ) : !summary ? (
        <EmptyState title="No summary yet" />
      ) : (
        <div className="space-y-3 text-sm">
          <p className="text-fg leading-relaxed">{stripTurnRefs(summary.narrative)}</p>
          {summary.key_points.length > 0 && (
            <ul className="list-disc pl-5 space-y-1 text-fg-muted">
              {summary.key_points.map((point, i) => {
                const citation = summary.citations.find((c) => c.claim === 'key_point' && c.index === i);
                return (
                  <li key={i}>
                    {stripTurnRefs(point)}
                    {citation && citation.turn_ids.length > 0 && (
                      <button type="button" className="ml-1 text-xs text-primer-blueFg hover:underline rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue" onClick={() => onJump(citation.turn_ids[0])}>
                        (jump to transcript)
                      </button>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
          {summary.rubric_highlights.length > 0 && (
            <div className="pt-2 border-t border-border-muted space-y-1.5">
              <p className="text-xs font-medium text-fg-muted">Rubric highlights</p>
              {summary.rubric_highlights.map((h) => {
                const d = verdictStatusDisplay(h.status);
                return (
                  <div key={h.criterion_id} className="flex items-start gap-2 text-xs">
                    <StatusPill tone={d.tone}>{d.label}</StatusPill>
                    <span className="text-fg-muted">
                      <span className="text-fg font-medium">{h.criterion_name}</span> — {h.note}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </Card>
  );
}

// --- scorecard, verdict override, escalation resolution ------------------------------------------

function ScorecardSection({
  call,
  review: latestReview,
  reviewLoading,
  reviewError,
  client,
  callId,
  session,
  onJump,
  onWrote,
}: {
  call: CallDetail;
  review: CallReviewState | undefined;
  reviewLoading: boolean;
  reviewError: unknown;
  client: WorkbenchViewProps['client'];
  callId: string;
  session: WorkbenchViewProps['session'];
  onJump: (turnId: number | null | undefined, timeHint?: number | null) => void;
  onWrote: () => void;
}) {
  const latestEvaluation = call.evaluation;
  const [selectedRubricId, setSelectedRubricId] = useState<string | null>(null);
  // Existing immutable evaluation-version reads let us show real rubric applications without
  // rerunning the call. Keep the most recent application of each distinct rubric.
  const historyQuery = useQuery({
    queryKey: [...queryKeys.call(callId), 'rubric-assessments', latestEvaluation?.version],
    enabled: !!latestEvaluation && latestEvaluation.version > 1,
    staleTime: Infinity,
    queryFn: async ({ signal }) => {
      const history: EvaluationView[] = [];
      for (let end = latestEvaluation!.version - 1; end > 0; end -= 4) {
        const batch = await Promise.all(Array.from({ length: Math.min(4, end) }, (_, i) =>
          client.get('/store/v1/calls/{call_id}/evaluations/{version}', { path: { call_id: callId, version: end - i }, signal })
            .catch(error => { if (error?.status === 404) return null; throw error; }),
        ));
        history.push(...batch.filter((item): item is EvaluationView => item !== null));
      }
      return history;
    },
  });
  const assessments = useMemo(() => {
    const byRubric = new Map<string, EvaluationView>();
    for (const assessment of [latestEvaluation, ...(historyQuery.data ?? [])]) {
      if (assessment && !byRubric.has(assessment.rubric.rubric_id)) byRubric.set(assessment.rubric.rubric_id, assessment);
    }
    return [...byRubric.values()];
  }, [latestEvaluation, historyQuery.data]);
  const assessmentIndex = Math.max(0, assessments.findIndex(a => a.rubric.rubric_id === selectedRubricId));
  const machineEvaluation = assessments[assessmentIndex] ?? latestEvaluation;
  const isCurrent = machineEvaluation?.version === latestEvaluation?.version;
  const review = isCurrent ? latestReview : undefined;
  const currentOverrides = new Map((review?.overrides ?? []).filter(o => o.evaluation_version === machineEvaluation?.version).map(o => [o.criterion_id, o]));
  const reviewedScore = review?.reviewed_score?.evaluation_version === machineEvaluation?.version ? review?.reviewed_score : undefined;
  const evaluation = machineEvaluation && reviewedScore ? {
    ...machineEvaluation, ...reviewedScore,
    verdicts: machineEvaluation.verdicts.map(v => ({ ...v, status: currentOverrides.get(v.criterion_id)?.status ?? v.status })),
  } : machineEvaluation;
  const qaGroup = resultOf(call, 'qa');
  const canOverride = isCurrent && session.can('override_verdict');
  const canResolve = isCurrent && session.can('resolve_escalation');
  const canRetain = isCurrent && session.can('retain_review');
  // Read the immutable version used for this score, never the currently active rubric.
  const scoredRubric = useQuery({
    queryKey: ['scorecard-rubric', evaluation?.rubric.rubric_id, evaluation?.rubric.rubric_version, evaluation?.rubric.digest],
    enabled: !!evaluation?.rubric.rubric_version,
    queryFn: ({ signal }) => client.get('/store/v1/rubrics/{rubric_id}/versions/{version}', {
      path: { rubric_id: evaluation!.rubric.rubric_id, version: evaluation!.rubric.rubric_version! }, signal,
    }),
    staleTime: Infinity,
  });
  const rubricDefinition = scoredRubric.data?.ref.digest === evaluation?.rubric.digest ? scoredRubric.data?.definition : undefined;
  const scoredCriteria = rubricDefinition?.criteria ?? [];
  const earnedWeight = scoredCriteria.reduce((sum, criterion) => sum + (evaluation?.verdicts.find(v => v.criterion_id === criterion.criterion_id)?.status === 'PASS' ? criterion.weight : 0), 0);
  const eligibleWeight = scoredCriteria.reduce((sum, criterion) => sum + (['NOT_APPLICABLE', 'FLAGGED'].includes(evaluation?.verdicts.find(v => v.criterion_id === criterion.criterion_id)?.status ?? 'FLAGGED') ? 0 : criterion.weight), 0);

  const [detailsOpen, setDetailsOpen] = useState(false);
  const [selectedCriterion, setSelectedCriterion] = useState<string | null>(null);
  const criterionAnchors = useRef(new Map<string, HTMLButtonElement>());
  const openDetails = () => {
    setSelectedCriterion(null);
    setDetailsOpen(true);
  };
  const jumpFromDetails = (turnId: number | null | undefined, timeHint?: number | null) => {
    setDetailsOpen(false);
    setSelectedCriterion(null);
    onJump(turnId, timeHint);
  };

  const [overrideOpen, setOverrideOpen] = useState<string | null>(null);
  const [reasonCode, setReasonCode] = useState<OverrideReasonCode | ''>('');
  const [notes, setNotes] = useState('');
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<unknown>(null);
  const [actionNotice, setActionNotice] = useState<string | null>(null);
  useEffect(() => {
    setActionError(null);
    setActionNotice(null);
  }, [selectedCriterion]);


  const expectedVersion = review?.review_version ?? call.review_version;

  const openOverride = (criterionId: string) => {
    setOverrideOpen(criterionId);
    setReasonCode('');
    setNotes('');
    setActionError(null);
  };

  async function submitOverride(status: VerdictStatus) {
    if (!overrideOpen || !evaluation) return;
    setBusy(true);
    setActionError(null);
    try {
      await client.post('/store/v1/calls/{call_id}/verdicts/{criterion_id}', {
        path: { call_id: callId, criterion_id: overrideOpen },
        body: {
          evaluation_version: evaluation.version,
          expected_version: expectedVersion,
          status,
          reason_code: reasonCode || null,
          reviewer_notes: notes || null,
        },
      });
      setOverrideOpen(null);
      setActionError(null);
      setActionNotice('Override saved.');
      onWrote();
    } catch (err) {
      setActionError(err);
      if (isVersionConflict(err)) {
        setActionNotice(null);
        onWrote(); // re-read the call and its review state; the next submit uses the new versions
      }
    } finally {
      setBusy(false);
    }
  }

  async function submitEscalation(status: 'APPROVED' | 'OVERRIDDEN') {
    if (!evaluation) return;
    setBusy(true);
    setActionError(null);
    try {
      await client.post('/store/v1/calls/{call_id}/escalation', {
        path: { call_id: callId },
        body: { escalation_status: status, evaluation_version: evaluation.version, expected_version: expectedVersion, reviewer_notes: notes || null },
      });
      setActionNotice(`Escalation ${status === 'APPROVED' ? 'approved' : 'overridden'}.`);
      setNotes('');
      onWrote();
    } catch (err) {
      setActionError(err);
      if (isVersionConflict(err)) {
        setActionNotice(null);
        onWrote(); // re-read the call and its review state; the next submit uses the new versions
      }
    } finally {
      setBusy(false);
    }
  }

  async function submitRetain() {
    if (!review) return;
    setBusy(true);
    setActionError(null);
    try {
      await client.post('/store/v1/calls/{call_id}/review/retain', { path: { call_id: callId }, body: { expected_version: review.review_version, note: notes || null } });
      setActionNotice('Kept the existing review decisions.');
      onWrote();
    } catch (err) {
      setActionError(err);
      if (isVersionConflict(err)) {
        setActionNotice(null);
        onWrote(); // re-read the call and its review state; the next submit uses the new versions
      }
    } finally {
      setBusy(false);
    }
  }

  const renderVerdict = (v: EvaluationView['verdicts'][number], popout = false) => {
    const criterion = scoredCriteria.find(c => c.criterion_id === v.criterion_id);
    const override = currentOverrides.get(v.criterion_id);
    const effectiveStatus = override?.status ?? v.status;
    const d = verdictStatusDisplay(effectiveStatus);
    return (
      <div id={`scorecard-detail-${v.criterion_id}`} key={v.criterion_id} className={`text-sm space-y-2 ${popout ? '' : 'scroll-mt-3 rounded-md border border-border-muted bg-canvas p-2.5'}`}>
        <div className="flex items-center justify-between gap-2">
          <span className="font-medium text-fg">{popout ? 'Verdict' : v.criterion_name}</span>
          <StatusPill tone={d.tone} title={override ? `Overridden from ${verdictStatusDisplay(override.original_status).label}` : undefined}>
            {d.label}
            {override && ' (overridden)'}
          </StatusPill>
        </div>
        {criterion && <p className="text-xs text-fg-muted tabular-nums">{criterion.critical ? 'Critical check · ' : ''}{['NOT_APPLICABLE', 'FLAGGED'].includes(v.status) ? `${criterion.weight} weight excluded from score${v.status === 'FLAGGED' ? ' · awaiting review' : ''}` : `${v.status === 'PASS' ? criterion.weight : 0} / ${criterion.weight} weight earned`}{override ? ' · reviewed decision' : ''}</p>}
        {override?.reviewer_notes && <p className="text-fg leading-relaxed">Review: {override.reviewer_notes}</p>}
        <p className="text-fg-muted leading-relaxed">{override && 'Original machine assessment: '}{v.reasoning}</p>
        {v.quoted_evidence ? (
          <button
            type="button"
            onClick={() => jumpFromDetails(v.quote_turn_id, v.timestamp_range?.[0])}
            className="block w-full text-left rounded-md bg-canvas-inset border-l-2 border-primer-blue px-3 py-2 italic text-fg hover:bg-primer-blueSubtle transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
          >
            &ldquo;{v.quoted_evidence}&rdquo;
          </button>
        ) : (
          <p className="text-xs text-fg-subtle italic">No supporting quote.</p>
        )}
        <div className="text-xs text-fg-muted">Confidence: {Math.round(v.confidence * 100)}%</div>

        {canOverride && (
          <div className="pt-1.5 border-t border-border-muted">
            {overrideOpen === v.criterion_id ? (
              <div className="space-y-2">
                <Field label="Reason code">
                  {(id) => (
                    <SelectInput id={id} value={reasonCode} onChange={(e) => setReasonCode(e.target.value as OverrideReasonCode)}>
                      <option value="">No reason given</option>
                      {OVERRIDE_REASONS.map((r) => (
                        <option key={r} value={r}>
                          {r.replace(/_/g, ' ')}
                        </option>
                      ))}
                    </SelectInput>
                  )}
                </Field>
                <Field label="Notes">{(id) => <TextInput id={id} value={notes} maxLength={2000} onChange={(e) => setNotes(e.target.value)} />}</Field>
                <div className="flex gap-1.5">
                  <Button size="sm" variant="primary" busy={busy} onClick={() => void submitOverride('PASS')}>
                    Pass
                  </Button>
                  <Button size="sm" variant="danger" busy={busy} onClick={() => void submitOverride('FAIL')}>
                    Fail
                  </Button>
                  <Button size="sm" busy={busy} onClick={() => void submitOverride('NOT_APPLICABLE')}>
                    N/A
                  </Button>
                  <Button size="sm" variant="ghost" disabled={busy} onClick={() => setOverrideOpen(null)}>
                    Cancel
                  </Button>
                </div>
              </div>
            ) : (
              <Button size="sm" variant="ghost" onClick={() => openOverride(v.criterion_id)}>
                Override
              </Button>
            )}
          </div>
        )}
      </div>
    );
  };


  const fullScorecard = (
    <Card
      title="Assessment details"
      subtitle={evaluation ? `Rubric ${evaluation.rubric.rubric_id}${evaluation.rubric.rubric_version != null ? ` · v${evaluation.rubric.rubric_version}` : ''}` : undefined}
      right={<SectionStatePill group={qaGroup} />}
    >
      {!isCurrent && <Notice tone="neutral">Earlier rubric application · assessment v{machineEvaluation?.version}. Review actions are available on the current assessment.</Notice>}
      {!qaGroup || qaGroup.state === 'disabled' ? (
        <EmptyState title="No scorecard requested">QA scoring has not run for this call.</EmptyState>
      ) : !evaluation && qaGroup.state === 'failed' ? (
        <EmptyState title="Scoring stopped">
          {resultOf(call, 'transcript')?.state === 'failed'
            ? 'There is no transcript to score, so this call has no scorecard.'
            : `QA scoring stopped without a scorecard${qaGroup.failure_code ? `: ${failureReason(qaGroup.failure_code)}` : ''}. An admin can retry it.`}
        </EmptyState>
      ) : !evaluation ? (
        <EmptyState title="Scoring in progress">The scorecard will appear once QA finishes.</EmptyState>
      ) : (
        <div className="space-y-3">
          <div className="flex items-center justify-between rounded-md border border-border-muted bg-canvas p-2.5">
            <div>
              <div className="text-2xl font-bold text-fg tabular-nums">{evaluation.verdicts.some(v => v.status === 'PASS' || v.status === 'FAIL') ? formatScore(evaluation.overall_score) : '—'}</div>
              <div className="text-xs text-fg-muted">{evaluation.verdicts.some(v => v.status === 'FLAGGED') ? 'Provisional · assessed checks only' : reviewedScore ? 'Reviewed score · out of 100' : 'out of 100'}</div>
            </div>
            <StatusPill tone={evaluation.critical_failure ? 'red' : evaluation.requires_human_review ? 'yellow' : evaluation.passed ? 'green' : 'red'}>
              {evaluation.critical_failure ? 'Critical fail' : evaluation.requires_human_review ? 'Needs Review' : evaluation.passed ? 'Pass' : 'Fail'}
            </StatusPill>
          </div>

          <div data-testid="score-explanation" className="rounded-md border border-border-muted bg-canvas p-3 space-y-3 text-sm">
            <p className="font-medium text-fg">How this score is calculated</p>
            <div className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
              {(['PASS', 'FAIL', 'FLAGGED', 'NOT_APPLICABLE'] as const).map(status => (
                <div key={status} className="rounded bg-canvas-inset px-2 py-1.5">
                  <span className="font-semibold tabular-nums text-fg">{evaluation.verdicts.filter(v => v.status === status).length}</span>{' '}
                  <span className="text-fg-muted">{({ PASS: 'passed', FAIL: 'failed', FLAGGED: 'need review', NOT_APPLICABLE: 'not applicable' })[status]}</span>
                </div>
              ))}
            </div>
            {rubricDefinition ? (
              <div className="space-y-1">
                <p className="font-medium text-fg tabular-nums">{eligibleWeight > 0 ? `${earnedWeight} earned weight ÷ ${eligibleWeight} assessed weight × 100 = ${formatScore(evaluation.overall_score)}` : 'No assessed checks to score yet'}</p>
                <p className="text-fg-muted">Passed checks earn their full weight. Confirmed failures earn zero. Checks needing review and not-applicable checks are excluded from the calculation.</p>
                <p className="text-fg-muted">To pass: at least {rubricDefinition.pass_threshold || 80}/100, no critical failures, and no checks requiring human review.</p>
              </div>
            ) : (
              <p className="text-fg-muted">Passed checks earn their weight; confirmed failures earn zero. Checks needing review and not-applicable checks are excluded. {scoredRubric.isFetching ? 'Loading rubric weights…' : 'The exact rubric weights are unavailable.'}</p>
            )}
            {evaluation.critical_failure && (
              <div className="border-l-2 border-primer-redBorder pl-2 text-primer-redFg">
                <p className="font-medium">A critical failure prevents a pass, regardless of the numeric score.</p>
                {scoredCriteria.filter(c => c.critical && evaluation.verdicts.some(v => v.criterion_id === c.criterion_id && v.status === 'FAIL')).map(c => <p key={c.criterion_id}>Failed critical check: {c.name}.</p>)}
              </div>
            )}
            {evaluation.verdicts.some(v => v.status === 'FLAGGED') && <p className="text-primer-yellowFg">This score is provisional. Unresolved checks do not lower it or earn passing credit. The call remains Needs Review until those checks are settled.</p>}
            {reviewedScore && <p className="text-fg-muted">This score includes current reviewer decisions. Original machine score: {formatScore(machineEvaluation!.overall_score)}. The original assessment and review history are preserved.</p>}
          </div>

          {review?.staleness === 'stale' && (
            <Notice tone="magenta">
              A newer machine result exists than the one these decisions were made against.
              {canRetain && (
                <div className="mt-1.5">
                  <Button size="sm" busy={busy} onClick={() => void submitRetain()}>
                    Keep existing decisions
                  </Button>
                </div>
              )}
            </Notice>
          )}
          {review?.staleness === 'retained' && <Notice tone="neutral">These decisions were retained despite a newer machine version.</Notice>}

          {isCurrent && machineEvaluation?.requires_human_review && (
            <div data-testid="escalation-banner" className="flex items-center justify-between rounded-md border border-primer-yellowBorder bg-primer-yellowSubtle px-2.5 py-2">
              <span className="text-xs font-medium text-primer-yellowFg">{escalationBannerText(review?.escalation_status ?? 'PENDING')}</span>
              {canResolve && (review?.escalation_status ?? 'PENDING') === 'PENDING' && (
                <div className="flex gap-1.5">
                  <Button size="sm" variant="primary" busy={busy} onClick={() => void submitEscalation('APPROVED')}>
                    Approve
                  </Button>
                  <Button size="sm" variant="danger" busy={busy} onClick={() => void submitEscalation('OVERRIDDEN')}>
                    Override
                  </Button>
                </div>
              )}
            </div>
          )}

          {reviewLoading && <Loading label="Loading review state…" />}
          <ErrorNotice error={reviewError} />
          <ErrorNotice error={actionError} />
          {actionNotice && (
            <Notice tone="green" icon={CheckCircle2}>
              {actionNotice}
            </Notice>
          )}

          <div className="space-y-2">
            {evaluation.verdicts.map(v => renderVerdict(v))}
          </div>
        </div>
      )}
    </Card>
  );

  const selectedIndex = evaluation?.verdicts.findIndex(v => v.criterion_id === selectedCriterion) ?? -1;
  const selectedVerdict = selectedIndex >= 0 ? evaluation?.verdicts[selectedIndex] : undefined;

  return (
    <>
      <Card title="Scorecard" className="[&>div]:p-3" right={evaluation ? <Button size="sm" onClick={() => openDetails()}>Review scorecard</Button> : <SectionStatePill group={qaGroup} />}>
        {evaluation ? (
          <div className="space-y-1.5" role="region" aria-roledescription="carousel" aria-label="Applied rubrics">
            <div className="flex items-center justify-between gap-2">
              <div className="min-w-0">
                <p className="text-xs font-semibold text-fg truncate" title={rubricDefinition?.name ?? evaluation.rubric.rubric_id}>{rubricDefinition?.name ?? evaluation.rubric.rubric_id}</p>
                <p className="text-[10px] text-fg-muted">Rubric v{evaluation.rubric.rubric_version} · {isCurrent ? 'Current assessment' : `Earlier application · assessment v${evaluation.version}`}</p>
              </div>
              <div className="flex items-center gap-1 shrink-0">
                <Button size="sm" variant="ghost" aria-label="Previous rubric" icon={ChevronLeft} disabled={assessmentIndex === 0} onClick={() => setSelectedRubricId(assessments[assessmentIndex - 1].rubric.rubric_id)} />
                <span className="text-[10px] text-fg-muted tabular-nums" aria-live="polite">{assessmentIndex + 1} / {assessments.length}</span>
                <Button size="sm" variant="ghost" aria-label="Next rubric" icon={ChevronRight} disabled={assessmentIndex === assessments.length - 1} onClick={() => setSelectedRubricId(assessments[assessmentIndex + 1].rubric.rubric_id)} />
              </div>
            </div>
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-baseline gap-2">
                <span className="text-3xl font-bold text-fg tabular-nums">{evaluation.verdicts.some(v => v.status === 'PASS' || v.status === 'FAIL') ? formatScore(evaluation.overall_score) : '—'}</span>
                <span className="text-xs text-fg-muted">{evaluation.requires_human_review ? 'provisional' : reviewedScore ? 'reviewed / 100' : '/ 100'}</span>
              </div>
              <StatusPill tone={evaluation.critical_failure ? 'red' : evaluation.requires_human_review ? 'yellow' : evaluation.passed ? 'green' : 'red'}>
                {evaluation.critical_failure ? 'Critical fail' : evaluation.requires_human_review ? 'Needs Review' : evaluation.passed ? 'Pass' : 'Fail'}
              </StatusPill>
            </div>
            <div className="flex gap-1.5 overflow-x-auto pb-0.5" role="group" aria-label="Criterion results">
              {evaluation.verdicts.map((v, index) => {
                const status = verdictStatusDisplay(v.status);
                const color = { PASS: 'bg-primer-greenSubtle text-primer-greenFg border-primer-greenBorder', FAIL: 'bg-primer-redSubtle text-primer-redFg border-primer-redBorder', FLAGGED: 'bg-primer-yellowSubtle text-primer-yellowFg border-primer-yellowBorder', NOT_APPLICABLE: 'bg-canvas-inset text-fg-muted border-border' }[v.status];
                return <button key={v.criterion_id} type="button" ref={el => { if (el) criterionAnchors.current.set(v.criterion_id, el); else criterionAnchors.current.delete(v.criterion_id); }} onClick={() => setSelectedCriterion(selectedCriterion === v.criterion_id ? null : v.criterion_id)} aria-haspopup="dialog" aria-expanded={selectedCriterion === v.criterion_id} aria-controls={selectedCriterion === v.criterion_id ? 'scorecard-criterion-popout' : undefined} title={`${v.criterion_name}: ${status.label}`} aria-label={`Review ${v.criterion_name}`} className={`h-7 w-7 shrink-0 rounded border text-xs font-semibold focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue hover:brightness-95 ${selectedCriterion === v.criterion_id ? 'ring-2 ring-primer-blue ring-offset-2 ring-offset-canvas-subtle' : ''} ${color}`}>
                  <span aria-hidden="true">{index + 1}</span><span className="sr-only">{status.label}</span>
                </button>;
              })}
            </div>
            <p className="text-[11px] text-fg-muted" data-testid="scorecard-counts">{evaluation.verdicts.filter(v => v.status === 'PASS').length} passed · {evaluation.verdicts.filter(v => v.status === 'FAIL').length} failed · {evaluation.verdicts.filter(v => v.status === 'FLAGGED').length} need review · {evaluation.verdicts.filter(v => v.status === 'NOT_APPLICABLE').length} N/A</p>
            {historyQuery.isFetching && <p className="text-xs text-fg-muted">Loading applied rubrics…</p>}
            <ErrorNotice error={historyQuery.error} />
          </div>
        ) : <EmptyState title={!qaGroup || qaGroup.state === 'disabled' ? 'No scorecard requested' : qaGroup.state === 'failed' ? 'Scoring stopped' : 'Scoring in progress'}>The scorecard appears here once QA finishes.</EmptyState>}
      </Card>
      {selectedVerdict && evaluation && <ScorecardPopout key={selectedVerdict.criterion_id}
        anchor={criterionAnchors.current.get(selectedVerdict.criterion_id) ?? null}
        title={selectedVerdict.criterion_name}
        eyebrow={`${rubricDefinition?.name ?? evaluation.rubric.rubric_id} · Check ${selectedIndex + 1} of ${evaluation.verdicts.length}`}
        onClose={() => setSelectedCriterion(null)}
        footer={<>
          <Button size="sm" variant="ghost" icon={ChevronLeft} aria-label="Previous criterion" disabled={selectedIndex === 0 || busy} onClick={() => setSelectedCriterion(evaluation.verdicts[selectedIndex - 1].criterion_id)}>Previous</Button>
          <span className="text-xs text-fg-muted tabular-nums">{selectedIndex + 1} / {evaluation.verdicts.length}</span>
          <Button size="sm" variant="ghost" icon={ChevronRight} aria-label="Next criterion" disabled={selectedIndex === evaluation.verdicts.length - 1 || busy} onClick={() => setSelectedCriterion(evaluation.verdicts[selectedIndex + 1].criterion_id)}>Next</Button>
        </>}>
        {!isCurrent && <Notice tone="neutral">Earlier rubric application · read-only assessment.</Notice>}
        {renderVerdict(selectedVerdict, true)}
        <ErrorNotice error={actionError} />
        {actionNotice && <Notice tone="green" icon={CheckCircle2}>{actionNotice}</Notice>}
      </ScorecardPopout>}
      {detailsOpen && <Dialog title="Review scorecard" size="wide" onClose={() => setDetailsOpen(false)}>{fullScorecard}</Dialog>}
    </>
  );
}

// --- reanalysis ---------------------------------------------------------------------------------

function ReanalysisMenu({
  client,
  callId,
  session,
  onWrote,
}: {
  client: WorkbenchViewProps['client'];
  callId: string;
  session: WorkbenchViewProps['session'];
  onWrote: () => void;
}) {
  const trigger = useRef<HTMLButtonElement>(null);
  const [requestOpen, setRequestOpen] = useState(false);
  const [kind, setKind] = useState<ReanalysisKind>('full');
  const [note, setNote] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [notice, setNotice] = useState<string | null>(null);
  // One key per logical request: reused for a retry of the same body, replaced when kind or note
  // changes (Store digests the body under the key) and after a success.
  const idempotency = useIdempotencyKey();

  if (!session.can('request_reanalysis')) return null;

  async function submit() {
    setBusy(true);
    setError(null);
    setNotice(null);
    const body = { kind, note: note || null, rescore_signals: false };
    try {
      await sendIdempotent(idempotency, { callId, ...body }, (key) =>
        client.post('/store/v1/calls/{call_id}/reanalysis-requests', {
          path: { call_id: callId },
          headers: { 'Idempotency-Key': key },
          body,
        }),
      );
      setNotice('Reanalysis requested.');
      setNote('');
      onWrote();
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  }

  const options: { kind: ReanalysisKind; label: string }[] = [
    { kind: 'full', label: 'Full (everything)' },
    { kind: 'qa', label: 'QA only' },
    { kind: 'summary', label: 'Summary only' },
    { kind: 'contact_signals', label: 'Contact signals only' },
    { kind: 'embeddings', label: 'Search index only (re-embed)' },
  ];
  const closeRequest = () => {
    setRequestOpen(false);
    // The menu item that opened this dialog no longer exists. Restore focus to its trigger.
    queueMicrotask(() => trigger.current?.focus());
  };

  return (
    <>
      <DropdownMenu.Root>
        <DropdownMenu.Trigger asChild>
          <button
            ref={trigger}
            type="button"
            disabled={busy}
            className="inline-flex h-8 items-center gap-2 rounded-md border border-border bg-canvas px-3 text-sm font-medium text-fg transition-colors hover:bg-canvas-inset focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue disabled:opacity-50"
          >
            <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />
            Reanalysis
            <ChevronDown className="h-3.5 w-3.5 text-fg-muted" aria-hidden="true" />
          </button>
        </DropdownMenu.Trigger>
        <DropdownMenu.Portal>
          <DropdownMenu.Content
            align="end"
            sideOffset={6}
            collisionPadding={12}
            onCloseAutoFocus={(event) => { if (requestOpen) event.preventDefault(); }}
            className="z-50 min-w-60 max-w-[calc(100vw-24px)] rounded-lg border border-border bg-canvas p-1.5 shadow-xl"
          >
            <DropdownMenu.Label className="px-2.5 py-2 text-xs font-medium text-fg-muted">
              Reanalyze this call
            </DropdownMenu.Label>
            {options.map((option) => (
              <DropdownMenu.Item
                key={option.kind}
                onSelect={() => {
                  setKind(option.kind);
                  setError(null);
                  setNotice(null);
                  setRequestOpen(true);
                }}
                className="cursor-pointer rounded-md px-2.5 py-2 text-sm text-fg outline-none data-[highlighted]:bg-canvas-inset data-[highlighted]:text-primer-blueFg"
              >
                {option.label}
              </DropdownMenu.Item>
            ))}
          </DropdownMenu.Content>
        </DropdownMenu.Portal>
      </DropdownMenu.Root>
      {requestOpen && (
        <Dialog title="Request reanalysis" onClose={closeRequest} footer={
          <>
            <Button onClick={closeRequest}>Close</Button>
            <Button variant="primary" icon={RefreshCw} busy={busy} onClick={() => void submit()}>Request</Button>
          </>
        }>
          <Field label="Kind">
            {(id) => (
              <SelectInput id={id} value={kind} disabled={busy} onChange={(e) => setKind(e.target.value as ReanalysisKind)}>
                {options.map((option) => <option key={option.kind} value={option.kind}>{option.label}</option>)}
              </SelectInput>
            )}
          </Field>
          <Field label="Note" hint="Optional; kept with the request.">
            {(id) => <TextInput id={id} value={note} disabled={busy} maxLength={1000} onChange={(e) => setNote(e.target.value)} />}
          </Field>
          <ErrorNotice error={error} />
          {notice && <Notice tone="green" icon={CheckCircle2}>{notice}</Notice>}
        </Dialog>
      )}
    </>
  );
}
