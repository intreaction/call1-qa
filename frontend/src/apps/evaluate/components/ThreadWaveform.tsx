// ThreadWaveform — Evaluate's port of the legacy "thread-inspired" three.js waveform
// (frontend/src/components/workbench/WaveformDeck.tsx, left untouched for the legacy app).
//
// Data-agnostic: it takes plain props, never imports a legacy module and never fetches anything
// itself. The caller hands it `loadAudio`, which in the Workbench reads
// `GET /store/v1/calls/{id}/audio` through the typed Store client (same origin, session cookie) —
// Store serves PII-muted audio, and the muted spans come out of the decoded PCM as a flat thread.
//
// Ported as-is: the PCM -> RMS envelope (AudioContext decode, AbortController, generation guard),
// the two-layer ShaderMaterial/DataTexture ribbon, the hover wake, the click ripple, the playhead
// and hover line, the time tip, the detail drawer, theme sync, visibility pausing (Intersection
// Observer + document.hidden), reduced motion, context-loss handling and disposal.
// Changed: markers come from props (contact signals and verdict evidence the Workbench already
// has) in a lane above the stage instead of pins inside the slider; a marker with an `end` (a
// contact-signals v2 span) also draws a range band in that lane, under its pin; a speaker lane under the ribbon
// shows transcript turns; digital-silence runs are hatched and labelled "silent or muted"; Space
// plays/pauses on the stage (Enter shows the detail); a native range input and a text state replace
// the stage when WebGL or decoding is unavailable, so the box is never blank.

import { useEffect, useRef, useState } from 'react';
import type * as THREE from 'three';
import { Flag, MapPin, Quote, X } from 'lucide-react';
import { COUNT, STAGE_BUFFER_H, computeEnvelope, createScene, disposeScene, syncPalette, type Scene3 } from './threadWaveformGl';

export interface WaveformTurn {
  id: number;
  start: number;
  end: number;
  /** 'AGENT' | 'CALLER' | anything else (drawn neutral). */
  speaker: string;
  channel?: number | null;
  text?: string;
}

export type WaveformMarkerTone = 'blue' | 'green' | 'red' | 'yellow' | 'magenta' | 'neutral';

export interface WaveformMarker {
  id: string;
  time: number;
  /** A span's end (contact signals v2): the marker also draws a range band in the lane, under its
   * pin. Clustering still uses `time`. */
  end?: number | null;
  kind: 'signal' | 'evidence';
  /** Short label for the pin, e.g. "Caller objective" or a criterion name. */
  label: string;
  /** Status text shown next to the label, e.g. "Pass" — color is never the only signal. */
  status?: string;
  tone: WaveformMarkerTone;
  quote?: string | null;
  speaker?: string | null;
  turnId?: number | null;
  /** A second label level shown after "›", e.g. a signal's subcategory. */
  detail?: string | null;
  /** Short text chips, e.g. extracted fields ("reason: price"). */
  chips?: string[];
  /** The color family in words (e.g. "Lifecycle", "Custom"), so color is never the only signal. */
  family?: string | null;
  /** Stable ID for the band's test hook (a signal's hit ID). */
  hitId?: string | null;
  /** How many segments a multi-segment contact signal spans (decision 25); shown as "×N segments"
   * when above 1, and the band then runs to the last segment's end (`end`). */
  segments?: number | null;
}

function segmentsText(m: Pick<WaveformMarker, 'segments'>): string | null {
  return m.segments != null && m.segments > 1 ? `×${m.segments} segments` : null;
}

export interface ThreadWaveformProps {
  /** Identifies the recording (the audio URL); a change reloads the envelope. */
  sourceKey: string;
  /** Reads the recording's bytes (same-origin, with credentials). */
  loadAudio: (signal: AbortSignal) => Promise<ArrayBuffer>;
  /** Text for a failed `loadAudio`; defaults to a generic line. */
  describeLoadError?: (error: unknown) => string;
  audioRef: React.RefObject<HTMLAudioElement | null>;
  currentTime: number;
  duration: number;
  /** Used for the time axis until the audio element reports its duration. */
  fallbackDuration?: number | null;
  playing: boolean;
  /** The audio element failed: no seeking, a text state instead. */
  audioError?: boolean;
  /** Text shown when `audioError` is set. */
  unavailableText?: string;
  onSeek: (seconds: number) => void;
  onTogglePlay: () => void;
  turns?: WaveformTurn[];
  markers?: WaveformMarker[];
  /** "Jump to moment" on a marker; defaults to `onSeek(marker.time)`. */
  onMarkerJump?: (marker: WaveformMarker) => void;
}

// Pins are at most 8.5rem wide and centred, so clusters closer than that would overlap.
const CLUSTER_PX = 140;
const SEEK_STEP = 5;
const SEEK_STEP_LARGE = 15;

function clock(seconds: number): string {
  if (!seconds || !Number.isFinite(seconds)) return '0:00';
  return `${Math.floor(seconds / 60)}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`;
}

// --- speaker mapping (ported from legacy; per-turn `channel` is trusted, not the layout) --------

function channelSpeakers(turns: WaveformTurn[]) {
  if (!turns.length) return null;
  const perChannel: Record<number, Set<string>> = {};
  const roles = new Set<string>();
  for (const t of turns) {
    if (t.channel == null || !Number.isInteger(t.channel) || t.channel < 0) continue;
    if (t.speaker !== 'AGENT' && t.speaker !== 'CALLER') continue;
    roles.add(t.speaker);
    (perChannel[t.channel] ??= new Set()).add(t.speaker);
  }
  for (const t of turns) if (t.speaker === 'AGENT' || t.speaker === 'CALLER') roles.add(t.speaker);
  const exclusive: Record<number, string | null> = {};
  for (const ch in perChannel) exclusive[ch] = perChannel[ch].size === 1 ? (perChannel[ch].values().next().value ?? null) : null;
  return { exclusive, roles };
}

const roleName = (r: string | null | undefined) => (r === 'AGENT' ? 'Agent' : r === 'CALLER' ? 'Caller' : null);

function channelInfo(turns: WaveformTurn[], channelCount: number) {
  const info = channelSpeakers(turns);
  const stereo = channelCount > 1;
  const roles: number[] = [];
  for (let ch = 0; ch < 2; ch++) roles.push(info?.exclusive[stereo ? ch : 0] === 'CALLER' ? 1 : 0);
  let labels: [string, string | null];
  let status: string;
  if (stereo) {
    const r0 = info?.exclusive[0] ? roleName(info.exclusive[0]) : null;
    const r1 = info?.exclusive[1] ? roleName(info.exclusive[1]) : null;
    const known = !!(r0 && r1 && r0 !== r1);
    labels = [known ? r0! : 'Channel 1', known ? r1! : 'Channel 2'];
    status = known ? `Stereo · ${r0} · ${r1}` : 'Stereo · speaker mapping unavailable';
  } else {
    const a = info?.roles.has('AGENT');
    const c = info?.roles.has('CALLER');
    labels = [a && c ? 'Mixed audio' : a ? 'Agent' : c ? 'Caller' : 'Mono', null];
    status = a && c ? 'Mono · mixed audio (Caller + Agent)' : 'Mono · single channel';
  }
  return { roles, labels, status };
}

// --- marker clusters (ported) ------------------------------------------------------------------

interface Cluster {
  key: string;
  frac: number;
  markers: WaveformMarker[];
}

function clusterMarkers(markers: WaveformMarker[], total: number, width: number): Cluster[] {
  if (!markers.length || total <= 0) return [];
  const w = width > 0 ? width : 600;
  const positioned = markers
    .map((m) => ({ m, px: (Math.min(Math.max(m.time, 0), total) / total) * w * 0.92 }))
    .sort((a, b) => a.px - b.px);
  const clusters: Cluster[] = [];
  let current: typeof positioned = [];
  let anchor = 0;
  const flush = () => {
    if (!current.length) return;
    const px = current.reduce((s, it) => s + it.px, 0) / current.length;
    clusters.push({ key: `c-${current[0].m.id}`, frac: px / (w * 0.92), markers: current.map((it) => it.m) });
  };
  for (const item of positioned) {
    if (!current.length || item.px - anchor <= CLUSTER_PX) {
      if (!current.length) anchor = item.px;
      current.push(item);
    } else {
      flush();
      current = [item];
      anchor = item.px;
    }
  }
  flush();
  return clusters;
}

const TONE: Record<WaveformMarkerTone, string> = {
  blue: 'border-primer-blueBorder bg-primer-blueSubtle text-primer-blueFg',
  green: 'border-primer-greenBorder bg-primer-greenSubtle text-primer-greenFg',
  red: 'border-primer-redBorder bg-primer-redSubtle text-primer-redFg',
  yellow: 'border-primer-yellowBorder bg-primer-yellowSubtle text-primer-yellowFg',
  magenta: 'border-primer-magentaBorder bg-primer-magentaSubtle text-primer-magentaFg',
  neutral: 'border-border bg-canvas-inset text-fg-muted',
};

const SPEAKER_TEXT: Record<string, string> = { AGENT: 'Agent', CALLER: 'Caller', SYSTEM: 'System', UNKNOWN: 'Unknown' };

/** Stage x (0..1 of the ribbon) <-> CSS left %, matching the ribbon's 4%/96% faded ends. */
const pct = (frac: number) => 4 + Math.max(0, Math.min(1, frac)) * 92;

// --- engine ------------------------------------------------------------------------------------

interface Engine {
  gl: Scene3 | null;
  envelope: Float32Array;
  frame: number;
  previous: number;
  time: number;
  hover: number;
  targetHover: number;
  pointerX: number;
  pointerY: number;
  cursor: number;
  clickTime: number;
  visible: boolean;
  reduced: MediaQueryList;
  controller: AbortController | null;
  generation: number;
  context: AudioContext | null;
  duration: number;
}

type GlState = 'ok' | 'unavailable' | 'lost';
type LoadState = 'loading' | 'ready' | 'failed';

function prefersReduced(): MediaQueryList {
  try {
    return window.matchMedia('(prefers-reduced-motion: reduce)');
  } catch {
    return { matches: false, addEventListener() {}, removeEventListener() {} } as unknown as MediaQueryList;
  }
}

export default function ThreadWaveform({
  sourceKey,
  loadAudio,
  describeLoadError,
  audioRef,
  currentTime,
  duration,
  fallbackDuration,
  playing,
  audioError = false,
  unavailableText = 'Audio unavailable for this call. Use the transcript to review it.',
  onSeek,
  onTogglePlay,
  turns = [],
  markers = [],
  onMarkerJump,
}: ThreadWaveformProps) {
  const stageRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const playheadRef = useRef<HTMLDivElement>(null);
  const hoverlineRef = useRef<HTMLDivElement>(null);
  const tipRef = useRef<HTMLSpanElement>(null);
  const engineRef = useRef<Engine | null>(null);

  const [glState, setGlState] = useState<GlState>('ok');
  const [loadState, setLoadState] = useState<LoadState>('loading');
  const [loadError, setLoadError] = useState('');
  const [channels, setChannels] = useState(0);
  const [silentRuns, setSilentRuns] = useState<Array<[number, number]>>([]);
  const [reduced, setReduced] = useState(() => prefersReduced().matches);
  const [info, setInfo] = useState<{ title: string; text: string; quiet: boolean } | null>(null);
  const [openCluster, setOpenCluster] = useState<string | null>(null);
  const [laneWidth, setLaneWidth] = useState(600);

  // Latest render-scope values for the animation loop and event handlers.
  const stateRef = useRef({ currentTime, duration, playing, fallbackDuration });
  stateRef.current = { currentTime, duration, playing, fallbackDuration };
  const turnsRef = useRef(turns);
  turnsRef.current = turns;

  const total = (): number => {
    const s = stateRef.current;
    if (Number.isFinite(s.duration) && s.duration > 0) return s.duration;
    const e = engineRef.current?.duration ?? 0;
    if (e > 0) return e;
    return s.fallbackDuration && s.fallbackDuration > 0 ? s.fallbackDuration : 0;
  };

  const updatePlayhead = () => {
    const playhead = playheadRef.current;
    if (!playhead) return;
    const d = total();
    playhead.style.left = `${pct(d ? stateRef.current.currentTime / d : 0)}%`;
  };

  const requestDraw = (engine: Engine) => {
    updatePlayhead();
    if (!engine.frame && engine.gl && engine.visible && !document.hidden) {
      engine.frame = requestAnimationFrame((now) => draw(engine, now));
    }
  };

  const draw = (engine: Engine, now: number) => {
    engine.frame = 0;
    const dt = engine.previous ? Math.min((now - engine.previous) / 1000, 0.05) : 0;
    engine.previous = now;
    const still = engine.reduced.matches;
    if (!still) engine.time += dt;
    engine.hover += (engine.targetHover - engine.hover) * Math.min(1, dt * 10);
    engine.cursor += (engine.pointerX - engine.cursor) * Math.min(1, dt * 14);
    const d = total();
    const progress = d ? Math.min(1, stateRef.current.currentTime / d) : 0;
    if (!engine.gl) return;
    for (const mesh of engine.gl.layers) {
      const u = (mesh.material as THREE.ShaderMaterial).uniforms;
      u.uTime.value = engine.time;
      u.uCursor.value = engine.cursor;
      u.uHover.value = still ? 0 : engine.hover;
      u.uY.value = engine.pointerY;
      u.uClick.value = engine.clickTime;
      u.uProgress.value = progress;
    }
    engine.gl.renderer.render(engine.gl.scene, engine.gl.camera);
    updatePlayhead();
    if (!still && (engine.hover > 0.001 || engine.targetHover || stateRef.current.playing || engine.time - engine.clickTime < 2)) {
      requestDraw(engine);
    }
  };

  // Mount: WebGL scene, observers, listeners. React owns the canvas, so reconciliation never
  // clobbers the WebGL surface. Cleanup disposes everything.
  useEffect(() => {
    const stage = stageRef.current;
    const canvas = canvasRef.current;
    if (!stage || !canvas) return;
    const engine: Engine = {
      gl: null,
      envelope: new Float32Array(COUNT * 4),
      frame: 0,
      previous: 0,
      time: 0,
      hover: 0,
      targetHover: 0,
      pointerX: 0.5,
      pointerY: 0,
      cursor: 0.5,
      clickTime: -10,
      visible: true,
      reduced: prefersReduced(),
      controller: null,
      generation: 0,
      context: null,
      duration: 0,
    };
    engineRef.current = engine;
    try {
      engine.gl = createScene(canvas, engine.envelope);
    } catch {
      engine.gl = null;
    }
    setGlState(engine.gl ? 'ok' : 'unavailable');

    const onReducedChange = () => {
      setReduced(engine.reduced.matches);
      requestDraw(engine);
    };
    engine.reduced.addEventListener('change', onReducedChange);

    let resizeObserver: ResizeObserver | null = null;
    let intersectionObserver: IntersectionObserver | null = null;
    const onContextLost = (event: Event) => {
      event.preventDefault();
      cancelAnimationFrame(engine.frame);
      engine.frame = 0;
      if (engine.gl) disposeScene(engine.gl);
      engine.gl = null;
      setGlState('lost');
    };
    const onTheme = () => {
      if (!engine.gl) return;
      syncPalette(engine.gl.layers);
      requestDraw(engine);
    };
    const themeObserver = new MutationObserver(onTheme);
    const onVisibility = () => {
      engine.previous = 0;
      requestDraw(engine);
    };

    if (engine.gl) {
      syncPalette(engine.gl.layers);
      // Stable backing buffer: resize only when the width changes (reallocating clears it).
      let bufferW = 0;
      resizeObserver = new ResizeObserver(() => {
        const w = stage.clientWidth;
        if (!engine.gl || !w) return;
        if (w !== bufferW) {
          bufferW = w;
          engine.gl.renderer.setSize(w, STAGE_BUFFER_H, false);
        }
        requestDraw(engine);
      });
      resizeObserver.observe(stage);
      intersectionObserver = new IntersectionObserver((entries) => {
        engine.visible = entries[0]?.isIntersecting ?? true;
        requestDraw(engine);
      });
      intersectionObserver.observe(stage);
      canvas.addEventListener('webglcontextlost', onContextLost);
      window.addEventListener('call1:theme-change', onTheme);
      themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
      document.addEventListener('visibilitychange', onVisibility);
      requestDraw(engine);
    }

    return () => {
      cancelAnimationFrame(engine.frame);
      engine.controller?.abort();
      engine.reduced.removeEventListener('change', onReducedChange);
      resizeObserver?.disconnect();
      intersectionObserver?.disconnect();
      themeObserver.disconnect();
      canvas.removeEventListener('webglcontextlost', onContextLost);
      window.removeEventListener('call1:theme-change', onTheme);
      document.removeEventListener('visibilitychange', onVisibility);
      if (engine.gl) disposeScene(engine.gl);
      engine.gl = null;
      void engine.context?.close().catch(() => undefined);
      engine.context = null;
      engineRef.current = null;
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Lane width for marker clustering.
  const laneRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = laneRef.current ?? stageRef.current;
    if (!el) return;
    setLaneWidth(el.clientWidth || 600);
    const ro = new ResizeObserver((entries) => setLaneWidth(entries[0]?.contentRect.width || el.clientWidth || 600));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Keep the loop live through playback and seeking: the element's own events (as legacy) and the
  // parent's state, whichever arrives first.
  useEffect(() => {
    const engine = engineRef.current;
    if (engine) requestDraw(engine);
  }, [currentTime, duration, playing]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    const onAudioEvent = () => {
      const engine = engineRef.current;
      if (engine) requestDraw(engine);
    };
    const events = ['timeupdate', 'seeked', 'play', 'pause', 'loadedmetadata', 'durationchange'] as const;
    for (const ev of events) audio.addEventListener(ev, onAudioEvent);
    return () => {
      for (const ev of events) audio.removeEventListener(ev, onAudioEvent);
    };
  }, [audioRef, sourceKey]); // eslint-disable-line react-hooks/exhaustive-deps

  // Load the recording and compute the envelope when the source changes.
  useEffect(() => {
    const engine = engineRef.current;
    if (!engine || !engine.gl) return;
    const version = ++engine.generation;
    engine.controller?.abort();
    const controller = new AbortController();
    engine.controller = controller;
    engine.duration = 0;
    engine.envelope.fill(0);
    engine.gl.texture.needsUpdate = true;
    setLoadState('loading');
    setLoadError('');
    setSilentRuns([]);
    setInfo(null);
    requestDraw(engine);

    (async () => {
      try {
        const bytes = await loadAudio(controller.signal);
        if (controller.signal.aborted || version !== engine.generation) return;
        const Ctor = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
        if (!Ctor) throw new Error('This browser cannot decode audio for the waveform.');
        engine.context ??= new Ctor();
        const buffer = await engine.context.decodeAudioData(bytes);
        if (version !== engine.generation || !engine.gl) return;
        engine.duration = buffer.duration;
        setSilentRuns(computeEnvelope(buffer, engine.envelope));
        engine.gl.texture.needsUpdate = true;
        setChannels(buffer.numberOfChannels);
        setLoadState('ready');
        requestDraw(engine);
      } catch (error) {
        if (version !== engine.generation || controller.signal.aborted) return;
        setLoadError(describeLoadError ? describeLoadError(error) : '');
        setLoadState('failed');
      }
    })();

    return () => controller.abort();
  }, [sourceKey, glState]); // eslint-disable-line react-hooks/exhaustive-deps

  // Speaker roles follow the transcript, which may arrive after the audio.
  const mapping = channelInfo(turns, channels);
  useEffect(() => {
    const gl = engineRef.current?.gl;
    if (!gl) return;
    gl.layers.forEach((mesh, ch) => {
      (mesh.material as THREE.ShaderMaterial).uniforms.uRole.value = mapping.roles[ch];
    });
    requestDraw(engineRef.current!);
  }, [mapping.roles[0], mapping.roles[1]]); // eslint-disable-line react-hooks/exhaustive-deps

  // --- interaction (ported) ---

  const position = (clientX: number, clientY: number) => {
    const engine = engineRef.current;
    const stage = stageRef.current;
    if (!engine || !stage) return;
    const bounds = stage.getBoundingClientRect();
    engine.pointerX = Math.max(0, Math.min(1, ((clientX - bounds.left) / bounds.width - 0.04) / 0.92));
    engine.pointerY = 1 - (2 * (clientY - bounds.top)) / bounds.height;
    if (hoverlineRef.current) hoverlineRef.current.style.left = `${pct(engine.pointerX)}%`;
    if (tipRef.current) {
      tipRef.current.style.left = `${Math.max(12, Math.min(88, pct(engine.pointerX)))}%`;
      tipRef.current.textContent = clock(engine.pointerX * total());
    }
  };

  const envelopeAt = (frac: number) => {
    const engine = engineRef.current;
    if (!engine) return 0;
    const i = Math.max(0, Math.min(COUNT - 1, Math.floor(frac * COUNT)));
    return Math.max(engine.envelope[i * 4], engine.envelope[i * 4 + 1]);
  };

  const reveal = (seconds: number, seek = true) => {
    const engine = engineRef.current;
    const d = total();
    if (!engine || !d) return;
    const t = Math.max(0, Math.min(d, seconds));
    if (seek) onSeek(t);
    const turn = turnsRef.current.find((tn) => t >= tn.start && t < tn.end);
    const frac = t / d;
    const quiet = loadState === 'ready' && silentRuns.some(([a, b]) => frac >= a && frac < b);
    setInfo({
      title: `${clock(t)} · ${turn ? (SPEAKER_TEXT[turn.speaker] ?? turn.speaker) : 'Between transcript turns'}`,
      text: turn?.text || 'No transcript segment at this moment. Play to hear the recording here.',
      quiet: quiet || (loadState === 'ready' && envelopeAt(frac) < 0.002),
    });
    engine.clickTime = engine.reduced.matches ? -10 : engine.time;
    requestDraw(engine);
  };

  const interactive = !audioError;

  const onPointerMove = (e: React.PointerEvent) => {
    const engine = engineRef.current;
    if (!engine || !interactive) return;
    position(e.clientX, e.clientY);
    engine.targetHover = 1;
    if (hoverlineRef.current) hoverlineRef.current.style.opacity = '1';
    if (tipRef.current) tipRef.current.style.opacity = '1';
    requestDraw(engine);
  };

  const onPointerLeave = () => {
    const engine = engineRef.current;
    if (!engine) return;
    engine.targetHover = 0;
    if (hoverlineRef.current) hoverlineRef.current.style.opacity = '0';
    if (tipRef.current) tipRef.current.style.opacity = '0';
    requestDraw(engine);
  };

  const onClick = (e: React.MouseEvent) => {
    const engine = engineRef.current;
    if (!engine || !interactive) return;
    position(e.clientX, e.clientY);
    reveal(engine.pointerX * total());
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    const engine = engineRef.current;
    if (!engine || !interactive) return;
    if (e.key === ' ') {
      e.preventDefault();
      onTogglePlay();
      return;
    }
    if (e.key === 'Escape') {
      if (info) {
        e.preventDefault();
        setInfo(null);
      }
      return;
    }
    const keys = ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'Home', 'End', 'Enter', 'j', 'l'];
    if (!keys.includes(e.key) || e.metaKey || e.ctrlKey || e.altKey) return;
    e.preventDefault();
    const d = total();
    let seconds = stateRef.current.currentTime;
    const step = e.shiftKey ? SEEK_STEP_LARGE : SEEK_STEP;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowDown' || e.key === 'j') seconds -= step;
    if (e.key === 'ArrowRight' || e.key === 'ArrowUp' || e.key === 'l') seconds += step;
    if (e.key === 'PageDown') seconds -= SEEK_STEP_LARGE;
    if (e.key === 'PageUp') seconds += SEEK_STEP_LARGE;
    if (e.key === 'Home') seconds = 0;
    if (e.key === 'End') seconds = d;
    engine.pointerX = d ? Math.max(0, Math.min(1, seconds / d)) : 0;
    if (hoverlineRef.current) hoverlineRef.current.style.opacity = '0';
    if (tipRef.current) tipRef.current.style.opacity = '0';
    reveal(seconds, e.key !== 'Enter');
  };

  // --- render ---

  const d = total();
  const shownDuration = d || 0;
  const mode: 'audio-error' | 'no-webgl' | 'no-waveform' | LoadState = audioError
    ? 'audio-error'
    : glState !== 'ok'
      ? 'no-webgl'
      : loadState === 'failed'
        ? 'no-waveform'
        : loadState;
  const fallback = mode === 'audio-error' || mode === 'no-webgl' || mode === 'no-waveform';
  const clusters = clusterMarkers(markers, shownDuration, laneWidth);
  const bands = shownDuration > 0 ? markers.filter((m) => m.end != null && m.end > m.time) : [];
  const status =
    mode === 'loading' ? 'Reading audio…' : mode === 'ready' ? mapping.status : mode === 'audio-error' ? 'Audio unavailable' : 'Waveform unavailable';

  const fallbackText =
    mode === 'audio-error'
      ? unavailableText
      : mode === 'no-webgl'
        ? glState === 'lost'
          ? 'The waveform stopped drawing (the graphics context was lost). Seek with the bar below.'
          : 'The waveform needs WebGL, which this browser has turned off. Seek with the bar below.'
        : `No waveform for this recording${loadError ? `: ${loadError}` : '.'} Seek with the bar below.`;

  const turnSpans = shownDuration > 0 ? turns.filter((t) => t.end > t.start) : [];

  return (
    <div
      className={`fabric-deck ${info && !fallback ? 'fabric-info-open' : ''} ${fallback ? 'fabric-error' : ''}`}
      data-testid="thread-waveform"
      data-state={mode}
      data-motion={reduced ? 'reduced' : 'full'}
    >
      {/* Legend and status strip */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 px-3 py-2 border-b border-border-muted text-[11px] text-fg-muted">
        {mode === 'ready' && (
          <>
            <span className="fabric-key">
              <span className={`fabric-dot ${mapping.roles[0] ? 'caller' : ''}`} aria-hidden="true" />
              {mapping.labels[0]}
            </span>
            {mapping.labels[1] && (
              <span className="fabric-key">
                <span className={`fabric-dot ${mapping.roles[1] ? 'caller' : ''}`} aria-hidden="true" />
                {mapping.labels[1]}
              </span>
            )}
          </>
        )}
        <span className="text-fg-subtle" data-testid="thread-waveform-status">
          {status}
        </span>
        {mode === 'ready' && silentRuns.length > 0 && (
          <span className="inline-flex items-center gap-1.5" title="Stretches of digital silence: Store mutes PII by zeroing the audio, so muted spans are flat">
            <span className="inline-block w-3 h-2.5 rounded-sm border border-border" style={{ backgroundImage: HATCH }} aria-hidden="true" />
            Flat = silent or muted ({silentRuns.length})
          </span>
        )}
        {reduced && !fallback && <span className="text-fg-subtle">Motion reduced</span>}
        {markers.length > 0 && (
          <span className="ml-auto">
            {markers.length} marker{markers.length === 1 ? '' : 's'}
          </span>
        )}
      </div>

      {/* Marker lane: contact signals and verdict evidence (ported milestone pins) */}
      {clusters.length > 0 && !fallback && (
        <div ref={laneRef} className={`relative ${bands.length ? 'h-10' : 'h-7'} mx-0 border-b border-border-muted`} aria-label="Waveform markers" role="group">
          {bands.map((m) => {
            const start = Math.max(0, Math.min(m.time, shownDuration));
            const end = Math.max(start, Math.min(m.end ?? m.time, shownDuration));
            const segs = segmentsText(m);
            const text = `${m.label}${m.detail ? ` › ${m.detail}` : ''}, ${clock(start)}–${clock(end)}${segs ? ` (${segs})` : ''}`;
            return (
              <div
                key={`band-${m.id}`}
                data-testid={`waveform-span-band-${m.hitId ?? m.id}`}
                data-signal-band={m.hitId ?? m.id}
                data-segments={m.segments ?? 1}
                title={text}
                aria-hidden="true"
                className={`absolute bottom-1 h-2 rounded-sm border opacity-80 pointer-events-none ${TONE[m.tone]}`}
                style={{ left: `${pct(start / shownDuration)}%`, width: `max(3px, ${((end - start) / shownDuration) * 92}%)` }}
              />
            );
          })}
          {clusters.map((cluster) => {
            const first = cluster.markers[0];
            const isOpen = openCluster === cluster.key;
            const Icon = first.kind === 'evidence' ? Quote : Flag;
            const label =
              cluster.markers.length === 1 ? `${first.status ? `${first.status} · ` : ''}${first.label}` : `${cluster.markers.length} markers`;
            // Near the ends, anchor the pin (and its popover) at its edge so it stays inside the deck.
            const edge = cluster.frac < 0.15 ? 'start' : cluster.frac > 0.85 ? 'end' : 'center';
            const shift = edge === 'start' ? '-8px' : edge === 'end' ? 'calc(-100% + 8px)' : '-50%';
            const popover = edge === 'start' ? 'left-0' : edge === 'end' ? 'right-0' : 'left-1/2 -translate-x-1/2';
            return (
              <div
                key={cluster.key}
                className="absolute top-1"
                style={{ left: `${pct(cluster.frac)}%`, transform: `translateX(${shift})`, zIndex: isOpen ? 40 : 20 }}
              >
                <button
                  type="button"
                  aria-expanded={isOpen}
                  aria-label={`${label} at ${clock(first.time)}`}
                  title={cluster.markers.map((m) => `${m.status ? `${m.status} · ` : ''}${m.label}${m.detail ? ` › ${m.detail}` : ''} (${clock(m.time)})`).join('\n')}
                  onClick={() => {
                    setOpenCluster(isOpen ? null : cluster.key);
                    onSeek(first.time);
                  }}
                  className={`flex items-center gap-1 px-1.5 py-0.5 rounded border text-[10px] font-semibold max-w-[8.5rem] ${TONE[first.tone]} ${
                    isOpen ? 'ring-2 ring-primer-blue' : ''
                  } focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue`}
                >
                  <Icon className="w-2.5 h-2.5 shrink-0" aria-hidden="true" />
                  <span className="truncate">{label}</span>
                </button>
                {isOpen && (
                  <div
                    className={`absolute ${popover} top-full mt-1.5 w-72 max-w-[90vw] p-2.5 rounded-lg bg-canvas border border-border shadow-xl space-y-2`}
                    onKeyDown={(e) => {
                      if (e.key === 'Escape') setOpenCluster(null);
                    }}
                  >
                    <div className="flex items-center justify-between pb-1 border-b border-border-muted text-[11px] text-fg-muted font-semibold">
                      <span>
                        {cluster.markers.length} marker{cluster.markers.length === 1 ? '' : 's'}
                      </span>
                      <button
                        type="button"
                        aria-label="Close markers"
                        onClick={() => setOpenCluster(null)}
                        className="text-fg-muted hover:text-fg p-0.5 rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
                      >
                        <X className="w-3.5 h-3.5" aria-hidden="true" />
                      </button>
                    </div>
                    <div className="max-h-60 overflow-y-auto space-y-1.5">
                      {cluster.markers.map((m) => (
                        <div key={m.id} className="p-2 rounded border border-border-muted bg-canvas-subtle space-y-1">
                          <div className="flex items-center gap-2 text-xs">
                            <span className="font-semibold text-fg">
                              {m.label}
                              {m.detail && <span className="font-normal text-fg-muted"> › {m.detail}</span>}
                            </span>
                            {m.status && <span className={`ml-auto px-1.5 py-0.5 rounded border text-[10px] font-semibold ${TONE[m.tone]}`}>{m.status}</span>}
                          </div>
                          <div className="text-[11px] text-fg-muted tabular-nums">
                            {m.kind === 'evidence' ? 'Verdict evidence' : 'Contact signal'}
                            {m.family ? ` · ${m.family}` : ''} · {clock(m.time)}
                            {m.end != null && m.end > m.time ? `–${clock(m.end)}` : ''}
                            {m.speaker ? ` · ${SPEAKER_TEXT[m.speaker] ?? m.speaker}` : ''}
                          </div>
                          {(segmentsText(m) || (m.chips && m.chips.length > 0)) && (
                            <div className="flex flex-wrap gap-1">
                              {segmentsText(m) && (
                                <span className="px-1.5 py-0.5 rounded border border-border bg-canvas-inset text-[10px] font-semibold text-fg" data-testid="marker-segments">
                                  {segmentsText(m)}
                                </span>
                              )}
                              {(m.chips ?? []).map((c) => (
                                <span key={c} className="px-1.5 py-0.5 rounded border border-border bg-canvas-inset text-[10px] font-medium text-fg-muted">
                                  {c}
                                </span>
                              ))}
                            </div>
                          )}
                          {m.quote && <blockquote className="text-xs text-fg italic border-l-2 border-border pl-2">&ldquo;{m.quote}&rdquo;</blockquote>}
                          <button
                            type="button"
                            onClick={() => {
                              if (onMarkerJump) onMarkerJump(m);
                              else onSeek(m.time);
                              setOpenCluster(null);
                            }}
                            className="inline-flex items-center gap-1.5 text-xs font-semibold text-primer-blueFg hover:underline rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
                          >
                            <MapPin className="w-3 h-3" aria-hidden="true" />
                            Jump to moment
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* The stage: the WebGL thread, the keyboard-reachable seek slider */}
      <div
        ref={stageRef}
        className="fabric-stage"
        style={{ height: 128 }}
        hidden={fallback}
        role="slider"
        tabIndex={fallback ? -1 : 0}
        aria-label="Seek audio"
        aria-valuemin={0}
        aria-valuemax={Math.round(shownDuration)}
        aria-valuenow={Math.round(Math.min(currentTime, shownDuration || currentTime))}
        aria-valuetext={`${clock(currentTime)} of ${clock(shownDuration)}`}
        aria-keyshortcuts="ArrowLeft ArrowRight Home End Space Enter"
        aria-describedby="thread-waveform-help"
        onPointerMove={onPointerMove}
        onPointerLeave={onPointerLeave}
        onClick={onClick}
        onKeyDown={onKeyDown}
      >
        <canvas ref={canvasRef} aria-hidden="true" />
        {mode === 'ready' &&
          silentRuns.map(([a, b]) => (
            <div
              key={`${a}-${b}`}
              className="thread-silence absolute top-2 bottom-2 pointer-events-none rounded-sm"
              style={{ left: `${pct(a)}%`, width: `${(b - a) * 92}%`, backgroundImage: HATCH }}
              title="Silent or muted"
              aria-hidden="true"
            />
          ))}
        {turnSpans.map((t) => (
          <div
            key={t.id}
            className={`absolute bottom-1 h-[3px] rounded-full pointer-events-none opacity-70 ${
              t.speaker === 'AGENT' ? 'bg-primer-blue' : t.speaker === 'CALLER' ? 'bg-primer-caller' : 'bg-fg-subtle'
            }`}
            style={{ left: `${pct(t.start / shownDuration)}%`, width: `${(Math.min(t.end, shownDuration) - t.start) / shownDuration * 92}%` }}
            aria-hidden="true"
          />
        ))}
        <div className="fabric-playhead" ref={playheadRef} aria-hidden="true" />
        <div className="fabric-hoverline" ref={hoverlineRef} aria-hidden="true" />
        <span className="fabric-tip" ref={tipRef} aria-hidden="true" />
      </div>
      <p id="thread-waveform-help" className="sr-only">
        Arrow keys seek 5 seconds, Shift plus arrow 15. Home and End jump to the ends. Space plays or pauses. Enter shows the transcript at the playhead.
      </p>

      {fallback && (
        <div className="px-3 py-3 space-y-2 bg-canvas-subtle rounded-b-lg" data-testid="thread-waveform-fallback">
          <p className="text-[13px] text-fg">{fallbackText}</p>
          {mode !== 'audio-error' && (
            <input
              type="range"
              aria-label="Seek audio"
              className="w-full accent-primer-blue"
              min={0}
              max={shownDuration}
              step={0.1}
              value={Math.min(currentTime, shownDuration)}
              onChange={(e) => onSeek(Number(e.target.value))}
            />
          )}
        </div>
      )}

      {!fallback && (
        <div className="fabric-info" aria-live="polite">
          <header>
            <span>{info?.title}</span>
            <button type="button" aria-label="Close waveform detail" tabIndex={info ? 0 : -1} onClick={() => setInfo(null)}>
              <X className="w-3.5 h-3.5 mx-auto" aria-hidden="true" />
            </button>
          </header>
          {info?.quiet && <p className="text-fg-muted">Flat here: silence, or audio Store muted for PII.</p>}
          <p>{info?.text}</p>
        </div>
      )}
    </div>
  );
}

const HATCH = 'repeating-linear-gradient(135deg, rgba(var(--fg-muted-rgb) / 0.22) 0 2px, transparent 2px 6px)';
