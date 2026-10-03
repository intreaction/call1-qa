/**
 * A private Store + Process stack for one test (tests/e2e/serve_stack.py with scripted fake
 * behaviour), for flows the shared stack cannot show: a job that fails on purpose, a job held
 * RUNNING long enough to cancel. Data lives under /private/tmp/call1-e2e/ and is deleted on close.
 *
 *   const priv = await startPrivateStack({ fakeBehavior: { acoustic_tone: ['fail:validation_rejected'] } });
 *   try { ... priv.info.process_console_url ... } finally { await priv.close(); }
 */
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { stackInfo, uniqueWav, samplePath, type IngestReceipt, type StackInfo } from './harness';

export interface PrivateStack {
  info: StackInfo;
  /** Process's loopback API on this stack (console token sent unless `token: false`). */
  process(method: string, p: string, options?: { json?: unknown; token?: boolean }): Promise<Response>;
  /** Store with this stack's service key. */
  service(method: string, p: string, body?: unknown): Promise<Response>;
  ingest(sample?: string, options?: { agentId?: string; filename?: string }): Promise<IngestReceipt>;
  close(): Promise<void>;
}

export async function startPrivateStack(options: {
  name?: string;
  fakeBehavior?: Record<string, string[]>;
  storeParameters?: Record<string, number>;
  processConfig?: Record<string, unknown>;
  /** Extra Store environment variables (e.g. `{ CALL1_STORE_DEMO: '1' }` for demo.spec.ts). */
  storeEnv?: Record<string, string>;
  /** Extra Process environment variables (e.g. on-device training's `CALL1_FAKE_TRAINING_OUTCOMES`). */
  processEnv?: Record<string, string>;
  /** Ingest calls and override their QA verdicts as a reviewer before reporting ready, until both
   * on-device training splits have a labelled call (tests/e2e/training_seed.py). */
  seedTrainingLabels?: boolean;
  /** Store only, no Process — for flows that need no queue/job data (demo.spec.ts). */
  noProcess?: boolean;
  /** The Contact Signals pipeline setting (default: the run's CALL1_SIGNALS_PIPELINE, else v1). */
  signalsPipeline?: 'v1' | 'shadow' | 'v2';
  /** A seed taxonomy (repo-relative SignalTaxonomySave JSON) to publish first, as `--demo` does. */
  signalsSeed?: string;
  /** An ASR vocabulary seed (repo-relative JSON) to install as the industry pack first, as `--demo`
   * does (docs/DualAsr.md): every ingest on the stack then plans dual transcription. */
  vocabularySeed?: string;
} = {}): Promise<PrivateStack> {
  const shared = stackInfo();
  const args = [path.join(shared.repo, 'tests', 'e2e', 'serve_stack.py'), '--name', options.name ?? 'pw-private'];
  if (options.fakeBehavior) args.push('--fake-behavior', JSON.stringify(options.fakeBehavior));
  if (options.storeParameters) args.push('--store-parameters', JSON.stringify(options.storeParameters));
  if (options.processConfig) args.push('--process-config', JSON.stringify(options.processConfig));
  if (options.storeEnv) args.push('--store-env', JSON.stringify(options.storeEnv));
  if (options.processEnv) args.push('--process-env', JSON.stringify(options.processEnv));
  if (options.seedTrainingLabels) args.push('--seed-training-labels');
  if (options.noProcess) args.push('--no-process');
  if (options.signalsPipeline) args.push('--signals-pipeline', options.signalsPipeline);
  if (options.signalsSeed) args.push('--signals-seed', path.join(shared.repo, options.signalsSeed));
  if (options.vocabularySeed) args.push('--vocabulary-seed', path.join(shared.repo, options.vocabularySeed));
  const child = spawn(shared.python, args, { cwd: shared.repo, stdio: ['pipe', 'pipe', 'pipe'] });
  const logPath = path.join(shared.run_dir, `${options.name ?? 'pw-private'}-${process.pid}-${Date.now()}.log`);
  const log = fs.openSync(logPath, 'a');
  child.stderr.on('data', (chunk: Buffer) => fs.writeSync(log, chunk));
  const info = await new Promise<StackInfo>((resolve, reject) => {
    const lines = readline.createInterface({ input: child.stdout });
    const timer = setTimeout(() => reject(new Error('private serve_stack.py did not report within 90 s')), 90_000);
    lines.once('line', (line) => {
      clearTimeout(timer);
      const parsed = JSON.parse(line) as StackInfo & { error?: string };
      if (!parsed.ready) reject(new Error(`private stack did not start: ${parsed.error}`));
      else resolve({ ...parsed, run_dir: shared.run_dir });
    });
    child.once('exit', (code) => {
      clearTimeout(timer);
      reject(new Error(`private serve_stack.py exited (${code}); see ${logPath}`));
    });
  });

  const processCall = (method: string, p: string, o: { json?: unknown; token?: boolean } = {}) => {
    const headers: Record<string, string> = {};
    if (o.token !== false) headers['X-Call1-Console-Token'] = info.console_token;
    if (o.json !== undefined) headers['Content-Type'] = 'application/json';
    return fetch(`${info.process_url}/process/api${p}`, { method, headers, body: o.json !== undefined ? JSON.stringify(o.json) : undefined });
  };
  return {
    info,
    process: processCall,
    service: (method, p, body) =>
      fetch(`http://127.0.0.1:${info.store_port}/store/v1${p}`, {
        method,
        headers: { Authorization: `Bearer ${info.service_key}`, ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}) },
        body: body !== undefined ? JSON.stringify(body) : undefined,
      }),
    async ingest(sample = 'call_01_compliant', o = {}) {
      const bytes = uniqueWav(fs.readFileSync(samplePath(sample)));
      const form = new FormData();
      form.append('file', new Blob([new Uint8Array(bytes)], { type: 'audio/wav' }), o.filename ?? `${sample}.wav`);
      if (o.agentId) form.append('agent_id', o.agentId);
      const response = await fetch(`${info.process_url}/process/api/recordings`, {
        method: 'POST',
        headers: { 'X-Call1-Console-Token': info.console_token },
        body: form,
      });
      const body = (await response.json()) as IngestReceipt;
      if (response.status !== 201) throw new Error(`private ingest answered ${response.status}: ${JSON.stringify(body)}`);
      return body;
    },
    async close() {
      const exited = new Promise<void>((resolve) => (child.exitCode !== null ? resolve() : child.once('exit', () => resolve())));
      child.stdin.end();
      const timer = setTimeout(() => child.kill('SIGTERM'), 20_000);
      await exited;
      clearTimeout(timer);
      fs.closeSync(log);
    },
  };
}
