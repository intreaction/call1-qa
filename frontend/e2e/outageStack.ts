/**
 * A private Store+Process stack that a single spec file can start, break (stop Store) and heal
 * (start Store) on command — for testing a real Store outage, which the shared harness stack
 * (fixtures.ts / harness.ts, one stack per whole run) has no way to do: nothing in the shared
 * fixtures stops Store, and stopping it would break every other test sharing that stack.
 *
 * This spawns `tests/e2e/outage_stack.py` (new test infrastructure, not part of the harness under
 * test) the same way `e2e/global-setup.ts` spawns `serve_stack.py`: one JSON "ready" line, then a
 * newline-delimited JSON command protocol over stdin/stdout. Only `consoles.spec` (store-console
 * spec's c5 test) uses this; every other test uses the shared `stack` fixture.
 */
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..', '..');
const E2E_ROOT = '/private/tmp/call1-e2e';

export interface OutageStackInfo {
  ready: boolean;
  dir: string;
  store_url: string;
  store_port: number;
  process_url: string;
  process_port: number;
  process_console_url: string;
  console_token: string;
  installation_id: string;
  service_key: string;
  [extra: string]: unknown;
}

export class OutageStack {
  readonly info: OutageStackInfo;
  private readonly child: ChildProcessWithoutNullStreams;
  private readonly rl: readline.Interface;
  private readonly waiters: Array<(line: string) => void> = [];
  private readonly runDir: string;

  private constructor(child: ChildProcessWithoutNullStreams, info: OutageStackInfo, runDir: string) {
    this.child = child;
    this.info = info;
    this.runDir = runDir;
    this.rl = readline.createInterface({ input: this.child.stdout });
    this.rl.on('line', (line) => {
      const next = this.waiters.shift();
      if (next) next(line);
    });
  }

  static async start(name = 'outage'): Promise<OutageStack> {
    fs.mkdirSync(E2E_ROOT, { recursive: true });
    const runDir = fs.mkdtempSync(path.join(E2E_ROOT, `outage-${name}-`));
    const infoFile = path.join(runDir, 'stack.json');
    const python = process.env.CALL1_E2E_PYTHON || path.join(REPO, '.venv-local', 'bin', 'python');
    const child = spawn(python, [path.join(REPO, 'tests', 'e2e', 'outage_stack.py'), '--name', name, '--info-file', infoFile], {
      cwd: REPO,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    const logPath = path.join(runDir, 'outage_stack.log');
    const log = fs.openSync(logPath, 'a');
    child.stderr.on('data', (chunk: Buffer) => fs.writeSync(log, chunk));

    const info = await new Promise<OutageStackInfo>((resolve, reject) => {
      const lines = readline.createInterface({ input: child.stdout });
      const timer = setTimeout(() => reject(new Error('outage_stack.py did not report within 120 s')), 120_000);
      lines.once('line', (line) => {
        clearTimeout(timer);
        lines.close(); // hand stdout to the instance's own reader below
        try {
          const parsed = JSON.parse(line) as OutageStackInfo;
          if (!parsed.ready) reject(new Error(`outage stack failed to start: ${String(parsed.error)}`));
          else resolve(parsed);
        } catch {
          reject(new Error(`outage_stack.py printed something other than JSON: ${line}`));
        }
      });
      child.once('exit', (code) => {
        clearTimeout(timer);
        reject(new Error(`outage_stack.py exited (${code}) before it was up; see ${logPath}`));
      });
    });
    return new OutageStack(child, info, runDir);
  }

  private send(cmd: string): Promise<{ ok: boolean; cmd?: string; error?: string }> {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(`outage stack command ${cmd} timed out`)), 30_000);
      this.waiters.push((line) => {
        clearTimeout(timer);
        try {
          resolve(JSON.parse(line));
        } catch {
          reject(new Error(`outage stack sent a non-JSON reply to ${cmd}: ${line}`));
        }
      });
      this.child.stdin.write(JSON.stringify({ cmd }) + '\n');
    });
  }

  async stopStore(): Promise<void> {
    const r = await this.send('stop_store');
    if (!r.ok) throw new Error(`stop_store failed: ${r.error}`);
  }

  async startStore(): Promise<void> {
    const r = await this.send('start_store');
    if (!r.ok) throw new Error(`start_store failed: ${r.error}`);
  }

  /** Re-issue Process's service key without `jobs:control` and restart Process to load it. */
  async downgradeScope(): Promise<void> {
    const r = await this.send('downgrade_scope');
    if (!r.ok) throw new Error(`downgrade_scope failed: ${r.error}`);
  }

  async close(): Promise<void> {
    const exited = new Promise<void>((resolve) => {
      if (this.child.exitCode !== null) resolve();
      else this.child.once('exit', () => resolve());
    });
    this.rl.close();
    this.child.stdin.end();
    const timer = setTimeout(() => this.child.kill('SIGTERM'), 15_000);
    await exited;
    clearTimeout(timer);
    // outage_stack.py's own `stack.close()` already removed its Stack data dir; this removes the
    // small runDir this helper made for stack.json/outage_stack.log.
    if (process.env.CALL1_E2E_KEEP !== '1') fs.rmSync(this.runDir, { recursive: true, force: true });
  }
}

/** Upload a WAV directly to a private outage stack's Process (no shared-stack `ingestSample`). */
export async function ingestOn(stack: OutageStack, wavPath: string, agentId: string): Promise<{ conversation_id: string; call_id: string }> {
  const bytes = fs.readFileSync(wavPath);
  const form = new FormData();
  form.append('file', new Blob([new Uint8Array(bytes)], { type: 'audio/wav' }), path.basename(wavPath));
  form.append('agent_id', agentId);
  const response = await fetch(`${stack.info.process_url}/process/api/recordings`, {
    method: 'POST',
    headers: { 'X-Call1-Console-Token': stack.info.console_token },
    body: form,
  });
  if (response.status !== 201) throw new Error(`ingest answered ${response.status}: ${await response.text()}`);
  return (await response.json()) as { conversation_id: string; call_id: string };
}

/** Poll a private outage stack's Store progress (service key) until settled. */
export async function waitUntilSettledOn(stack: OutageStack, conversationId: string, timeoutMs = 60_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const response = await fetch(`${stack.info.store_url}/store/v1/conversations/${conversationId}/progress`, {
      headers: { Authorization: `Bearer ${stack.info.service_key}` },
    });
    if (response.ok) {
      const body = (await response.json()) as { settled: boolean };
      if (body.settled) return;
    }
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error(`conversation ${conversationId} on the outage stack never settled`);
}
