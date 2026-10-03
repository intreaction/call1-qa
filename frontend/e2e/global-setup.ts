/**
 * Starts one Store + Process stack for the whole Playwright run by running
 * tests/e2e/serve_stack.py (the Python harness), which prints the stack's ports and tokens as one
 * JSON line and keeps both servers alive until its stdin closes. The facts go to
 * $CALL1_E2E_STACK_FILE (read by e2e/harness.ts in every worker). The returned teardown stops the
 * stack and deletes the run directory (set CALL1_E2E_KEEP=1 to keep both for a post-mortem).
 *
 * Environment:
 *   CALL1_E2E_PYTHON   interpreter (default <repo>/.venv-local/bin/python)
 *   CALL1_E2E_HANDLERS fake (default) | real;  with CALL1_REAL_MODELS=1 and real: --real-models
 *   CALL1_E2E_KEEP=1   keep /private/tmp/call1-e2e/pw-run-… and the stack directory
 */
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..', '..');
const E2E_ROOT = '/private/tmp/call1-e2e';

function newestMtime(dir: string): number {
  let newest = 0;
  if (!fs.existsSync(dir)) return newest;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (/ \d+\./.test(entry.name)) continue; // iCloud conflict copies
    const full = path.join(dir, entry.name);
    newest = Math.max(newest, entry.isDirectory() ? newestMtime(full) : fs.statSync(full).mtimeMs);
  }
  return newest;
}

function warnIfEvaluateBuildIsStale(): void {
  const built = path.join(REPO, 'call1', 'store', 'static', 'evaluate', 'index.html');
  if (!fs.existsSync(built)) {
    console.warn('[e2e] Evaluate is not built (call1/store/static/evaluate/index.html): run `npm run build:evaluate` first.');
    return;
  }
  const source = Math.max(newestMtime(path.join(REPO, 'frontend', 'src', 'apps', 'evaluate')), newestMtime(path.join(REPO, 'frontend', 'src', 'contracts')));
  if (source > fs.statSync(built).mtimeMs) {
    console.warn('[e2e] Evaluate sources are newer than the build Store serves; the tests exercise the BUILT app. Run `npm run build:evaluate` to test the current source.');
  }
}

export default async function globalSetup(): Promise<() => Promise<void>> {
  warnIfEvaluateBuildIsStale();
  fs.mkdirSync(E2E_ROOT, { recursive: true });
  const runDir = fs.mkdtempSync(path.join(E2E_ROOT, `pw-run-${new Date().toISOString().replace(/[:.]/g, '').slice(0, 15)}-`));
  const infoFile = path.join(runDir, 'stack.json');
  const python = process.env.CALL1_E2E_PYTHON || path.join(REPO, '.venv-local', 'bin', 'python');
  const args = [path.join(REPO, 'tests', 'e2e', 'serve_stack.py'), '--name', 'playwright', '--info-file', infoFile];
  if (process.env.CALL1_E2E_HANDLERS === 'real') {
    args.push('--handlers', 'real');
    if (process.env.CALL1_REAL_MODELS === '1') args.push('--real-models');
  }
  const log = fs.openSync(path.join(runDir, 'serve_stack.log'), 'a');
  const child: ChildProcessWithoutNullStreams = spawn(python, args, { cwd: REPO, stdio: ['pipe', 'pipe', 'pipe'] });
  child.stderr.on('data', (chunk: Buffer) => fs.writeSync(log, chunk));

  const info = await new Promise<Record<string, unknown>>((resolve, reject) => {
    const lines = readline.createInterface({ input: child.stdout });
    const timer = setTimeout(() => reject(new Error('serve_stack.py did not report within 240 s')), 240_000);
    lines.once('line', (line) => {
      clearTimeout(timer);
      try {
        resolve(JSON.parse(line) as Record<string, unknown>);
      } catch {
        reject(new Error(`serve_stack.py printed something other than JSON: ${line}`));
      }
    });
    child.once('exit', (code) => {
      clearTimeout(timer);
      reject(new Error(`serve_stack.py exited (${code}) before the stack was up; see ${path.join(runDir, 'serve_stack.log')}`));
    });
  });
  if (!info.ready) throw new Error(`the e2e stack did not start: ${String(info.error)}`);

  process.env.CALL1_E2E_STACK_FILE = infoFile;
  process.env.CALL1_E2E_RUN_DIR = runDir;
  console.log(`[e2e] Store ${String(info.store_url)}  Process ${String(info.process_url)}  (stack ${String(info.dir)})`);

  return async () => {
    const exited = new Promise<void>((resolve) => {
      if (child.exitCode !== null) resolve();
      else child.once('exit', () => resolve());
    });
    child.stdin.end(); // serve_stack.py stops the stack when its stdin closes
    const timer = setTimeout(() => child.kill('SIGTERM'), 20_000);
    await exited;
    clearTimeout(timer);
    fs.closeSync(log);
    if (process.env.CALL1_E2E_KEEP !== '1') fs.rmSync(runDir, { recursive: true, force: true });
    else console.log(`[e2e] kept ${runDir} and ${String(info.dir)}`);
  };
}
