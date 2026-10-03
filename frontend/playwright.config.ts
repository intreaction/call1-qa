import os from 'node:os';
import path from 'node:path';
import { defineConfig } from '@playwright/test';

// Test real local services with scripted handlers; no model downloads or production data.
const ARTIFACTS = path.join(process.env.CALL1_E2E_ROOT || path.join(os.tmpdir(), 'call1-e2e'), 'playwright');
const CHANNEL = process.env.CALL1_E2E_BROWSER || (process.env.CI ? 'chromium' : 'chrome');

// Unique per invocation by default: two runs that overlap (e.g. two writers testing in parallel)
// used to share `test-results/`, which Playwright empties at the start of a run — a run in
// progress would then lose in-flight trace/screenshot files out from under it (ENOENT). Set
// CALL1_E2E_RUN_ID to pin a value (or pass --output on the CLI, which still wins over this).
const RUN_ID = process.env.CALL1_E2E_RUN_ID || `${Date.now()}-${process.pid}`;

export default defineConfig({
  testDir: './e2e',
  testMatch: /.*\.spec\.ts$/,
  // iCloud conflict copies ("x.spec 2.ts", gitignored) must never run.
  testIgnore: /\s\d+\.ts$/,
  outputDir: `${ARTIFACTS}/test-results/${RUN_ID}`,
  globalSetup: './e2e/global-setup.ts',
  timeout: 90_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: 0,
  workers: process.env.CALL1_E2E_WORKERS ? Number(process.env.CALL1_E2E_WORKERS) : 3,
  reporter: [['list'], ['html', { outputFolder: `${ARTIFACTS}/report`, open: 'never' }]],
  use: {
    channel: CHANNEL,
    headless: true,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'off',
    actionTimeout: 15_000,
    navigationTimeout: 30_000,
  },
  projects: [
    { name: 'dark', use: { channel: CHANNEL, colorScheme: 'dark' } },
    { name: 'light', use: { channel: CHANNEL, colorScheme: 'light' } },
  ],
});
