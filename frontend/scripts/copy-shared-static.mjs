// Copy shared fonts and icons from frontend/public into the served apps.
import { existsSync, mkdirSync, cpSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const frontendRoot = path.resolve(here, '..');
const source = path.resolve(frontendRoot, 'public');

const FILES = [
  'favicon.ico',
  'apple-touch-icon.png',
  'call1-dark-32.png',
  'call1-dark-192.png',
  'call1-light-32.png',
  'call1-light-192.png',
];
const DIRS = ['fonts'];

const targets = process.argv.slice(2).map((t) => path.resolve(frontendRoot, t));
if (targets.length === 0) {
  console.error('usage: copy-shared-static.mjs <target-static-dir> [...more]');
  process.exit(1);
}
if (!existsSync(source)) {
  console.error(`source not found: ${source} (expected frontend/public with fonts/ and icons)`);
  process.exit(1);
}

for (const target of targets) {
  mkdirSync(target, { recursive: true });
  for (const file of FILES) {
    const from = path.join(source, file);
    if (!existsSync(from)) continue; // optional icon variants
    cpSync(from, path.join(target, file));
  }
  for (const dir of DIRS) {
    const from = path.join(source, dir);
    if (!existsSync(from)) continue;
    cpSync(from, path.join(target, dir), { recursive: true });
  }
  console.log(`copied shared static assets -> ${path.relative(frontendRoot, target)}`);
}
