#!/usr/bin/env node
// Removes an app's previous hashed bundles before a build whose outDir is shared and therefore
// built with `emptyOutDir: false` (the Process console: call1/process/static also holds the
// copied fonts and icons). Without this, every build leaves its old `<prefix>-<hash>.js/.css`
// behind. Only files named `<prefix>-*.js` or `<prefix>-*.css` directly in the assets directory
// are removed; nothing else is touched.
//
// Usage: node scripts/clean-app-assets.mjs <assets-dir> <prefix>

import { existsSync, readdirSync, rmSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const frontendRoot = path.resolve(here, '..');
const [dirArg, prefix] = process.argv.slice(2);
if (!dirArg || !prefix || !/^[A-Za-z0-9_-]+$/.test(prefix)) {
  console.error('usage: clean-app-assets.mjs <assets-dir> <prefix>');
  process.exit(1);
}
const dir = path.resolve(frontendRoot, dirArg);
if (!existsSync(dir)) process.exit(0);

const pattern = new RegExp(`^${prefix}-[A-Za-z0-9_-]+\\.(js|css)$`);
let removed = 0;
for (const name of readdirSync(dir)) {
  if (name.includes(' ') || !pattern.test(name)) continue; // never touch iCloud "* 2.*" copies
  rmSync(path.join(dir, name));
  removed += 1;
}
console.log(`removed ${removed} stale ${prefix} bundle(s) from ${path.relative(frontendRoot, dir)}`);
