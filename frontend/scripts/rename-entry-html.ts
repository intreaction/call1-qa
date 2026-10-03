import { existsSync, renameSync } from 'node:fs';
import path from 'node:path';
import type { Plugin } from 'vite';

/**
 * Vite keeps an HTML entry's output filename equal to its source filename (`store-console.html`
 * in, `store-console.html` out). `call1/store/static.py`'s SpaSite looks for `index.html` inside
 * each app's own static subdirectory (or the app's flat `*.html` name only as the index — never
 * for its assets, which are always resolved under the subdirectory). Renaming the entry to
 * `index.html` after the build is the smallest change that fits `static.py` as committed,
 * without editing it.
 */
export function renameEntryHtml(from: string, to = 'index.html'): Plugin {
  return {
    name: 'call1-rename-entry-html',
    apply: 'build',
    writeBundle(options) {
      const dir = options.dir;
      if (!dir) return;
      const fromPath = path.join(dir, from);
      const toPath = path.join(dir, to);
      if (existsSync(fromPath)) {
        renameSync(fromPath, toPath);
      }
    },
  };
}
