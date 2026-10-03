import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';
import { renameEntryHtml } from './scripts/rename-entry-html';

// The Store operations console: read-only health/coverage/admin panels served by Store itself
// at /console/. Build into call1/store/static/console/ (docs/SplitBuild.md "Frontend layout").
// Run `npm run copy:store-static` first (or `npm run build:store-console`, which does it for
// you) so the shared fonts/icons land at call1/store/static/ before this build's index.html
// references them.
export default defineConfig({
  base: '/console/',
  plugins: [react(), renameEntryHtml('store-console.html')],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  build: {
    outDir: '../call1/store/static/console',
    emptyOutDir: true,
    assetsDir: 'assets',
    sourcemap: false,
    rollupOptions: {
      input: fileURLToPath(new URL('./store-console.html', import.meta.url)),
    },
  },
  server: {
    port: 5174,
    proxy: {
      '/store': 'http://127.0.0.1:8010',
    },
  },
});
