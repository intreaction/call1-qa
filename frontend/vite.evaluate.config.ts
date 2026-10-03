import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';
import { renameEntryHtml } from './scripts/rename-entry-html';

// Evaluate: the reviewer browser app, served by Store at `/` (docs/SplitBuild.md "Frontend
// layout"). Source: src/apps/evaluate/. Run `npm run copy:store-static` first (or
// `npm run build:evaluate`, which does it for you).
//
// Dev server: http://localhost:5175 proxies /store/v1 to Store on 127.0.0.1:8010. Store checks the
// Origin of writes and the WebAuthn origin, so start Store with
//   CALL1_STORE_DEV_ORIGINS=http://localhost:5175 python -m call1.store serve
// and open the app at http://localhost:5175 (not 127.0.0.1: the relying-party ID is `localhost`).
export default defineConfig({
  base: '/',
  plugins: [react(), renameEntryHtml('evaluate.html')],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  build: {
    outDir: '../call1/store/static/evaluate',
    emptyOutDir: true,
    assetsDir: 'assets',
    sourcemap: false,
    rollupOptions: {
      input: fileURLToPath(new URL('./evaluate.html', import.meta.url)),
    },
  },
  server: {
    port: 5175,
    strictPort: true, // the origin must match CALL1_STORE_DEV_ORIGINS exactly
    proxy: {
      '/store/v1': 'http://127.0.0.1:8010',
    },
  },
});
