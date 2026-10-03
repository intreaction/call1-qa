import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';

// The Process operator console, served by Process itself at `/` (docs/SplitBuild.md "Frontend
// layout"). Source: src/apps/process/. Use `npm run build:process`, which copies the shared fonts
// and icons in and removes the previous process-* bundles before building.
export default defineConfig({
  base: '/',
  plugins: [react()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  build: {
    outDir: '../call1/process/static',
    emptyOutDir: false, // shared with the copied fonts and icons; build:process cleans old process-* bundles first
    assetsDir: 'assets',
    sourcemap: false,
    rollupOptions: {
      input: fileURLToPath(new URL('./process.html', import.meta.url)),
    },
  },
  server: {
    port: 5176,
    proxy: {
      '/process': 'http://127.0.0.1:8020',
    },
  },
});
