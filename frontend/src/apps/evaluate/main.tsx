// Evaluate: the reviewer browser app, served by Store at `/` and talking only to `/store/v1`
// (docs/SplitBuild.md), with one documented exception: `/demo/*` (api/demo.ts) for the
// localhost-only demo mode. See README.md in this directory.
import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import App from './App';
import { StoreClient, isSignedOutError, StoreError } from './api';
import { wireConnectivity } from './state/connectivity';
import { normalizeInitialLocation } from './state/router';
import '@/index.css';

normalizeInitialLocation();

const client = new StoreClient();
wireConnectivity(client);

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 15_000,
      refetchOnWindowFocus: false,
      // Never retry a definitive answer (4xx); retry transient ones twice.
      retry: (count, err) => !isSignedOutError(err) && !(err instanceof StoreError && err.status < 500) && count < 2,
    },
    mutations: { retry: false },
  },
});

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App client={client} />
    </QueryClientProvider>
  </React.StrictMode>,
);
