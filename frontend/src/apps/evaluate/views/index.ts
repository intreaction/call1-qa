// One file per view; the shell imports them from here. See ../README.md "View contract".
export { default as CallsView } from './CallsView';
export { default as WorkbenchView } from './WorkbenchView';
export { default as RubricsView } from './RubricsView';
export { default as SignalsView } from './SignalsView';
export { default as QueueView } from './QueueView';
export { default as EscalationsView } from './EscalationsView';
import { lazy } from 'react';
export const MetricsView = lazy(() => import('./MetricsView'));
export type * from './types';
