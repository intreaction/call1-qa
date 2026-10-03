// The contract between the Evaluate shell and its views. The shell (App.tsx) renders exactly one
// view per route and passes these props; see ../README.md "View contract".

import type { ContractInfo, StoreClient } from '../api';
import type { SignedInSession } from '../state/app';
import type { CallsFilters, Route, SignalsTab } from '../state/router';

export interface ViewProps {
  /** The typed Store client. The only way a view reaches Store. */
  client: StoreClient;
  /** `GET /store/v1/contract`: `parameters` holds Store's effective timings. Never hardcode them. */
  contract: ContractInfo;
  /** The signed-in reviewer: `session`, `role`, `can(permission)`, `atLeast(role)`. */
  session: SignedInSession;
  /** Hash navigation, e.g. `navigate({ name: 'workbench', callId })`. */
  navigate(route: Route, options?: { replace?: boolean }): void;
}

/** `#/calls/:callId` */
export interface WorkbenchViewProps extends ViewProps {
  callId: string;
  /** `?turn=N`: seek to this transcript turn once the transcript loads. */
  turn?: number;
}

/** `#/rubrics` and `#/rubrics/:rubricId` */
export interface RubricsViewProps extends ViewProps {
  rubricId?: string;
}

/** `#/signals`, `#/signals/:categoryId`, `#/signals/alerts`, `#/signals/versions` */
export interface SignalsViewProps extends ViewProps {
  tab: SignalsTab;
  categoryId?: string;
}

/** `#/calls`, optionally with signal filters from the address (`?signal_category=…`). */
export interface CallsViewProps extends ViewProps {
  filters?: CallsFilters;
}
export type QueueViewProps = ViewProps;
export type EscalationsViewProps = ViewProps;
export type MetricsViewProps = ViewProps;
