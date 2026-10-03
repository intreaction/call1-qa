// Typed access to the frozen Store contract (call1/contracts). Evaluate's Store client is built
// on these types and nothing else: no legacy /api/v1 shapes, no hand-written mirrors.
//
// `store-v1.ts` is generated from call1/contracts/openapi.json by
// `python -m call1.contracts.generate` (or `npm run contracts:types`). Never edit it by hand.

export type { components, operations, paths } from './store-v1';
import type { components, operations, paths } from './store-v1';

/** A named schema from the contract, e.g. `Schema<'CallDetail'>`. */
export type Schema<K extends keyof components['schemas']> = components['schemas'][K];

type Schemas = components['schemas'];

/**
 * A model as Store sends it. Models used both in requests and in responses appear twice in the
 * OpenAPI document (`Name-Input`, `Name-Output`); response shapes mark every field Store always
 * sends as required. `Output<'Job'>` resolves either spelling.
 */
export type Output<K extends string> = `${K}-Output` extends keyof Schemas
  ? Schemas[`${K}-Output`]
  : K extends keyof Schemas
    ? Schemas[K]
    : never;

/** A model as a client sends it (optional fields with defaults may be omitted). */
export type Input<K extends string> = `${K}-Input` extends keyof Schemas
  ? Schemas[`${K}-Input`]
  : K extends keyof Schemas
    ? Schemas[K]
    : never;

/** The JSON request body of an operation, by operationId. */
export type RequestBody<Op extends keyof operations> = operations[Op] extends {
  requestBody: { content: { 'application/json': infer B } };
}
  ? B
  : never;

/** The JSON 2xx response of an operation, by operationId. */
export type ResponseBody<Op extends keyof operations> = operations[Op] extends {
  responses: { 200: { content: { 'application/json': infer R } } };
}
  ? R
  : operations[Op] extends { responses: { 201: { content: { 'application/json': infer C } } } }
    ? C
    : never;

export const STORE_API_PREFIX = '/store/v1' as const;
export type StorePath = keyof paths;

/**
 * Sent on every state-changing reviewer request. The value is `SessionInfo.csrf_token`: returned by
 * sign-in and enrollment, and again by `GET /store/v1/auth/session`, so a reload or a new tab
 * recovers it. It is session-bound, not one-time; keep it in memory, never in web storage.
 */
export const CSRF_HEADER = 'X-Call1-CSRF' as const;
/** Sent on reanalysis requests so repeated clicks do not duplicate work. */
export const IDEMPOTENCY_HEADER = 'Idempotency-Key' as const;
