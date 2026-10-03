// Evaluate's Store client: the only way Evaluate reaches Store (docs/SplitBuild.md rule 3).
export * from './client';
export * from './errors';
export * from './types';
export * from './contract';
export * from './changes';
export * from './derive';
export * from './queryKeys';
export * from './signals';
export * from './signalRules';
export * from './vocabulary';
export * as passkeys from './passkeys';
export * from './idempotency';
// Demo mode (docs/SplitBuild.md rule 3's one documented exception — see api/demo.ts's header).
export * as demo from './demo';
