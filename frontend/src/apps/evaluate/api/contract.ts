// Contract-version check (contracts/README.md "Versioning and change rules"): read
// `GET /store/v1/contract` at start and refuse to run against a different major. Store also
// reports its effective `ContractParameters` there; Evaluate reads timings from it and never
// hardcodes them.

import type { StoreClient } from './client';
import { ContractMismatchError } from './errors';
import type { ContractInfo } from './types';

/**
 * The contract version `frontend/src/contracts/store-v1.ts` was generated from
 * (`call1.contracts.common.CONTRACT_VERSION`). Bump it with every regeneration.
 */
export const BUILT_FOR_CONTRACT = '1.4.0';

export function majorOf(version: string): number {
  const major = Number.parseInt(version.split('.')[0] ?? '', 10);
  return Number.isFinite(major) ? major : -1;
}

export function isCompatible(storeVersion: string, builtFor = BUILT_FOR_CONTRACT): boolean {
  return majorOf(storeVersion) === majorOf(builtFor) && majorOf(builtFor) >= 0;
}

/** Fetch the contract info; throws `ContractMismatchError` on a different major. */
export async function checkContract(client: StoreClient): Promise<ContractInfo> {
  const info = await client.get('/store/v1/contract');
  if (!isCompatible(info.contract_version)) {
    throw new ContractMismatchError(info.contract_version, BUILT_FOR_CONTRACT);
  }
  return info;
}
