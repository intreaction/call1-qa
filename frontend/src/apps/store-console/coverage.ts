// Contract coverage, derived entirely from the committed, generated openapi.json — never a
// live probe of every route. docs/SplitBuild.md: "derive from GET /store/v1/contract plus a
// small console-only JSON endpoint only if needed ... prefer reading the committed openapi.json
// x-call1-stage." No endpoint was needed: every field this panel shows (operationId, tags,
// x-call1-stage) is already in the contract's OpenAPI document, so this imports it directly as
// a build-time JSON module — it never changes at runtime and regenerating it
// (`python -m call1.contracts.generate`) is call1/contracts' job, not the console's.
//
// This groups operations by their OpenAPI `tags` (the contract's own area labels — "jobs",
// "reviews", "auth", etc.) and by `x-call1-stage` (2 = in scope for this split now; 4/5 =
// deferred to a later stage per docs/SplitBuild.md "Deferred"). It is honest about what it
// does NOT know: whether an in-scope (stage 2) route has an actual handler registered yet, vs.
// still answering 501, is live server state this panel does not probe. The Health panel's
// contract version is the only live cross-check offered here.

import openapi from '../../../../call1/contracts/openapi.json';

interface RawOperation {
  operationId?: string;
  tags?: string[];
  'x-call1-stage'?: number;
}

export interface StageCounts {
  stage2: number;
  stage4: number;
  stage5: number;
  other: number;
}

export interface TagCoverage {
  tag: string;
  total: number;
  counts: StageCounts;
}

export interface ContractCoverage {
  contractVersion: string;
  totalOperations: number;
  byTag: TagCoverage[];
  totals: StageCounts;
}

function emptyCounts(): StageCounts {
  return { stage2: 0, stage4: 0, stage5: 0, other: 0 };
}

function bump(counts: StageCounts, stage: number | undefined) {
  if (stage === 2) counts.stage2 += 1;
  else if (stage === 4) counts.stage4 += 1;
  else if (stage === 5) counts.stage5 += 1;
  else counts.other += 1;
}

let cached: ContractCoverage | null = null;

/** Pure, synchronous and cheap after the first call — safe to call from render. */
export function loadContractCoverage(): ContractCoverage {
  if (cached) return cached;

  const paths = (openapi as { paths?: Record<string, Record<string, RawOperation>> }).paths ?? {};
  const info = (openapi as { info?: { version?: string } }).info;
  const byTag = new Map<string, TagCoverage>();
  const totals = emptyCounts();
  let totalOperations = 0;

  for (const methods of Object.values(paths)) {
    for (const op of Object.values(methods)) {
      if (!op.operationId) continue;
      totalOperations += 1;
      const stage = op['x-call1-stage'];
      bump(totals, stage);
      const tags = op.tags && op.tags.length > 0 ? op.tags : ['untagged'];
      for (const tag of tags) {
        let entry = byTag.get(tag);
        if (!entry) {
          entry = { tag, total: 0, counts: emptyCounts() };
          byTag.set(tag, entry);
        }
        entry.total += 1;
        bump(entry.counts, stage);
      }
    }
  }

  cached = {
    contractVersion: info?.version ?? 'unknown',
    totalOperations,
    byTag: Array.from(byTag.values()).sort((a, b) => a.tag.localeCompare(b.tag)),
    totals,
  };
  return cached;
}
