import { useMemo, useState } from 'react';
import { LayoutList, ChevronDown, ChevronRight } from 'lucide-react';
import { loadContractCoverage } from '../coverage';
import Panel from './Panel';
import StatusPill from './StatusPill';

const STAGE_LABEL: Record<'stage2' | 'stage4' | 'stage5', string> = {
  stage2: 'Stage 2 — in scope now',
  stage4: 'Stage 4 — deferred (TLS/CA, backup)',
  stage5: 'Stage 5 — deferred (updater, egress)',
};

export default function CoveragePanel() {
  const coverage = useMemo(loadContractCoverage, []);
  const [open, setOpen] = useState(false);

  const { totals } = coverage;
  const deferred = totals.stage4 + totals.stage5;

  return (
    <Panel
      title="Contract coverage"
      icon={LayoutList}
      subtitle={`openapi.json · contract v${coverage.contractVersion}`}
      right={<StatusPill tone="blue">{coverage.totalOperations} routes</StatusPill>}
    >
      <div className="flex flex-wrap gap-2 mb-3">
        <StatusPill tone="green">{totals.stage2} in scope (Stage 2)</StatusPill>
        <StatusPill tone="neutral">{deferred} deferred (Stage 4/5)</StatusPill>
      </div>
      <p className="text-xs text-fg-muted mb-3">
        Grouped by the contract&apos;s own <code>tags</code> and <code>x-call1-stage</code>. This
        reflects contract scope, not which stage-2 routes already have a live handler — Store
        still answers 501 for a route whose area hasn&apos;t built it yet.
      </p>

      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1.5 text-xs font-medium text-fg-muted hover:text-fg mb-2"
        aria-expanded={open}
      >
        {open ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
        By area ({coverage.byTag.length} tags)
      </button>

      {open && (
        <div className="space-y-1.5 max-h-64 overflow-y-auto pr-1">
          {coverage.byTag.map((t) => (
            <div key={t.tag} className="flex items-center justify-between gap-2 text-xs py-1 border-b border-border-muted last:border-0">
              <span className="text-fg font-medium truncate">{t.tag}</span>
              <span className="text-fg-muted shrink-0 flex items-center gap-2">
                <span title={STAGE_LABEL.stage2}>{t.counts.stage2} live-scope</span>
                {(t.counts.stage4 || t.counts.stage5) > 0 && (
                  <span className="text-fg-subtle" title={`${STAGE_LABEL.stage4} / ${STAGE_LABEL.stage5}`}>
                    +{t.counts.stage4 + t.counts.stage5} deferred
                  </span>
                )}
                <span className="text-fg-subtle">/ {t.total}</span>
              </span>
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}
