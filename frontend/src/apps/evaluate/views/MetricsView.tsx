// `#/metrics` — `GET /store/v1/metrics/executive`, `/metrics/rubrics/{rubric_id}`, (supervisor)
// `/metrics/review-agreement`, and the Signals card (`/metrics/signals`, contract 1.3.0,
// docs/ContactSignalsV2.md §10.3) with "Top caller needs".

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { BarChart3 } from "lucide-react";
import { isNotImplemented, queryKeys, type RubricSummary } from "../api";
import { useDemoStatus } from "../state/demo";
import { BenchmarkExplorer } from "./metrics/BenchmarkExplorer";
import { DailyChart, OutcomesChart } from "./metrics/MetricCharts";
import { SignalsMetricsCard } from "./metrics/SignalsMetricsCard";
import {
  Card,
  EmptyState,
  ErrorNotice,
  Field,
  Loading,
  NotBuiltYet,
  PageHeader,
  SelectInput,
  TextInput,
  formatDateTime,
} from "../components/ui";
import type { MetricsViewProps } from "./types";

/**
 * A local calendar day from `<input type="date">` (`YYYY-MM-DD`) as an RFC 3339 timestamp with the
 * browser's own UTC offset for that instant, e.g. `2024-01-01T00:00:00-07:00`. Store's
 * `MetricsQuery.start`/`end` are aware timestamps and reject a bare date. `dayOffset` 1 gives the
 * start of the next day, which is the end of the chosen day because `end` is exclusive.
 */
export function localDayBoundary(day: string, dayOffset = 0): string | null {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(day);
  if (!m) return null;
  const d = new Date(
    Number(m[1]),
    Number(m[2]) - 1,
    Number(m[3]) + dayOffset,
    0,
    0,
    0,
    0,
  );
  if (Number.isNaN(d.getTime())) return null;
  const pad = (n: number) => String(Math.trunc(Math.abs(n))).padStart(2, "0");
  const offset = -d.getTimezoneOffset(); // minutes east of UTC
  const sign = offset >= 0 ? "+" : "-";
  return (
    `${String(d.getFullYear()).padStart(4, "0")}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` +
    `T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}` +
    `${sign}${pad(offset / 60)}:${pad(offset % 60)}`
  );
}

/** Audited time for the tile: minutes under an hour (so one short call is not "0.0"), else hours. */
export function formatAuditedHours(hours: number): string {
  if (!(hours > 0)) return "0 h";
  if (hours < 1) {
    const minutes = Math.round(hours * 60);
    return minutes < 1 ? "< 1 min" : `${minutes} min`;
  }
  return `${hours.toFixed(1)} h`;
}

export default function MetricsView({ client, session }: MetricsViewProps) {
  const [tab, setTab] = useState<"center" | "peers">("center");
  const [volume, setVolume] = useState(false);
  const demo = useDemoStatus();
  const [rubricId, setRubricId] = useState("");
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");

  // From is the start of that local day; To includes the whole local day (Store's `end` is exclusive).
  // The executive summary and review agreement cover every rubric; the rubric picker only drives
  // the "Per rubric" card.
  const range = {
    start: localDayBoundary(start),
    end: localDayBoundary(end, 1),
    rubric_id: null,
  };

  const rubricsQuery = useQuery({
    queryKey: queryKeys.rubrics,
    queryFn: ({ signal }) =>
      client.get("/store/v1/rubrics", { query: { limit: 200 }, signal }),
  });
  const rubrics: RubricSummary[] = rubricsQuery.data?.items ?? [];
  // Until someone picks one, show the first published rubric (or the first rubric at all).
  const selectedRubricId =
    rubricId ||
    (rubrics.find((r) => r.current_version != null) ?? rubrics[0])?.rubric_id ||
    "";
  const rubricRange = { ...range, rubric_id: selectedRubricId || null };

  const execQuery = useQuery({
    queryKey: [...queryKeys.metrics, "executive", range],
    queryFn: ({ signal }) =>
      client.get("/store/v1/metrics/executive", { query: range, signal }),
  });

  const rubricMetricsQuery = useQuery({
    queryKey: [...queryKeys.metrics, "rubric", selectedRubricId, rubricRange],
    queryFn: ({ signal }) =>
      client.get("/store/v1/metrics/rubrics/{rubric_id}", {
        path: { rubric_id: selectedRubricId },
        query: rubricRange,
        signal,
      }),
    enabled: !!selectedRubricId,
  });

  const agreementQuery = useQuery({
    queryKey: [...queryKeys.metrics, "agreement", range],
    queryFn: ({ signal }) =>
      client.get("/store/v1/metrics/review-agreement", {
        query: range,
        signal,
      }),
    enabled: session.atLeast("supervisor"),
  });

  const focus = rubricMetricsQuery.data?.criteria
    .filter((c) => c.counts.FAIL + c.counts.FLAGGED > 0)
    .sort(
      (a, b) =>
        b.counts.FAIL + b.counts.FLAGGED - (a.counts.FAIL + a.counts.FLAGGED),
    )[0];

  return (
    <div className="max-w-7xl mx-auto space-y-5">
      <PageHeader
        title="Metrics"
        description="See the trend. Find the gap. Focus the next coaching conversation."
        right={
          tab === "center" ? (
            <>
              <Field label="From">
                {(id) => (
                  <TextInput
                    id={id}
                    type="date"
                    value={start}
                    className="w-36"
                    onChange={(e) => setStart(e.target.value)}
                  />
                )}
              </Field>
              <Field label="To">
                {(id) => (
                  <TextInput
                    id={id}
                    type="date"
                    value={end}
                    className="w-36"
                    onChange={(e) => setEnd(e.target.value)}
                  />
                )}
              </Field>
            </>
          ) : undefined
        }
      />

      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border pb-3">
        <div className="flex gap-2" role="group" aria-label="Metrics view">
          {(
            [
              ["center", "Your center"],
              ["peers", "Peer comparison"],
            ] as const
          ).map(([key, label]) => (
            <button
              key={key}
              aria-pressed={tab === key}
              onClick={() => setTab(key)}
              className={`px-4 py-2 rounded-md text-sm font-medium ${tab === key ? "bg-primer-blueSubtle text-primer-blue" : "text-fg-muted hover:bg-canvas-subtle"}`}
            >
              {label}
            </button>
          ))}
        </div>
        <p className="text-xs text-fg-muted">
          {tab === "center"
            ? demo.data?.demo
              ? "Demo Store data · includes synthetic sessions"
              : "Live Store data"
            : "Benchmark concept · demo only"}
        </p>
      </div>
      {tab === "peers" ? (
        demo.data?.demo ? (
          <BenchmarkExplorer />
        ) : (
          <Card title="Peer comparison">
            <EmptyState title="Explore benchmarks in demo mode">
              Shared industry benchmarks are planned. Start the localhost demo
              to explore a clearly labeled fictional comparison.
            </EmptyState>
          </Card>
        )
      ) : (
        <>
          <Card title="Executive summary" icon={BarChart3}>
            {execQuery.isLoading && <Loading label="Loading…" />}
            <ErrorNotice error={execQuery.error} />
            {execQuery.data && (
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                <Stat
                  label="Calls audited"
                  value={execQuery.data.total_audited_calls}
                />
                <Stat
                  label="Audio audited"
                  value={formatAuditedHours(execQuery.data.total_hours_audited)}
                  title={`${execQuery.data.total_hours_audited.toFixed(2)} hours of audited call audio`}
                />
                <Stat
                  label="Average score"
                  value={
                    execQuery.data.total_audited_calls
                      ? execQuery.data.average_score.toFixed(1)
                      : "—"
                  }
                />
                <Stat
                  label="Pass rate"
                  value={
                    execQuery.data.total_audited_calls
                      ? `${execQuery.data.pass_rate_pct.toFixed(1)}%`
                      : "—"
                  }
                />
                <Stat
                  label="Critical breaches"
                  value={execQuery.data.critical_compliance_breaches}
                />
                <Stat
                  label="Supervisor escalations"
                  value={execQuery.data.supervisor_escalations}
                />
                <Stat
                  label="Still analyzing"
                  value={execQuery.data.calls_pending_analysis}
                  title="Calls whose scorecard is still being produced. Calls that stopped with an error are not counted; find them on Calls under Needs attention."
                />
              </div>
            )}
          </Card>

          <Card
            title="Per rubric"
            right={
              <SelectInput
                aria-label="Rubric"
                value={selectedRubricId}
                onChange={(e) => setRubricId(e.target.value)}
                className="w-40 sm:w-56 max-w-full"
              >
                {rubrics.length === 0 && (
                  <option value="">No rubrics yet</option>
                )}
                {rubrics.map((r) => (
                  <option key={r.rubric_id} value={r.rubric_id}>
                    {r.name || r.rubric_id}
                  </option>
                ))}
              </SelectInput>
            }
          >
            {!selectedRubricId ? (
              rubricsQuery.isLoading ? (
                <Loading label="Loading…" />
              ) : (
                <EmptyState title="No rubrics yet">
                  Publish a rubric under Rubrics to see its results here.
                </EmptyState>
              )
            ) : rubricMetricsQuery.isLoading ? (
              <Loading label="Loading…" />
            ) : rubricMetricsQuery.error ? (
              <ErrorNotice error={rubricMetricsQuery.error} />
            ) : rubricMetricsQuery.data ? (
              <div className="space-y-5">
                <p className="text-xs text-fg-muted">
                  Results for the selected rubric across its evaluated versions.
                  Date filters apply to all center metrics.
                </p>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                  <Stat
                    label="Calls evaluated"
                    value={rubricMetricsQuery.data.total_calls_evaluated}
                  />
                  <Stat
                    label="Average score"
                    value={
                      rubricMetricsQuery.data.total_calls_evaluated
                        ? rubricMetricsQuery.data.average_score.toFixed(1)
                        : "—"
                    }
                  />
                  <Stat
                    label="Pass rate"
                    value={
                      rubricMetricsQuery.data.total_calls_evaluated
                        ? `${rubricMetricsQuery.data.pass_rate_pct.toFixed(1)}%`
                        : "—"
                    }
                  />
                </div>
                {focus && (
                  <div className="border-l-2 border-primer-blue pl-3 text-sm">
                    <span className="font-medium">
                      Review focus: {focus.criterion_name}.
                    </span>
                    <span className="text-fg-muted">
                      {" "}
                      {focus.counts.FAIL} failures and {focus.counts.FLAGGED}{" "}
                      flagged verdicts—the largest review volume in this rubric.
                    </span>
                  </div>
                )}
                {rubricMetricsQuery.data.total_calls_evaluated === 0 && (
                  <EmptyState title="No evaluated calls in this range">
                    Choose another date range or evaluate calls to build a
                    quality trend.
                  </EmptyState>
                )}
                <div className="grid xl:grid-cols-2 gap-6 min-w-0">
                  {rubricMetricsQuery.data.daily.length > 0 && (
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                        <h3 className="text-sm font-medium">
                          {volume ? "Evaluation volume" : "Quality over time"}
                        </h3>
                        <div
                          className="flex gap-1"
                          role="group"
                          aria-label="Trend measure"
                        >
                          {[false, true].map((v) => (
                            <button
                              key={String(v)}
                              aria-pressed={volume === v}
                              onClick={() => setVolume(v)}
                              className={`text-xs rounded px-3 py-1.5 ${volume === v ? "bg-primer-blueSubtle text-primer-blue" : "text-fg-muted"}`}
                            >
                              {v ? "Volume" : "Score"}
                            </button>
                          ))}
                        </div>
                      </div>
                      <DailyChart
                        data={rubricMetricsQuery.data.daily}
                        volume={volume}
                      />
                      <p className="text-xs text-fg-subtle mt-3">
                        {volume
                          ? "Evaluated calls by day."
                          : "Average QA score by evaluation day (UTC), out of 100. Date filters select call creation time."}
                      </p>
                    </div>
                  )}
                  {rubricMetricsQuery.data.total_calls_evaluated > 0 &&
                    rubricMetricsQuery.data.criteria.length > 0 && (
                      <div className="min-w-0">
                        <h3 className="text-sm font-medium mb-3">
                          Where outcomes differ
                        </h3>
                        <OutcomesChart
                          criteria={rubricMetricsQuery.data.criteria}
                        />
                      </div>
                    )}
                </div>
                {rubricMetricsQuery.data.total_calls_evaluated > 0 &&
                  rubricMetricsQuery.data.criteria.length > 0 && (
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="text-left text-xs text-fg-muted border-b border-border">
                            <th className="font-medium py-1.5 pr-2">
                              Criterion
                            </th>
                            <th className="font-medium py-1.5 pr-2">Pass</th>
                            <th className="font-medium py-1.5 pr-2">Fail</th>
                            <th className="font-medium py-1.5 pr-2">Flagged</th>
                            <th className="font-medium py-1.5 pr-2">N/A</th>
                            <th className="font-medium py-1.5 pr-2">
                              Pass rate
                            </th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-border-muted">
                          {rubricMetricsQuery.data.criteria.map((c) => (
                            <tr key={c.criterion_id}>
                              <td className="py-1.5 pr-2 text-fg">
                                {c.criterion_name}
                              </td>
                              <td className="py-1.5 pr-2 tabular-nums text-fg-muted">
                                {c.counts.PASS}
                              </td>
                              <td className="py-1.5 pr-2 tabular-nums text-fg-muted">
                                {c.counts.FAIL}
                              </td>
                              <td className="py-1.5 pr-2 tabular-nums text-fg-muted">
                                {c.counts.FLAGGED}
                              </td>
                              <td className="py-1.5 pr-2 tabular-nums text-fg-muted">
                                {c.counts.NOT_APPLICABLE}
                              </td>
                              <td className="py-1.5 pr-2 tabular-nums text-fg-muted">
                                {c.counts.pass_rate_pct.toFixed(1)}%
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                {rubricMetricsQuery.data.daily.length > 0 && (
                  <details>
                    <summary className="text-xs font-medium text-fg-muted cursor-pointer mb-2">
                      View daily values
                    </summary>
                    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-fg-muted">
                      {rubricMetricsQuery.data.daily.map((d) => (
                        <span key={d.date}>
                          {d.date}: {d.evaluated} calls, avg{" "}
                          {d.mean_score.toFixed(1)}
                        </span>
                      ))}
                    </div>
                  </details>
                )}
              </div>
            ) : null}
          </Card>

          <SignalsMetricsCard
            client={client}
            session={session}
            start={range.start}
            end={range.end}
          />

          <Card title="Review agreement" subtitle="Supervisors only">
            {!session.atLeast("supervisor") ? (
              <p className="text-sm text-fg-muted">
                Your role does not include review-agreement metrics.
              </p>
            ) : agreementQuery.isLoading ? (
              <Loading label="Loading…" />
            ) : agreementQuery.error ? (
              isNotImplemented(agreementQuery.error) ? (
                <NotBuiltYet what="Review agreement metrics" />
              ) : (
                <ErrorNotice error={agreementQuery.error} />
              )
            ) : agreementQuery.data ? (
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-2">
                  <Stat
                    label="Calls reviewed"
                    value={agreementQuery.data.calls_reviewed}
                  />
                  <Stat
                    label="Overall agreement"
                    value={
                      agreementQuery.data.overall_agreement_rate !== null
                        ? `${(agreementQuery.data.overall_agreement_rate * 100).toFixed(1)}%`
                        : "—"
                    }
                  />
                </div>
                {agreementQuery.data.per_criterion.length > 0 && (
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="text-left text-xs text-fg-muted border-b border-border">
                          <th className="font-medium py-1.5 pr-2">Criterion</th>
                          <th className="font-medium py-1.5 pr-2">
                            Human reviewed
                          </th>
                          <th className="font-medium py-1.5 pr-2">Overrides</th>
                          <th className="font-medium py-1.5 pr-2">Agreement</th>
                          <th className="font-medium py-1.5 pr-2">Precision</th>
                          <th className="font-medium py-1.5 pr-2">Recall</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border-muted">
                        {agreementQuery.data.per_criterion.map((c) => (
                          <tr key={c.criterion_id}>
                            <td className="py-1.5 pr-2 text-fg">
                              {c.criterion_name}
                            </td>
                            <td className="py-1.5 pr-2 tabular-nums text-fg-muted">
                              {c.human_reviewed}
                            </td>
                            <td className="py-1.5 pr-2 tabular-nums text-fg-muted">
                              {c.overrides}
                            </td>
                            <td className="py-1.5 pr-2 tabular-nums text-fg-muted">
                              {c.agreement_rate !== null
                                ? `${(c.agreement_rate * 100).toFixed(1)}%`
                                : "—"}
                            </td>
                            <td className="py-1.5 pr-2 tabular-nums text-fg-muted">
                              {c.precision !== null
                                ? `${(c.precision * 100).toFixed(1)}%`
                                : "—"}
                            </td>
                            <td className="py-1.5 pr-2 tabular-nums text-fg-muted">
                              {c.recall !== null
                                ? `${(c.recall * 100).toFixed(1)}%`
                                : "—"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            ) : null}
          </Card>
          {execQuery.data && (
            <p className="text-xs text-fg-subtle">
              Generated {formatDateTime(execQuery.data.generated_at)} · QA
              results describe evaluated calls; they do not establish customer
              satisfaction or first-contact resolution.
            </p>
          )}
        </>
      )}
    </div>
  );
}

function Stat({
  label,
  value,
  title,
}: {
  label: string;
  value: string | number;
  title?: string;
}) {
  return (
    <div
      className="rounded-lg border border-border-muted bg-canvas px-4 py-4"
      title={title}
    >
      <div className="text-2xl sm:text-3xl font-semibold tracking-tight text-fg tabular-nums">
        {value}
      </div>
      <div className="text-xs text-fg-muted mt-2">{label}</div>
    </div>
  );
}
