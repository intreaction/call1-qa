import { useState } from "react";
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  ReferenceLine,
  XAxis,
  YAxis,
} from "recharts";
import { Card, SelectInput } from "../../components/ui";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "../../components/charts/chart";
import {
  benchmarkData,
  coaching,
  cohorts,
  measures,
  type BenchmarkCohort,
  type BenchmarkMeasure,
} from "./benchmarkDemo";
import { chartColors } from "./MetricCharts";

const config = {
  center: { label: "Demo center", color: chartColors.center },
  peer: { label: "Peer median", color: chartColors.peer },
  value: { label: "Rate", color: chartColors.center },
};
const pct = (value: number) => `${Number(value.toFixed(1))}%`;

export function BenchmarkExplorer() {
  const [cohort, setCohort] = useState<BenchmarkCohort>("retail");
  const [measure, setMeasure] = useState<BenchmarkMeasure>("quality");
  const data = benchmarkData(cohort, measure);
  const delta = data.demo - data.median;
  const favorable = measures[measure].lower ? delta < 0 : delta > 0;
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-sm text-fg-muted mt-2">
            Weekly snapshots · Aug 10–Sep 28, 2026 · {cohorts[cohort].size} ·{" "}
            {data.peers} fictional peers
          </p>
        </div>
        <label className="text-xs text-fg-muted space-y-1.5">
          Peer cohort
          <SelectInput
            className="block w-full sm:w-56"
            value={cohort}
            onChange={(e) => setCohort(e.target.value as BenchmarkCohort)}
          >
            {Object.entries(cohorts).map(([key, c]) => (
              <option key={key} value={key}>
                {c.label}
              </option>
            ))}
          </SelectInput>
        </label>
      </div>
      <div
        className="flex flex-wrap gap-2"
        role="group"
        aria-label="Benchmark measure"
      >
        {Object.entries(measures).map(([key, m]) => (
          <button
            key={key}
            aria-pressed={key === measure}
            onClick={() => setMeasure(key as BenchmarkMeasure)}
            className={`px-4 py-2 rounded-md text-sm border transition-colors ${key === measure ? "border-primer-blue text-primer-blue bg-primer-blueSubtle" : "border-border text-fg-muted hover:text-fg hover:bg-canvas-subtle"}`}
          >
            {m.label}
          </button>
        ))}
      </div>
      <div className="rounded-xl border border-border bg-canvas-subtle p-4 sm:p-6 grid grid-cols-2 lg:grid-cols-4 gap-5 items-end">
        <div>
          <p className="text-sm text-fg-muted">
            Demo center · {measures[measure].short}
          </p>
          <div className="text-4xl sm:text-5xl font-semibold tracking-tight tabular-nums mt-2 text-primer-blue">
            {pct(data.demo)}
          </div>
          <p className="text-xs text-fg-muted mt-3">
            Sep 28 snapshot · {measures[measure].direction}
          </p>
        </div>
        <div>
          <p className="text-xs text-fg-muted">Versus peer median</p>
          <p
            className={`text-3xl tabular-nums font-medium mt-2 ${favorable ? "text-primer-green" : "text-primer-yellow"}`}
          >
            {delta > 0 ? "+" : ""}
            {Number(delta.toFixed(1))} <span className="text-base">pp</span>
          </p>
          <p className="text-xs text-fg-muted mt-2">
            {favorable
              ? "Ahead of the peer median"
              : "Opportunity to close the gap"}
          </p>
        </div>
        <div>
          <p className="text-xs text-fg-muted">Position in this cohort</p>
          <p className="text-3xl tabular-nums font-medium mt-2">
            {data.rank}
            <span className="text-base text-fg-muted">
              {" "}
              / {data.ranked.length}
            </span>
          </p>
          <p className="text-xs text-fg-muted mt-2">
            Best-performing center ranks first
          </p>
        </div>
        <div>
          <p className="text-xs text-fg-muted">Top-quartile threshold</p>
          <p className="text-3xl tabular-nums font-medium mt-2">
            {pct(data.frontier)}
          </p>
          <p className="text-xs text-fg-muted mt-2">
            Peer median {pct(data.median)}
          </p>
        </div>
      </div>
      <div className="grid xl:grid-cols-[1.4fr_1fr] gap-5 min-w-0">
        <Card
          title="Progress against peers"
          subtitle={`${measures[measure].label} · weekly rate (%)`}
        >
          <Legend />
          <ChartContainer
            config={config}
            className="h-72 w-full mt-4"
            aria-label="Demo center and peer median weekly trend"
          >
            <ComposedChart
              accessibilityLayer
              data={data.trend}
              margin={{ right: 12, top: 8, bottom: 4 }}
            >
              <CartesianGrid
                vertical={false}
                stroke="var(--border-muted)"
                strokeDasharray="3 5"
              />
              <XAxis
                dataKey="week"
                axisLine={false}
                tickLine={false}
                minTickGap={25}
                tickMargin={12}
              />
              <YAxis
                domain={[0, measure === "escalation" ? 30 : 100]}
                width={35}
                axisLine={false}
                tickLine={false}
              />
              <ChartTooltip content={<ChartTooltipContent />} />
              <Area
                type="monotone"
                dataKey="center"
                stroke={chartColors.center}
                fill={chartColors.center}
                fillOpacity={0.1}
                strokeWidth={3}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="peer"
                stroke={chartColors.peer}
                strokeDasharray="5 5"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </ComposedChart>
          </ChartContainer>
          <p className="text-sm text-fg-muted border-t border-border-muted pt-4 mt-4 leading-relaxed">
            The demo center improves by 12 percentage points over eight
            snapshots. Use this view to discuss whether coaching progress is
            keeping pace with comparable teams.
          </p>
          <details className="mt-3 text-xs text-fg-muted">
            <summary className="cursor-pointer">View weekly values</summary>
            <table className="w-full mt-2 text-left">
              <thead>
                <tr>
                  <th>Week</th>
                  <th>Demo center</th>
                  <th>Peer median</th>
                </tr>
              </thead>
              <tbody>
                {data.trend.map((row) => (
                  <tr key={row.week}>
                    <td className="py-1">{row.week}</td>
                    <td>{pct(row.center)}</td>
                    <td>{pct(row.peer)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        </Card>
        <Card
          title="Where the center stands"
          subtitle={`${measures[measure].direction} · Sep 28 snapshot`}
        >
          <ChartContainer
            config={config}
            className="w-full"
            style={{ height: Math.max(280, data.ranked.length * 30) }}
            aria-label="Fictional contact center ranking"
          >
            <BarChart
              accessibilityLayer
              data={data.ranked}
              layout="vertical"
              margin={{ right: 12, top: 8 }}
            >
              <CartesianGrid horizontal={false} stroke="var(--border-muted)" />
              <XAxis
                type="number"
                domain={[0, measure === "escalation" ? 30 : 100]}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                type="category"
                dataKey="name"
                width={90}
                axisLine={false}
                tickLine={false}
              />
              <ChartTooltip content={<ChartTooltipContent />} />
              <ReferenceLine
                x={data.median}
                stroke={chartColors.peer}
                strokeDasharray="3 3"
              />
              <Bar
                dataKey="value"
                barSize={14}
                radius={[0, 3, 3, 0]}
                isAnimationActive={false}
              >
                {data.ranked.map((c) => (
                  <Cell
                    key={c.name}
                    fill={c.demo ? chartColors.center : chartColors.peer}
                    fillOpacity={c.demo ? 1 : 0.5}
                  />
                ))}
              </Bar>
            </BarChart>
          </ChartContainer>
          <p className="text-xs text-fg-muted mt-3">
            Dashed line: peer median. Demo center is excluded from peer
            statistics.
          </p>
          <details className="mt-3 text-xs text-fg-muted">
            <summary className="cursor-pointer">View center values</summary>
            <table className="w-full mt-2 text-left">
              <thead>
                <tr>
                  <th>Center</th>
                  <th>{measures[measure].label}</th>
                </tr>
              </thead>
              <tbody>
                {data.ranked.map((c) => (
                  <tr key={c.name}>
                    <td className="py-1">{c.name}</td>
                    <td>{pct(c.value)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        </Card>
      </div>
      <div className="min-w-0">
        <Card
          title="Turn comparison into coaching"
          subtitle="Separate retail example · QA criterion pass rates (%)"
        >
          <Legend />
          <ChartContainer
            config={config}
            className="h-64 w-full mt-3"
            aria-label="Illustrative coaching opportunities by criterion"
          >
            <BarChart
              accessibilityLayer
              data={coaching}
              layout="vertical"
              margin={{ right: 12 }}
            >
              <CartesianGrid horizontal={false} stroke="var(--border-muted)" />
              <XAxis
                type="number"
                domain={[0, 100]}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                type="category"
                dataKey="criterion"
                width={90}
                axisLine={false}
                tickLine={false}
              />
              <ChartTooltip content={<ChartTooltipContent />} />
              <Bar
                dataKey="center"
                fill={chartColors.center}
                barSize={10}
                radius={[0, 3, 3, 0]}
                isAnimationActive={false}
              />
              <Bar
                dataKey="peer"
                fill={chartColors.peer}
                fillOpacity={0.5}
                barSize={10}
                radius={[0, 3, 3, 0]}
                isAnimationActive={false}
              />
            </BarChart>
          </ChartContainer>
          <details className="text-xs text-fg-muted">
            <summary className="cursor-pointer">View coaching values</summary>
            <table className="w-full mt-2 text-left">
              <thead>
                <tr>
                  <th>Criterion</th>
                  <th>Demo center</th>
                  <th>Peer median</th>
                </tr>
              </thead>
              <tbody>
                {coaching.map((c) => (
                  <tr key={c.criterion}>
                    <td className="py-1">{c.criterion}</td>
                    <td>{pct(c.center)}</td>
                    <td>{pct(c.peer)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        </Card>
      </div>
      <p className="text-xs text-fg-subtle leading-relaxed">
        Method: equally weighted center rates; interpolated peer median and
        quartiles. QA pass rate = passing evaluations / evaluated calls.
        Resolution confirmed = calls marked resolved / evaluated calls.
        Escalation rate = escalated calls / evaluated calls. A production
        benchmark would need consent, minimum cohort sizes and comparable
        rubric, model and sampling definitions. Shared benchmarks are not
        connected.
      </p>
    </div>
  );
}
function Legend() {
  return (
    <div className="flex flex-wrap gap-5 text-xs text-fg-muted">
      <span className="flex items-center gap-2">
        <span className="w-3 h-2 rounded-sm bg-primer-blue" />
        Demo center
      </span>
      <span className="flex items-center gap-2">
        <span className="w-3 border-t-2 border-dashed border-fg-subtle" />
        Peer median
      </span>
    </div>
  );
}
