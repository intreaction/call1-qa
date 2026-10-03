import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  XAxis,
  YAxis,
} from "recharts";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "../../components/charts/chart";
import type { RubricMetrics } from "../../api";

export const chartColors = {
  center: "var(--primer-blue)",
  peer: "var(--fg-subtle)",
  pass: "var(--primer-green)",
  fail: "var(--primer-red)",
  flagged: "var(--primer-yellow)",
  na: "var(--fg-subtle)",
};
export const scoreConfig = {
  mean_score: { label: "Average score", color: chartColors.center },
  evaluated: { label: "Calls evaluated", color: chartColors.center },
} satisfies ChartConfig;
export function DailyChart({
  data,
  volume = false,
}: {
  data: RubricMetrics["daily"];
  volume?: boolean;
}) {
  return (
    <ChartContainer
      config={scoreConfig}
      className="h-64 w-full"
      aria-label={
        volume ? "Daily evaluated call volume" : "Daily average QA score"
      }
    >
      <AreaChart
        accessibilityLayer
        data={data}
        margin={{ left: 0, right: 12, top: 14, bottom: 4 }}
      >
        <CartesianGrid
          vertical={false}
          stroke="var(--border-muted)"
          strokeDasharray="3 5"
        />
        <XAxis
          dataKey="date"
          tickLine={false}
          axisLine={false}
          minTickGap={35}
          tickMargin={12}
          tickFormatter={(v) => String(v).slice(5)}
        />
        <YAxis
          domain={volume ? [0, "auto"] : [0, 100]}
          allowDecimals={!volume}
          tickLine={false}
          axisLine={false}
          width={36}
        />
        <ChartTooltip content={<ChartTooltipContent />} />
        <Area
          dataKey={volume ? "evaluated" : "mean_score"}
          type="monotone"
          stroke={chartColors.center}
          fill={chartColors.center}
          fillOpacity={0.12}
          strokeWidth={2.5}
          dot={data.length <= 12 ? { r: 3, fill: chartColors.center } : false}
          isAnimationActive={false}
        />
      </AreaChart>
    </ChartContainer>
  );
}
const outcomeConfig = {
  PASS: { label: "Pass", color: chartColors.pass },
  FAIL: { label: "Fail", color: chartColors.fail },
  FLAGGED: { label: "Flagged", color: chartColors.flagged },
  NOT_APPLICABLE: { label: "N/A", color: chartColors.na },
} satisfies ChartConfig;
export function OutcomesChart({
  criteria,
}: {
  criteria: RubricMetrics["criteria"];
}) {
  const rows = criteria.map((c) => ({ name: c.criterion_name, ...c.counts }));
  return (
    <div>
      <ChartContainer
        config={outcomeConfig}
        className="w-full"
        style={{ height: Math.max(220, rows.length * 44) }}
        aria-label="QA criterion outcome counts"
      >
        <BarChart
          accessibilityLayer
          data={rows}
          layout="vertical"
          margin={{ left: 0, right: 12, top: 8, bottom: 4 }}
        >
          <CartesianGrid horizontal={false} stroke="var(--border-muted)" />
          <XAxis
            type="number"
            allowDecimals={false}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="name"
            width={125}
            axisLine={false}
            tickLine={false}
            tick={{ fontSize: 11 }}
            tickFormatter={(v) =>
              String(v).length > 20 ? String(v).slice(0, 19) + "…" : String(v)
            }
          />
          <ChartTooltip content={<ChartTooltipContent />} />
          {(["PASS", "FAIL", "FLAGGED", "NOT_APPLICABLE"] as const).map(
            (key) => (
              <Bar
                key={key}
                dataKey={key}
                stackId="outcomes"
                fill={outcomeConfig[key].color}
                barSize={18}
                isAnimationActive={false}
              />
            ),
          )}
        </BarChart>
      </ChartContainer>
      <div className="flex flex-wrap gap-4 text-xs text-fg-muted pt-3">
        {Object.entries(outcomeConfig).map(([key, c]) => (
          <span key={key} className="flex items-center gap-1.5">
            <span
              className="w-2 h-2 rounded-sm"
              style={{ background: c.color }}
            />
            {c.label}
          </span>
        ))}
      </div>
    </div>
  );
}
