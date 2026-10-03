/** Fictional presentation data. Never derived from Store, posted to an API, or used as live results. */
export type BenchmarkMeasure = "quality" | "resolution" | "escalation";
export type BenchmarkCohort = "retail" | "ecommerce";
export const measures = {
  quality: {
    label: "QA pass rate",
    short: "Quality",
    direction: "Higher is better",
    lower: false,
  },
  resolution: {
    label: "Resolution confirmed",
    short: "Resolution",
    direction: "Higher is better",
    lower: false,
  },
  escalation: {
    label: "Escalation rate",
    short: "Escalations",
    direction: "Lower is better",
    lower: true,
  },
} as const;
export const cohorts = {
  retail: { label: "Retail & e-commerce", size: "10–50 agents", count: 12 },
  ecommerce: { label: "E-commerce support", size: "10–50 agents", count: 8 },
} as const;
const centers = [
  { name: "Center 01", quality: 72, resolution: 66, escalation: 18 },
  { name: "Center 02", quality: 76, resolution: 70, escalation: 15 },
  { name: "Center 03", quality: 79, resolution: 74, escalation: 13 },
  { name: "Center 04", quality: 81, resolution: 77, escalation: 11 },
  { name: "Center 05", quality: 83, resolution: 78, escalation: 10 },
  { name: "Demo center", quality: 86, resolution: 82, escalation: 8 },
  { name: "Center 06", quality: 90, resolution: 88, escalation: 5 },
  { name: "Center 07", quality: 93, resolution: 91, escalation: 4 },
  { name: "Center 08", quality: 74, resolution: 68, escalation: 17 },
  { name: "Center 09", quality: 78, resolution: 73, escalation: 14 },
  { name: "Center 10", quality: 80, resolution: 76, escalation: 12 },
  { name: "Center 11", quality: 88, resolution: 85, escalation: 6 },
];
export function quantile(values: number[], q: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  const position = (sorted.length - 1) * q;
  const lo = Math.floor(position),
    hi = Math.ceil(position);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (position - lo);
}
export function benchmarkData(
  cohort: BenchmarkCohort,
  measure: BenchmarkMeasure,
) {
  const selected = centers.slice(0, cohorts[cohort].count);
  const demo = selected.find((c) => c.name === "Demo center")![measure];
  // Benchmarks exclude the demonstration center, so it cannot improve its own comparator.
  const peers = selected
    .filter((c) => c.name !== "Demo center")
    .map((c) => c[measure]);
  const median = quantile(peers, 0.5);
  const frontier = quantile(peers, measures[measure].lower ? 0.25 : 0.75);
  const ranked = selected
    .map((c) => ({
      name: c.name,
      value: c[measure],
      demo: c.name === "Demo center",
    }))
    .sort((a, b) =>
      measures[measure].lower ? a.value - b.value : b.value - a.value,
    );
  const drift = measures[measure].lower ? 1 : -1;
  const weeks = [
    "Aug 10",
    "Aug 17",
    "Aug 24",
    "Aug 31",
    "Sep 7",
    "Sep 14",
    "Sep 21",
    "Sep 28",
  ];
  const trend = weeks.map((week, i) => ({
    week,
    center: demo + drift * [12, 10, 11, 7, 6, 4, 2, 0][i],
    peer: median + drift * [4, 3, 4, 2, 2, 1, 1, 0][i],
  }));
  return {
    demo,
    median,
    frontier,
    ranked,
    trend,
    rank: ranked.findIndex((c) => c.demo) + 1,
    peers: peers.length,
  };
}
export const coaching = [
  { criterion: "Policy clarity", center: 71, peer: 82 },
  { criterion: "Ownership", center: 85, peer: 79 },
  { criterion: "Empathy", center: 89, peer: 83 },
  { criterion: "Verification", center: 94, peer: 88 },
];
