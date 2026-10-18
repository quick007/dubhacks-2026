import llm from "../fieldwork/llm-run.json";
import { modelRun } from "./model";

/**
 * Hosted LLMs timed on the same 16 ranking requests as the local model, by
 * scripts/benchmark-llm-ranking.mjs. Every number on the speed slide comes from here.
 */
export const llmRun = llm;

export interface Series {
  id: string;
  label: string;
  detail: string;
  ours: boolean;
  ms: number[];
  /** Median of a trivial "reply ok" turn through the same client, when measured. */
  floorMs?: number;
  topOne: number;
  cases: number;
}

const names: Record<string, string> = {
  "gpt-6.1-sol": "GPT-6.1 Sol",
  "gpt-6-luna": "GPT-6 Luna",
};

export function latencySeries(): Series[] {
  return [
    {
      id: "ours",
      label: "Our model",
      detail: "Qwen3.5-9B features, trained head, laptop",
      ours: true,
      ms: modelRun.requests.map((request) => request.latencyMs),
      topOne: modelRun.arms.learned.topOne,
      cases: modelRun.cases,
    },
    ...llmRun.models.map((model) => ({
      id: model.model,
      label: names[model.model] ?? model.model,
      detail: `${model.effort} reasoning, hosted`,
      ours: false,
      ms: model.requests.map((request) => request.turnMs),
      floorMs: model.floorTurnMs.median,
      topOne: model.topOne,
      cases: model.cases,
    })),
  ];
}

export function median(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/** Small seeded generator, so the interval is the same on every render and in tests. */
function mulberry32(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Percentile bootstrap 95% interval of the median. */
export function medianInterval(values: number[], resamples = 4000, seed = 1): [number, number] {
  const random = mulberry32(seed);
  const medians = Array.from({ length: resamples }, () =>
    median(values.map(() => values[Math.floor(random() * values.length)])),
  ).sort((a, b) => a - b);
  return [medians[Math.floor(resamples * 0.025)], medians[Math.ceil(resamples * 0.975) - 1]];
}

/** Maps milliseconds to a percentage of the axis on a log scale, clamped to the axis. */
export function logScale(min: number, max: number) {
  return (ms: number) =>
    (100 * Math.log10(Math.min(Math.max(ms, min), max) / min)) / Math.log10(max / min);
}

/** Axis label for a tick: milliseconds below a second, seconds above. */
export const tickLabel = (ms: number) => (ms < 1000 ? `${ms} ms` : `${ms / 1000} s`);

/** Centers a tick label under its tick, except the ends, which stay inside the axis. */
export const tickAlign = (index: number, count: number) =>
  index === 0 ? "" : index === count - 1 ? "-translate-x-full" : "-translate-x-1/2";

export function seconds(ms: number) {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}
