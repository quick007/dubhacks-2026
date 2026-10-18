import { latencySeries, llmRun, median } from "../lib/latency";
import { modelRun } from "../lib/model";
import { LatencyChart } from "./LatencyChart";
import { SlideFrame } from "./SlideFrame";

export function SpeedSlide() {
  const series = latencySeries();
  const [ours, ...llms] = series;
  const fastest = Math.min(...llms.map((item) => median(item.ms)));
  const ratio = Math.round(fastest / median(ours.ms));
  return (
    <SlideFrame
      className="max-w-7xl py-6"
      title={`Ranking takes ${ratio} times longer through the hosted client`}
      lead={`Historical ranking-only comparison · ${modelRun.cases} matched public requests`}
    >
      <LatencyChart series={series} />
      <p className="mt-6 max-w-[110ch] text-sm leading-relaxed text-muted">
        Copy, images and rendering excluded. Hosted time includes Codex CLI turn overhead; startup excluded.
        Measured {new Date(llmRun.measuredAt).toLocaleDateString("en-US", { dateStyle: "medium" })}.
      </p>
    </SlideFrame>
  );
}
