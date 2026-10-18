import { latencySeries, llmRun, median, seconds } from "../lib/latency";
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
      lead={`Historical comparison: ${modelRun.cases} public requests against ${Object.keys(modelRun.products).length} products. Local ranking takes ${seconds(median(ours.ms))}; the fastest hosted model takes ${seconds(fastest)}.`}
    >
      <LatencyChart series={series} />
      <p className="mt-6 max-w-[110ch] text-sm leading-relaxed text-muted">
        Ranking only; copy, images and rendering excluded. Each dot is a request; bars show median
        uncertainty. Hosted times include Codex CLI turn overhead, with startup excluded; the dashed
        line is that client's trivial reply. A direct API comparison could differ.
        Measured {new Date(llmRun.measuredAt).toLocaleDateString("en-US", { dateStyle: "medium" })}.
      </p>
    </SlideFrame>
  );
}
