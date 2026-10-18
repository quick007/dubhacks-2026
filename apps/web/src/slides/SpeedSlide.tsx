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
      title={`An LLM takes ${ratio} times longer to personalize a page`}
      lead={`Same ${modelRun.cases} held-out requests, same product facts, same task: rank the ${modelRun.cases} products. The LLMs pick better, but the shopper waits ${seconds(fastest)} for each pick. Ours answers in ${seconds(median(ours.ms))}.`}
    >
      <LatencyChart series={series} />
      <p className="mt-6 max-w-[110ch] text-sm leading-relaxed text-muted">
        <span className="font-semibold text-ink">Figure 1.</span> Time to rank, log scale. Each dot
        is one request. The tall tick is the median, the capped bar its 95% bootstrap interval
        (4,000 resamples). Our time runs from encoding the request to a serialized ranking on an
        Apple silicon laptop. LLM times run from turn start to turn end through the Codex CLI, with
        JSON output and startup excluded. The dashed line is a reply of &ldquo;ok&rdquo; through the
        same client, so part of every LLM time is client overhead a direct API call would skip.
        Measured {new Date(llmRun.measuredAt).toLocaleDateString("en-US", { dateStyle: "medium" })}.
      </p>
    </SlideFrame>
  );
}
