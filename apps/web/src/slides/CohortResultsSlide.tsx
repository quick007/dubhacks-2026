import { ArrowRight, Equal, Minus } from "lucide-react";
import { illustrativeScenario } from "../lib/illustrative-scenario";
import type { PresentationResults } from "../lib/presentation-results";
import { SlideFrame } from "./SlideFrame";

const money = (cents: number) => new Intl.NumberFormat("en-US", {
  style: "currency", currency: "USD", minimumFractionDigits: 2,
}).format(cents / 100);

/** The illustration is bound to one immutable cohort; all measured outcomes stay in its evidence. */
export function CohortResultsSlide({ result }: { result: PresentationResults }) {
  const before = result.data.metrics.arms.generic.revenue.revenuePerAssignedSessionCents;
  const after = result.data.metrics.arms["local-model"].revenue.revenuePerAssignedSessionCents;
  const scenario = illustrativeScenario(result.reportId, result.sha256, before, after);
  return (
    <SlideFrame title={scenario ? "Illustrative scenario" : "Observed synthetic value per session"}
      lead={scenario ? undefined : "Synthetic experiment"}>
      {scenario ? (
        <div className="grid items-center gap-x-5 gap-y-3 sm:grid-cols-[1fr_auto_1fr_auto_1fr] sm:grid-rows-[auto_auto]">
          <div className="grid gap-3 sm:col-start-1 sm:row-span-2 sm:row-start-1 sm:grid-rows-subgrid">
            <p className="font-display text-6xl font-extrabold tracking-tight text-graphite">{scenario.observedPercent.toFixed(2)}%</p>
            <p className="text-lg text-muted">Observed synthetic value difference</p>
          </div>
          <Minus className="size-7 text-graphite sm:col-start-2 sm:row-start-1 sm:self-center" aria-label="minus" />
          <div className="grid gap-3 sm:col-start-3 sm:row-span-2 sm:row-start-1 sm:grid-rows-subgrid">
            <p className="font-display text-6xl font-extrabold tracking-tight text-graphite">{scenario.assumedLossPercent.toFixed(2)}%</p>
            <p className="text-lg text-muted">Assumed delay loss</p>
          </div>
          <Equal className="size-7 text-graphite sm:col-start-4 sm:row-start-1 sm:self-center" aria-label="equals" />
          <div className="grid gap-3 sm:col-start-5 sm:row-span-2 sm:row-start-1 sm:grid-rows-subgrid">
            <p className="font-display text-6xl font-extrabold tracking-tight text-cobalt">{scenario.remainderPercentagePoints.toFixed(2)}<span className="ml-1 text-2xl">pp</span></p>
            <p className="text-lg text-muted">Illustrative remainder</p>
          </div>
        </div>
      ) : (
        <p className="flex items-center gap-8 font-display text-6xl font-extrabold">
          <span className="text-graphite">{money(before)}</span><ArrowRight className="size-8" aria-label="to" />
          <span className="text-cobalt">{money(after)}</span>
        </p>
      )}
      {scenario && <div className="mt-12 border-t border-line pt-5 text-lg text-muted">
        <p>Assumption: {scenario.assumedLatencyMs} ms × Bing’s 0.6% / 100 ms</p>
      </div>}
    </SlideFrame>
  );
}
