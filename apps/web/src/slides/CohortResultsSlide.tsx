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
        <div className="grid items-center gap-5 sm:grid-cols-[1fr_auto_1fr_auto_1fr]">
          <div>
            <p className="font-display text-6xl font-extrabold tracking-tight text-graphite">{scenario.observedPercent.toFixed(2)}%</p>
            <p className="mt-3 text-lg text-muted">Observed synthetic value difference</p>
          </div>
          <Minus className="size-7 text-graphite" aria-label="minus" />
          <div>
            <p className="font-display text-6xl font-extrabold tracking-tight text-graphite">{scenario.assumedLossPercent.toFixed(2)}%</p>
            <p className="mt-3 text-lg text-muted">Assumed delay loss</p>
          </div>
          <Equal className="size-7 text-graphite" aria-label="equals" />
          <div>
            <p className="font-display text-6xl font-extrabold tracking-tight text-cobalt">{scenario.remainderPercentagePoints.toFixed(2)}<span className="ml-1 text-2xl">pp</span></p>
            <p className="mt-3 text-lg text-muted">Illustrative remainder</p>
          </div>
        </div>
      ) : (
        <p className="flex items-center gap-8 font-display text-6xl font-extrabold">
          <span className="text-graphite">{money(before)}</span><ArrowRight className="size-8" aria-label="to" />
          <span className="text-cobalt">{money(after)}</span>
        </p>
      )}
      <div className="mt-12 flex flex-wrap gap-x-10 gap-y-3 border-t border-line pt-5 text-lg text-muted">
        <p>Frozen v5 synthetic: {money(before)} → {money(after)} per session</p>
        {scenario && <p>Assumption: {scenario.incrementalMs} ms × Bing’s 0.6% / 100 ms</p>}
      </div>
    </SlideFrame>
  );
}
