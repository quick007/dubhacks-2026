import { ArrowRight } from "lucide-react";
import type { PresentationResults } from "../lib/presentation-results";
import { SlideFrame } from "./SlideFrame";

const money = (cents: number) => new Intl.NumberFormat("en-US", {
  style: "currency", currency: "USD", maximumFractionDigits: 0,
}).format(cents / 100);

/** The immutable assigned-cohort result; no mock runs enter these figures. */
export function CohortResultsSlide({ result }: { result: PresentationResults }) {
  const before = result.data.metrics.arms.generic;
  const after = result.data.metrics.arms["local-model"];
  const figures = [
    ["Checkouts that fit", String(before.verifiedPurchases), String(after.verifiedPurchases)],
    ["Value of checkouts that fit", money(before.revenue.totalCents), money(after.revenue.totalCents)],
    ["All accepted order value", money(before.allServerConfirmedOrders.merchandiseSubtotalCents),
      money(after.allServerConfirmedOrders.merchandiseSubtotalCents)],
  ];
  return (
    <SlideFrame title="Did shoppers get what they needed?" lead="60 paired synthetic shoppers. Every assigned session counts.">
      <p className="flex flex-wrap items-baseline gap-x-8 font-display text-6xl leading-[1.1] font-semibold tracking-[-0.025em] sm:text-8xl">
        <span className="text-graphite">{before.intentCorrectSuccess.numerator}<span className="ml-2 text-2xl">of 60</span></span>
        <ArrowRight className="size-[0.35em] self-center" aria-label="to" />
        <span className="text-cobalt">{after.intentCorrectSuccess.numerator}<span className="ml-2 text-2xl">of 60</span></span>
      </p>
      <p className="mt-4 text-xl">met their goal, including choosing nothing when nothing fit</p>
      <dl className="mt-12 grid gap-8 sm:grid-cols-3">
        {figures.map(([label, first, second]) => (
          <div key={label} className="border-t border-line pt-3">
            <dt className="text-muted">{label}</dt>
            <dd className="mt-2 flex flex-wrap items-center gap-3 font-display text-4xl font-extrabold tracking-[-0.015em]">
              <span className="text-graphite">{first}</span>
              <ArrowRight className="size-5" aria-label="to" />
              <span className="text-cobalt">{second}</span>
            </dd>
          </div>
        ))}
      </dl>
      <p className="mt-8 text-sm text-muted">Generic shop → personalized shop. Server-confirmed simulated orders; merchandise value only.</p>
    </SlideFrame>
  );
}
