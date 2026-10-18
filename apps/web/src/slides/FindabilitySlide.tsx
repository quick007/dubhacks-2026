import { ArrowRight } from "lucide-react";
import { useMemo } from "react";
import { ledger, type Pair } from "../lib/story";
import { SlideFrame } from "./SlideFrame";

export function FindabilitySlide({ pairs }: { pairs: Pair[] }) {
  const data = useMemo(() => ledger(pairs), [pairs]);
  return (
    <SlideFrame
      title="How easy it was to find"
      lead="Share of shoppers whose purchase met each requirement, across every shopper."
    >
      <dl className="grid gap-x-12 gap-y-8 sm:grid-cols-2">
        {data.rows.map((row) => (
          <div key={row.id} className="border-t border-line pt-3">
            <dt className="text-muted">{row.label}</dt>
            <dd className="mt-1 flex items-center gap-4 font-display text-5xl font-extrabold tracking-[-0.015em]">
              <span className="text-graphite">
                {row.before.met}/{row.before.total}
              </span>
              <ArrowRight className="size-7" aria-label="to" />
              <span className="text-cobalt">
                {row.after.met}/{row.after.total}
              </span>
            </dd>
          </div>
        ))}
      </dl>
      {data.giveUpPage.baseline && (
        <p className="mt-10 text-xl">
          Most common place to give up in today's shop:{" "}
          <span className="font-bold">{data.giveUpPage.baseline}</span>
        </p>
      )}
    </SlideFrame>
  );
}
