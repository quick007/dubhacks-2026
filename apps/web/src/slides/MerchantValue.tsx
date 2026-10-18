import { ArrowRight } from "lucide-react";
import { useMemo } from "react";
import { merchant, type Pair } from "../lib/story";

const money = (value: number) => `$${value.toLocaleString("en-US")}`;
const count = (value: number) => String(value);

// Fewer returns is the improvement, so a rise is shown as a regression.
function tone(before: number, after: number, lowerIsBetter: boolean) {
  if (after === before) return "text-ink";
  return after < before === lowerIsBetter ? "text-cobalt" : "text-signal";
}

export function MerchantValue({ pairs }: { pairs: Pair[] }) {
  const { baseline, candidate } = useMemo(() => merchant(pairs), [pairs]);
  const rows = [
    ["Suitable orders", baseline.orders, candidate.orders, count, false],
    ["Suitable order value", baseline.value, candidate.value, money, false],
    ["Likely returns", baseline.wrongItems, candidate.wrongItems, count, true],
  ] as const;
  return (
    <dl className="grid gap-8 sm:grid-cols-3">
      {rows.map(([label, before, after, format, lowerIsBetter]) => (
        <div key={label} className="border-t border-line pt-3">
          <dt className="text-muted">{label}</dt>
          <dd className="mt-2 flex flex-wrap items-center gap-3 font-display text-4xl font-extrabold tracking-[-0.015em] sm:text-5xl">
            <span className="text-graphite">{format(before)}</span>
            <ArrowRight className="size-6" aria-label="to" />
            <span className={tone(before, after, lowerIsBetter)}>{format(after)}</span>
          </dd>
        </div>
      ))}
    </dl>
  );
}
