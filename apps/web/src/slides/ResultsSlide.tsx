import type { Run } from "@storefront/contracts";
import { ArrowRight } from "lucide-react";
import type { Pair, Verdict } from "../lib/story";
import { MerchantValue } from "./MerchantValue";
import { ShopperOutcome } from "./ShopperOutcome";

const Share = ({
  value,
  total,
  className,
}: {
  value: number;
  total: number;
  className: string;
}) => (
  <span className={className}>
    {value}
    <span className="ml-[0.12em] text-[0.32em] font-bold tracking-normal">of {total}</span>
  </span>
);

export function ResultsSlide({
  result,
  pairs,
  onEvidence,
}: {
  result: Verdict;
  pairs: Pair[];
  onEvidence: (run: Run) => void;
}) {
  const broken = result.technical.length;
  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col justify-center px-4 py-10 sm:px-8">
      {result.total > 0 && (
        <>
          <p
            className="flex flex-wrap items-baseline gap-x-8 font-display text-8xl leading-none font-black tracking-[-0.03em] sm:text-[11rem]"
            style={{ fontVariationSettings: '"wdth" 75' }}
          >
            <Share value={result.before} total={result.total} className="text-graphite" />
            <ArrowRight className="size-[0.35em] self-center" aria-label="to" />
            <Share value={result.after} total={result.total} className="text-cobalt" />
          </p>
          <p className="mt-4 text-2xl">new shoppers bought something that fits</p>
          <ul className="mt-12 flex flex-wrap gap-x-12 gap-y-8">
            {result.shoppers.map((pair) => (
              <li key={pair.persona.id}>
                <ShopperOutcome pair={pair} onOpen={onEvidence} />
              </li>
            ))}
          </ul>
        </>
      )}
      <div className="mt-14">
        <MerchantValue pairs={pairs} />
      </div>
      {broken > 0 && (
        <p className="mt-6 text-muted">
          {broken === 1 ? "1 shopper" : `${broken} shoppers`} had a run break and{" "}
          {broken === 1 ? "isn't" : "aren't"} counted.
        </p>
      )}
    </div>
  );
}
