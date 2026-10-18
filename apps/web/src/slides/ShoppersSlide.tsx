import type { Run } from "@storefront/contracts";
import { ArrowRight } from "lucide-react";
import type { Pair, Verdict } from "../lib/story";

function Names({ pairs, onEvidence }: { pairs: Pair[]; onEvidence: (run: Run) => void }) {
  return pairs.map((pair, i) => (
    <span key={pair.persona.id}>
      {i > 0 && ", "}
      <button
        type="button"
        className="underline underline-offset-4"
        onClick={() => pair.candidate && onEvidence(pair.candidate)}
      >
        {pair.persona.name}
      </button>
    </span>
  ));
}

export function ShoppersSlide({
  result,
  onEvidence,
}: {
  result: Verdict;
  onEvidence: (run: Run) => void;
}) {
  const broken = result.technical.length;
  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col justify-center px-4 sm:px-8">
      {result.total > 0 && (
        <>
          <p className="flex flex-wrap items-center gap-x-6 font-display text-7xl leading-none font-extrabold tracking-[-0.02em] sm:text-9xl">
            <span className="text-graphite">
              {result.before} of {result.total}
            </span>
            <ArrowRight className="size-[0.6em]" aria-label="to" />
            <span className="text-cobalt">
              {result.after} of {result.total}
            </span>
          </p>
          <p className="mt-6 max-w-[36ch] text-2xl">
            new shoppers found and bought what they came for.
          </p>
          <p className="mt-2 max-w-[60ch] text-muted">
            None of these shoppers were used to find or choose the fix.
          </p>
        </>
      )}
      <div className="mt-10 space-y-2 text-xl">
        {result.better.length > 0 && (
          <p>
            Better with the fix: <Names pairs={result.better} onEvidence={onEvidence} />
          </p>
        )}
        {result.worse.length > 0 && (
          <p className="text-signal">
            Worse with the fix: <Names pairs={result.worse} onEvidence={onEvidence} />
          </p>
        )}
        {broken > 0 && (
          <p className="text-muted">
            {broken === 1 ? "1 shopper" : `${broken} shoppers`} had a run break for technical
            reasons and {broken === 1 ? "isn't" : "aren't"} counted.
          </p>
        )}
      </div>
    </div>
  );
}
