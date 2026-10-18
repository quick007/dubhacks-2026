import type { Run } from "@storefront/contracts";
import { ArrowRight, Check, X } from "lucide-react";
import { purchased, type Pair } from "../lib/story";
import { cn } from "../lib/utils";

function Mark({ run }: { run?: Run }) {
  const bought = purchased(run);
  const Icon = bought ? Check : X;
  return (
    <span
      className={cn(
        "grid size-9 place-items-center rounded-full sm:size-11",
        bought ? "bg-cobalt text-paper" : "border-2 border-signal text-signal",
      )}
    >
      <Icon aria-label={bought ? "bought" : "did not buy"} className="size-5" strokeWidth={3} />
    </span>
  );
}

/** One new shopper: their name, then today's shop and the fix side by side. */
export function ShopperOutcome({ pair, onOpen }: { pair: Pair; onOpen: (run: Run) => void }) {
  const run = pair.candidate ?? pair.baseline;
  return (
    <button
      type="button"
      disabled={!run}
      onClick={() => run && onOpen(run)}
      className="group text-left"
    >
      <span className="block font-voice text-2xl italic group-hover:underline group-hover:underline-offset-4 sm:text-3xl">
        {pair.persona.name}
      </span>
      <span className="mt-3 flex items-center gap-2">
        <Mark run={pair.baseline} />
        <ArrowRight aria-label="then" className="size-4 text-graphite" />
        <Mark run={pair.candidate} />
      </span>
    </button>
  );
}
