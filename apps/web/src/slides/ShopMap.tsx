import { useMemo } from "react";
import { flows, type Arm, type Pair } from "../lib/story";
import { RouteGraph } from "./RouteGraph";

export function ShopMap({ pairs, arm }: { pairs: Pair[]; arm: Arm }) {
  const flow = useMemo(() => flows(pairs, arm), [pairs, arm]);
  const bought = pairs.filter((pair) => {
    const run = pair[arm];
    const outcome = run?.evaluation?.findability?.outcome;
    return (
      run?.status === "completed" && (outcome === "purchased" || outcome === "constraint-failure")
    );
  }).length;
  const stopped = [...flow.giveUps.values()].reduce((total, count) => total + count, 0);
  const color = arm === "baseline" ? "var(--color-graphite)" : "var(--color-ink)";
  return (
    <section className="min-w-0">
      <header className="flex flex-col gap-1 sm:flex-row sm:items-baseline sm:justify-between">
        <h3 className="text-lg font-semibold sm:text-xl">
          {arm === "baseline" ? "Today's shop" : "Proposed version"}
        </h3>
        <p className="text-sm text-muted sm:text-base">
          <span className="font-semibold text-ink">{bought} bought</span>
          {stopped > 0 && <span className="text-signal">, {stopped} stopped</span>}
        </p>
      </header>
      <RouteGraph flow={flow} color={color} />
      <RouteGraph flow={flow} color={color} mobile />
    </section>
  );
}
