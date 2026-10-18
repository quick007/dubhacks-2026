import type { Run } from "@storefront/contracts";
import { frames, type Pair } from "../lib/story";
import { ShopComparison } from "./ShopComparison";

const listing = (run?: Run) =>
  frames(run).find(
    (frame) => frame.event.page?.kind === "category" || frame.event.page?.kind === "search",
  );

export function FixSlide({ pair }: { pair: Pair }) {
  const proposal = pair.baseline?.proposedChanges[0];
  if (!pair.baseline || !proposal) return null;
  return (
    <section className="mx-auto flex w-full max-w-6xl flex-col items-center justify-center gap-4 px-6 py-4 sm:px-8">
      <h2 className="font-display text-4xl leading-tight font-extrabold tracking-[-0.015em]">
        A proposed fix
      </h2>
      <div className="w-[min(100%,calc((100dvh-14rem)*1.6))]">
        <ShopComparison before={listing(pair.baseline)} after={listing(pair.candidate)} />
      </div>
    </section>
  );
}
