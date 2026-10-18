import type { Run } from "@storefront/contracts";
import { modelRequest, ms, ordinal, product } from "../lib/model";
import { frames, type Pair } from "../lib/story";
import { ShopComparison } from "./ShopComparison";
import { PersonalizationSequence } from "./PersonalizationSequence";

const listing = (run?: Run) => frames(run).find((frame) => frame.event.page?.kind === "category");

export function FixSlide({ pair }: { pair: Pair }) {
  const name = pair.persona.name;
  const request = modelRequest(pair.persona.id);
  const target = request && product(request.target);
  if (!pair.baseline || !pair.candidate) return null;
  if (pair.persona.id === "dev-sofa") return <PersonalizationSequence />;
  const before = listing(pair.baseline);
  const after = listing(pair.candidate);
  return (
    <section className="mx-auto flex w-full max-w-[1440px] flex-col items-center justify-center gap-6 px-6 py-4 sm:px-8">
      <div className="text-center">
        <h2 className="font-display text-4xl leading-tight font-extrabold tracking-[-0.015em]">
          {name}'s personalized page
        </h2>
        {request && target && (
          <p className="mt-2 text-lg text-muted sm:text-xl">
            The {target.name.toLowerCase()} moves from {ordinal(target.defaultRank)} to{" "}
            {ordinal(request.modelRank)}. Personalizing took {ms(request.latencyMs)}.
          </p>
        )}
      </div>
      <div className="w-[min(100%,calc((100dvh-16rem)*1.6))]">
        <div className="min-w-0">
          <ShopComparison
            before={before}
            after={after}
            beforeLabel="Default page"
            afterLabel={`Personalized for ${name}`}
          />
        </div>
      </div>
    </section>
  );
}
