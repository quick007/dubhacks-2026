import type { Run } from "@storefront/contracts";
import { modelRequest, ms, ordinal, product } from "../lib/model";
import { frames, type Pair } from "../lib/story";
import { ShopComparison } from "./ShopComparison";
import { PersonalizationChanges } from "./PersonalizationChanges";

const listing = (run?: Run) => frames(run).find((frame) => frame.event.page?.kind === "category");

export function FixSlide({ pair }: { pair: Pair }) {
  const name = pair.persona.name;
  const request = modelRequest(pair.persona.id);
  const target = request && product(request.target);
  const bob = pair.persona.id === "dev-sofa";
  if (!pair.baseline || !pair.candidate) return null;
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
            {bob
              ? "His sofa throw, with the details that explain the fit."
              : `The ${target.name.toLowerCase()} moves from ${ordinal(target.defaultRank)} to ${ordinal(request.modelRank)}. Personalizing took ${ms(request.latencyMs)}.`}
          </p>
        )}
      </div>
      <div
        className={
          bob
            ? "grid w-full items-center gap-8 lg:w-[min(100%,calc((100dvh-23rem)*1.6+308px))] lg:grid-cols-[minmax(0,1fr)_260px]"
            : "w-[min(100%,calc((100dvh-16rem)*1.6))]"
        }
      >
        <div className="min-w-0">
          <ShopComparison
            before={
              bob && before
                ? {
                    ...before,
                    src: "/fieldwork/bob/default.jpg",
                    event: {
                      ...before.event,
                      message: "Default collection with product names, prices and color options.",
                    },
                  }
                : before
            }
            after={
              bob && after
                ? {
                    ...after,
                    src: "/fieldwork/bob/personalized.jpg",
                    event: {
                      ...after.event,
                      message:
                        "Designed page for Bob with a Picked for Bob badge, Cotton, Throw blanket and For your sofa chips, and a Why this fits explanation.",
                    },
                  }
                : after
            }
            beforeLabel="Default page"
            afterLabel={`Personalized for ${name}`}
            initialPosition={bob ? 0 : 50}
          />
          {bob && request && target && (
            <p className="mt-3 text-center text-sm text-muted">
              The throw moves from {ordinal(target.defaultRank)} to {ordinal(request.modelRank)} in
              the measured ranking run ({ms(request.latencyMs)}).
            </p>
          )}
        </div>
        {bob && <PersonalizationChanges />}
      </div>
    </section>
  );
}
