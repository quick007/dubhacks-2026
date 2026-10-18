import type { Persona } from "@storefront/contracts";
import { ArrowRight } from "lucide-react";
import stages from "../fieldwork/title-stages.json";
import { modelRequest, ordinal, product } from "../lib/model";
import { profiles } from "../lib/profiles";
import { StoreShot, type ShotStage } from "./StoreShot";
import { SlideFrame } from "./SlideFrame";

const shown: { personaId: string; stage: ShotStage }[] = [
  { personaId: stages.featured, stage: "fit" },
  { personaId: stages.secondShopper, stage: "second" },
];

/** Opening: two shoppers the store knows well, each on the same shop personalized for them. */
export function TwoShoppersSlide({ personas }: { personas: Persona[] }) {
  return (
    <SlideFrame title="Two shoppers, one store">
      <div className="grid gap-10 md:grid-cols-2">
        {shown.map(({ personaId, stage }) => {
          const persona = personas.find((item) => item.id === personaId);
          const request = modelRequest(personaId);
          const target = request && product(request.target);
          const profile = profiles[personaId];
          if (!persona || !request || !target || !profile) return null;
          return (
            <figure key={personaId}>
              <p className="font-display text-3xl font-semibold tracking-[-0.015em]">
                {persona.name}, {profile.age}
              </p>
              <ul className="mt-3 grid gap-x-6 gap-y-1.5 text-muted sm:grid-cols-2">
                {profile.facts.map(({ icon: Icon, fact }) => (
                  <li key={fact} className="flex items-center gap-2">
                    <Icon aria-hidden className="size-4 shrink-0 text-graphite" />
                    {fact}
                  </li>
                ))}
              </ul>
              <p className="mt-4 text-2xl leading-snug font-medium text-cobalt">
                "{request.request}"
              </p>
              <div className="mt-3">
                <StoreShot stage={stage} names={[persona.name]} />
              </div>
              <figcaption className="mt-3 flex items-center gap-2 text-lg">
                {target.name}
                <span className="text-graphite">{ordinal(target.defaultRank)}</span>
                <ArrowRight aria-label="becomes" className="size-4" />
                <span className="font-bold text-cobalt">{ordinal(request.modelRank)}</span>
              </figcaption>
            </figure>
          );
        })}
      </div>
    </SlideFrame>
  );
}
