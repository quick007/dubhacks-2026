import type { Persona } from "@storefront/contracts";
import { ChevronRight } from "lucide-react";
import { ShopperQueries } from "./ShopperQueries";

export function TitleSlide({ personas, featuredId }: { personas: Persona[]; featuredId: string }) {
  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col justify-center gap-12 sm:justify-between px-4 pt-10 sm:px-8 sm:pt-14">
      <ShopperQueries
        personas={personas.filter((persona) => persona.task)}
        featuredId={featuredId}
      />
      <div>
        <h1
          className="font-display text-7xl leading-[0.85] font-black tracking-[-0.03em] sm:text-[10rem]"
          style={{ fontVariationSettings: '"wdth" 75' }}
        >
          Storefront Lab
        </h1>
        <p className="mt-5 text-xl sm:text-3xl">
          Better personalized shopping for customers and stores.
        </p>
      </div>
      <ChevronRight
        aria-hidden="true"
        className="title-navigation-hint pointer-events-none fixed top-1/2 right-3 size-6 text-muted/40 sm:right-8 sm:size-8"
        strokeWidth={1.5}
      />
    </div>
  );
}
