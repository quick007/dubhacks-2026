import { ArrowRight, RefreshCw } from "lucide-react";
import { Fragment } from "react";
import { StoreSketch, type SketchStage } from "./StoreSketch";

const STAGGER_MS = 450;
const steps: { stage: SketchStage; caption: string }[] = [
  { stage: "browse", caption: "AI shoppers browse your store" },
  { stage: "stuck", caption: "See where each one gets stuck" },
  { stage: "fit", caption: "Shoppers find it, stores sell it" },
];

export function TitleSlide({ names }: { names: string[] }) {
  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col justify-center gap-14 px-4 py-10 sm:px-8">
      <div className="text-center">
        <h1
          className="font-display text-7xl leading-[0.85] font-black tracking-[-0.03em] sm:text-9xl"
          style={{ fontVariationSettings: '"wdth" 75' }}
        >
          Storefront Lab
        </h1>
        <p className="mt-5 text-xl sm:text-3xl">
          Better personalized shopping for customers and stores.
        </p>
      </div>
      <ol className="grid items-start gap-8 md:grid-cols-[1fr_auto_1fr_auto_1fr] md:gap-4">
        {steps.map((step, i) => (
          <Fragment key={step.stage}>
            {i > 0 && (
              <ArrowRight
                aria-hidden
                className="hidden size-6 self-center text-graphite md:block"
                style={{ animationDelay: `${300 + i * STAGGER_MS}ms` }}
              />
            )}
            <li
              className="grid animate-rise grid-cols-[9rem_1fr] items-center gap-4 md:block"
              style={{ animationDelay: `${300 + i * STAGGER_MS}ms` }}
            >
              <StoreSketch stage={step.stage} names={names} />
              <p className="text-lg leading-snug sm:text-xl md:mt-4">{step.caption}</p>
            </li>
          </Fragment>
        ))}
      </ol>
      <p
        className="flex animate-rise items-center justify-center gap-3 text-xl text-muted sm:text-2xl"
        style={{ animationDelay: `${300 + steps.length * STAGGER_MS}ms` }}
      >
        <RefreshCw aria-hidden className="size-5 shrink-0" />
        Simulated shoppers test, agents fix, repeat
      </p>
    </div>
  );
}
