import { ArrowDown, ArrowUp } from "lucide-react";
import { personalizationSteps } from "../lib/personalization";

export function PersonalizationChanges({
  step,
  onChange,
}: {
  step: number;
  onChange: (step: number) => void;
}) {
  return (
    <aside aria-label="Current change" className="flex flex-col gap-6">
      <div role="status" className="grid min-h-40">
        {/* Shared grid space reserves the tallest caption without moving the text or arrows. */}
        {personalizationSteps.map((item, index) => (
          <div
            key={item.image}
            aria-hidden={index !== step}
            className={`col-start-1 row-start-1 ${index === step ? "" : "invisible"}`}
          >
            <h3 className="text-2xl font-semibold lg:text-3xl">{item.title}</h3>
            <p className="mt-3 max-w-[23ch] text-2xl leading-snug text-muted lg:text-4xl">
              {item.detail}
            </p>
          </div>
        ))}
      </div>
      <nav aria-label="Personalization changes" className="flex items-center gap-4">
        <button
          type="button"
          aria-label="Previous change"
          disabled={step === 0}
          onClick={() => onChange(step - 1)}
          className="grid size-10 place-items-center rounded-sm hover:bg-line/50 disabled:invisible"
        >
          <ArrowUp aria-hidden className="size-5" />
        </button>
        <button
          type="button"
          aria-label="Next change"
          disabled={step === personalizationSteps.length - 1}
          onClick={() => onChange(step + 1)}
          className="grid size-10 place-items-center rounded-sm hover:bg-line/50 disabled:invisible"
        >
          <ArrowDown aria-hidden className="size-5" />
        </button>
      </nav>
    </aside>
  );
}
