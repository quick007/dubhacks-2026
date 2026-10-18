import { Slider } from "@base-ui/react/slider";
import { ChevronsLeftRight } from "lucide-react";
import { useState } from "react";
import type { Frame } from "../lib/story";
import { RaceFrame } from "./RaceFrame";

export function ShopComparison({ before, after }: { before?: Frame; after?: Frame }) {
  const [position, setPosition] = useState(74);
  const [interacted, setInteracted] = useState(false);

  return (
    <Slider.Root
      value={position}
      onValueChange={(value) => {
        setInteracted(true);
        setPosition(Number(value));
      }}
      onPointerDown={() => setInteracted(true)}
      onKeyDown={() => setInteracted(true)}
      min={0}
      max={100}
      step={1}
      thumbAlignment="edge"
      data-no-swipe
      className="relative isolate aspect-[8/5] w-full"
    >
      <RaceFrame frame={after} className="[&_figcaption]:sr-only [&_span]:hidden" />
      <div className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}>
        <RaceFrame frame={before} className="[&_figcaption]:sr-only [&_span]:hidden" />
      </div>
      <Slider.Control className="absolute inset-0 touch-none cursor-ew-resize">
        <Slider.Track className="relative h-full w-full">
          <Slider.Thumb
            aria-label="Compare before and after"
            getAriaValueText={(_, value) => `${value}% before, ${100 - value}% after`}
            className="relative flex h-full w-1 items-center justify-center bg-ink shadow-[0_0_0_1px_#ffffff,0_0_12px_#17241d40] outline-none [&_input]:outline-none"
          >
            <span
              className={`flex size-11 shrink-0 items-center justify-center rounded-full border-2 border-paper bg-ink text-paper shadow-[0_4px_14px_#17241d55] ${interacted ? "" : "comparison-drag-hint"}`}
            >
              <ChevronsLeftRight aria-hidden size={22} />
            </span>
          </Slider.Thumb>
        </Slider.Track>
      </Slider.Control>
    </Slider.Root>
  );
}
