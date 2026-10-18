import { Slider } from "@base-ui/react/slider";
import { ChevronsLeftRight } from "lucide-react";
import { useState } from "react";
import type { Frame } from "../lib/story";
import { cn } from "../lib/utils";
import { RaceFrame } from "./RaceFrame";

/** A side's label hides once the divider leaves it less room than this, in percent. */
const LABEL_ROOM = 22;

export function ShopComparison({
  before,
  after,
  beforeLabel,
  afterLabel,
  initialPosition = 50,
}: {
  before?: Frame;
  after?: Frame;
  beforeLabel: string;
  afterLabel: string;
  initialPosition?: number;
}) {
  const [position, setPosition] = useState(initialPosition);
  const [interacted, setInteracted] = useState(false);

  return (
    <div>
      <div
        role="group"
        aria-label="Page version"
        className="mb-3 flex flex-wrap justify-center gap-2"
      >
        {[
          { label: beforeLabel, value: 100 },
          { label: afterLabel, value: 0 },
        ].map(({ label, value }) => (
          <button
            key={value}
            type="button"
            aria-pressed={position === value}
            onClick={() => {
              setPosition(value);
              setInteracted(true);
            }}
            className={cn(
              "rounded-full border px-4 py-2 text-sm font-semibold transition-colors",
              position === value
                ? "border-cobalt bg-cobalt text-paper"
                : "border-line bg-paper text-muted hover:border-cobalt",
            )}
          >
            {label}
          </button>
        ))}
      </div>
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
        className="relative isolate aspect-[8/5] w-full overflow-hidden rounded-md"
      >
        <RaceFrame frame={after} still className="[&_figcaption]:sr-only" />
        <div className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}>
          <RaceFrame frame={before} still className="[&_figcaption]:sr-only" />
        </div>
        <span
          aria-hidden
          className={cn(
            "pointer-events-none absolute bottom-3 w-max -translate-x-1/2 rounded-md bg-ink px-3 py-2 text-center text-sm leading-tight font-bold text-paper shadow-lg transition-opacity sm:text-base",
            (position < LABEL_ROOM || position === 100) && "opacity-0",
          )}
          style={{ left: `${position / 2}%`, maxWidth: `calc(${position}% - 3rem)` }}
        >
          {beforeLabel}
        </span>
        <span
          aria-hidden
          className={cn(
            "pointer-events-none absolute bottom-3 w-max -translate-x-1/2 rounded-md bg-cobalt px-3 py-2 text-center text-sm leading-tight font-bold text-paper shadow-lg transition-opacity sm:text-base",
            (position > 100 - LABEL_ROOM || position === 0) && "opacity-0",
          )}
          style={{ left: `${(100 + position) / 2}%`, maxWidth: `calc(${100 - position}% - 3rem)` }}
        >
          {afterLabel}
        </span>
        <Slider.Control className="absolute inset-0 touch-none cursor-ew-resize">
          <Slider.Track className="relative h-full w-full">
            <Slider.Thumb
              aria-label="Compare before and after"
              getAriaValueText={(_, value) =>
                `${value}% ${beforeLabel}, ${100 - value}% ${afterLabel}`
              }
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
      <p className="mt-2 text-center text-xs text-muted">
        Choose a page above or drag the divider to compare.
      </p>
    </div>
  );
}
