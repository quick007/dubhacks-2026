import { Bot, MousePointer2 } from "lucide-react";
import type { CSSProperties } from "react";

type Point = { x: number; y: number };
const colors = ["#2448d6", "#8b2399", "#006b60"];

/** A looping illustration of browsing, independent of recorded session replays. */
export function ShopperCursor({
  name,
  at,
  via,
  to,
  index = 0,
}: {
  name?: string;
  at: Point;
  via: Point;
  to: Point;
  index?: number;
}) {
  return (
    <span
      data-shopper={name}
      className={`shopper-cursor pointer-events-none absolute inset-0 ${index > 0 ? "hidden md:block" : ""}`}
      style={
        {
          "--shopper-color": colors[index % colors.length],
          "--start-x": `${at.x * 100}%`,
          "--start-y": `${at.y * 100}%`,
          "--via-x": `${via.x * 100}%`,
          "--via-y": `${via.y * 100}%`,
          "--end-x": `${to.x * 100}%`,
          "--end-y": `${to.y * 100}%`,
          animationDelay: `${index * -2.7}s`,
        } as CSSProperties
      }
    >
      <span className="absolute top-0 left-0 flex items-start">
        <MousePointer2
          className="size-5 shrink-0 fill-[var(--shopper-color)] text-paper drop-shadow-md sm:size-7"
          strokeWidth={2}
        />
        <span className="absolute top-4 left-3 flex -translate-x-1/2 items-center gap-1 rounded-full bg-[var(--shopper-color)] px-2 py-0.5 text-[10px] font-bold whitespace-nowrap text-paper shadow-md ring-2 ring-paper sm:top-6 sm:text-xs">
          <Bot className="size-3 shrink-0" />
          {name}
        </span>
      </span>
    </span>
  );
}
