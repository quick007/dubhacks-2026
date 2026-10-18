import { Bot, Check, X } from "lucide-react";
import { cn } from "../lib/utils";

export type SketchStage = "browse" | "stuck" | "fit";

// Six jackets, drawn as colour swatches; the sketch is an illustration, not a screenshot.
const swatches = ["#26365f", "#5d6b3a", "#1e2421", "#3b4f7a", "#8a8f86", "#26365f"];
// Where each agent sits while browsing, as % of the frame.
const cursors = [
  { left: "18%", top: "38%" },
  { left: "56%", top: "72%" },
  { left: "82%", top: "30%" },
];

/** A tiny storefront: agents browsing it, getting stuck in it, then finding what fits. */
export function StoreSketch({ stage, names }: { stage: SketchStage; names: string[] }) {
  return (
    <div aria-hidden className="relative overflow-hidden rounded-lg border border-line bg-paper">
      <div className="flex h-7 items-center gap-2 bg-ink px-3">
        <span className="h-2 w-10 rounded-full bg-paper/70" />
        <span className="ml-auto h-3 w-1/3 rounded-sm bg-paper/20" />
      </div>
      <ul className="grid grid-cols-3 gap-2 p-3">
        {swatches.map((color, i) => {
          const stuck = stage === "stuck" && i === 0;
          const fit = stage === "fit" ? [5, 3, 1].indexOf(i) : -1;
          return (
            <li
              key={i}
              className={cn(
                "relative rounded-md bg-fog p-1.5",
                stuck && "ring-2 ring-signal",
                fit >= 0 && "ring-2 ring-cobalt",
                stage === "fit" && fit < 0 && "opacity-40",
              )}
            >
              <span
                className={cn("block aspect-[4/3] rounded-sm", stuck && "opacity-50")}
                style={{ background: color }}
              />
              <span className="mt-1.5 block h-1.5 w-3/4 rounded-full bg-line" />
              <span className="mt-1 block h-1.5 w-1/3 rounded-full bg-line" />
              {stuck && (
                <span className="absolute -top-2 left-1 flex items-center gap-1 rounded-full max-md:px-1 bg-signal px-2 py-0.5 text-xs font-bold whitespace-nowrap text-paper">
                  <X className="size-3" strokeWidth={3} />
                  <span className="max-md:hidden">No size M</span>
                </span>
              )}
              {fit >= 0 && names[fit] && (
                <span className="absolute -top-2 left-1 flex items-center gap-1 rounded-full max-md:px-1 bg-cobalt px-2 py-0.5 text-xs font-bold whitespace-nowrap text-paper">
                  <Check className="size-3" strokeWidth={3} />
                  <span className="max-md:hidden">{names[fit]}</span>
                </span>
              )}
            </li>
          );
        })}
      </ul>
      {stage !== "fit" &&
        cursors.slice(0, stage === "stuck" ? 1 : cursors.length).map((at, i) => (
          <span
            key={i}
            className="absolute flex -translate-x-1/2 -translate-y-1/2 items-center gap-1"
            style={stage === "stuck" ? { left: "26%", top: "52%" } : at}
          >
            <span className="grid size-7 place-items-center rounded-full bg-ink text-paper ring-2 ring-paper">
              <Bot className="size-4" />
            </span>
            <span className="rounded bg-ink px-1.5 py-0.5 text-xs font-bold text-paper ring-2 ring-paper max-md:hidden">
              {names[i]}
            </span>
          </span>
        ))}
    </div>
  );
}
