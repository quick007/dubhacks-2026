import type { Persona, Run } from "@storefront/contracts";
import { Check, X } from "lucide-react";
import { outcomeLabels } from "../lib/format";
import { requirementLabel, type Frame } from "../lib/story";
import { cn } from "../lib/utils";
import { RaceFrame } from "./RaceFrame";

export function RaceLane({
  title,
  tone,
  frames,
  index,
  playing,
  run,
  persona,
}: {
  title?: string;
  tone: "graphite" | "cobalt";
  frames: Frame[];
  index: number;
  playing: boolean;
  run?: Run;
  persona: Persona;
}) {
  const at = Math.min(index, frames.length - 1);
  const findability = index >= frames.length - 1 ? run?.evaluation?.findability : undefined;
  return (
    <div className="min-w-0">
      {title && (
        <h2 className="flex items-center gap-3 font-display text-4xl font-extrabold tracking-[-0.015em]">
          <span
            aria-hidden
            className={cn(
              "h-8 w-1.5 rounded-full",
              tone === "cobalt" ? "bg-cobalt" : "bg-graphite",
            )}
          />
          {title}
        </h2>
      )}
      <RaceFrame frame={frames[at]} name={persona.name} playing={playing} className={title ? "mt-3" : undefined} />
      {findability && (
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 text-sm">
          <p
            className={cn(
              "font-semibold",
              findability.outcome === "purchased" ? "text-cobalt" : "text-signal",
            )}
          >
            {outcomeLabels[findability.outcome]}
          </p>
          {findability.outcome === "abandoned" ? (
            <p className="max-w-[65ch] text-muted">{run?.evaluation?.failureReasons[0]}</p>
          ) : (
            <ul className="flex flex-wrap gap-x-4 gap-y-1">
              {findability.requirements.map((requirement) => (
                <li
                  key={requirement.id}
                  className={cn("flex items-center gap-1", !requirement.met && "text-signal")}
                >
                  {requirement.met ? <Check className="size-4" /> : <X className="size-4" />}
                  {requirementLabel(persona, requirement.id)}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
