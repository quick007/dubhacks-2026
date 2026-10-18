import type { Persona, Run } from "@storefront/contracts";
import { Check, X } from "lucide-react";
import { outcomeLabels } from "../lib/format";
import { counters, requirementLabel, type Frame } from "../lib/story";
import { cn } from "../lib/utils";
import { RaceFrame } from "./RaceFrame";

export function RaceLane({
  title,
  tone,
  frames,
  index,
  run,
  persona,
  side = false,
}: {
  title?: string;
  tone: "graphite" | "cobalt";
  frames: Frame[];
  index: number;
  run?: Run;
  persona: Persona;
  /** Lay the frame and its numbers side by side, for a slide with a single lane. */
  side?: boolean;
}) {
  const at = Math.min(index, frames.length - 1);
  const findability = index >= frames.length - 1 ? run?.evaluation?.findability : undefined;
  const totals = counters(frames, at);
  return (
    <div className={cn("min-w-0", side && "grid items-start gap-x-10 lg:grid-cols-[3fr_2fr]")}>
      {title && (
        <h2
          className={cn(
            "font-display text-4xl font-extrabold tracking-[-0.015em]",
            tone === "cobalt" ? "text-cobalt" : "text-graphite",
            side && "lg:col-span-2",
          )}
        >
          {title}
        </h2>
      )}
      <RaceFrame frame={frames[at]} className={title ? "mt-3" : undefined} />
      <div>
        <dl className="mt-2 flex gap-8">
          {(
            [
              ["Steps", totals.steps],
              ["Searches", totals.searches],
              ["Pages revisited", totals.revisits],
            ] as const
          ).map(([label, value]) => (
            <div key={label}>
              <dt className="text-sm text-muted">{label}</dt>
              <dd className="font-display text-5xl font-extrabold tracking-[-0.015em]">{value}</dd>
            </div>
          ))}
        </dl>
        {findability && (
          <div className="mt-4">
            <p
              className={cn(
                "text-xl font-bold",
                findability.outcome === "purchased" ? "text-cobalt" : "text-signal",
              )}
            >
              {outcomeLabels[findability.outcome]}
            </p>
            {findability.outcome === "abandoned" ? (
              <p className="mt-1 max-w-[50ch] text-muted">{run?.evaluation?.failureReasons[0]}</p>
            ) : (
              <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
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
    </div>
  );
}
