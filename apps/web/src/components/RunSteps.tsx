import type { StepEvent } from "@storefront/contracts";
import { Compass, Eye, MousePointerClick, X } from "lucide-react";
import { cn } from "../lib/utils";

const icons = { navigation: Compass, action: MousePointerClick, observation: Eye, failure: X };

export function RunSteps({ events }: { events: StepEvent[] }) {
  const steps = events.filter((event) => event.kind !== "status");
  if (steps.length === 0) return <p className="text-muted">No steps recorded yet.</p>;
  return (
    <ol className="relative space-y-3 before:absolute before:top-2 before:bottom-2 before:left-[11px] before:w-px before:bg-line">
      {steps.map((event) => {
        const Icon = icons[event.kind as keyof typeof icons];
        return (
          <li key={event.id} className="relative flex gap-3">
            <span
              className={cn(
                "z-10 flex size-6 shrink-0 items-center justify-center rounded-full border bg-paper",
                event.kind === "failure" ? "border-signal text-signal" : "border-line text-muted",
              )}
            >
              <Icon className="size-3.5" />
            </span>
            <span className={cn("pt-0.5", event.kind === "failure" && "text-signal")}>
              {event.message}
            </span>
          </li>
        );
      })}
    </ol>
  );
}
