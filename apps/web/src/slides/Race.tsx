import { Slider } from "@base-ui/react/slider";
import type { Persona, Run } from "@storefront/contracts";
import { Pause, Play } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { Button } from "../components/ui/button";
import { frames } from "../lib/story";
import { useReducedMotion } from "../lib/useReducedMotion";
import { cn } from "../lib/utils";
import { REPLAY_STEP_MS } from "../lib/replay";
import { RaceLane } from "./RaceLane";

export interface Lane {
  title?: string;
  tone: "graphite" | "cobalt";
  run?: Run;
}

/** Replays one or more runs step by step; starts on mount, since a slide mounts when shown. */
export function Race({
  persona,
  lanes,
  actions,
}: {
  persona: Persona;
  lanes: Lane[];
  actions?: ReactNode;
}) {
  const laneFrames = lanes.map((lane) => frames(lane.run));
  const last = Math.max(0, ...laneFrames.map((list) => list.length)) - 1;
  const reduced = useReducedMotion();
  const [index, setIndex] = useState(() => (reduced ? Math.max(last, 0) : 0));
  const [playing, setPlaying] = useState(!reduced);
  const running = playing && !reduced && index <= last;

  // Turning on reduced motion mid-replay stops it on the final frames.
  const [wasReduced, setWasReduced] = useState(reduced);
  if (reduced !== wasReduced) {
    setWasReduced(reduced);
    if (reduced) {
      setPlaying(false);
      setIndex(Math.max(last, 0));
    }
  }

  useEffect(() => {
    if (!running) return;
    const timer = window.setTimeout(() => {
      if (index < last) setIndex(index + 1);
      else setPlaying(false);
    }, REPLAY_STEP_MS);
    return () => window.clearTimeout(timer);
  }, [running, index, last]);

  if (last < 0)
    return <p className="text-lg text-muted">No steps recorded for this shopper yet.</p>;

  return (
    <div>
      <div className={cn("grid gap-10", lanes.length > 1 && "md:grid-cols-2")}>
        {lanes.map((lane, i) => (
          <RaceLane
            key={lane.title ?? i}
            title={lane.title}
            tone={lane.tone}
            frames={laneFrames[i]}
            index={index}
            playing={playing && !reduced}
            run={lane.run}
            persona={persona}
          />
        ))}
      </div>
      <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2">
        <Button
          variant="outline"
          size="sm"
          aria-label={running ? "Pause" : "Play"}
          onClick={() => {
            if (running) return setPlaying(false);
            if (index >= last) setIndex(0);
            setPlaying(true);
          }}
        >
          {running ? <Pause /> : <Play />}
        </Button>
        <Slider.Root
          value={index}
          min={0}
          max={last}
          onValueChange={(value) => {
            setPlaying(false);
            setIndex(Array.isArray(value) ? value[0] : value);
          }}
          className="min-w-20 flex-1"
          data-no-swipe
        >
          <Slider.Control className="flex h-8 items-center">
            <Slider.Track className="relative h-1 w-full rounded bg-line">
              <Slider.Indicator className="rounded bg-ink" />
              {Array.from({ length: last + 1 }, (_, step) => (
                <span
                  key={step}
                  aria-hidden
                  className={cn(
                    "pointer-events-none absolute top-1/2 size-2 -translate-1/2 rounded-full",
                    step <= index ? "bg-ink" : "bg-line",
                  )}
                  style={{ left: `${last > 0 ? (step / last) * 100 : 0}%` }}
                />
              ))}
              <Slider.Thumb
                aria-label="Step"
                aria-valuetext={`Step ${index + 1} of ${last + 1}`}
                className="size-4 rounded-full bg-ink"
              />
            </Slider.Track>
          </Slider.Control>
        </Slider.Root>
        <span aria-hidden className="text-sm whitespace-nowrap text-muted tabular-nums">
          Step {index + 1} of {last + 1}
        </span>
        {actions}
      </div>
    </div>
  );
}
