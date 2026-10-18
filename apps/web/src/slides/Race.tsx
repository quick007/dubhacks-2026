import { Slider } from "@base-ui/react/slider";
import type { Persona, Run } from "@storefront/contracts";
import { Pause, Play } from "lucide-react";
import { useEffect, useState } from "react";
import { Button } from "../components/ui/button";
import { frames } from "../lib/story";
import { useReducedMotion } from "../lib/useReducedMotion";
import { cn } from "../lib/utils";
import { RaceLane } from "./RaceLane";

const STEP_MS = 1100;

export interface Lane {
  title?: string;
  tone: "graphite" | "cobalt";
  run?: Run;
}

/** Replays one or more runs step by step; starts on mount, since a slide mounts when shown. */
export function Race({ persona, lanes }: { persona: Persona; lanes: Lane[] }) {
  const laneFrames = lanes.map((lane) => frames(lane.run));
  const last = Math.max(0, ...laneFrames.map((list) => list.length)) - 1;
  const reduced = useReducedMotion();
  const [index, setIndex] = useState(() => (reduced ? Math.max(last, 0) : 0));
  const [playing, setPlaying] = useState(!reduced);
  const running = playing && !reduced && index < last;

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
    const timer = window.setTimeout(() => setIndex(index + 1), STEP_MS);
    return () => window.clearTimeout(timer);
  }, [running, index]);

  if (last < 0)
    return <p className="text-lg text-muted">No steps recorded for this shopper yet.</p>;

  return (
    <div>
      <div className={cn("grid gap-10", lanes.length > 1 && "md:grid-cols-2")}>
        {lanes.map((lane, i) => (
          <RaceLane
            key={lane.title ?? i}
            side={lanes.length === 1}
            title={lane.title}
            tone={lane.tone}
            frames={laneFrames[i]}
            index={index}
            run={lane.run}
            persona={persona}
          />
        ))}
      </div>
      <div className="mt-6 flex items-center gap-4">
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
          className="flex-1"
          data-no-swipe
        >
          <Slider.Control className="flex h-8 items-center">
            <Slider.Track className="h-1 w-full rounded bg-line">
              <Slider.Indicator className="rounded bg-ink" />
              <Slider.Thumb aria-label="Step" className="size-4 rounded-full bg-ink" />
            </Slider.Track>
          </Slider.Control>
        </Slider.Root>
        <span className="text-sm whitespace-nowrap text-muted">
          Step {index + 1} of {last + 1}
        </span>
      </div>
    </div>
  );
}
