import { Bot, MousePointer2 } from "lucide-react";
import { useEffect, useState } from "react";
import { browsingMotion } from "../lib/replay";
import { cn } from "../lib/utils";
import type { Frame } from "../lib/story";
import { useReducedMotion } from "../lib/useReducedMotion";
import { ReplayBrowsingCursor } from "./ReplayBrowsingCursor";

/** Where the cursor rests on a step without a click: mid-page, as if scrolling. */
const REST = { x: 0.6, y: 0.55 };
/** The new screen lands first, then the cursor sets off; the glide itself takes 750 ms (styles.css). */
const TRAVEL_DELAY_MS = 300;
const CLICK_DELAY_MS = TRAVEL_DELAY_MS + 750;
const scrolls = new Set(["home", "category", "search"]);

/**
 * One recorded step, replayed like a session recording: the screen changes, then the cursor
 * glides to where the shopper clicked and the click pulses. `still` shows the screenshot alone.
 */
export function RaceFrame({
  frame,
  name,
  still = false,
  playing = true,
  className,
}: {
  frame?: Frame;
  name?: string;
  still?: boolean;
  playing?: boolean;
  className?: string;
}) {
  const reduced = useReducedMotion();
  const animate = !still && !reduced;
  // A listed artifact whose file is gone falls back to the placeholder; a new src retries.
  const [failedSrc, setFailedSrc] = useState<string>();
  const src = frame?.src !== failedSrc ? frame?.src : undefined;

  // Keep the previous screen underneath so the next one slides or fades in over it.
  const [current, setCurrent] = useState(frame);
  const [prior, setPrior] = useState<Frame>();
  if (frame !== current) {
    setPrior(current);
    setCurrent(frame);
  }
  const scrolled =
    prior?.event.page?.kind === frame?.event.page?.kind &&
    scrolls.has(frame?.event.page?.kind ?? "");

  const target = frame?.event.click ?? REST;
  const [cursor, setCursor] = useState(target);
  useEffect(() => {
    if (!animate) return;
    const timer = window.setTimeout(() => setCursor(target), TRAVEL_DELAY_MS);
    return () => window.clearTimeout(timer);
  }, [animate, target.x, target.y]);
  const at = animate ? cursor : target;
  const click = frame?.event.click;
  const motion = browsingMotion(frame);

  return (
    <figure className={className}>
      <div className="relative aspect-[8/5] overflow-hidden rounded-md border border-line bg-paper">
        {animate && prior?.src && prior.src !== src && (
          <img src={prior.src} alt="" className="absolute inset-0 size-full object-cover" />
        )}
        {src ? (
          <img
            key={src}
            src={src}
            alt={frame?.event.message}
            onError={() => setFailedSrc(src)}
            className={cn(
              "absolute inset-0 size-full object-cover",
              animate && prior && (scrolled ? "animate-scroll-in" : "animate-frame-in"),
            )}
          />
        ) : (
          <p className="grid size-full place-items-center text-muted">
            No screenshot for this step
          </p>
        )}
        {!still && frame && (
          <>
            {click && animate && (
              <span
                key={frame.event.id}
                aria-hidden
                className="replay-click absolute size-14 rounded-full border-[3px] border-cobalt bg-cobalt/15"
                style={{
                  left: `${click.x * 100}%`,
                  top: `${click.y * 100}%`,
                  animationDelay: `${CLICK_DELAY_MS}ms`,
                }}
              />
            )}
            {animate && motion ? (
              <ReplayBrowsingCursor
                key={frame.event.id}
                motion={motion}
                name={name}
                playing={playing}
              />
            ) : (
              <span
                aria-hidden
                className="replay-cursor pointer-events-none absolute flex items-start gap-1"
                style={{ left: `${at.x * 100}%`, top: `${at.y * 100}%` }}
              >
                <MousePointer2
                  className="size-8 shrink-0 -translate-x-1 -translate-y-1 fill-cobalt text-paper drop-shadow-md"
                  strokeWidth={2}
                />
                {name && (
                  <span className="mt-4 flex items-center gap-1 rounded-full bg-cobalt px-2 py-0.5 text-sm font-bold text-paper shadow-md ring-2 ring-paper">
                    <Bot className="size-3.5" />
                    {name}
                  </span>
                )}
              </span>
            )}
          </>
        )}
      </div>
      <figcaption className={cn("mt-2 min-h-12 text-lg text-ink", !frame && "invisible")}>
        {frame?.event.message}
      </figcaption>
    </figure>
  );
}
