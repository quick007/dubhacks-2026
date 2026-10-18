import { useState } from "react";
import { cn } from "../lib/utils";
import type { Frame } from "../lib/story";

export function RaceFrame({ frame, className }: { frame?: Frame; className?: string }) {
  const click = frame?.event.click;
  // A listed artifact whose file is gone falls back to the placeholder; a new src retries.
  const [failedSrc, setFailedSrc] = useState<string>();
  const src = frame?.src !== failedSrc ? frame?.src : undefined;
  return (
    <figure className={className}>
      <div className="relative aspect-[8/5] overflow-hidden rounded-md border border-line bg-paper">
        {src ? (
          <img
            src={src}
            alt={frame?.event.message}
            onError={() => setFailedSrc(src)}
            className="size-full object-cover"
          />
        ) : (
          <p className="grid size-full place-items-center text-muted">
            No screenshot for this step
          </p>
        )}
        {click && (
          <span
            aria-hidden
            className="absolute size-7 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-ink bg-ink/15"
            style={{ left: `${click.x * 100}%`, top: `${click.y * 100}%` }}
          />
        )}
      </div>
      <figcaption className={cn("mt-2 min-h-12 text-muted", !frame && "invisible")}>
        {frame?.event.message}
      </figcaption>
    </figure>
  );
}
