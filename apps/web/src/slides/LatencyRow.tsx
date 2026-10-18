import { useState } from "react";
import { median, medianInterval, seconds, type Series } from "../lib/latency";
import { modelRun } from "../lib/model";
import { cn } from "../lib/utils";

/** Vertical offsets in px, so the 16 dots in a row overlap less. */
const jitter = [-8, 0, 8, -4, 4];

/**
 * One method's strip: a dot per request, the median as a tall tick and its 95% interval as a
 * capped bar. `x` maps milliseconds to a percentage of the plot width.
 */
export function LatencyRow({ series, x }: { series: Series; x: (ms: number) => number }) {
  const [active, setActive] = useState<number>();
  const mid = median(series.ms);
  const [low, high] = medianInterval(series.ms);
  const tone = series.ours ? "bg-cobalt" : "bg-graphite";
  const hovered = active === undefined ? undefined : modelRun.requests[active];
  return (
    <div className="relative h-24" onPointerLeave={() => setActive(undefined)}>
      <div className="absolute inset-x-0 top-1/2 h-px bg-line" />
      {series.floorMs !== undefined && (
        <div
          className="absolute top-1/2 h-10 -translate-x-1/2 -translate-y-1/2 border-l-2 border-dashed border-graphite/60"
          style={{ left: `${x(series.floorMs)}%` }}
        />
      )}
      {series.ms.map((ms, index) => (
        <button
          key={modelRun.requests[index].caseId}
          type="button"
          aria-label={`${modelRun.requests[index].request} ${seconds(ms)}`}
          onPointerEnter={() => setActive(index)}
          onFocus={() => setActive(index)}
          onBlur={() => setActive(undefined)}
          className="absolute top-1/2 grid size-5 -translate-x-1/2 -translate-y-1/2 place-items-center"
          style={{ left: `${x(ms)}%`, marginTop: jitter[index % jitter.length] }}
        >
          <span
            className={cn(
              "size-2.5 rounded-full opacity-45 ring-2 ring-fog transition-opacity",
              tone,
              active === index && "opacity-100",
            )}
          />
        </button>
      ))}
      <div
        className={cn("pointer-events-none absolute top-1/2 h-0.5 -translate-y-1/2", tone)}
        style={{ left: `${x(low)}%`, width: `${x(high) - x(low)}%` }}
      >
        <span className={cn("absolute -top-1.5 left-0 h-3.5 w-0.5", tone)} />
        <span className={cn("absolute -top-1.5 right-0 h-3.5 w-0.5", tone)} />
      </div>
      <div
        className={cn(
          "pointer-events-none absolute top-1/2 h-9 w-1 -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-fog",
          tone,
        )}
        style={{ left: `${x(mid)}%` }}
      />
      <span
        className={cn(
          "pointer-events-none absolute top-0 -translate-x-1/2 text-sm font-semibold whitespace-nowrap tabular-nums",
          series.ours ? "text-cobalt" : "text-ink",
        )}
        style={{ left: `${x(mid)}%` }}
      >
        {seconds(mid)}
      </span>
      {hovered && active !== undefined && (
        <div
          role="tooltip"
          className="pointer-events-none absolute bottom-full z-10 mb-1 w-64 -translate-x-1/2 rounded-md bg-ink px-3 py-2 text-sm text-fog shadow-lg"
          style={{ left: `${x(series.ms[active])}%` }}
        >
          <p className="leading-snug">{hovered.request}</p>
          <p className="mt-1 tabular-nums text-fog/75">{seconds(series.ms[active])}</p>
        </div>
      )}
    </div>
  );
}
