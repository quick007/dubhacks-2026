import {
  linearScale,
  linearAxisTicks,
  median,
  medianInterval,
  seconds,
  tickAlign,
  tickLabel,
  type Series,
} from "../lib/latency";
import { cn } from "../lib/utils";
import { LatencyRow } from "./LatencyRow";

/** Per-request ranking time for each method, with medians and 95% intervals. */
export function LatencyChart({ series }: { series: Series[] }) {
  const ticks = linearAxisTicks(series.flatMap((item) => [
    ...item.ms, ...(item.floorMs === undefined ? [] : [item.floorMs]),
  ]));
  const x = linearScale(0, ticks.at(-1)!);
  const major = new Set([0, ticks[Math.floor(ticks.length / 2)], ticks.at(-1)!]);
  return (
    <figure>
      <div className="grid grid-cols-1 items-center gap-x-8 sm:grid-cols-[minmax(8rem,13rem)_1fr]">
        {series.map((item) => (
          <div key={item.id} className="contents">
            <div>
              <p className={`text-lg font-semibold ${item.ours ? "text-cobalt" : ""}`}>
                {item.label}
              </p>
              <p className="text-sm leading-tight text-muted">
                {item.detail}
              </p>
            </div>
            <div className="relative">
              {ticks.map((tick) => (
                <span
                  key={tick}
                  className="absolute -inset-y-px w-px bg-line/70"
                  style={{ left: `${x(tick)}%` }}
                />
              ))}
              <LatencyRow series={item} x={x} />
            </div>
          </div>
        ))}
        <span className="hidden sm:block" />
        <div className="relative h-7 border-t border-ink">
          {ticks.map((tick, index) => (
            <span
              key={tick}
              className={cn(
                "absolute top-0 pt-1.5 text-sm whitespace-nowrap text-muted tabular-nums",
                tickAlign(index, ticks.length),
              )}
              style={{ left: `${x(tick)}%` }}
            >
              <span
                className={cn(
                  "absolute top-0 h-1.5 w-px bg-ink",
                  index === 0 ? "left-0" : index === ticks.length - 1 ? "right-0" : "left-1/2",
                )}
              />
              <span className={major.has(tick) ? undefined : "hidden sm:inline"}>
                {tickLabel(tick)}
              </span>
            </span>
          ))}
        </div>
        <span className="hidden sm:block" />
        <p className="mt-3 flex flex-wrap items-center gap-x-6 gap-y-2 text-sm text-muted">
          <span className="flex items-center gap-2 whitespace-nowrap">
            <span className="size-2.5 rounded-full bg-graphite/45" /> Request
          </span>
          <span className="flex items-center gap-2 whitespace-nowrap">
            <span className="h-4 w-1 rounded-full bg-graphite" /> Median
          </span>
          <span className="flex items-center gap-2 whitespace-nowrap">
            <span className="relative h-0.5 w-6 bg-graphite">
              <span className="absolute -top-1 left-0 h-2.5 w-0.5 bg-graphite" />
              <span className="absolute -top-1 right-0 h-2.5 w-0.5 bg-graphite" />
            </span>
            95% median interval
          </span>
          <span className="flex items-center gap-2 whitespace-nowrap">
            <span className="h-4 border-l-2 border-dashed border-graphite/60" /> Trivial CLI reply
          </span>
        </p>
      </div>
      <table className="sr-only">
        <caption>Time to rank 16 products, per method</caption>
        <thead>
          <tr>
            <th>Method</th>
            <th>Median</th>
            <th>95% interval</th>
          </tr>
        </thead>
        <tbody>
          {series.map((item) => {
            const [low, high] = medianInterval(item.ms);
            return (
              <tr key={item.id}>
                <td>{item.label}</td>
                <td>{seconds(median(item.ms))}</td>
                <td>
                  {seconds(low)} to {seconds(high)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </figure>
  );
}
