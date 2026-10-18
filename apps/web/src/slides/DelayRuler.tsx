import { latencySeries, linearAxisTicks, linearScale, median, seconds, tickAlign, tickLabel } from "../lib/latency";
import { cn } from "../lib/utils";

/** Delays the studies measured, on the same axis as the waits each method adds. */
const studied = [
  { from: 100, to: 250, label: "Bing slowdown", ref: 1 },
  { from: 1000, to: 3000, label: "Mobile load-time comparison", ref: 3 },
];

/** One linear axis: published delays above, our measured waits below. */
export function DelayRuler() {
  const series = latencySeries();
  const ticks = linearAxisTicks([...studied.flatMap(({ from, to }) => [from, to]),
    ...series.map((item) => median(item.ms))]);
  const x = linearScale(0, ticks.at(-1)!);
  return (
    <figure className="mt-8 hidden md:block">
      <div className="relative mx-2 h-40">
        {studied.map((span, index) => (
          <div
            key={span.ref}
            className="absolute"
            style={{ top: 12 + index * 24, left: `${x(span.from)}%`, width: `${x(span.to) - x(span.from)}%` }}
          >
            <p className="absolute left-full ml-2 -top-1 whitespace-nowrap text-sm text-muted">
              {span.label} <span className="tabular-nums">[{span.ref}]</span>
            </p>
            <div className="h-3 rounded-sm bg-graphite/25 ring-1 ring-graphite/40" />
          </div>
        ))}
        <div className="absolute inset-x-0 top-[4.25rem] border-t border-ink">
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
              {tickLabel(tick)}
            </span>
          ))}
        </div>
        {series.map((item, index) => (
          <div
            key={item.id}
            className="absolute top-[6.75rem] -translate-x-1/2"
            style={{ left: `${x(median(item.ms))}%`, marginTop: item.ours ? 0 : index * 22 - 22 }}
          >
            <span
              className={cn(
                "mx-auto block size-3 rounded-full ring-2 ring-fog",
                item.ours ? "bg-cobalt" : "bg-graphite",
              )}
            />
            <span
              className={cn(
                "absolute top-1/2 -translate-y-1/2 text-sm font-semibold whitespace-nowrap",
                item.ours ? "left-full ml-2 text-cobalt" : "right-full mr-2 text-ink",
              )}
            >
              {item.label}, {seconds(median(item.ms))}
            </span>
          </div>
        ))}
      </div>
      <figcaption className="text-sm text-muted">
        <span className="font-semibold text-ink">Figure 2.</span> Local warm backend / hosted ranking client.
        Linear scale; shaded spans show the cited studies' delays.
      </figcaption>
    </figure>
  );
}
