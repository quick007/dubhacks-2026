import { shopperFacts, shopperNumbers } from "../lib/inputs";
import { latencySeries, median, seconds } from "../lib/latency";
import { findings } from "../lib/research";
import { cn } from "../lib/utils";
import { SlideFrame } from "./SlideFrame";

const sizes = ["text-xl", "text-2xl", "text-lg", "text-3xl", "text-xl"];

const numberStyles = [
  "bg-cobalt text-paper text-2xl sm:text-4xl",
  "border-2 border-ink bg-paper text-3xl sm:text-5xl",
  "bg-ink text-paper text-xl sm:text-3xl",
  "border-2 border-cobalt bg-paper text-cobalt text-2xl sm:text-4xl",
];

/** Fixed offset in [-range/2, range/2] per index, so the pile looks thrown but never reshuffles. */
const jitter = (index: number, step: number, range: number) =>
  ((index * step) % (range + 1)) - range / 2;

export function InputsSlide() {
  const series = latencySeries()
    .map((item) => ({ ...item, median: median(item.ms) }))
    .sort((a, b) => b.median - a.median);
  const slowest = series[0].median;
  const ours = series.find((item) => item.ours)!;
  const fastestLlm = Math.min(...series.filter((item) => !item.ours).map((item) => item.median));
  const conversion = findings.find((finding) => finding.value === "+8.4%")!;
  return (
    <SlideFrame
      className="max-w-[110rem] py-6"
      title="A shopper's context can guide the next recommendation"
      lead={`Illustrative shopping context. In the separate ranking-only comparison, the local head takes ${seconds(ours.median)} and the fastest hosted client takes ${seconds(fastestLlm)}.`}
    >
      <div className="grid gap-10 lg:grid-cols-[1fr_20rem]">
        <figure>
          <div className="relative h-[70dvh] [clip-path:inset(-4rem_-4rem_0_-4rem)] lg:h-[calc(100dvh-19rem)] lg:min-h-[28rem]">
            <ul className="flex flex-wrap content-start items-center justify-center px-2 pt-3">
              {shopperFacts.map((fact, index) => (
                <li
                  key={fact}
                  className={cn(
                    "-mx-2.5 -my-2 rounded-lg border border-line bg-paper px-5 py-3 font-medium whitespace-nowrap shadow-sm motion-safe:animate-rise",
                    sizes[index % sizes.length],
                  )}
                  style={{
                    rotate: `${jitter(index, 17, 14)}deg`,
                    translate: `${jitter(index, 53, 12)}px ${jitter(index, 29, 10)}px`,
                    zIndex: jitter(index, 41, 20) + 10,
                    animationDelay: `${index * 12}ms`,
                  }}
                >
                  {fact}
                </li>
              ))}
            </ul>
            <ul aria-label="Numbers about the same shopper">
              {shopperNumbers.map((number, index) => (
                <li
                  key={number}
                  className={cn(
                    "absolute rounded-lg px-4 py-2 font-display font-extrabold tracking-[-0.01em] whitespace-nowrap tabular-nums shadow-md motion-safe:animate-rise max-sm:nth-[n+9]:hidden",
                    numberStyles[index % numberStyles.length],
                  )}
                  style={{
                    [index % 2 ? "right" : "left"]: `${(index * 37) % 30}%`,
                    top: `${3 + ((index * 53) % 80)}%`,
                    rotate: `${jitter(index, 23, 24)}deg`,
                    zIndex: 40 + index,
                    animationDelay: `${600 + index * 40}ms`,
                  }}
                >
                  {number}
                </li>
              ))}
            </ul>
            <div className="pointer-events-none absolute inset-x-0 bottom-0 z-[100] h-32 bg-linear-to-t from-fog" />
          </div>
        </figure>
        <div className="grid content-start gap-8">
          <figure>
            <p className="font-medium">Historical time to rank the catalog</p>
            <ul className="mt-3 grid gap-3">
              {series.map((item) => (
                <li key={item.id}>
                  <div className="flex items-baseline justify-between gap-4 text-sm">
                    <span className={cn(item.ours && "font-semibold text-cobalt")}>
                      {item.label}
                    </span>
                    <span className="tabular-nums">{seconds(item.median)}</span>
                  </div>
                  <div
                    className={cn(
                      "mt-1 h-3 min-w-1 rounded-r-sm",
                      item.ours ? "bg-cobalt" : "bg-graphite/60",
                    )}
                    style={{ width: `${(100 * item.median) / slowest}%` }}
                  />
                </li>
              ))}
            </ul>
          </figure>
          <div className="border-t-2 border-ink pt-3">
            <p className="font-display text-6xl font-extrabold tracking-[-0.02em] text-cobalt tabular-nums">
              {conversion.value}
            </p>
            <p className="mt-2 text-lg leading-snug font-medium">{conversion.claim}</p>
          </div>
        </div>
      </div>
    </SlideFrame>
  );
}
