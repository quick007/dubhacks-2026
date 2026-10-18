import { shopperFacts, shopperNumbers } from "../lib/inputs";
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

/** Original teammate context pile, without its unrelated speed/conversion sidebar. */
export function ShopperContextSlide() {
  return (
    <SlideFrame className="max-w-[110rem] py-6" title="A shopper's context">
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
    </SlideFrame>
  );
}
