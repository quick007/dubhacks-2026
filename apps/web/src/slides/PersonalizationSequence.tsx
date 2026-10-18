import { useEffect, useRef, useState } from "react";
import { personalizationSteps, PERSONALIZATION_STEP_HEIGHT } from "../lib/personalization";
import { useReducedMotion } from "../lib/useReducedMotion";
import { PersonalizationChanges } from "./PersonalizationChanges";

export function PersonalizationSequence() {
  const scroll = useRef<HTMLDivElement>(null);
  const [step, setStep] = useState(0);
  const reduced = useReducedMotion();

  useEffect(() => {
    const element = scroll.current;
    if (!element) return;
    const update = () =>
      setStep(
        Math.min(
          personalizationSteps.length - 1,
          Math.max(
            0,
            Math.round(element.scrollTop / (element.clientHeight * PERSONALIZATION_STEP_HEIGHT)),
          ),
        ),
      );
    const resize = new ResizeObserver(update);
    resize.observe(element);
    element.addEventListener("scroll", update, { passive: true });
    return () => {
      resize.disconnect();
      element.removeEventListener("scroll", update);
    };
  }, []);

  const go = (next: number) => {
    const element = scroll.current;
    if (!element) return;
    element.scrollTo({
      top: next * element.clientHeight * PERSONALIZATION_STEP_HEIGHT,
      behavior: reduced ? "instant" : "smooth",
    });
  };

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey) return;
      if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
      const target = event.target instanceof Element ? event.target : null;
      if (target?.closest("input, textarea, select, [contenteditable], [role=dialog], [role=slider]"))
        return;
      event.preventDefault();
      go(
        Math.max(
          0,
          Math.min(personalizationSteps.length - 1, step + (event.key === "ArrowDown" ? 1 : -1)),
        ),
      );
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  return (
    <div
      ref={scroll}
      data-no-swipe
      data-scroll-story
      data-step={step}
      role="region"
      aria-label="Bob's page changes"
      tabIndex={0}
      className="h-[calc(100dvh-7rem)] w-full overflow-y-auto overscroll-contain [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
    >
      <div
        style={{
          height: `${(1 + (personalizationSteps.length - 1) * PERSONALIZATION_STEP_HEIGHT) * 100}%`,
        }}
      >
        <section className="sticky top-0 flex h-[calc(100dvh-7rem)] flex-col items-center justify-center gap-7 px-6 py-5 sm:px-8">
          <div className="w-full max-w-[1440px] lg:w-[min(100%,calc((100dvh-16rem)*1.6+392px))]">
            <h2 className="mb-7 text-3xl font-semibold tracking-tight">Bob's personalized page</h2>
            <div className="grid items-center gap-7 lg:grid-cols-[minmax(0,1fr)_360px] lg:gap-8">
              <div className="relative isolate aspect-[8/5] overflow-hidden rounded-sm border border-line bg-paper">
                {personalizationSteps.map((item, index) => (
                  <img
                    key={item.image}
                    src={`/fieldwork/bob/${item.image}.jpg`}
                    alt={item.alt}
                    aria-hidden={index !== step}
                    width={1280}
                    height={800}
                    className="absolute inset-0 size-full object-cover transition-opacity duration-500 motion-reduce:transition-none"
                    style={{ opacity: index <= step ? 1 : 0 }}
                  />
                ))}
              </div>
              <PersonalizationChanges step={step} onChange={go} />
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
