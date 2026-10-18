import { useEffect, useRef, useState, type ReactNode } from "react";
import { slideFromHash, stepFor, type SlideId } from "../lib/deck";
import { cn } from "../lib/utils";

export interface SlideDef {
  id: SlideId;
  label: string;
  render: () => ReactNode;
  /** Slides about the featured shopper show the session indicator. */
  featured?: boolean;
}

const ignoredTargets = "input:not([type=range]), textarea, select, [role=dialog]";
const SWIPE_PX = 60;

/** One slide at a time. The URL hash names the slide, so links and reloads land on it. */
export function Deck({ slides, status }: { slides: SlideDef[]; status?: ReactNode }) {
  const [currentId, setCurrentId] = useState(() => window.location.hash.slice(1));
  const index = slideFromHash(
    currentId,
    slides.map((item) => item.id),
  );
  const slide = slides[index];
  const swipeStart = useRef<number | undefined>(undefined);

  const go = (target: number) => {
    const id = slides[target].id;
    window.history.replaceState(null, "", `#${id}`);
    setCurrentId(id);
  };

  // In-page links such as href="#sessions" navigate the deck too.
  useEffect(() => {
    const onHash = () => setCurrentId(window.location.hash.slice(1));
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.altKey || event.ctrlKey || event.metaKey) return;
      const target = event.target instanceof Element ? event.target : null;
      if (target?.closest(ignoredTargets)) return;
      // Keep native vertical scrolling inside the staged personalization demo.
      if (
        target?.closest("[data-scroll-story]") &&
        [" ", "PageDown", "PageUp", "Home", "End"].includes(event.key)
      )
        return;
      const fromSlider = target?.closest("[role=slider], input[type=range]");
      if (fromSlider) {
        if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
        // Reserve deck arrows even at the first/last slide, without moving the slider.
        event.preventDefault();
      }
      if (event.key === " " && target?.closest("button, a")) return;
      const next = stepFor(event.key, index, slides.length);
      if (next === undefined) return;
      event.preventDefault();
      go(next);
    }
    // Capture before slider controls consume arrow keys.
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  });

  return (
    <div
      className="flex min-h-dvh flex-col"
      onPointerDown={(event) => {
        const fromControl =
          event.target instanceof Element && event.target.closest("[data-no-swipe]");
        if (event.pointerType === "touch" && !fromControl) swipeStart.current = event.clientX;
      }}
      onPointerUp={(event) => {
        if (swipeStart.current === undefined) return;
        const dx = event.clientX - swipeStart.current;
        swipeStart.current = undefined;
        const next = Math.abs(dx) < SWIPE_PX ? undefined : index + (dx < 0 ? 1 : -1);
        if (next !== undefined && next >= 0 && next < slides.length) go(next);
      }}
    >
      <nav aria-label="Slides" className="flex gap-1 px-4 pt-3 sm:px-8">
        {slides.map((item, i) => (
          <button
            key={item.id}
            type="button"
            aria-label={`Slide ${i + 1}: ${item.label}`}
            aria-current={i === index ? "step" : undefined}
            onClick={() => go(i)}
            className="group h-4 flex-1 py-1.5"
          >
            <span
              className={cn(
                "block h-1 rounded-full transition-colors group-hover:bg-graphite",
                i <= index ? "bg-ink" : "bg-line",
              )}
            />
          </button>
        ))}
      </nav>
      <main
        key={slide.id}
        aria-label={slide.label}
        className={cn("flex flex-1 animate-slide-in", slide.id === "stakes" ? "pb-12" : "pb-20")}
      >
        {slide.render()}
      </main>
      <footer className="fixed inset-x-0 bottom-0 z-10 bg-fog/90 backdrop-blur print:hidden">
        <div className="flex items-center gap-4 px-4 py-3 text-sm sm:px-8">
          {status}
          <span className="ml-auto text-muted">
            {index + 1} of {slides.length}
          </span>
        </div>
      </footer>
    </div>
  );
}
