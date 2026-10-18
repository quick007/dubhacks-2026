export const allSlides = [
  "title",
  "meet",
  "struggle",
  "fix",
  "retest",
  "sessions",
  "results",
] as const;
export type SlideId = (typeof allSlides)[number];

export interface Available {
  runs: boolean;
  proposal: boolean;
  result: boolean;
}

// The session grid has no entry: it is where a round starts, so it shows even before any run.
const needs: Partial<Record<SlideId, keyof Available>> = {
  struggle: "runs",
  fix: "proposal",
  retest: "runs",
  results: "result",
};

/** Slides whose data exists, in story order. */
export function visibleSlides(available: Available): SlideId[] {
  return allSlides.filter((id) => {
    const need = needs[id];
    return !need || available[need];
  });
}

export function slideFromHash(hash: string, ids: SlideId[]) {
  return Math.max(0, ids.indexOf(hash.replace(/^#/, "") as SlideId));
}

/** Target slide index for a navigation key, or undefined when the key does nothing. */
export function stepFor(key: string, index: number, count: number) {
  const target =
    key === "ArrowRight" || key === " " || key === "PageDown"
      ? index + 1
      : key === "ArrowLeft" || key === "PageUp"
        ? index - 1
        : key === "Home"
          ? 0
          : key === "End"
            ? count - 1
            : undefined;
  return target === undefined || target < 0 || target >= count || target === index
    ? undefined
    : target;
}
