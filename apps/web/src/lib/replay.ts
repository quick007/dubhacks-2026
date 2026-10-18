import type { Run } from "@storefront/contracts";
import { frames, type Frame } from "./story";

export const REPLAY_STEP_MS = 2400;

/** Skip the opening hero when the next capture already shows those same products. */
export function replayFrames(run?: Run): Frame[] {
  const list = frames(run);
  return list.filter((frame, index) => {
    const next = list[index + 1];
    return !(
      index === 0 &&
      frame.event.page?.kind === "home" &&
      !frame.event.click &&
      next?.event.page?.kind === "category" &&
      next.event.page.label === "Products"
    );
  });
}

const point = (offset: number, x: number, y: number): Keyframe => ({
  offset,
  transform: `translate(${x * 100}%, ${y * 100}%)`,
  easing: "cubic-bezier(0.4, 0, 0.2, 1)",
});

// Read the search prompt, then inspect the first photo before the page scrolls.
const homeMotion = [
  point(0, 0.32, 0.36),
  point(0.2, 0.32, 0.36),
  point(0.48, 0.18, 0.82),
  point(1, 0.18, 0.82),
];

function productMotion(furtherDown: boolean): Keyframe[] {
  // The first row is 150px below the viewport top; later rows are 60px below it.
  const photoY = furtherDown ? 0.3 : 0.44;
  return [
    point(0, furtherDown ? 0.82 : 0.18, furtherDown ? 0.44 : photoY),
    point(0.12, 0.18, photoY),
    point(0.24, 0.18, photoY),
    point(0.46, 0.5, photoY),
    point(0.58, 0.5, photoY),
    point(0.8, 0.82, photoY),
    point(1, 0.82, photoY),
  ];
}

const firstRowMotion = productMotion(false);
const nextRowMotion = productMotion(true);

/** Presentation waypoints matched to the captured Fieldwork shop, not recorded mouse events. */
export function browsingMotion(frame?: Frame): Keyframe[] | undefined {
  const page = frame?.event.page;
  if (!page || frame?.event.click) return undefined;
  if (page.kind === "home") return homeMotion;
  if (page.kind !== "category") return undefined;
  if (page.label === "Products") return firstRowMotion;
  if (page.label === "Products, further down") return nextRowMotion;
  return undefined;
}
