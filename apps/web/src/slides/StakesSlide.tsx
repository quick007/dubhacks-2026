import { findings } from "../lib/research";
import { DelayRuler } from "./DelayRuler";
import { SlideFrame } from "./SlideFrame";

export function StakesSlide() {
  return (
    <SlideFrame
      className="max-w-[1600px] py-4 lg:pb-0"
      title="Slow or lost shoppers buy somewhere else"
      lead="Study findings · not a universal revenue formula"
    >
      <dl className="grid gap-x-8 gap-y-8 sm:grid-cols-2 lg:grid-cols-4">
        {findings.map((finding, index) => (
          <div key={finding.reference} className="border-t-2 border-ink pt-3">
            <dt className="font-display text-6xl font-extrabold tracking-[-0.02em] text-cobalt tabular-nums">
              {finding.value}
              <sup className="ml-3 align-super text-base font-semibold text-muted">
                [{index + 1}]
              </sup>
            </dt>
            <dd className="mt-2 text-2xl leading-snug font-medium xl:text-3xl">
              {finding.claim}
            </dd>
          </div>
        ))}
      </dl>
      <DelayRuler />
      <ol className="mt-6 grid gap-x-8 gap-y-2 border-t border-line pt-4 text-sm text-muted lg:grid-cols-2">
        {findings.map((finding, index) => (
          <li key={finding.url} className="flex gap-2">
            <span className="tabular-nums">[{index + 1}]</span>
            <a href={finding.url} target="_blank" rel="noreferrer" className="hover:text-ink">
              Source {index + 1}
            </a>
          </li>
        ))}
      </ol>
    </SlideFrame>
  );
}
