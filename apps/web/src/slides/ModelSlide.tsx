import { ArrowRight } from "lucide-react";
import measured from "../fieldwork/current-model.json";
import { SlideFrame } from "./SlideFrame";

export function ModelSlide() {
  const { rank, loadedBrowser, server } = measured;
  const figures = [
    { label: "Right product first", before: rank.before.topOneCorrect, after: rank.after.topOneCorrect },
    { label: "Right product in six", before: rank.before.targetInTopSix, after: rank.after.targetInTopSix },
  ];
  return (
    <SlideFrame title="The model, measured" lead={`${rank.cases} independent requests against ${rank.catalogProducts} familiar products. A balanced fit improves the earlier trained head.`}>
      <dl className="grid gap-8 sm:grid-cols-3">
        {figures.map(({ label, before, after }) => (
          <div key={label}>
            <dt className="text-muted">{label}</dt>
            <dd className="mt-2 flex items-baseline gap-3 font-display text-5xl font-extrabold tracking-[-0.015em]">
              <span className="text-graphite">{before}</span><ArrowRight className="size-6 self-center" aria-label="to" />
              <span className="text-cobalt">{after}</span><span className="text-xl">of {rank.cases}</span>
            </dd>
          </div>
        ))}
        <div>
          <dt className="text-muted">Fresh personalized cards, p50 / p95</dt>
          <dd className="mt-2 font-display text-5xl font-extrabold tracking-[-0.015em] text-cobalt">
            {Math.round(loadedBrowser.local.p50Ms)} / {Math.round(loadedBrowser.local.p95Ms)}<span className="ml-2 text-xl">ms</span>
          </dd>
        </div>
      </dl>
      <div className="mt-12 grid gap-8 border-t border-line pt-5 sm:grid-cols-2">
        <p className="text-xl leading-relaxed">One frozen Qwen backbone, learned product selection and fresh parallel text. Six selected cards contain {server.freshFields} factual generated fields.</p>
        <p className="text-xl leading-relaxed">The same generic shop is ready in {Math.round(loadedBrowser.generic.p50Ms)} ms p50. Personalization adds {Math.round(loadedBrowser.incremental.p50Ms)} ms p50 on the matched loaded view.</p>
      </div>
      <p className="mt-8 text-sm text-muted">20 paired loaded-shop observations; image cache eligible. First three cards fully visible with decoded images. Physical paint and cold navigation excluded.</p>
    </SlideFrame>
  );
}
