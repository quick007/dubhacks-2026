import { ArrowRight } from "lucide-react";
import measured from "../fieldwork/current-model.json";
import currentSpeed from "../fieldwork/current-candidate-speed.json";
import { SlideFrame } from "./SlideFrame";

export function ModelSlide() {
  const { rank, loadedBrowser, server } = measured;
  const figures = [
    { label: "Right product first", before: rank.before.topOneCorrect, after: rank.after.topOneCorrect },
    { label: "Right product in six", before: rank.before.targetInTopSix, after: rank.after.targetInTopSix },
  ];
  return (
    <SlideFrame title="Relevance and latency, measured" lead="Frozen v5 · synthetic known-catalog test">
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
          <dt className="text-muted">Frozen v5 · added loaded-view time, p50</dt>
          <dd className="mt-2 font-display text-5xl font-extrabold tracking-[-0.015em] text-cobalt">
            +{Math.round(loadedBrowser.incremental.p50Ms)}<span className="ml-2 text-xl">ms</span>
          </dd>
          <p className="mt-2 text-muted">{Math.round(loadedBrowser.local.p50Ms)} ms personalized / {Math.round(loadedBrowser.generic.p50Ms)} ms generic</p>
        </div>
      </dl>
      <div className="mt-12 grid gap-8 border-t border-line pt-5 sm:grid-cols-2">
        <p className="text-lg text-muted">v5: {server.freshFields} fresh fields · six cards<br />First three visible: {Math.round(loadedBrowser.local.p50Ms)} / {Math.round(loadedBrowser.local.p95Ms)} ms p50 / p95</p>
        <p className="text-lg text-muted">Separate v4 candidate: backend {Math.round(currentSpeed.fullBackend.p50Ms)} / {Math.round(currentSpeed.fullBackend.p95Ms)} ms<br />First three visible: {Math.round(currentSpeed.firstThreeVisible.p50Ms)} / {Math.round(currentSpeed.firstThreeVisible.p95Ms)} ms p50 / p95</p>
      </div>
      <p className="mt-6 text-sm text-muted">Loaded views; decoded images, no physical-paint claim. v4: 10 searches; quality unvalidated.</p>
    </SlideFrame>
  );
}
