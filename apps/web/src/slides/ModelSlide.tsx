import { ArrowRight } from "lucide-react";
import { Fragment } from "react";
import { SlideFrame } from "./SlideFrame";

const stages = [
  { title: "Frozen Qwen", detail: "Sequential encoder · reusable input features" },
  { title: "Learned selection", detail: "Query/catalog projections · 16-family route · variant choice" },
  { title: "Selected fact rows", detail: "Preencoded public catalog features" },
  { title: "Parallel copy", detail: "14 word slots · short titles, badges, body and sections" },
];

export function ModelSlide() {
  return (
    <SlideFrame title="One backbone. Selection, then parallel copy.">
      <div className="border-t border-line pt-5">
        <p className="text-lg text-muted">Autoregressive generation</p>
        <ol className="mt-5 flex flex-wrap items-center gap-4 font-display text-2xl font-bold text-graphite">
          {["Token 1", "Token 2", "Token 3", "…"].map((token, index) => <Fragment key={token}>
            {index > 0 && <ArrowRight aria-hidden className="size-5" />}
            <li className="border-b-2 border-line pb-2">{token}</li>
          </Fragment>)}
        </ol>
      </div>
      <div className="mt-12 border-t border-line pt-5">
        <p className="text-lg text-muted">Our model</p>
        <ol className="mt-5 grid items-start gap-5 md:grid-cols-[1fr_auto_1fr_auto_1fr_auto_1fr]">
          {stages.map(({ title, detail }, index) => <Fragment key={title}>
            {index > 0 && <ArrowRight aria-hidden className="mt-2 hidden size-5 text-cobalt md:block" />}
            <li>
              <h3 className="font-display text-2xl font-bold text-cobalt">{title}</h3>
              <p className="mt-3 text-lg text-muted">{detail}</p>
            </li>
          </Fragment>)}
        </ol>
      </div>
    </SlideFrame>
  );
}
