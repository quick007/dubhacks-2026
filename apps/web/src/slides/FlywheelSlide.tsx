import { ArrowRight, Database, Files, MousePointer2, RefreshCw } from "lucide-react";
import { Fragment } from "react";
import { SlideFrame } from "./SlideFrame";

const steps = [
  { Icon: MousePointer2, title: "Shopping signals", detail: "Automatic, configured collection" },
  { Icon: Database, title: "Tiger Data", detail: "Durable, traceable evidence" },
  { Icon: Files, title: "Training export", detail: "Versioned choices and splits" },
  { Icon: RefreshCw, title: "Model update", detail: "Explicit training and evaluation" },
];

export function FlywheelSlide() {
  return (
    <SlideFrame title="A learning loop from the store" lead="Shopping → evidence → training inputs → next model">
      <ol className="grid items-start gap-6 md:grid-cols-[1fr_auto_1fr_auto_1fr_auto_1fr]">
        {steps.map(({ Icon, title, detail }, index) => (
          <Fragment key={title}>
            {index > 0 && <ArrowRight aria-hidden className="mt-8 hidden size-6 text-graphite md:block" />}
            <li className="border-t-2 border-line pt-6">
              <Icon aria-hidden className="size-10 text-cobalt" />
              <h3 className="mt-5 font-display text-2xl font-bold">{title}</h3>
              <p className="mt-2 text-lg text-muted">{detail}</p>
            </li>
          </Fragment>
        ))}
      </ol>
    </SlideFrame>
  );
}
