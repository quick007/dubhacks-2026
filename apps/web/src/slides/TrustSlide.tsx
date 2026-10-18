import { EyeOff, GitBranch, ShieldCheck } from "lucide-react";
import { SlideFrame } from "./SlideFrame";

const controls = [
  { Icon: ShieldCheck, title: "Trained for relevance", detail: "Post-trained selection, grounded in catalog facts" },
  { Icon: GitBranch, title: "Tested beyond training", detail: "Held-out groups; hidden goals excluded from model inputs" },
  { Icon: EyeOff, title: "Data with boundaries", detail: "Minimized events, masked replay and user opt-out" },
];

export function TrustSlide() {
  return (
    <SlideFrame title="Trust and responsibility">
      <dl className="grid gap-x-12 gap-y-7 sm:grid-cols-3">
        {controls.map(({ Icon, title, detail }) => (
          <div key={title} className="flex gap-4 border-t border-line pt-4">
            <Icon aria-hidden className="mt-1 size-7 shrink-0 text-cobalt" />
            <div>
              <dt className="font-display text-2xl font-bold">{title}</dt>
              <dd className="mt-2 text-lg text-muted">{detail}</dd>
            </div>
          </div>
        ))}
      </dl>
    </SlideFrame>
  );
}
