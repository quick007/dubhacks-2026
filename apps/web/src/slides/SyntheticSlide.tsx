import type { Experiment } from "@storefront/contracts";
import { RotateCcw } from "lucide-react";
import { Button } from "../components/ui/button";
import { SlideFrame } from "./SlideFrame";

const simulated = [
  "The shoppers, their goals and their choices",
  "The page screenshots, drawn as simple mock frames",
  "Every outcome and order on these slides",
];
const real = [
  "The catalog facts each purchase is checked against",
  "The evaluation rules, applied the same way to both versions",
  "The fix, limited to facts the shop already has",
];

export function SyntheticSlide({
  experiment,
  busy,
  onRun,
}: {
  experiment: Experiment;
  busy: boolean;
  onRun: () => void;
}) {
  return (
    <SlideFrame
      title="What's simulated, what's real"
      lead="These results show how the loop works. They are not evidence of how real people shop or how revenue changes."
      actions={
        <Button variant="outline" disabled={busy} onClick={onRun}>
          <RotateCcw />
          Run the shoppers again
        </Button>
      }
    >
      <div className="grid gap-10 sm:grid-cols-2">
        {(
          [
            ["Simulated", simulated],
            ["Real", real],
          ] as const
        ).map(([title, items]) => (
          <div key={title} className="border-t-2 border-ink pt-4">
            <h3 className="text-xl font-bold">{title}</h3>
            <ul className="mt-3 space-y-2 text-lg">
              {items.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>
      <p className="mt-10 text-sm text-muted">
        {experiment.name}, seed {experiment.seed}, contract version {experiment.contractVersion}
      </p>
    </SlideFrame>
  );
}
