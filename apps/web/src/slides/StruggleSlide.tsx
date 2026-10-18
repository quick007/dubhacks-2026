import type { Run } from "@storefront/contracts";
import { Button } from "../components/ui/button";
import type { Pair } from "../lib/story";
import { Race } from "./Race";
import { SlideFrame } from "./SlideFrame";

export function StruggleSlide({
  pair,
  onEvidence,
}: {
  pair: Pair;
  onEvidence: (run: Run) => void;
}) {
  const run = pair.baseline;
  return (
    <SlideFrame title={`${pair.persona.name} on the default page`} className="max-w-4xl">
      <Race
        persona={pair.persona}
        lanes={[{ tone: "graphite", run }]}
        actions={
          run && (
            <Button variant="ghost" onClick={() => onEvidence(run)}>
              See every step
            </Button>
          )
        }
      />
    </SlideFrame>
  );
}
