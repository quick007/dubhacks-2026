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
    <SlideFrame
      title={`${pair.persona.name} tries today's shop`}
      actions={
        run && (
          <Button variant="outline" onClick={() => onEvidence(run)}>
            See every step
          </Button>
        )
      }
    >
      <Race persona={pair.persona} lanes={[{ tone: "graphite", run }]} />
    </SlideFrame>
  );
}
