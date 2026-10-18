import type { Pair } from "../lib/story";
import { Race } from "./Race";
import { SlideFrame } from "./SlideFrame";

export function RetestSlide({ pair }: { pair: Pair }) {
  return (
    <SlideFrame title={`Same ${pair.persona.name}, with the fix`}>
      <Race
        persona={pair.persona}
        lanes={[
          { title: "Today's shop", tone: "graphite", run: pair.baseline },
          { title: "With the fix", tone: "cobalt", run: pair.candidate },
        ]}
      />
    </SlideFrame>
  );
}
