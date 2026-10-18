import type { Pair } from "../lib/story";
import { Race } from "./Race";
import { SlideFrame } from "./SlideFrame";

export function RetestSlide({ pair }: { pair: Pair }) {
  return (
    <SlideFrame title={`Same ${pair.persona.name}, on the model's page`}>
      <Race
        persona={pair.persona}
        lanes={[
          { title: "Default page", tone: "graphite", run: pair.baseline },
          { title: "Personalized page", tone: "cobalt", run: pair.candidate },
        ]}
      />
    </SlideFrame>
  );
}
