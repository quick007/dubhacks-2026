import { SlideFrame } from "./SlideFrame";

const notes = [
  "Shopper results are synthetic; the context pile illustrates possible store data, not what our telemetry collects.",
  "The illustrative scenario assumes Bing’s delay rate transfers. It is not measured net revenue; one $96 timeout drives the observed difference.",
  "Frozen v5 and the history-aware v4 candidate are separate experiments. v4 quality is unvalidated.",
  "Telemetry is off in normal local development. The collection-to-export pipeline was tested with synthetic traffic.",
  "Training and promotion are explicit steps, not autonomous production retraining.",
  "Demo checkout places no real order and makes no charge.",
];

export function DisclaimerSlide() {
  return (
    <SlideFrame title="Disclaimers">
      <ul className="grid gap-5 text-lg leading-relaxed text-muted">
        {notes.map((note) => <li key={note} className="border-t border-line pt-4">{note}</li>)}
      </ul>
    </SlideFrame>
  );
}
