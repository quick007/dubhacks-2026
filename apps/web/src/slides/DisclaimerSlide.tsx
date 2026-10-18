import { SlideFrame } from "./SlideFrame";

const notes = [
  "Current local: ten public profiles, 35 products, six cards/20 fresh fields and warm exact-query features. Full backend clock; startup, network, browser and images excluded.",
  "Shopper results use the earlier frozen 16-product service and synthetic shoppers; the 35-product demo has not been rerun through that cohort.",
  "The illustrative scenario assumes Bing’s delay rate transfers. It is not measured net revenue; one $96 timeout drives the observed difference.",
  "Hosted series: earlier 16-product ranking tasks timed through Codex CLI. Request sets, catalogs and timing boundaries differ; no matched speedup ratio. Expanded-catalog quality is unvalidated.",
  "Telemetry is off in normal local development. The pipeline was tested with synthetic traffic; the context pile illustrates possible store data, not collected telemetry.",
  "Training and promotion are explicit steps, not autonomous production retraining.",
  "Demo checkout places no real order and makes no charge.",
];

export function DisclaimerSlide({ reportId }: { reportId: string }) {
  return (
    <SlideFrame title="Disclaimers">
      <ul className="grid gap-5 text-lg leading-relaxed text-muted">
        {notes.map((note) => <li key={note} className="border-t border-line pt-4">{note}</li>)}
      </ul>
      <a className="mt-5 inline-block text-sm text-muted underline underline-offset-4" target="_blank" rel="noreferrer"
        href={`/api/experiment-results/presentation?reportId=${encodeURIComponent(reportId)}&source=tiger`}>Full measured cohort evidence</a>
    </SlideFrame>
  );
}
