import { ArrowRight } from "lucide-react";
import { modelRun, ms } from "../lib/model";
import { ModelRequestTile } from "./ModelRequestTile";
import { SlideFrame } from "./SlideFrame";

const Change = ({ before, after, of }: { before: number; after: number; of: number }) => (
  <span className="flex items-baseline gap-3">
    <span className="text-graphite">{before}</span>
    <ArrowRight aria-label="to" className="size-6 self-center" />
    <span className="text-cobalt">{after}</span>
    <span className="text-xl font-bold tracking-normal text-ink">of {of}</span>
  </span>
);

const gigabytes = (bytes: number) => `${(bytes / 1e9).toFixed(1)} GB`;

export function ModelSlide() {
  const { arms, cases, latencyMs, source, keywordSearch, copy, requests, peakMemoryBytes } =
    modelRun;
  const figures = [
    {
      label: "Right product first",
      value: <Change before={arms.prior.topOne} after={arms.learned.topOne} of={cases} />,
    },
    {
      label: "Right product in the top 6",
      value: <Change before={arms.prior.topSix} after={arms.learned.topSix} of={cases} />,
    },
    {
      label: `Median time to personalize a page (range ${Math.round(latencyMs.min)} to ${Math.round(latencyMs.max)} ms)`,
      value: <span>{ms(latencyMs.median)}</span>,
    },
  ];
  return (
    <SlideFrame
      className="max-w-7xl py-6"
      title="The model, measured"
      lead={`${cases} requests held out of training, against the shop's ${cases} products. Grey is the default page, blue is the personalized page.`}
    >
      <dl className="grid gap-6 sm:grid-cols-3">
        {figures.map(({ label, value }) => (
          <div key={label}>
            <dt className="text-muted">{label}</dt>
            <dd className="mt-1 font-display text-5xl font-extrabold tracking-[-0.015em]">
              {value}
            </dd>
          </div>
        ))}
      </dl>
      <ul className="mt-8 grid gap-x-6 gap-y-3 sm:grid-cols-2 lg:grid-cols-4">
        {requests.map((request) => (
          <li key={request.caseId}>
            <ModelRequestTile request={request} />
          </li>
        ))}
      </ul>
      <div className="mt-6 grid gap-2 text-muted lg:grid-cols-2 lg:gap-8">
        <p>
          {source.backbone} on a laptop, {gigabytes(peakMemoryBytes)} peak. We trained a{" "}
          {Math.round(source.trainedParameters / 1000)}k-parameter head on {source.trainingRequests}{" "}
          other requests. Training covered every product but none of these wordings. A random head
          gets {arms.random.topOne} of {cases}, shuffled requests get {arms.permutedQuery.topOne} of{" "}
          {cases}.
        </p>
        <p>
          Keyword search finds {keywordSearch.topOne} of {cases} when the shopper types the exact
          request. The model personalizes the page before any search. Generated product copy failed
          checks on {copy.total - copy.valid} of {copy.total} test pages, so pages use catalog
          titles.
        </p>
      </div>
    </SlideFrame>
  );
}
