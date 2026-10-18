import type { Run } from "@storefront/contracts";
import { Download } from "lucide-react";
import { artifactUrl } from "../lib/api";
import { configFields, percent, splitLabel } from "../lib/format";
import { outcomeOf } from "../lib/runs";
import { cn } from "../lib/utils";
import { RunHistory } from "./RunHistory";
import { RunSteps } from "./RunSteps";

const outcomeText = {
  passed: ["Task passed", "text-cobalt"],
  failed: ["Task failed", "text-signal"],
  error: ["Run error", "text-muted"],
  pending: ["In progress", "text-muted"],
} as const;

export function RunDetail({
  run,
  variantName,
  history,
  onSelectRun,
}: {
  run?: Run;
  variantName: string;
  history: Run[];
  onSelectRun: (id: string) => void;
}) {
  if (!run)
    return (
      <aside className="rounded-xl border border-dashed border-line px-5 py-10 text-center text-muted">
        Select a result to see what the shopper did.
      </aside>
    );
  const [label, color] = outcomeText[outcomeOf(run)];
  const screenshot = [...run.artifacts]
    .reverse()
    .find((artifact) => artifact.kind === "screenshot");
  const reasons = [...(run.error ? [run.error] : []), ...(run.evaluation?.failureReasons ?? [])];
  return (
    <aside
      data-testid="run-detail"
      className="space-y-5 rounded-xl border border-line bg-paper p-5"
    >
      <header>
        <h2 className="text-lg leading-tight font-semibold">{run.persona.name}</h2>
        <p className="mt-0.5 text-muted">
          {variantName}, {splitLabel(run.personaSplit).toLowerCase()} split
        </p>
        <p className={cn("mt-3 text-base font-semibold", color)}>
          {label}
          {run.evaluation && (
            <span className="ml-2 font-normal text-muted">
              score {percent(run.evaluation.score)}
            </span>
          )}
        </p>
        {reasons.map((reason) => (
          <p key={reason} className="mt-1 text-signal">
            {reason}
          </p>
        ))}
      </header>

      <div>
        <p>{run.persona.goal}</p>
        <ul className="mt-2 flex flex-wrap gap-1.5">
          {[...run.persona.traits, ...(run.persona.history.length > 0 ? ["has history"] : [])].map(
            (trait) => (
              <li key={trait} className="rounded bg-fog px-1.5 py-0.5 text-xs text-muted">
                {trait}
              </li>
            ),
          )}
        </ul>
      </div>

      <section>
        <h3 className="mb-2.5 font-semibold">Steps</h3>
        <RunSteps events={run.events} />
      </section>

      {screenshot && (
        <a href={artifactUrl(screenshot.id)} target="_blank" rel="noreferrer" className="block">
          <img
            src={artifactUrl(screenshot.id)}
            alt={`Final screen for ${run.persona.name}`}
            loading="lazy"
            className="max-h-48 w-full rounded-md border border-line object-cover object-top"
          />
        </a>
      )}

      <section>
        <h3 className="mb-2 font-semibold">Page configuration</h3>
        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-[13px]">
          {configFields.map((field) => (
            <div key={field.key} className="contents">
              <dt className="text-muted">{field.label}</dt>
              <dd>{field.format(run.config)}</dd>
            </div>
          ))}
        </dl>
      </section>

      {history.length > 1 && (
        <section>
          <h3 className="mb-1.5 font-semibold">Runs ({history.length})</h3>
          <RunHistory runs={history} selectedId={run.id} onSelect={onSelectRun} />
        </section>
      )}

      <footer className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-line pt-3 text-xs text-muted">
        <span>Seed {run.seed}</span>
        <span>{run.adapter === "mock" ? "Mock shopper" : "Browser"}</span>
        <span className="truncate" title={run.id}>
          Run {run.id.slice(0, 8)}
        </span>
        {run.artifacts
          .filter((artifact) => artifact.kind !== "screenshot")
          .map((artifact) => (
            <a
              key={artifact.id}
              href={artifactUrl(artifact.id)}
              download
              className="inline-flex items-center gap-1 text-ink hover:underline"
            >
              <Download className="size-3.5" />
              {artifact.kind}
            </a>
          ))}
      </footer>
    </aside>
  );
}
