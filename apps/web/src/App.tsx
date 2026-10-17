import { LOCAL_ORIGINS } from "@storefront/contracts";
import { useCallback, useEffect, useState } from "react";
import {
  CatalogSchema,
  ComparisonRowSchema,
  RunSchema,
  type Catalog,
  type ComparisonRow,
  type Run,
  type RunRequest,
  type Variant,
} from "@storefront/contracts";
import { Button } from "./components/ui/button";
import { Badge } from "./components/ui/badge";
import { Card, CardContent, CardHeader } from "./components/ui/card";
import { percent, splitLabel } from "./lib/format";

async function api(path: string, body?: RunRequest): Promise<unknown> {
  const response = await fetch(
    `/api${path}`,
    body
      ? {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }
      : undefined,
  );
  const data: unknown = await response.json();
  if (!response.ok)
    throw new Error(
      typeof data === "object" && data !== null && "error" in data
        ? String(data.error)
        : `Request failed (${response.status})`,
    );
  return data;
}

function storefrontUrl(variant: Variant) {
  const config = variant.config;
  return `${LOCAL_ORIGINS.storefront}/?${new URLSearchParams({ headline: config.headline, showShipping: String(config.showShipping), recommendationOrder: config.recommendationOrder })}`;
}

export function App() {
  const [catalog, setCatalog] = useState<Catalog>();
  const [runs, setRuns] = useState<Run[]>([]);
  const [comparison, setComparison] = useState<ComparisonRow[]>([]);
  const [error, setError] = useState("");
  const [actionError, setActionError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [loading, setLoading] = useState(true);
  const [notice, setNotice] = useState("");

  const refresh = useCallback(async () => {
    try {
      const [catalogData, runData, comparisonData] = await Promise.all([
        api("/catalog"),
        api("/runs"),
        api("/comparison"),
      ]);
      setCatalog(CatalogSchema.parse(catalogData));
      setRuns(RunSchema.array().parse(runData));
      setComparison(ComparisonRowSchema.array().parse(comparisonData));
      setError("");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not reach the local API.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const interval = window.setInterval(() => {
      void refresh();
    }, 1000);
    return () => window.clearInterval(interval);
  }, [refresh]);

  const active = runs.filter((run) => run.status === "queued" || run.status === "running").length;
  const completed = runs.filter((run) => run.status === "completed").length;
  const proposals = runs
    .filter((run) => run.personaSplit === "development")
    .flatMap((run) => run.proposedChanges);

  async function start(adapter: "mock" | "playwright") {
    if (!catalog) return;
    setSubmitting(true);
    setActionError("");
    setNotice("");
    let count = 0;
    try {
      if (catalog.personas.length === 0)
        throw new Error("Seed at least one persona before starting a run.");
      const requests: RunRequest[] =
        adapter === "mock"
          ? catalog.variants.flatMap((variant) =>
              catalog.personas.map((persona) => ({
                experimentId: catalog.experiment.id,
                variantId: variant.id,
                personaId: persona.id,
                adapter,
              })),
            )
          : [
              {
                experimentId: catalog.experiment.id,
                variantId: catalog.experiment.baselineVariantId,
                personaId: catalog.personas[0].id,
                adapter,
              },
            ];
      if (requests.length > 20)
        throw new Error("This starter supports at most 20 runs in a comparison batch.");
      for (const request of requests) {
        RunSchema.parse(await api("/runs", request));
        count++;
      }
      setNotice(
        `${count} ${adapter === "mock" ? "synthetic mock runs" : "local browser run"} queued. Progress updates automatically.`,
      );
    } catch (cause) {
      setNotice(count ? `${count} runs were queued before submission stopped.` : "");
      setActionError(cause instanceof Error ? cause.message : "Could not queue runs.");
    } finally {
      setSubmitting(false);
      await refresh();
    }
  }

  return (
    <div className="mx-auto max-w-7xl px-5 py-7 sm:px-8">
      <header className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-widest text-teal-800">
            <span className="inline-flex h-7 w-7 items-center justify-center rounded-lg bg-teal-800 text-white">
              S
            </span>{" "}
            Storefront Lab <Badge>LOCAL / v1</Badge>
          </div>
          <h1 className="text-2xl font-semibold tracking-tight">
            A test bench for better storefronts.
          </h1>
          <p className="mt-1 text-sm text-slate-500">
            Compare structured changes across shopper personas. Keep the evidence.
          </p>
        </div>
        <a
          className="text-sm font-medium text-teal-800 underline underline-offset-4"
          href={LOCAL_ORIGINS.storefront}
          target="_blank"
          rel="noreferrer"
        >
          Open demo shop ↗
        </a>
      </header>

      <div className="mb-6 rounded-xl border border-amber-200 bg-amber-50 px-5 py-4 text-sm text-amber-950">
        <strong>Synthetic experiment · no human conversion claims.</strong>
        <p className="mt-1">
          Mock shoppers and scores are deterministic demo data. Browser smoke checks only the local
          fake checkout. Results do not establish human behavior or proven conversion lift.
        </p>
      </div>
      {(error || actionError) && (
        <div
          role="alert"
          className="mb-4 rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800"
        >
          {error || actionError}{" "}
          {error && (
            <Button variant="outline" size="sm" onClick={() => void refresh()}>
              Retry connection
            </Button>
          )}
        </div>
      )}
      {notice && (
        <p role="status" className="mb-4 text-sm text-teal-800">
          {notice}
        </p>
      )}
      {loading && (
        <p role="status" className="py-12 text-center text-slate-500">
          Loading local experiment…
        </p>
      )}

      {catalog && (
        <>
          <Card className="mb-6">
            <CardContent className="flex flex-wrap items-center justify-between gap-5">
              <div>
                <Badge className="mb-2 bg-teal-50 text-teal-800">EXPERIMENT</Badge>
                <h2 className="text-lg font-semibold">{catalog.experiment.name}</h2>
                <p className="mt-1 text-xs text-slate-500">
                  Seed {catalog.experiment.seed} · contract v{catalog.experiment.contractVersion} ·{" "}
                  {catalog.personas.length} personas · {catalog.variants.length} configurations
                </p>
              </div>
              <div className="flex flex-wrap gap-2">
                <Button onClick={() => void start("mock")} disabled={submitting || active > 0}>
                  Run mock comparison
                </Button>
                <Button
                  variant="outline"
                  onClick={() => void start("playwright")}
                  disabled={submitting || active > 0}
                >
                  Run browser smoke
                </Button>
              </div>
            </CardContent>
          </Card>

          <div className="mb-6 grid gap-4 sm:grid-cols-3">
            {[
              [String(active), "Queued / running"],
              [String(completed), "Completed runs"],
              [String(proposals.length), "Development suggestions"],
            ].map(([value, label]) => (
              <Card key={label}>
                <CardContent className="flex items-center gap-4">
                  <span className="text-3xl font-semibold text-teal-900">{value}</span>
                  <span className="text-sm text-slate-500">{label}</span>
                </CardContent>
              </Card>
            ))}
          </div>

          <div className="mb-6 grid gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <h2 className="font-semibold">Variant configurations</h2>
                <p className="mt-1 text-xs text-slate-500">
                  Allowlisted settings. Suggestions never execute generated code.
                </p>
              </CardHeader>
              <CardContent className="space-y-5">
                {catalog.variants.map((variant) => (
                  <div key={variant.id} className="rounded-lg border border-slate-200 p-4">
                    <div className="mb-3 flex items-center justify-between gap-2">
                      <h3 className="text-sm font-semibold">{variant.name}</h3>
                      <Badge className={variant.isBaseline ? "" : "bg-teal-50 text-teal-800"}>
                        {variant.isBaseline ? "Baseline" : "Variant"}
                      </Badge>
                    </div>
                    <dl className="grid grid-cols-[112px_1fr] gap-2 text-xs">
                      <dt className="text-slate-500">Headline</dt>
                      <dd>{variant.config.headline}</dd>
                      <dt className="text-slate-500">Shipping detail</dt>
                      <dd>{variant.config.showShipping ? "Visible" : "Hidden"}</dd>
                      <dt className="text-slate-500">Recommendation</dt>
                      <dd>{variant.config.recommendationOrder}</dd>
                    </dl>
                    <a
                      href={storefrontUrl(variant)}
                      target="_blank"
                      rel="noreferrer"
                      className="mt-3 inline-block text-xs font-medium text-teal-800 underline underline-offset-4"
                    >
                      Preview configuration ↗
                    </a>
                  </div>
                ))}
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <h2 className="font-semibold">Persona split</h2>
                <p className="mt-1 text-xs text-slate-500">
                  Tune on development personas; compare separately on held-out personas.
                </p>
              </CardHeader>
              <CardContent className="space-y-4">
                {(["development", "held-out"] as const).map((split) => (
                  <section key={split}>
                    <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-500">
                      {splitLabel(split)}
                    </h3>
                    <div className="space-y-2">
                      {catalog.personas
                        .filter((persona) => persona.split === split)
                        .map((persona) => (
                          <div className="rounded-lg bg-slate-50 p-3" key={persona.id}>
                            <p className="text-sm font-medium">{persona.name}</p>
                            <p className="mt-1 text-xs text-slate-500">{persona.goal}</p>
                            <p className="mt-1 text-xs text-slate-400">
                              {persona.traits.join(" · ")}
                              {persona.history.length > 0
                                ? ` · ${persona.history.length} supplied history items`
                                : ""}
                            </p>
                          </div>
                        ))}
                    </div>
                  </section>
                ))}
              </CardContent>
            </Card>
          </div>

          <Card className="mb-6" data-testid="comparison">
            <CardHeader>
              <h2 className="font-semibold">Synthetic comparison</h2>
              <p className="mt-1 text-xs text-slate-500">
                Completed mock runs only, grouped by persona split. Scores are synthetic task
                scores, not conversion rates.
              </p>
            </CardHeader>
            <CardContent>
              {comparison.length === 0 ? (
                <p className="py-5 text-center text-sm text-slate-500">
                  Run a mock comparison to see baseline and variant results.
                </p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-sm">
                    <thead>
                      <tr className="border-b border-slate-200 text-xs text-slate-500">
                        <th className="pb-3 pr-4 font-medium">Configuration</th>
                        <th className="pb-3 pr-4 font-medium">Persona split</th>
                        <th className="pb-3 pr-4 font-medium">Tasks complete</th>
                        <th className="pb-3 font-medium">Mean synthetic score</th>
                      </tr>
                    </thead>
                    <tbody>
                      {comparison.map((row) => (
                        <tr
                          key={`${row.variantId}-${row.personaSplit}`}
                          className="border-b border-slate-100 last:border-0"
                        >
                          <td className="py-3 pr-4 font-medium">
                            {catalog.variants.find((variant) => variant.id === row.variantId)
                              ?.name ?? row.variantId}
                          </td>
                          <td className="py-3 pr-4">{splitLabel(row.personaSplit)}</td>
                          <td className="py-3 pr-4">
                            {row.completedTasks} / {row.runs}
                          </td>
                          <td className="py-3 font-mono text-teal-800">{percent(row.meanScore)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </CardContent>
          </Card>

          <div className="grid items-start gap-6 lg:grid-cols-[1.5fr_1fr]">
            <Card>
              <CardHeader>
                <h2 className="font-semibold">Run evidence</h2>
                <p className="mt-1 text-xs text-slate-500">
                  Immutable run IDs retain configuration, seed and persona split.
                </p>
              </CardHeader>
              <CardContent className="space-y-3">
                {runs.length === 0 && (
                  <p className="py-5 text-sm text-slate-500">
                    No runs yet. Start with the deterministic mock adapters.
                  </p>
                )}
                {runs.map((run) => (
                  <RunCard
                    key={run.id}
                    run={run}
                    variantName={
                      catalog.variants.find((variant) => variant.id === run.variantId)?.name ??
                      run.variantId
                    }
                  />
                ))}
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <h2 className="font-semibold">Suggested changes</h2>
                <p className="mt-1 text-xs text-slate-500">
                  Development evidence only. Review before creating a future variant.
                </p>
              </CardHeader>
              <CardContent className="space-y-3">
                {proposals.length === 0 && (
                  <p className="py-5 text-sm text-slate-500">
                    Suggestions from the optimizer adapter appear after development runs.
                  </p>
                )}
                {proposals.map((proposal) => (
                  <article key={proposal.id} className="rounded-lg border border-slate-200 p-4">
                    <Badge className="mb-2">{proposal.kind} · suggested</Badge>
                    <p className="text-sm">{proposal.rationale}</p>
                    <pre className="mt-3 overflow-x-auto rounded-md bg-slate-50 p-3 text-xs">
                      {JSON.stringify(proposal.configPatch, null, 2)}
                    </pre>
                    <p className="mt-2 break-all text-xs text-slate-400">
                      Source {proposal.sourceRunId} · {proposal.evidenceStepIds.length} evidence
                      steps
                    </p>
                  </article>
                ))}
              </CardContent>
            </Card>
          </div>
        </>
      )}
      <footer className="mt-7 text-center text-xs text-slate-400">
        Local-first scaffold · deterministic mocks · fake orders only
      </footer>
    </div>
  );
}

function RunCard({ run, variantName }: { run: Run; variantName: string }) {
  return (
    <article data-testid="run-card" className="rounded-lg border border-slate-200 p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-medium">
          {run.persona.name} <span className="font-normal text-slate-400">/ {variantName}</span>
        </h3>
        <Badge
          className={
            run.status === "completed"
              ? "bg-teal-50 text-teal-800"
              : run.status === "failed"
                ? "bg-red-50 text-red-700"
                : ""
          }
        >
          {run.status}
        </Badge>
      </div>
      <p className="mt-2 text-xs text-slate-500">
        {splitLabel(run.personaSplit)} · {run.adapter} · seed {run.seed}
        {run.evaluation ? ` · synthetic score ${percent(run.evaluation.score)}` : ""}
      </p>
      <p className="mt-1 break-all font-mono text-[10px] text-slate-400">{run.id}</p>
      {run.error && <p className="mt-2 text-xs text-red-700">{run.error}</p>}
      {run.evaluation?.failureReasons.map((reason) => (
        <p className="mt-2 text-xs text-red-700" key={reason}>
          Failure: {reason}
        </p>
      ))}
      <details className="mt-3 text-xs">
        <summary className="font-medium text-teal-800">
          Steps, artifacts and frozen configuration ({run.events.length} events)
        </summary>
        <ol className="my-3 space-y-2 border-l-2 border-slate-100 pl-3">
          {run.events.map((event) => (
            <li
              key={event.id}
              className={event.kind === "failure" ? "text-red-700" : "text-slate-600"}
            >
              <span className="font-mono text-slate-400">{event.sequence}.</span> [{event.kind}]{" "}
              {event.message}
            </li>
          ))}
        </ol>
        {run.artifacts.length === 0 ? (
          <p className="text-slate-400">No saved artifacts yet.</p>
        ) : (
          <div className="flex flex-wrap gap-3">
            {run.artifacts.map((artifact) => (
              <a
                key={artifact.id}
                className="font-medium text-teal-800 underline underline-offset-4"
                href={`/api/artifacts/${encodeURIComponent(artifact.id)}`}
                download
              >
                {artifact.kind} ↓
              </a>
            ))}
          </div>
        )}
        <pre className="mt-3 overflow-x-auto rounded-md bg-slate-50 p-3">
          {JSON.stringify(
            {
              baselineVariantId: run.baselineVariantId,
              variantId: run.variantId,
              config: run.config,
            },
            null,
            2,
          )}
        </pre>
      </details>
    </article>
  );
}
