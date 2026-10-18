import {
  CatalogSchema,
  RunSchema,
  type Catalog,
  type Run,
  type RunRequest,
} from "@storefront/contracts";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { api } from "./lib/api";
import { visibleSlides, type SlideId } from "./lib/deck";
import { mergeRuns } from "./lib/runs";
import { buildPairs, featured, sessions, verdict } from "./lib/story";
import { Deck, type SlideDef } from "./slides/Deck";
import { EvidenceDrawer } from "./slides/EvidenceDrawer";
import { FixSlide } from "./slides/FixSlide";
import { MeetSlide } from "./slides/MeetSlide";
import { ResultsSlide } from "./slides/ResultsSlide";
import { RetestSlide } from "./slides/RetestSlide";
import { SessionsSlide } from "./slides/SessionsSlide";
import { StruggleSlide } from "./slides/StruggleSlide";
import { TitleSlide } from "./slides/TitleSlide";

const message = (cause: unknown) =>
  cause instanceof Error ? cause.message : "Could not reach the local API.";

const labels: Record<SlideId, string> = {
  title: "Title",
  meet: "Meet the shopper",
  struggle: "Today's shop",
  fix: "A proposed fix",
  retest: "With the fix",
  sessions: "All sessions",
  results: "Results",
};
const featuredSlides = new Set<SlideId>(["meet", "struggle", "fix", "retest"]);

export function App() {
  const [catalog, setCatalog] = useState<Catalog>();
  const [runs, setRuns] = useState<Run[]>([]);
  // Polling errors clear on the next good poll; a failed submission stays until the next attempt.
  const [pollError, setPollError] = useState("");
  const [submitError, setSubmitError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [evidenceId, setEvidenceId] = useState<string>();
  const busy =
    submitting || runs.some((run) => run.status === "queued" || run.status === "running");

  // Poll fast only while shoppers are running, so the session grid fills in live.
  useEffect(() => {
    let cancelled = false;
    async function refresh() {
      try {
        const [catalogData, runData] = await Promise.all([
          catalog ? undefined : api("/catalog"),
          api("/runs"),
        ]);
        if (cancelled) return;
        if (catalogData) setCatalog(CatalogSchema.parse(catalogData));
        const server = RunSchema.array().parse(runData);
        setRuns((current) => mergeRuns(server, current));
        setPollError("");
      } catch (cause) {
        if (!cancelled) setPollError(message(cause));
      }
    }
    void refresh();
    const timer = window.setInterval(() => void refresh(), busy ? 500 : 5000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [catalog, busy]);

  const story = useMemo(() => {
    if (!catalog) return undefined;
    const pairs = buildPairs(catalog, runs);
    return { pairs, list: sessions(catalog, runs), result: verdict(pairs) };
  }, [catalog, runs]);

  async function runShoppers() {
    if (!catalog) return;
    const requests: RunRequest[] = catalog.experiment.variantIds.flatMap((variantId) =>
      catalog.personas
        .filter((persona) => persona.task)
        .map((persona) => ({
          experimentId: catalog.experiment.id,
          variantId,
          personaId: persona.id,
          adapter: "mock",
        })),
    );
    setSubmitting(true);
    setSubmitError("");
    try {
      await Promise.all(
        requests.map(async (request) => {
          const run = RunSchema.parse(await api("/runs", request));
          setRuns((current) => [run, ...current]);
        }),
      );
    } catch (cause) {
      setSubmitError(message(cause));
    } finally {
      setSubmitting(false);
    }
  }

  const error = submitError || pollError;
  // Resolve evidence from the latest poll so an open drawer follows a session that is still running.
  const evidence = runs.find((candidate) => candidate.id === evidenceId);
  const openEvidence = (target: Run) => setEvidenceId(target.id);

  if (!catalog || !story)
    return <p className="p-8 text-lg text-muted">{error || "Loading the study"}</p>;
  const feature = featured(story.pairs);
  if (!feature) return <p className="p-8 text-lg">This study has no shoppers with a goal yet.</p>;

  const hasRuns = story.pairs.some((pair) => pair.baseline || pair.candidate);
  const run = () => void runShoppers();
  const views: Record<SlideId, () => ReactNode> = {
    title: () => (
      <TitleSlide
        names={[
          feature.persona.name,
          ...story.pairs
            .filter((pair) => pair.persona.id !== feature.persona.id)
            .map((pair) => pair.persona.name),
        ]}
      />
    ),
    meet: () => <MeetSlide persona={feature.persona} />,
    struggle: () => <StruggleSlide pair={feature} onEvidence={openEvidence} />,
    fix: () => <FixSlide pair={feature} />,
    retest: () => <RetestSlide pair={feature} />,
    sessions: () => (
      <SessionsSlide
        list={story.list}
        name={feature.persona.name}
        personaId={feature.persona.id}
        busy={busy}
        onEvidence={openEvidence}
        onRun={run}
      />
    ),
    results: () => (
      <ResultsSlide result={story.result} pairs={story.pairs} onEvidence={openEvidence} />
    ),
  };
  const slides: SlideDef[] = visibleSlides({
    runs: hasRuns,
    proposal: Boolean(feature.baseline?.proposedChanges[0]),
    result: story.result.total > 0 || story.result.technical.length > 0,
  }).map((id) => ({
    id,
    label: labels[id],
    render: views[id],
    featured: featuredSlides.has(id),
  }));

  return (
    <>
      <Deck
        slides={slides}
        status={
          (busy || error) && (
            <span role="status" className={error ? "text-signal" : "text-muted"}>
              {error || "Shoppers running"}
            </span>
          )
        }
      />
      <EvidenceDrawer
        run={evidence}
        variantName={
          catalog.variants.find((variant) => variant.id === evidence?.variantId)?.name ?? ""
        }
        onClose={() => setEvidenceId(undefined)}
      />
    </>
  );
}
