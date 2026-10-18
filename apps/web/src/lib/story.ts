import type { Catalog, Persona, Run, StepEvent } from "@storefront/contracts";
import { artifactUrl } from "./api";
import { cellKey, groupRuns } from "./runs";

export type Arm = "baseline" | "candidate";
export type PageKind = NonNullable<StepEvent["page"]>["kind"];
export interface Pair {
  persona: Persona;
  baseline?: Run;
  candidate?: Run;
}
const arms: Arm[] = ["baseline", "candidate"];

function finished(run: Run) {
  return run.status === "completed" || run.status === "failed";
}
const shoppingOutcomes = new Set(["purchased", "abandoned", "constraint-failure"]);

/** Findability of a finished run whose outcome is about shopping; timeouts and technical failures are not. */
function findabilityOf(run?: Run) {
  const findability = run?.status === "completed" ? run.evaluation?.findability : undefined;
  return findability && shoppingOutcomes.has(findability.outcome) ? findability : undefined;
}
function broke(run?: Run) {
  const outcome = run?.evaluation?.findability?.outcome;
  return run?.status === "failed" || outcome === "technical-failure" || outcome === "timeout";
}
export function purchased(run?: Run) {
  return findabilityOf(run)?.outcome === "purchased";
}

/** Latest finished mock run per arm for every persona with a shopping task, so a rerun never blanks the story. */
export function buildPairs(catalog: Catalog, runs: Run[]): Pair[] {
  const cells = groupRuns(runs);
  const baselineId = catalog.experiment.baselineVariantId;
  const candidateId = catalog.experiment.variantIds.find((id) => id !== baselineId);
  const latest = (personaId: string, variantId?: string) =>
    variantId ? cells.get(cellKey("mock", personaId, variantId))?.find(finished) : undefined;
  return catalog.personas
    .filter((persona) => persona.task)
    .map((persona) => ({
      persona,
      baseline: latest(persona.id, baselineId),
      candidate: latest(persona.id, candidateId),
    }));
}

export function featured(pairs: Pair[], personaId?: string) {
  const development = pairs.filter((pair) => pair.persona.split === "development");
  return (
    pairs.find((pair) => pair.persona.id === personaId) ??
    development.find(
      (pair) =>
        findabilityOf(pair.baseline) && !purchased(pair.baseline) && purchased(pair.candidate),
    ) ??
    development[0]
  );
}

export interface Verdict {
  /** Held-out shoppers whose two runs both finished, in catalog order. */
  shoppers: Pair[];
  total: number;
  before: number;
  after: number;
  better: Pair[];
  worse: Pair[];
  technical: Pair[];
}

/** Held-out shoppers whose two runs both finished. */
function heldOutSettled(pairs: Pair[]) {
  return pairs.filter(
    (pair) =>
      pair.persona.split === "held-out" &&
      findabilityOf(pair.baseline) &&
      findabilityOf(pair.candidate),
  );
}

/** Technical failures are reported, never counted as giving up. */
export function verdict(pairs: Pair[]): Verdict {
  const heldOut = pairs.filter((pair) => pair.persona.split === "held-out");
  const both = heldOutSettled(pairs);
  return {
    shoppers: both,
    total: both.length,
    before: both.filter((pair) => purchased(pair.baseline)).length,
    after: both.filter((pair) => purchased(pair.candidate)).length,
    better: both.filter((pair) => !purchased(pair.baseline) && purchased(pair.candidate)),
    worse: both.filter((pair) => purchased(pair.baseline) && !purchased(pair.candidate)),
    technical: heldOut.filter((pair) => broke(pair.baseline) || broke(pair.candidate)),
  };
}

/** What the merchant gets from the same held-out shoppers: orders, order value and wrong-item purchases. */
export function merchant(
  pairs: Pair[],
): Record<Arm, { orders: number; value: number; wrongItems: number }> {
  const both = heldOutSettled(pairs);
  const tally = (arm: Arm) => {
    const totals = { orders: 0, value: 0, wrongItems: 0 };
    for (const pair of both) {
      const findability = findabilityOf(pair[arm]);
      if (findability?.outcome === "purchased") {
        totals.orders++;
        totals.value += findability.orderValue ?? 0;
      }
      if (findability?.outcome === "constraint-failure") totals.wrongItems++;
    }
    return totals;
  };
  return { baseline: tally("baseline"), candidate: tally("candidate") };
}

/** Give-up notes from other shoppers on today's shop, as recorded. */
export function quotes(pairs: Pair[], featuredId?: string) {
  return pairs.flatMap((pair) => {
    const run = pair.baseline;
    const text = run?.evaluation?.failureReasons[0];
    return run &&
      text &&
      pair.persona.id !== featuredId &&
      findabilityOf(run)?.outcome === "abandoned"
      ? [{ persona: pair.persona, text, run }]
      : [];
  });
}

/** Development shoppers on today's shop: the evidence the fix was chosen from. */
export function observed(pairs: Pair[]) {
  const development = pairs.filter(
    (pair) => pair.persona.split === "development" && findabilityOf(pair.baseline),
  );
  return {
    shoppers: development.length,
    gaveUp: development.filter((pair) => !purchased(pair.baseline)).length,
  };
}

export type SessionState = "pending" | "bought" | "missed" | "broke";
export interface Session {
  persona: Persona;
  arm: Arm;
  run?: Run;
  state?: SessionState;
}

/** Every shopper on both shop versions, newest run first, including runs still in flight. */
export function sessions(catalog: Catalog, runs: Run[]): Session[] {
  const cells = groupRuns(runs);
  const ids: Record<Arm, string | undefined> = {
    baseline: catalog.experiment.baselineVariantId,
    candidate: catalog.experiment.variantIds.find(
      (id) => id !== catalog.experiment.baselineVariantId,
    ),
  };
  return catalog.personas
    .filter((persona) => persona.task)
    .flatMap((persona) =>
      arms.map((arm) => {
        const variantId = ids[arm];
        const run = variantId ? cells.get(cellKey("mock", persona.id, variantId))?.[0] : undefined;
        return { persona, arm, run, state: run && sessionState(run) };
      }),
    );
}

function sessionState(run: Run): SessionState {
  if (run.status === "queued" || run.status === "running") return "pending";
  if (broke(run)) return "broke";
  return purchased(run) ? "bought" : "missed";
}

/**
 * Finished count, most sessions running at the same moment, and wall time.
 * Wall time is left out until every session finished, and when a span over ten minutes
 * shows the tiles come from different batches.
 */
export function batchStats(list: Session[]) {
  const timed = list.flatMap(({ run }) =>
    run?.startedAt
      ? [
          {
            start: Date.parse(run.startedAt),
            end: run.completedAt ? Date.parse(run.completedAt) : Infinity,
          },
        ]
      : [],
  );
  const edges = timed
    .flatMap(({ start, end }) => [
      { at: start, delta: 1 },
      { at: end, delta: -1 },
    ])
    .sort((a, b) => a.at - b.at || a.delta - b.delta);
  let running = 0;
  let peak = 0;
  for (const edge of edges) {
    running += edge.delta;
    peak = Math.max(peak, running);
  }
  const finished = list.filter((session) => session.state && session.state !== "pending").length;
  const span =
    timed.length > 0
      ? Math.max(...timed.map((t) => t.end)) - Math.min(...timed.map((t) => t.start))
      : Infinity;
  return {
    total: list.length,
    finished,
    peak,
    seconds:
      finished === list.length && span <= 10 * 60_000 ? Math.round(span / 100) / 10 : undefined,
  };
}

export interface Rate {
  met: number;
  total: number;
}
export interface Ledger {
  rows: { id: string; label: string; before: Rate; after: Rate }[];
  extraSteps: Record<Arm, number | undefined>;
  emptySearches: Record<Arm, number>;
  giveUpPage: Record<Arm, string | undefined>;
}

export function ledger(pairs: Pair[]): Ledger {
  const rows = new Map<string, Ledger["rows"][number]>();
  const extra: Record<Arm, number[]> = { baseline: [], candidate: [] };
  const giveUps: Record<Arm, Map<string, number>> = { baseline: new Map(), candidate: new Map() };
  const emptySearches: Record<Arm, number> = { baseline: 0, candidate: 0 };
  for (const pair of pairs) {
    for (const arm of arms) {
      const run = pair[arm];
      const findability = findabilityOf(run);
      if (!run || !findability) continue;
      for (const requirement of findability.requirements) {
        const row = rows.get(requirement.id) ?? {
          id: requirement.id,
          label: requirement.id === "budget" ? "Within budget" : capitalize(requirement.id),
          before: { met: 0, total: 0 },
          after: { met: 0, total: 0 },
        };
        const rate = arm === "baseline" ? row.before : row.after;
        rate.total++;
        if (requirement.met) rate.met++;
        rows.set(requirement.id, row);
      }
      if (findability.outcome === "purchased")
        extra[arm].push(Math.max(0, findability.steps - findability.optimalSteps));
      emptySearches[arm] += findability.emptySearches;
      const page = run.events.find((event) => event.id === findability.abandonedAt)?.page?.label;
      if (page) giveUps[arm].set(page, (giveUps[arm].get(page) ?? 0) + 1);
    }
  }
  return {
    rows: [...rows.values()],
    extraSteps: { baseline: median(extra.baseline), candidate: median(extra.candidate) },
    emptySearches,
    giveUpPage: {
      baseline: mostCommon(giveUps.baseline),
      candidate: mostCommon(giveUps.candidate),
    },
  };
}

/** Shoppers per route between page kinds, and where they gave up. */
export function flows(pairs: Pair[], arm: Arm) {
  const edges = new Map<string, { from: PageKind; to: PageKind; shoppers: Set<string> }>();
  const giveUps = new Map<PageKind, number>();
  for (const pair of pairs) {
    const run = pair[arm];
    const findability = findabilityOf(run);
    if (!run || !findability) continue;
    let previous: PageKind | undefined;
    for (const event of run.events) {
      const kind = event.page?.kind;
      if (!kind) continue;
      if (previous && previous !== kind) {
        const key = `${previous}>${kind}`;
        const edge = edges.get(key) ?? { from: previous, to: kind, shoppers: new Set<string>() };
        edge.shoppers.add(pair.persona.id);
        edges.set(key, edge);
      }
      if (event.id === findability.abandonedAt) giveUps.set(kind, (giveUps.get(kind) ?? 0) + 1);
      previous = kind;
    }
  }
  return {
    edges: [...edges.values()].map(({ from, to, shoppers }) => ({
      from,
      to,
      count: shoppers.size,
    })),
    giveUps,
  };
}

export interface Frame {
  event: StepEvent;
  src?: string;
}
export function frames(run?: Run): Frame[] {
  if (!run) return [];
  const saved = new Set(run.artifacts.map((artifact) => artifact.id));
  return run.events
    .filter((event) => event.page)
    .map((event) => ({
      event,
      src:
        event.screenshotArtifactId && saved.has(event.screenshotArtifactId)
          ? artifactUrl(event.screenshotArtifactId)
          : undefined,
    }));
}

/** Running totals up to and including step `index`; revisits match the evaluator (same page kind and label). */
export function counters(list: Frame[], index: number) {
  const seen = new Set<string>();
  let searches = 0;
  let revisits = 0;
  for (const { event } of list.slice(0, index + 1)) {
    if (event.search) searches++;
    const key = `${event.page?.kind}:${event.page?.label ?? ""}`;
    if (seen.has(key)) revisits++;
    seen.add(key);
  }
  return { steps: Math.min(index + 1, list.length), searches, revisits };
}

export function requirementLabel(persona: Persona, id: string) {
  if (id === "budget") return `Under $${persona.task?.budget}`;
  return persona.task?.requirements.find((requirement) => requirement.id === id)?.label ?? id;
}

function capitalize(value: string) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}
function median(values: number[]) {
  if (values.length === 0) return undefined;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}
function mostCommon(counts: Map<string, number>) {
  let best: string | undefined;
  for (const [key, count] of counts)
    if (best === undefined || count > counts.get(best)!) best = key;
  return best;
}
