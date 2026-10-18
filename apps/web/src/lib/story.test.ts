import assert from "node:assert/strict";
import test from "node:test";
import type { Catalog, Findability, Persona, Run, StepEvent } from "@storefront/contracts";
import {
  batchStats,
  buildPairs,
  counters,
  featured,
  flows,
  frames,
  ledger,
  merchant,
  observed,
  quotes,
  sessions,
  verdict,
} from "./story.ts";

const task = {
  query: "navy jacket",
  budget: 150,
  requirements: [
    { id: "color", label: "Navy", kind: "hard" as const, attribute: "color", value: "navy" },
  ],
};
const persona = (id: string, split: Persona["split"], withTask = true): Persona => ({
  id,
  name: id,
  goal: "",
  split,
  traits: [],
  history: [],
  ...(withTask ? { task } : {}),
});
const catalog: Catalog = {
  experiment: {
    id: "e",
    name: "E",
    baselineVariantId: "base",
    variantIds: ["base", "cand"],
    developmentPersonaIds: ["dev", "dev2"],
    heldOutPersonaIds: ["up", "down", "broken"],
    seed: 1,
    createdAt: "2026-01-01T00:00:00Z",
    contractVersion: 1,
  },
  personas: [
    persona("legacy", "development", false),
    persona("dev", "development"),
    persona("up", "held-out"),
    persona("down", "held-out"),
    persona("broken", "held-out"),
  ],
  variants: [],
};
const page = (
  id: string,
  kind: NonNullable<StepEvent["page"]>["kind"],
  extra: Partial<StepEvent> = {},
): StepEvent => ({
  id,
  runId: "r",
  sequence: 0,
  at: "2026-01-01T00:00:00Z",
  kind: "navigation",
  message: id,
  page: { kind, label: kind },
  ...extra,
});
let clock = 0;
function run(
  personaId: string,
  variantId: string,
  outcome: Findability["outcome"] | "queued" | "failed",
  events: StepEvent[] = [],
): Run {
  clock++;
  const finished = outcome !== "queued" && outcome !== "failed";
  const createdAt = new Date(Date.UTC(2026, 0, 1, 0, 0, clock)).toISOString();
  return {
    id: `${personaId}-${variantId}-${clock}`,
    experimentId: "e",
    baselineVariantId: "base",
    variantId,
    personaId,
    personaSplit: catalog.personas.find((p) => p.id === personaId)!.split,
    seed: 1,
    config: { headline: "h", showShipping: true, recommendationOrder: "popular" },
    persona: catalog.personas.find((p) => p.id === personaId)!,
    adapter: "mock",
    synthetic: true,
    contractVersion: 1,
    status: finished ? "completed" : outcome,
    createdAt,
    ...(outcome === "queued"
      ? {}
      : {
          startedAt: createdAt,
          completedAt: new Date(Date.parse(createdAt) + 1000).toISOString(),
        }),
    events,
    artifacts: [],
    evaluation: finished
      ? {
          runId: "r",
          synthetic: true,
          taskCompleted: outcome === "purchased",
          score: outcome === "purchased" ? 1 : 0,
          failureReasons: outcome === "abandoned" ? [`${personaId} gave up`] : [],
          notes: [],
          findability: {
            outcome,
            requirements: [
              { id: "color", met: outcome === "purchased" },
              { id: "budget", met: outcome === "purchased" },
            ],
            steps: events.filter((event) => event.page).length,
            optimalSteps: 2,
            searches: events.filter((event) => event.search).length,
            emptySearches: events.filter((event) => event.search?.resultCount === 0).length,
            revisits: 0,
            abandonedAt: outcome === "abandoned" ? events.at(-1)?.id : undefined,
            orderValue:
              outcome === "purchased" || outcome === "constraint-failure" ? 100 : undefined,
          },
        }
      : undefined,
    proposedChanges: [],
  };
}

const devBase = run("dev", "base", "abandoned", [
  page("h", "home"),
  page("c", "category"),
  page("p", "product"),
]);
const devCand = run("dev", "cand", "purchased", [
  page("h", "home"),
  page("c", "category"),
  page("p", "product"),
  page("k", "cart"),
]);
const runs = [
  devBase,
  devCand,
  run("dev", "cand", "queued"), // rerun in progress: must not hide the finished run
  run("legacy", "base", "purchased"),
  run("up", "base", "abandoned", [
    page("h2", "home"),
    page("s", "search", { search: { query: "x", resultCount: 0 } }),
  ]),
  run("up", "cand", "purchased", [page("h3", "home"), page("c3", "category")]),
  run("down", "base", "purchased", [page("h4", "home")]),
  run("down", "cand", "constraint-failure", [page("h5", "home")]),
  run("broken", "base", "purchased"),
  run("broken", "cand", "failed"),
];
const pairs = buildPairs(catalog, runs);

void test("pairs skip personas without a task and keep finished runs during a rerun", () => {
  assert.deepEqual(
    pairs.map((pair) => pair.persona.id),
    ["dev", "up", "down", "broken"],
  );
  assert.equal(pairs[0].candidate?.id, devCand.id);
});

void test("featured shopper is the first development shopper the fix helped", () => {
  assert.equal(featured(pairs)?.persona.id, "dev");
  assert.equal(featured(pairs, "down")?.persona.id, "down");
});

void test("verdict is held-out only, names regressions and sets technical failures aside", () => {
  const result = verdict(pairs);
  assert.equal(result.total, 2);
  assert.equal(result.shoppers.length, 2);
  assert.equal(result.before, 1);
  assert.equal(result.after, 1);
  assert.deepEqual(
    result.better.map((pair) => pair.persona.id),
    ["up"],
  );
  assert.deepEqual(
    result.worse.map((pair) => pair.persona.id),
    ["down"],
  );
  assert.deepEqual(
    result.technical.map((pair) => pair.persona.id),
    ["broken"],
  );
});

void test("ledger rates requirements and summarises effort", () => {
  const result = ledger(pairs);
  const color = result.rows.find((row) => row.id === "color")!;
  assert.deepEqual(color.before, { met: 2, total: 4 });
  assert.deepEqual(color.after, { met: 2, total: 3 });
  assert.equal(result.rows.find((row) => row.id === "budget")?.label, "Within budget");
  assert.equal(result.extraSteps.candidate, 1);
  assert.equal(result.emptySearches.baseline, 1);
  assert.equal(result.giveUpPage.baseline, "product");
});

void test("flows count shoppers per route and where they gave up", () => {
  const baseline = flows(pairs, "baseline");
  assert.equal(
    baseline.edges.find((edge) => edge.from === "home" && edge.to === "category")?.count,
    1,
  );
  assert.equal(baseline.giveUps.get("product"), 1);
  assert.equal(baseline.giveUps.get("search"), 1);
});

void test("frames fall back when a screenshot is missing and counters track revisits", () => {
  const withShot = {
    ...devBase,
    artifacts: [
      {
        id: "a",
        runId: "r",
        kind: "screenshot" as const,
        path: "x",
        mimeType: "image/svg+xml",
        createdAt: "2026-01-01T00:00:00Z",
      },
    ],
  };
  withShot.events = [
    page("1", "category", { screenshotArtifactId: "a" }),
    page("2", "category", { screenshotArtifactId: "gone" }),
  ];
  const list = frames(withShot);
  assert.equal(list[0].src, "/api/artifacts/a");
  assert.equal(list[1].src, undefined);
  assert.deepEqual(counters(list, 1), { steps: 2, searches: 0, revisits: 1 });
  assert.deepEqual(frames(undefined), []);
});

void test("merchant value compares held-out orders, value and wrong items", () => {
  assert.deepEqual(merchant(pairs), {
    baseline: { orders: 1, value: 100, wrongItems: 0 },
    candidate: { orders: 1, value: 100, wrongItems: 1 },
  });
});

void test("quotes are other shoppers' own give-up notes", () => {
  const list = quotes(pairs, "dev");
  assert.deepEqual(
    list.map((quote) => [quote.persona.id, quote.text]),
    [["up", "up gave up"]],
  );
});

void test("observe step counts development shoppers who gave up on the baseline", () => {
  assert.deepEqual(observed(pairs), { shoppers: 1, gaveUp: 1 });
});

void test("timeouts and technical failures are set aside, never counted as worse", () => {
  const slow = {
    persona: persona("slow", "held-out"),
    baseline: run("up", "base", "purchased"),
    candidate: run("up", "cand", "timeout"),
  };
  const result = verdict([slow]);
  assert.equal(result.total, 0);
  assert.deepEqual(result.worse, []);
  assert.deepEqual(
    result.technical.map((pair) => pair.persona.id),
    ["slow"],
  );
});

const timed = (base: Run, start: number, end?: number): Run => ({
  ...base,
  startedAt: new Date(Date.UTC(2026, 0, 2, 0, 0, start)).toISOString(),
  completedAt:
    end === undefined ? undefined : new Date(Date.UTC(2026, 0, 2, 0, 0, end)).toISOString(),
});

void test("sessions show the latest run per shopper and version, including pending", () => {
  const list = sessions(catalog, runs);
  assert.equal(list.length, 8);
  const dev = list.filter((session) => session.persona.id === "dev");
  assert.deepEqual(
    dev.map((session) => [session.arm, session.state]),
    [
      ["baseline", "missed"],
      ["candidate", "pending"],
    ],
  );
  assert.equal(
    list.find((s) => s.persona.id === "broken" && s.arm === "candidate")?.state,
    "broke",
  );
  assert.equal(list.find((s) => s.persona.id === "up" && s.arm === "candidate")?.state, "bought");
});

void test("batch stats report peak parallel sessions and wall time once all finish", () => {
  const p = persona("dev", "development");
  const done = [
    timed(run("dev", "base", "purchased"), 0, 4),
    timed(run("dev", "cand", "purchased"), 1, 3),
    timed(run("up", "base", "abandoned"), 2, 6),
  ].map((r) => ({ persona: p, arm: "baseline" as const, run: r, state: "bought" as const }));
  assert.deepEqual(batchStats(done), { total: 3, finished: 3, peak: 3, seconds: 6 });
  const running = [
    ...done,
    {
      persona: p,
      arm: "candidate" as const,
      run: run("up", "cand", "queued"),
      state: "pending" as const,
    },
  ];
  assert.deepEqual(batchStats(running), { total: 4, finished: 3, peak: 3, seconds: undefined });
});
