# Case file demo page — design

Status: approved in conversation (story and contract); page architecture and visual system below await review. Expect change: the project is evolving fast, and real agent data will replace the mock scenario.

## Purpose

The dashboard is a demo stage. Its job is to convince an audience (hackathon judges, merchants) that the tool finds why shoppers fail on a store, proposes a factual fix, and measurably helps simulated shoppers find and buy what they came for. It is a scrolling story page, closer to Spotify Wrapped or an interactive article than to a monitoring grid. Every big number opens its evidence; credibility is the differentiator.

The current results-grid dashboard in `apps/web` is replaced. Its run detail panel survives as the evidence drawer.

Scope of this spec:

1. Findability fields in the shared contract (Zod and Go).
2. A deterministic mock scenario (outdoor jacket shop) that fills those fields.
3. The case file page built against that data.

Out of scope, each a later spec plugging into the same contract: an apparel catalog with real search and filters in `apps/storefront`; a model-driven browser runner; latency and cost accounting.

## Honesty rules

- Every result is labeled synthetic in the fine print and in the presenter bar. No claim of human conversion or revenue lift.
- The headline result uses held-out shoppers only. Development shoppers feed the snag and the fix.
- Regressions are shown by name, not netted away.
- Outcome categories are kept separate: a technical failure or timeout never counts as a shopper giving up.
- Findability is judged by the evaluator against scenario facts, never by the shopper's own claim.

## Story (top to bottom)

1. **Opening.** The featured shopper's goal types itself out like a search query ("waterproof navy jacket, medium, under $150"). One line of persona beneath.
2. **The race.** Two screenshot flipbooks side by side, baseline and candidate, for the featured shopper. A click marker shows where the agent clicked. Live counters: steps, searches, pages revisited. A requirement checklist ticks as requirements are met. Autoplays when scrolled into view; scrub bar for the presenter.
3. **The snag.** Freezes on the baseline frame where the shopper gave up, annotated with the cause from the proposal rationale. Links to the evidence steps.
4. **The fix.** The proposal's config patch shown as a before/after pair of frames, with the factual guard line (uses existing catalog facts only).
5. **Everyone else.** Shop map drawn as a transit map. Stations are page kinds: home, category, search, product, cart, checkout, confirmation. Line thickness is shopper count; ✗ marks where shoppers gave up. Toggle redraws baseline vs candidate. Clicking a line replays that shopper in the race.
6. **Findability ledger.** Ruled rows, not tiles. Per requirement: found rate before → after. Then median wasted steps, empty searches, most common give-up page.
7. **Result.** One sentence: "3 of 10 → 8 of 10 shoppers found and bought what they came for", held-out only, with paired better / worse / same counts and any worse shoppers named.
8. **Fine print.** Synthetic label, seeds, adapter, evaluator version, scenario id.

**Featured shopper:** the first development persona (fixture order) whose latest baseline run did not purchase and whose latest candidate run did. Fallback: first development persona. The presenter can switch it from the shop map.

**Presenter bar:** fixed, thin, bottom. Run again (mock), adapter label, keyboard hint. `←`/`→` jump between story beats (scroll, not slides). Hidden in print.

## Contract additions

All optional, additive. `CONTRACT_VERSION` stays 1 so stored runs keep parsing. Zod in `packages/contracts/src/index.ts` and Go in `apps/api/internal/contracts/types.go` change in the same commit.

```ts
// Persona (named `task` because `goal: string` already exists)
task?: {
  query: string;
  requirements: { id: string; label: string; kind: "hard" | "preference" }[];
  budget?: number;
};

// StepEvent
page?: { kind: "home" | "category" | "search" | "product" | "cart" | "checkout" | "confirmation" | "other"; label?: string };
click?: { x: number; y: number };          // 0–1 viewport fractions
screenshotArtifactId?: string;
search?: { query: string; resultCount: number };

// EvaluationResult
findability?: {
  outcome: "purchased" | "abandoned" | "constraint-failure" | "technical-failure" | "timeout";
  requirements: { id: string; met: boolean; observed?: string }[];
  steps: number;
  optimalSteps: number;
  searches: number;
  emptySearches: number;
  revisits: number;
  abandonedAt?: string; // step event id
};
```

The shop map needs no contract data: its stations are the fixed page kinds.

## Mock scenario

- `apps/api/fixtures/scenarios/jacket.json`: about 10 personas (5 development, 5 held-out) with `task` requirements, and for each persona × variant a scripted path: ordered steps with page kind, label, click point, optional search, and a frame name. Also each persona's scenario facts (which product satisfies the task) for the evaluator.
- `apps/api/fixtures/catalog.json`: experiment renamed for the jacket study; variants become baseline (sizes hidden on product cards, as an example) and candidate (sizes shown). Exact snag chosen while writing the scenario.
- `apps/api/fixtures/mock-frames/*.svg`: simple drawn page frames (product grid without sizes, search with 0 results, product page, cart, confirmation), each stamped "mock".
- The Go `Mock` adapter loads the scenario when the persona has a `task`, emits enriched events, copies each referenced frame into the run's artifact directory as a `screenshot` artifact, and returns the path. Personas without a `task` keep today's shipping rule.
- The evaluator computes `findability` from the emitted path and scenario facts: requirements met, steps vs `optimalSteps`, searches, empty searches, revisits (repeat page label), outcome, and the abandon step.
- `runner.Event` and `Context.Emit` gain the optional step fields. The browser runner protocol is untouched in this spec.
- Artifact serving adds `image/svg+xml` with `Content-Security-Policy: default-src 'none'; style-src 'unsafe-inline'` and `X-Content-Type-Options: nosniff`. Frames render through `<img>` only.

## Page architecture (`apps/web/src`)

One component per file. Pure derivations live in `lib/` and are unit tested.

- `App.tsx`: loads catalog once, polls runs (1 s while anything is queued or running, else 5 s), renders the story.
- `lib/story.ts`: latest run per persona × variant, featured shopper, paired held-out result, ledger rows, shop-map flows (station-to-station edge counts and give-up stations per variant).
- `story/Opening.tsx`, `story/Race.tsx`, `story/RaceFrame.tsx`, `story/RaceCounters.tsx`, `story/Snag.tsx`, `story/Fix.tsx`, `story/ShopMap.tsx`, `story/Ledger.tsx`, `story/Verdict.tsx`, `story/FinePrint.tsx`, `story/PresenterBar.tsx`, `story/EvidenceDrawer.tsx` (Base UI Dialog wrapping the existing `RunDetail`).
- Kept: `RunDetail`, `RunSteps`, `RunHistory`, `lib/api.ts`, `lib/format.ts`, `lib/runs.ts`. Deleted: `OutcomeMatrix`, `OutcomeCell`, `ConfigDiff`, `SuggestionList`, `AppHeader`.
- Charts and the shop map are hand-built SVG. No chart library.
- Optimistic: "Run again" marks the race and verdict as rerunning immediately.

Motion: one orchestrated moment, the race. The opening's typing is the only other animation. `prefers-reduced-motion` shows the final frames and skips typing, checked live via `matchMedia` listener.

## Visual system

- **Type.** Display: Big Shoulders Display (condensed signage face; fits the transit-map metaphor and big reveal numbers). Text: Atkinson Hyperlegible Next (legible on a projector). Both self-hosted via `@fontsource-variable`. Sentence case everywhere; no all-caps, no eyebrows.
- **Color.**
  - fog `#ECEFEA` page background
  - ink `#17241D` text
  - graphite `#8B938D` baseline lines and before values
  - cobalt `#2448D6` candidate lines and after values
  - signal `#E2462A` give-up marks only
  - paper `#FFFFFF` screenshot frames
- **Layout.** Full-bleed sections separated by space and type scale, not boxes. Rules for ledgers. The only framed objects are the screenshot frames, because they depict screens. Left-aligned text, max 68ch. Works at 390 px with 16 px gutters; race stacks vertically on phones.

## Testing

- Contract: Zod parse round-trip of a run with every new field; old v1 run still parses.
- Go: scenario loading; mock emits enriched events and screenshot artifacts; evaluator findability values for one passing and one abandoning path; SVG served with the CSP header.
- Web unit (`node --test`): `lib/story.ts` featured shopper, paired result with a regression, ledger rates, map flows.
- Playwright (`tests/local-demo.spec.ts`): page shows the race and a held-out result after "Run again"; evidence drawer opens from a number. Note: the config sets `reuseExistingServer: false`, so stop `pnpm dev` before running.
