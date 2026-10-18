# Slide deck demo — design

Status: approved in conversation on 2026-10-17. Supersedes the scrolling layout in `2026-10-17-case-file-demo-design.md`; its data contract, mock scenario, evaluator and honesty rules still apply.

## Purpose

The demo page is a presenter-driven slide deck: one focused idea per slide, advanced with the arrow keys, space, a click or a swipe. It tells one shopper's story ("Bob"), then zooms out to show Bob is one of many simulated sessions run in parallel, then shows the fix, the retest and what it means for shoppers and the store. Each deck covers one shop; all copy comes from the loaded experiment so a different shop or the later, nicer storefront retells the story without code changes.

## Story (slides)

Slides with no data yet are skipped.

1. **Title.** Experiment name, "Synthetic study", run shoppers.
2. **Meet Bob.** Persona name and bio, then the goal types itself out.
3. **Bob is one of 20.** Live grid of every session in the batch (personas × shop versions), each tile a tiny replay ending in its outcome. Counter: sessions finished, peak sessions running at once, wall time. Bob's tile highlighted. Tiles open evidence.
4. **Bob's struggle.** Today's shop replay; ends frozen where Bob gave up, with the reason. Replay, scrub, see every step.
5. **Not just Bob.** Two or three other shoppers' give-up notes, named.
6. **The fix.** Rationale and factual change; flip the product listing between today's shop and the fix.
7. **Same Bob, with the fix.** Side-by-side race.
8. **Every shopper's route.** Shop map with today/fixed toggle; picking a shopper replays them on slide 7.
9. **How findable it was.** Three or four big before → after figures from the ledger.
10. **For shoppers.** Held-out result ("2 of 5 → 4 of 5 new shoppers found their jacket"), better and worse named.
11. **For the store.** Suitable orders, suitable order value, wrong-item orders.
12. **Iterate.** Observe → propose → edit → retest for this round, sessions and wall time, and that round 2 starts from here; a live A/B test needs weeks of traffic per comparison.
13. **What's synthetic.** Provenance and run again.

A corner indicator on Bob's slides reads "Bob · 1 of N sessions" with one dot per session coloured by outcome; it jumps to slide 3.

## Navigation

`→`/`Space`/`PageDown` next, `←`/`PageUp` previous, `Home`/`End`. Next/back buttons (Lucide chevrons) bottom right. Top progress bar with one clickable segment per slide. The URL hash holds the slide id (`#fix`), so reloads and shared links land on the same slide. Swipe on touch. One short fade between slides; none with reduced motion. Keys are ignored while a dialog, slider or input has focus.

## Data changes

- `Persona.bio?: string` (Zod and Go, optional). Fixture personas get first names and one-line bios set in Seattle; fictional and labelled synthetic.
- The API queue runs jobs on a pool of 8 workers; real browser runs are capped at 2 at once. SQLite access is already serialised through one connection.

## Code

- `apps/web/src/slides/`: one component per slide, plus `Deck` (navigation, hash, progress, transitions) and `SessionIndicator`. Reused: replay lane and frame, shop map, evidence drawer, run detail.
- `lib/deck.ts`: slide list from available data and hash parsing. `lib/story.ts` gains `sessions(catalog, runs)` for the grid (latest run per persona × variant, including pending) and `batchStats` (finished, peak concurrency, wall time).
- The scrolling page, `PresenterBar` and `FinePrint` section are replaced.

## Testing

- Go: concurrent sessions overlap (several runs start before the first completes); browser runs never exceed 2 at once is enforced by a semaphore.
- Contract: persona bio round-trips.
- Web unit: slide skipping, hash parsing, session tiles including pending, peak concurrency and wall time.
- Playwright: run shoppers from the title slide, step through the deck with the keyboard, open evidence.
