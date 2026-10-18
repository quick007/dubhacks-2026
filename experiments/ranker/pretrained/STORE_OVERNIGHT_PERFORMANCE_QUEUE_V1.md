# Overnight performance queue

Prepared on 2026-10-18. The user's latest timing instruction is **not now** for
major architecture work. The live six-card native service remains unchanged;
there has been no new native profiling, training, model download or package
installation in this preparation. Work below is queued for the later coordinated
performance window. No low-hundreds-of-milliseconds result is claimed.

The current usable checkpoint includes one completed independent Sol pair:
[repaired smoke](store_shopper_sol_smoke_v2/smoke-summary.json). Both shoppers
completed a simulated checkout with evidence, constraints and budgets verified.
Local-model query `a bottle for drinking water under $40` reached the browser
first-three paint opportunity in5450.7ms with10/10fields accepted; full-six in
6384ms with19/20 accepted. This measures React commit plus two frames, excluding
physical paint and image transfer. The [native receipt](store_demo_results/v5/requests/18bf1940-31bb-4ae1-9f8f-badfe767f170/receipt.json)
records rank204.920ms, serialized first3 5423.797ms and full6 6367.897ms. The
first section has three finished factual bodies and matching names/colors/prices.
The remaining nine paired shopper scenarios have the next exclusive serving
window; no GPU profiling or training may overlap them. Only the bottle directly answers the narrow request; a mug and towel are broad
store fill. This is a working development example, not six relevant alternatives,
a population comparison or conversion lift. Eighteen shopper slots remain open.

## Measurement first

[Profiler v2](store_demo_decode_profile_v2.py) and the
[exclusive driver](store_demo_performance_run_v1.py) are CPU-tested. The final
[CPU preflight v4](store_demo_performance_preflight_v4/preflight.json) pins sources
and retains the same three public requests, six cards,20fields and416-token cap.
Baseline/profile order is AB/BA/AB, with one excluded baseline warmup. Native
execution must first coordinate shopper/browser inactivity, gracefully close the
sole service, confirm no other owned worker, and use the shared native-owner lock.
The driver refuses an active8766 service before and after acquiring its lock.

The observer separates suffix prefill, initial/subsequent AR steps, cache
merge/split/filter/extend and synchronization waits. Inclusive phases nest;
exclusive times plus separate pre-boundary waits avoid double-counting. Added
barriers and materialization disturb asynchronous overlap: compare ordinary
baseline totals and require exact emitted tokens/EOS/bindings. These are
synchronized native host-phase measurements, not GPU kernel traces. Restore the
service and health-check it before releasing the next shopper window.

## Candidate order after the profile

1. Test full width20 and exact common-root reuse if the observed breakdown
   justifies them. All20fields and all6cards remain present; shorter caps alone
   do not establish successful generation. Keep the initial7GiB guard. The latest
   inspected24GiB machine had8.51GB available and4.58GB swap in use; those are a
   historical sample, so inspect again before any guard increase.
2. Measure [native-generated copy caching](GENERATED_COPY_CACHE_V1_DESIGN.md).
   Eight complete native warm requests cover48product/color bindings. The helper
   validates native provenance, exact facts/prompts/adapter and original EOS;
   caches the first attempt including failures; and reports generation age,
   warm hit, miss and original cold cost. A hit is **generated-earlier copy**.
   Learned ordering still uses the fresh public query. Misses retain genuine
   generation and cannot silently use templates. No native warming or hit latency
   has yet been measured. Source-conditioned neutral copy makes this reuse
   applicable to the current model; it does not supply query-conditioned wording.
3. Compare shorter genuinely fresh9B copy and a
   [parallel learned text-head candidate](store_parallel_text_v1/manifest.json).
   The latter predicts word/EOS slots from cached Qwen variant features, while
   keeping the capable Qwen ranking backbone. CPU engineered-feature fit is only
   a wiring check. Native features, training quality and latency remain unmeasured;
   it must not replace the live decoder on the strength of CPU tests.
4. Read [fresh-generation feasibility](STORE_FAST_GENERATION_FEASIBILITY_V1.md)
   before alternative backends. Stock installed speculation rejects this hybrid
   cache. DFlash provides a separate implementation worth an isolated test, with
   different dependency pins and single-stream behavior. No install or draft
   download has been performed; inspect dependency changes and use an isolated
   environment before testing it. Published long-output gains are not this short parallel workload.

## Quality and frontier

The [v2 offline usefulness scorer](store_usefulness_v2_candidate/README.md)
recognizes equivalent source-bound dollar notation without replacing generated
text. It preserves the frozen benchmark and null historical readiness timestamps.
Optional headings/badges are separate from complete title/body core. It is a
conservative lexical screen, not a comprehensive semantic judge or exact teacher
wording test. Real unsupported facts and unfinished/ambiguous bodies still fail.

The [prepared frontier](store_speed_quality_frontier_v1/manifest.json) has32
public requests spanning16product functions with two wordings each, all with16
eligible products and a fixed six-card/20-field workload. Three wordings overlap
TRAIN exactly; the packet is curated development, not unseen generalization.
Its [independent audit](STORE_SPEED_QUALITY_FRONTIER_V1_AUDIT.md) preserves two
corrected draft issues. Pass only each `request` to inference: case IDs and all
relevance labels are scheduler/offline data. Top-one labels do not assert that
all six cards answer every narrow request. Separate existing guard probes cover
budgets, exact names, colors, unknown stock and size with legitimate shortfalls.

Report p50/p95 by declared condition: startup, cold fresh generation, warm fresh
generation, generated-earlier hits and cold misses. Include above-fold core
completeness, full completion, factual errors/withheld fields, query relevance,
all fixed failures, tokens/forward counts and memory. Preserve a separate broad
honest evaluation lane even if the demo uses deliberately favorable cache/catalog
conditions. Neither rank-only nor cache-only timing is complete browser readiness.

