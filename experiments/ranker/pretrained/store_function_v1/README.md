# Actual-store function adaptation candidate

Frozen CPU preparation only: 16 actual-store products, 32 TRAIN public-use requests, 16 independently authored evaluation requests in 8 preference-flip pairs, and 8 deterministic hard-guard probes. Each pair holds the task, catalog and guards fixed while changing the public preference. Full product and query inputs preserve authoritative public facts; opaque routing IDs, targets, support rationales and image metadata never enter predictive text. All 16 product identities/functions occur in TRAIN, so this is known-catalog adaptation with held-out request wording. Prices and wording remain class-correlated. There is no unseen-catalog, latent-preference, human conversion or sealed population claim.

The existing seven-factor backpack head is unsuitable as an established store scorer: the authoritative name/category/searchTerms fields support only `organizer` on one product. Restricting product inputs to those seven exact atoms gives 15 identical empty-atom product texts and 2 distinct texts in total. Waterproof is not silently substituted with water-resistant. Missing factors are unknown rather than negative facts.

The new candidate uses 64 whole B1 native forwards (16 catalog + 32 TRAIN query + 16 evaluation query), each 64–103 tokens under the fixed 512-token cap. Two 4096-to-16 linear role classifiers have 131,104 parameters. Fixed 80-step AdamW training uses equal-role cross-entropy, seed 728, LR .03, weight decay .001, one CPU thread, TRAIN-only role means and L2 centering. Controls are an independently trained item prior, seeded random heads, permuted queries, and a postfreeze public-label oracle ceiling. The head checkpoint contains exact mean membership/hashes, feature provenance, targets hash and dataset/packet bindings. Predictions for all four model/control arms freeze before the evaluator opens evaluation labels.

Hard guards run before top-one/top-six selection: exact declared category/color membership and inclusive item-price ceilings. Required stock or size information is unknown and returns zero eligible products. Item prices do not establish delivered prices. Full 16-product scores and raw/eligible orders are retained; no suitable fillers are invented. Top-one/top-six use one query feature. This module does not choose images, generate copy, learn grouping or measure UI behavior. Existing image heads require separately verified representation/feature compatibility; their dimensions alone are insufficient. Backpack copy validators require unsupported shipping/stock/tag/capacity schema assumptions and must not be made compatible by inventing facts; the separate actual-store copy diagnostic owns its explicit contract.

**CPU review blocker for endpoint latency:** current extraction encodes all 64 inputs before fitting and later evaluates cached query vectors. Native query-encoding timings, verified 16-product warm-read timings, and cached-vector head/guard timings are separate components. No contiguous fresh-request endpoint was measured; summing them is not a measured request latency. Before a future endpoint benchmark, stage 16 catalog + 32 TRAIN vectors, fit while the native model remains resident, then encode each of 16 fresh evaluation queries and immediately perform verified warm catalog reads, head scoring and guards before freezing predictions/opening labels.

Additional native gate follow-ups from independent review: enforce the 180-second extraction budget after each forward and at completion, and preserve a fixed-denominator failed/count receipt if native extraction fails. The current runner preserves partial immutable feature artifacts and does not fabricate a completed report on failure. These follow-ups should be resolved in a new source/data/preflight version before native launch; preserve this packet.

Validation: 9 CPU tests pass; the frozen token preflight passed with no MLX/model execution. No fit or native forward has run for this packet. Root deferred actual-store execution at the migration boundary. Recheck source freshness and independent review before continuing.

```sh
.tools/pretrained-venv/bin/python -m unittest experiments.ranker.pretrained.test_store_function_adaptation -q
```

Prepared commands for a later root-owned run, **after the review follow-ups and a new freeze**:

```sh
.tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.store_function_adaptation freeze --data experiments/ranker/pretrained/store_function_v1_reviewed2
.tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.store_function_adaptation prepare --data experiments/ranker/pretrained/store_function_v1_reviewed2 --packet experiments/ranker/pretrained/STORE_FUNCTION_PREFLIGHT_V2.json
.tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.store_function_adaptation run --owner root --data experiments/ranker/pretrained/store_function_v1_reviewed2 --packet experiments/ranker/pretrained/STORE_FUNCTION_PREFLIGHT_V2.json --out data/ranker/pretrained/store-function-adaptation-v1-attempt-1
```

Root-only execution acquires the existing native owner lock and freshly hashes both pinned model shards, tokenizer/config/template before loading. The active/observed-peak guard stays fixed at 6 GiB. No downloads or fallback execution route is provided.

The future completed report interface is `status=completed`, `modelInputSha256`, `heads.learned.sha256`, `featureProvenance`, `rawFeaturesFreezeSha256`, `publicCasesSha256`, and `results.evaluation.learned.cases[caseId]` with `scores`, probabilities and `topSix.productIndices/eligibleRank/rawRank/guards/shortfall`. The fixed first six public evaluation rows feed the separate store-copy diagnostic only after a completed verified learned run. Their labels must not choose the cases or repair predictions.
