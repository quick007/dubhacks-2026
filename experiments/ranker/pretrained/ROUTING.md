# Optional grounded sub-label routing

This small model-level experiment trains one shared 4096→3 linear head on actual cached shared-prefix pretrained readouts. It predicts `NONE`, `SELECT`, or `GENERATE` for the product/presentation action chosen by the existing trained heads. It changes no storefront UI and does not load a second backbone.

The teacher is deliberately transparent and synthetic: zero explicitly requested catalog tags means NONE; one means SELECT; two or more means GENERATE. Only observable `statedPreferences` and `statedRequirements` matched to product tags count. Private/unknown preferences do not justify a label. This measures imitation of a toy public policy, not human label relevance, human preferences or architecture novelty. Selection versus writing has precedent.

Training uses only the existing public training requests, catalog, split manifest and saved actual backbone training embeddings. It checks the cached source hashes and action mapping, uses disjoint initial/adaptation families, and fixes seed 37, AdamW and 80 epochs per stage before evaluation. Receipts include nonzero gradients, prediction changes, exact checkpoint reload and SHA-256. The backbone and decoder stay frozen. CPU training neither loads MLX nor reads private truth.

```sh
pnpm pretrained:test:routing
pnpm pretrained:route:train
pnpm pretrained:route:benchmark
```

Run the benchmark only after other 9B processes finish. Defaults use `data/ranker/pretrained/shared-prefix/adaptation`, dataset v2, `data/ranker/pretrained/fusion-proof/case-plan.json` and ignored output `data/ranker/pretrained/routing`. `--heads`, `--dataset`, `--out` and `--baselines` explicitly select those paths. The eight existing case IDs are fixed before routing evaluation; every case is retained, including sparse/no-label and unfavorable cases.

The result schema contains `rawRoute`, `guardedRoute`, `servedRoute`, optional `selectedLabelId`/`text`, exact `{observablePreference, productFact}` grounding, generation-attempt status and any rejection. The approved dictionary derives only `attribute:<existing-tag>` labels and human-readable versions of those exact tags. A deterministic guard forces NONE without compatible, eligible and supported evidence; this is masking, not learned behavior. NONE and SELECT never call the decoder.

Only guarded GENERATE calls the same resident backbone's native decoder, reusing the complete exact-prefix recurrent, convolution and KV state once. Generation is capped at 64 tokens with no retry. Existing independent factual validators check the generated title and optional copy separately. The sub-label must be at most 90 characters and itself mention an observable supported tag; generic “for you” is insufficient. An accepted novel title can survive rejected optional copy. If neither field qualifies, the result records the failed attempt and serves an approved SELECT label.

The benchmark reports raw and guarded routing accuracy against the toy policy, exact-evidence/lexically validated grounded-label rate, attempted and served generation counts, route distributions and complete uncached latency p50/p95. It compares matching IDs against previous actual always-generate and selection-only fusion measurements. These are prior sequential timings, not a concurrent paired comparison; full title/copy generation and optional sub-label generation also differ in output requirements. A tag-based relevance proxy and restrictive lexical factuality check are not semantic validation or human-quality evidence. No synthetic metric establishes human conversion or revenue lift.

CPU/stub tests verify teacher 0/1/2 behavior, missing-context NONE, compatible exact evidence, no decoder calls for NONE/SELECT, real parameter updates, exact reload/hash rejection, and independent acceptance of a supported title with a rejected sentence. Those tests are not proof that a trained router generalizes or that the real 9B generation benchmark completed. Only saved actual training and benchmark artifacts support those claims.

## Actual local result

The fixed 80+80-epoch CPU training ran on 96 actual cached 9B readouts per stage. Adaptation cross-entropy decreased from 0.6940 to 0.07591 with nonzero gradients, changed predictions and exact checkpoint reload. This is training-fit evidence only.

The actual eight-case 9B benchmark generalized poorly: raw routing accuracy was 37.5%, guarded accuracy 50%. Raw routes were six NONE and two SELECT; served results were seven NONE and one approved SELECT label. There were **zero GENERATE attempts**. The sole served label had exact supported evidence, so its 100% grounding proxy has a denominator of one. No real routed GENERATE path occurred; the independent core fusion benchmark supplies actual decoder evidence.

Resident full routed latency p50/p95 was 1373/1418 ms, versus prior sequential selection-only 1333/1372 ms and always-generate 2938/3284 ms on the same eight case IDs. Skipping generation explains the lower latency; this is not evidence of a capable selective generator. No held-out cases were discarded, and no tuning followed this negative result. See [RESULTS.json](RESULTS.json) for committed training receipts, all metrics and local evidence hashes. Full records remain ignored at `data/ranker/pretrained/routing/benchmark.json`.
