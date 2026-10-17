# Reusable catalog embeddings with fresh user-conditioned scoring

Proposal only: retain the same frozen quantized Qwen3.5-9B backbone, cache user-independent catalog features, and train a small fusion scorer against fresh request features. This is a new representation and checkpoint, not a cache-equivalent implementation of the current ranker. It must preserve complete twelve-/twenty-four-card product order and image choices, followed by genuine new copy from the same backbone. No UI, template, smaller model or vendor service is proposed.

## Existing boundary and evidence

The current corrected `ParallelHeads(4096)` has 8,194 trainable parameters and consumes joint causal readouts from one conversation containing user context and every candidate's facts. Its cache is indexed by request, not just catalog content. `evidence_plan.train` fits 24 requests / 192 variant rows; `ranking_loss` uses best-product cross-entropy on mean variant product scores plus within-product presentation cross-entropy on informative compatible variants. Existing weights cannot be applied to standalone catalog vectors and called equivalent.

The v2 fixture has 84 products / 168 text-described presentation variants, 24 training requests, six validation requests, twelve test requests and six preference-counterfactual diagnostics. Its ordinary requests have at most four candidates. Manifest metadata specifies disjoint persona-family/product assignments and shuffled candidate/variant order. Public records do not expose the family labels. The previously inspected test and diagnostic cases are now replay evidence; they are not an untouched test for this new design and contain no labeled twelve-/twenty-four-card grids.

The earlier `catalog_cache.py` experiment caches complete catalog-first **decoder state** while recomputing the unchanged joint ranker. It therefore duplicates work; all ten generated pages failed validation. That result does not establish the quality or speed of reusable ranking embeddings. This proposal caches small feature vectors, not native decoder states, generated responses or personalized scores.

## Minimal trainable scorer

Define a new named representation, `catalog-user-fusion-v1`:

- `c[p,v] = normalize(F(catalog_variant_text[p,v]))`, one frozen 4,096-dimensional vector per product/approved presentation variant. Use canonical, request-independent allowlisted product facts and text image descriptors. Retain current `clean_product` and `variant_text` facts, including price/shipping/stock and compatibility metadata; do not introduce opaque IDs or catalog positions into text.
- `q = normalize(F(public_request_text))`, recomputed once per request. Include current task, budget, explicit requirements/preferences, presentation surface/preferences and the same cutoff-filtered latest-history summary as the existing compact representation. No candidate roster, private persona, latent intent, labels, user ID or split/family metadata enters this encoder. Exact IDs remain external mapping keys only.
- Freeze `F`, its tokenizer and quantization. Fix independent catalog/query prompt formats, last-valid-token pooling and normalization before feature extraction. Use a controlled extraction batch shape/chunking, retain their receipts, and refuse silent truncation. These encodings are different from joint readouts even though the backbone is shared.

A small first trainable fusion is rank four with shared projections:

```text
z_user = tanh(U q)                 U: 4 × 4096
z_item = tanh(V c[p,v])            V: 4 × 4096
score_k(p,v) = bias_k
             + prior_k · z_item
             + interaction_k · (z_user * z_item)
k = product, presentation
```

This has 32,786 trainable parameters: 32,768 projection weights and eighteen output weights/biases. It is a practical fixed small hypothesis, not a proven sufficient bottleneck. The catalog-only term permits a learned prior; the multiplicative term makes scores explicitly user-conditioned. A query-only additive bias is unnecessary because it cancels across candidates within a request. Use nonzero deterministic initialization and verify actual gradients/parameter changes; no backbone training or compatibility with old head weights is implied.

Preserve the existing action interface: average product-head scores over each product's variants, apply current known stock/budget/requirement/compatibility masks against a consistent live catalog snapshot, rank **every** candidate, then choose the best compatible presentation for **each selected product**, not only the first. Stable ID tie breaking remains outside learned features. Twelve cards require at least twelve eligible unique products; twenty-four require at least twenty-four. Infeasible requests expose an explicit shortfall/failure and never count as complete grids. Section grouping remains deterministic.

Product/variant scores can be computed for all candidates in one CPU tensor operation after `q` is available; that algorithmic property is not a wall-time claim. Raw float32 catalog storage for the current 168 vectors is 2,752,512 bytes (2.625 MiB), excluding metadata. Optional projected `z_item` storage is only 2,688 bytes at rank four, but its key must also bind the new fusion checkpoint because `V` is trainable. Keep raw frozen features to avoid re-encoding the catalog after every head update. Neither vector type can be concatenated/spliced into the model's native recurrent/KV state.

## Training and evaluation boundary

The first implementation should make one fixed fit of the new projections/heads on freshly extracted **training** features, using the same product/presentation loss definitions as the current corrective fit. Use only `public/train.jsonl` and `labels/train.jsonl` during optimization. Select a fixed seed/rank/optimizer schedule before inspecting outcomes; retain failed fits, gradients, loss, checkpoint hashes and exact reload predictions. The existing 24-query set is a pipeline proof with substantial overfit risk for even this small scorer. It is insufficient to establish full-grid ranking quality.

Keep these comparisons distinct:

| Comparison | Purpose |
| --- | --- |
| Existing joint features + current trained heads | Preserved quality/latency reference; retain its unsupported full-grid cases honestly |
| New separated frozen features + random fusion | Detect training benefit, not merely feature extraction |
| New features + trained item-prior-only scorer | Test whether user interaction improves over static catalog popularity/prior |
| New features + trained user/item fusion | Proposed architecture |
| Same trained fusion, query shuffled across users | Diagnose reliance on current user context; not a serving configuration |
| Fresh catalog feature extraction versus cached exact same features | Isolate caching correctness/latency within the new representation |

Report old validation/test/diagnostic replays without calling them untouched. Before promotion, freeze a new sealed family- and SKU-disjoint evaluation with sufficient eligible candidates for full twelve-/twenty-four-card requests, for example fixed 24-/48-candidate pools. Obtain fresh per-request relevance and per-product presentation labels for the whole pool from the independent synthetic grading pipeline. Do not manufacture cross-family labels by concatenating old candidate lists or reuse one user's relevance labels for another user. Keep test labels and family metadata out of feature extraction/training. Existing attributes may overlap; unseen combinations/SKUs do not establish unseen-attribute generalization.

Evaluate best-product accuracy plus NDCG at three, twelve and twenty-four as applicable, full requested-card fulfillment, unique/eligible selection, and compatible-image accuracy/utility for every selected card. Report informative-image denominators (two or more compatible, differently scored variants) and image quality on oracle-product versus model-selected-product sets separately. Include stronger current-preference counterfactual pairs and sparse/no-history controls. Masks are deterministic constraints, not learned safety; text image descriptors are not pixel understanding. A/B preference changes must affect appropriate scores/choices where labels require it, not every arbitrary request.

Only after decision quality passes should the new scorer feed the same strict full-grid copy evaluation. Keep fresh title/badge/heading generation, raw rejections, novelty checks, full cardinality and total wall time. The separate main-lane evidence-prefix experiment must retain its existing decisions until this new representation is actually trained/evaluated. Replacing ranking cannot be credited with solving badge truncation, false universal headings or unsupported title attributes.

## Cache keys, invalidation and serving sequence

Raw feature keys bind: exact model/weight revision receipt, tokenizer/template hashes, canonical encoder text/token hash, encoder-format version, pooling/normalization/dtype, extraction batch shape/chunking, and product/variant content version. IDs identify entries but are never predictive inputs. An optional projected-feature key additionally binds the fusion checkpoint. Keep immutable feature artifacts plus receipts/checksums and a roster mapping; reject corrupt, unknown-dimension or stale entries. A candidate-order change updates mapping/order only, not unchanged per-item features.

Request processing is explicit: snapshot current public catalog/compatibility; resolve/cache each required candidate variant; encode the current allowlisted user context once; score the complete roster; apply masks; select/order every required card and its image; then generate all requested copy freshly. Snapshot consistency prevents combining yesterday's embedding with today's facts in one response. Exact price/shipping/stock changes invalidate affected embeddings because those fields occur in the proposed encoder text. Live masks also use the new snapshot immediately; they must never permit an unavailable item while waiting for a rebuild. If a required embedding cannot be rebuilt, expose failure rather than silently falling back to stale scores or dropping a card.

| Control | Required evidence |
| --- | --- |
| Resident model, cold catalog, user A | Every required catalog feature built; actual extraction/load/hash costs included; fresh query encode and scores |
| Resident model, warm catalog, user A | Catalog hits; fresh query encode; same within-design decisions as the matched fresh-feature control |
| Warm catalog, changed user B | Catalog bytes/keys unchanged; new query text/hash, fresh query forward and fresh scores; appropriate labeled preference changes |
| Return to A | Catalog still unchanged; query recomputed; A's scores/ranks/image actions reproduce under fixed numerical configuration; no B-state leakage |
| Changed product facts/price/shipping/stock | Affected variant keys miss/rebuild; other entries hit; current mask/facts govern selection; no stale unavailable product |
| Changed image descriptor/style/facts or removed image | Affected variant invalidated/removed; no old image ID survives by slot position |
| Changed presentation surface/compatibility | Fresh request/masks; cached intrinsic variant features reused only if their catalog content is unchanged |
| Changed model/tokenizer/template/encoder/pooling | All incompatible raw catalog vectors rejected/rebuilt; no stale dimensional reinterpretation |
| Changed fusion head only | Raw vectors may remain valid; projected vectors and all personalized scores recomputed |

Model-cold startup, model-resident/catalog-cold, and catalog-warm/fresh-user timings are separate conditions. Report catalog maintenance cost and misses rather than amortizing them away without disclosure. A full-response cache or reused A scores is not this experiment. Native extraction can show batch/segmentation numerical sensitivity; compare cached bytes exactly with their saved artifacts, and compare fresh controls at matched extraction shape before diagnosing a cache defect. Preserve score margins/ties and mismatches rather than widening tolerances to claim parity.

## Required implementation receipts and stop conditions

Before native extraction, CPU tests should cover canonical input separation, no private/ID/position leakage, cache-key invalidation, immutable A/B/A catalog entries, complete roster scoring/image mapping, deterministic masks/shortfall behavior, real trainable interaction gradients and exact checkpoint reload. Record feature membership, dimensions, finite values and provenance. Then bound native extraction and fresh-user encoding with the same pinned model and an explicitly declared memory guard; no new download, hidden worker or automatic guard increase is part of the design.

Measure catalog extraction/loading/hash verification, fresh user construction/tokenization/prefill, host transfer, CPU fusion/masks/full-grid assembly, generation prefill/decode, validation, complete response and peak memory. A cached-feature head-only timing is not full decision latency, and neither establishes a successful storefront until every required generated field passes. Do not promise a fixed speed factor: this design removes repeated catalog **encoding** from warm scoring, but the new quality, cold extraction cost, fresh-user latency and full-grid copy cost remain unmeasured. Stop promotion if personalized ranking/image quality regresses, invalidation is stale, or full-grid output is incomplete.
