# Bounded same-backbone copy latency proposal

The next useful experiment is fresh C supervision plus shared exact query/card fact prefixes. The saved evidence does not support spending effort on Python cache-copy micro-optimization, moving to a larger model, removing visible fields, supplying template text, or weakening factual validation. This report is CPU/source analysis only: no native call, new fit, model/source edit or new inference result.

## Measured cost, and the attribution limit

The completed three-card endpoint generated the same ten fields at every width: S1 and T/B/C for each of three cards, total cap 208 sampled tokens. It accepted 0/36 complete views, so all times below are attempt completion, never useful-view latency. Each row pools the same six requests under both frozen/paired-STB modes (twelve attempts), with one observation per shape:

| Native active width | Cache-copy/materialization median | Boundary preparation median | Combined native suffix prefill/decode median | Continuous decode wall median | Validated/serialized attempt median | Returned rounds median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.76 ms | 17.38 ms | 10482.88 ms | 10515.41 ms | 11462.24 ms | 106.5 |
| 3 | 2.53 ms | 12.34 ms | 8884.51 ms | 8905.17 ms | 9741.53 ms | 46.0 |
| 6 | 2.26 ms | 10.38 ms | 7836.27 ms | 7853.75 ms | 8543.95 ms | 28.0 |

These component medians are separately calculated and must not be summed. Median sampled row AR steps stay about 106.5–108. The generator's returned-round count is not its complete forward/kernel count. Fewer rounds give only a modest total reduction because each round may perform prompt processing, cache merge/extract and generation work.

The saved native timer combines prefill and decode; per-step synchronization historically uses the default stream, while the completed outer endpoint explicitly fences the actual generation stream. Consequently there is **no defensible exact millisecond split** between admission/prefill and AR compute in these receipts. The outer actual payload completion is the trustworthy latency endpoint. Do not report all native time as prefill or infer per-token kernel latency by dividing by returned rounds.

The earlier fixed six-card STB-only comparison independently points to the same cost region: width-six combined generation 8900.2 ms, cache copy 3.9 ms, boundary preparation 36.2 ms, validation 2.5 ms, total 11034.6 ms. Its fourteen fields omit visible C copy and all factorized scale pages fail, so it is neither a six-card complete-body baseline nor a usable-page result. `DECOMPOSED_COST_AUDIT.md` preserves the source/case/denominator distinctions and comparisons with manifest.

The active store bridge's recorded 625.1 ms is cold CPU archive/source/81-asset verification plus 195 candidate text preparations, not inference. Perform immutable catalog validation/preparation on startup/version change and disclose it separately; a warm request must still fail stale through a declared current-version gate. Do not charge a cold catalog build to one method and omit it from another, or invent a cheap warm path before implementing/measuring it.

## Exact-token saving available before prompt changes

Using the pinned local tokenizer only, this review split each existing job at its already-present **root + bound public-fact user message**. Every full prompt begins with that exact token prefix. No message text, field, evidence fact, output cap or generation target changed.

Across the six saved three-card cases:

- Existing root: 367–370 tokens, already shared once per request.
- Existing repeated fact-plus-field suffix volume: 2562–2626 tokens.
- Prefill each unique bound fact message once: 1100–1132 tokens, including the section's full member evidence and each card's own facts.
- The ten remaining independent field requests are exactly 36 tokens each: 360 tokens total.
- Proposed fact-cache append volume: 1460–1492 tokens, saving 1102–1134 repeated input tokens, about **43%**, before any content compression.

This is a CPU token-volume result, not a native speedup estimate. Longer-lived exact fact caches consume additional recurrent/KV memory and unequal card-prefix lengths change native cache merging; those costs may offset some saving. A 43% input-token reduction does not promise a 43% latency reduction.

## Shortest safe cache topology to measure

1. Fresh observable query/rules root under the actual adapter mode/checkpoint. Keep ranking and copy evidence semantics separate where their prompts differ; a cached joint user/catalog feature is not an equivalent prompt cache.
2. Append each selected card's complete allowed public facts once to that root. Snapshot **all** recurrent, convolution and KV state with exact materialization; do not trim, splice or borrow a prefix from another adapter/request. The initial low-risk comparison can preserve the exact existing fact message, proving reuse before changing wording.
3. Fork that card's exact fact state independently for its T, B and C short requests. Generated siblings never enter another field's evidence. The heading gets only its declared all-member/universal evidence under its own exact prefix. Any later heading/evidence compression is a separately labeled prompt change.
4. Admit at most the declared width and release a fact state after all its required child fields finish. Preserve continuous refill if supported. Across-card mixed prefixes require new actual-width controls; existing equal-root cache controls do not establish unequal-prefix merge correctness. Alternatively start with same-card T/B/C lanes at width three and disclose the scheduling change.
5. Same-width fork versus independently rebuilt prefix versus repeat controls must cover actual 1/3 lanes (and six if six is selected), both actual adapter modes, heading/title/badge/body, full state hashes and sampled/logprob receipts. Preserve strict mismatches. Cross-width numerical wording differences are a semantic comparison, not themselves a state defect. Guard both active and cumulative peak memory; do not raise a stopped worker's limit silently.

Fresh C training is required before promoting this topology. The completed adapter has STB supervision only; body EOS/cap success did not provide factual bodies. The existing feature-fit helper's 256-token trainable suffix gate cannot simply admit a roughly 601-token heading/card evidence suffix. A valid route is to append full facts into the exact mode's cache first, then expose the short field/teacher continuation to training, with a new final-down adapter/cache/gradient proof. The training and serving token streams must agree, and teacher text remains supervision only. The new helpers/data owned by other lanes need their own review/freeze; this report implements none of them.

## Minimum useful-quality decision gate

Use one small, prospectively fixed six-request packet, organized as three explicit preference-intervention pairs on a declared current catalog. Choose known facts and soft preferences so at least six products genuinely remain eligible for the six-card endpoint; do not shrink to two cards or backfill hard failures. Keep every original candidate in ranking and retain all tie-aware valid action sets. Unknown physical/stock/size requirements must follow a declared abstention contract, not become invented facts.

Suggested pilot gate, to freeze before inference:

- **Hard compliance:** every evaluated selected action satisfies all known typed hard constraints; no invented product, image, price, stock or delivery fact. Any hard violation blocks promotion.
- **Minimum personalization:** both requests correct in at least two of the three predeclared preference pairs, with strictly more pairwise successes than the unconditioned prior. Count product and image preference objectives separately when present. This is a minimum engineering signal, not a generalization confidence claim.
- **Useful fresh copy:** at least four of six complete three-card payloads pass the prospectively frozen strict factual/protocol/novelty policy, including all required C fields. Require a complete accepted result in each tested semantic family. Preserve every failure; do not repair raw values, inject NONE, substitute a catalog string or use teacher templates at runtime. Fresh titles must follow the chosen prospectively versioned identity/subtitle policy; never retro-rescore old pass rates.
- **Six-card completion:** retain a separate six-case denominator for all twenty required fields: two headings and T/B/C for six cards. Under the existing caps that would be 416 total sampled tokens, a prospective endpoint rather than the old fourteen-field/224-cap run. Report its complete valid payload rate separately; no three-card acceptance implies full-six validity. A six-card benchmark is not the full sixteen-product storefront claim.

The four-of-six and two-of-three thresholds are proposed bounded engineering gates, not measured outcomes or a reason to accept unsupported copy. Do not keep growing this packet or chase perfect synthetic accuracy before measuring the declared implementation. The final consumer still receives only fully validated complete payloads; invalid attempts remain explicit failures in the fixed denominator.

## Timing endpoint and smallest isolated comparison

The primary above-fold metric is actual **complete validated serialized three-card payload handoff**, including fresh query encoding/scoring and deterministic planning/guards, copy root/fact preparation, all ten field completions, strict validation and exact serialization. Catalog cold preparation and model load are disclosed separately; warm cache contents/version/checkpoint must be identical across arms. Browser asset download/render/paint and conversion remain unmeasured. A locked-plan experiment can measure only copy-stage handoff and must keep that label.

Full-six readiness is the corresponding actual twenty-field validated/serialized payload receipt. Below-fold failure cannot erase a separately completed valid three-card payload, but it must fail the full-six denominator. In a combined progressive producer, record the three-card receipt as soon as **all** its required fields and metadata are ready and validated; do not wait for lower fields unnecessarily, and do not call first token or last-token receipt a completed view.

After one fixed C-trained checkpoint and cache controls pass, the smallest isolated cache comparison is six cases × two topologies at the same declared width/checkpoint/prompts: twelve attempts. Compare the current repeated-fact layout against exact shared fact caches before stacking prompt compression or new batch-width searches. If the new producer's prompts necessarily differ, label the result a combined layout/supervision change and do not attribute all gain to caching. A separate width-one versus width-three comparison can establish batching after the topology is sound; there is no need to repeat a full thirty-six-arm search by default.

Report all-attempt and valid-only times, fixed denominators, per-case paired comparisons when both payloads pass, text-token/AR counts, native rounds/admissions, root/fact-prefix tokens, shared/suffix volume, serialization receipt and active/cumulative memory. A faster invalid attempt does not count as a useful-view win. Use directly synchronized root/fact-prefill phase clocks and the actual generation stream if adding attribution instrumentation; verify any instrumentation preserves output/quality and do not sum overlapping internal timers. No sub-two-second guarantee or stable throughput estimate is justified by these saved runs.

Evidence: immutable `above-fold-copy-v1/results.json` SHA-256 `330ecbe9fa68dff2dec2fdc5aec6a3c21a1c64afbc4f9d8c297df3911f90b1df`, its frozen `BENCHMARK_PACKET_V1.json`, existing `BENCHMARK_AUDIT.json/.md`, `DECOMPOSED_COST_AUDIT.md`, `parallel_decode.py`, `branch_local_copy.py`, `semantic_copy_feature_fit.py`, installed MLX/MLX-LM stream/cache source and the verified store bridge receipt. All numbers above are saved measurements or explicitly labeled CPU-only token accounting/proposals.
