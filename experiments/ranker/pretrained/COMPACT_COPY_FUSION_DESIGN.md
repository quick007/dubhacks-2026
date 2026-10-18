# Compact selected-evidence copy after catalog/request fusion

The proposed boundary keeps decisions and copy on the same resident quantized 9B object while giving copy a fresh, selected-evidence cache. Catalog variant features may be reused from verified base-backbone feature receipts; every request still gets a fresh observable query encoding, CPU fusion/scoring, current constraint/image checks and deterministic grouping. Copy then starts from empty native caches with public user evidence and the selected facts. No hidden ranking prompt, catalog/query KV history or recurrent state is trimmed, transplanted or spliced into the copy cache.

This note and its local-tokenizer inventory are CPU-only design work. No model/native execution, adapter fit, generated quality or speed result is supplied here. `FULL_GRID_WORKLOAD.json` remains unchanged at SHA-256 `c8f4e5c07fb74ffd0d3deba2f8e71aedff9d5a133e8f41ee52fa804316f71ef4`.

## Representation and model boundary

Raw catalog features and fresh query features require the new catalog/request fusion checkpoint trained for that representation. Existing joint-query decision heads cannot be applied to those raw features merely because their dimensions match. Raw encoding runs with the copy adapter disabled and under the exact model/tokenizer/pooling/normalization receipts used to train the new scorer. The copy branch enables the existing learned final-layer down-projection adapter only for genuine new wording. Catalog-feature provenance remains independent of the copy adapter; no adapted representation is silently substituted into frozen decision features.

The candidate copy serialization retains every allowlisted `clean_product` fact for each selected product and every `variant_text` fact for its chosen image variant, plus every approved title for novelty exclusion. Product/image IDs stay in the trusted scheduler mapping outside model text. Identity is written once; approved titles are encoded as verified exact identity-plus-suffix values, with standalone literal fallback for titles without that prefix. The chosen image description uses an exactly reversible identity placeholder. The inventory mechanically checks full public product/selected-image reconstruction and all approved titles. No description, price, stock, shipping value, tag or chosen-image fact is silently removed to meet a token cap.

Observable context, cutoff-valid history and supported tags remain explicit. Support is serialized once and applies to both the bound title and optional badge; absent support requires the model's exact `NONE`, and support permits abstention. Sections are deterministic groups of three with exact member indices and their true tag intersection. Their grouping/count/evidence projection remain deterministic; only fresh heading text is generated. Host validation still has the complete public catalog, including unselected products, for identity ambiguity, foreign names, factuality, novelty and support checks. More supplied public facts do not relax the existing forbidden-claim policy.

The exact messages and per-field token/hash receipts are saved in `COMPACT_COPY_TOKEN_INVENTORY.json`, SHA-256 `2be6503f9decf6f9633f758c490df0aa568d1aca1129cb1543fca2a09f8db591`. It uses the existing local tokenizer at model revision `8b2b98c00a6b4d291155e4890773ca8f769aee53`, tokenizer SHA `87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4` and template SHA `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`. Thinking is disabled explicitly and cached chat boundaries must extend exactly.

## Fixed token inventory

The unchanged full-grid packet has no learned selected roster. Sizing therefore uses one explicit fixture rule: first eligible public candidate order, first compatible image respecting intrinsic surface restrictions, then groups of three. Six-card sizing derives from the twelve-card workload's request/pool. These are sizing fixtures, not trained rankings, heldout quality evidence or serving selections; actual trained selections require their own preflight. Profiles A/B retain their different fresh preferences. There was no prompt sweep or generated-outcome selection.

| Cards | Required jobs / sampled cap | Fresh raw query A/B | All-selected common A/B | Maximum full prompt + output cap A/B |
| --- | ---: | ---: | ---: | ---: |
| 6 | 14 / 224 | 82 / 84 | 1,549 / 1,561 | 1,609 / 1,621 |
| 12 | 28 / 448 | 82 / 84 | 2,570 / 2,588 | 2,631 / 2,649 |
| 24 | 56 / 896 | 82 / 84 | 4,630 / 4,667 | 4,691 / 4,728 |

Fresh query counts use the existing plain raw-request encoder text, not a chat generation template. The old all-candidate joint ranking prompt for this 36-candidate packet is 8,211/8,213 tokens, independent of whether twelve or twenty-four are requested. It is removed from copy context entirely. The roughly 1,600-token ranking prefix of the smaller current training cases should not be extrapolated to this pool.

The all-selected copy layout fits the 4,096-token context guard at six/twelve cards but fails it at twenty-four. This result does not authorize truncation or raising a cap. Fresh-prefix construction would also be a new entry point: the existing shared-evidence append helper retains its separate 2,048-token append guard and must not be silently repurposed for a 2,570-token fresh prefix.

## Branch-local architectural follow-on

One separately requested architectural follow-on shares the observable-user/rules root, then creates complete native fact branches for each selected card and each three-member section. A card branch holds its entire relevant selected public record, approved-title exclusions and support; its complete cache can be forked independently for title and badge. A section branch contains all three complete member records and their exact universal tags, then forks its heading job. No field needs facts from all twenty-four competing products. All fifty-six fields remain required at twenty-four cards.

For a fair CPU inventory, this follow-on keeps the same fixed rules and complete observable history as the all-selected layout. The complete generation root is **515/517 tokens**, rather than approximately 100; only the separately encoded raw request query is 82/84. Card fact appends are 170–190 tokens; section fact appends are 514–554. No alternative prompt wording was tuned against model output.

| Cards | Card / section fact branches | Largest prompt + cap A/B | Total root + fact appends + field suffixes A/B | All-selected total A/B |
| --- | ---: | ---: | ---: | ---: |
| 6 | 6 / 2 | 1,098 / 1,104 | 3,143 / 3,165 | 2,053 / 2,065 |
| 12 | 12 / 4 | 1,098 / 1,104 | 5,756 / 5,790 | 3,584 / 3,602 |
| 24 | 24 / 8 | 1,117 / 1,123 | 11,021 / 11,093 | 6,676 / 6,713 |

Totals count each card fact prefill once for its title/badge pair and each section prefill once. Without complete-card-cache reuse, follow-on totals rise to 4,209/4,241, 7,876/7,926 and 15,258/15,365 respectively. These are exact tokenizer-work inventories, not latency estimates; they exclude cache-copy/materialization cost and decode steps. Branch-local context fits 4,096 at every requested size while increasing aggregate prefill work. Shorter attention contexts and less cross-product interference are hypotheses requiring measurements.

The proposed native scheduler has six active field lanes and continuous refill, with ordinary token autoregression inside each field. Prepare fact branches lazily, count preparation and live full caches against the admission/memory bound, fork only complete states and release a card fact cache after both field branches own their clones. Do not retain twenty-four prepared native card caches waiting for slots. Keep one immutable root; never trim a recurrence, merge unrelated history or splice a card suffix into a section cache. Width six is a proposed measured arm, not established seven-GiB feasibility.

## Required comparisons and timing

This is a new copy input distribution: the current adapter was fitted on global evidence plus the joint ranking prefix. It may not transfer to selected-only or card/section-local evidence. Freeze one candidate protocol and compare adapter-disabled and adapter-enabled fresh generation with exactly the same selected plan, facts, width and caps. Retain the current six-card global-context result as separately identified reference; use a matched six-card comparison to isolate the new context boundary. Twelve/twenty-four-card tests must preserve complete grids and denominators. If a branch-local adapter fit is later needed, give it a separate fixed training plan and fresh prospective quality cases; do not tune on the already observed holdouts.

Before timing, run cache clone/rebuild/repeat controls with the nonzero learned adapter, identical fixed native input shapes, finite distributions and exact complete-state/source hashes for the new root/card/section seams. Compare a fresh branch prefill against root-cache cloning plus the exact same turns. Preserve raw sampled IDs/EOS, field finishes and unsupported/malformed output. Scheduler labels remain trusted transport; they contribute no generated words and do not repair values.

Warm full completion includes fresh query encoding, verified feature lookup/loading, CPU fusion, constraints/image mapping, deterministic grouping, selected-evidence serialization, **fresh copy-root prefill**, every fact-branch prefill, cache cloning, field suffix prefill, all fresh title/badge/heading decoding, strict validation and response serialization. Include raw catalog extraction/missing-feature work for resident cache-cold requests and model load for model-cold requests; report all three conditions separately. Feature-cache hits do not make fresh query or copy-context work free.

Use the unchanged full-grid conditions: profile A→B→A, current price/stock invalidation, resident cold/warm and model cold. Raw-feature keys must bind current public facts and trained-fusion provenance; stale catalog rows cannot silently survive a changed price/availability snapshot. A newly conditioned copy root must not reuse a previous user's/request's native state. Report selection/image quality separately from copy quality and full completion; a fast decision, first field, teacher-fit loss or failed page is not a usable-grid speedup.

Current page-plan/decoder serving guards still accept only 1/3/6 cards. Twelve/twenty-four execution requires an explicit versioned structural/validation extension while preserving old behavior; this sizing note does not bypass those guards or change them. No UI, shopper harness, paid compute, external integration or Perceptron work is included.
