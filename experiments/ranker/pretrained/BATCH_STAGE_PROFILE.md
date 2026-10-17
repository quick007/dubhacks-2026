# Native batch stage profile and chunk comparison

The current independent-field path spends about 59–63% of synchronized native time processing repeated suffix prompts and their consuming cache operations, 30–35% advancing active tokens and lookahead, and 5–6% starting fields. This is a larger architectural bottleneck than Python queueing, explicit prefix forking or validation. Doubling the native prefill chunk size helps only modestly in these samples and does not produce a valid page. The nine-billion-parameter per-field design has no credible measured route to near-instant complete-grid fresh copy through batch-width/chunk tuning alone.

## Actual measured stages

Fixed width three; unchanged compact-heading/title/badge prompts, per-field caps, greedy sampler, frozen model/heads and strict validator. Each case runs plain256, synchronized trace256 and plain512; the three-card case reverses chunk order. No manifest source/output or external Q001 files were modified. All nine attempts completed; all three trace pairs matched exact raw sampled IDs/EOS, text, finish, strict quality, product/image IDs, source native state, admission roster, finish order, returned rounds and emitting widths. All plain256 outputs replayed their saved historical counterparts.

| Cards | Suffix prefill | Active decode + lookahead | Field startup | Other native stages | Native total | Shared rank/structure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1147.0 ms (59.1%) | 681.3 ms (35.1%) | 93.0 ms | 17.9 ms | 1939.2 ms | 1933.9 ms |
| 3 | 2816.2 ms (61.5%) | 1447.5 ms (31.6%) | 268.5 ms | 44.2 ms | 4576.4 ms | 1942.6 ms |
| 6 | 6078.0 ms (63.5%) | 2837.6 ms (29.6%) | 561.8 ms | 96.3 ms | 9573.8 ms | 1950.0 ms |

Timers explicitly synchronize the native generator stream before/after each hooked method. Nested intervals use exclusive subtraction; root sums reconcile. No full prefill vocabulary logits or additional cache graphs were forced. These spans include Python, synchronization and memory-check overhead. Labels ending in Host denote dispatch boundaries, not pure CPU measurements: native cache merge/extend/filter operations are lazy, so their later materialization is charged to the consuming suffix/decode stage. The experiment cannot separately isolate memory traffic from compute inside prefill.

Explicit source-prefix fork/materialization takes 0.42/1.40/3.64 ms; strict assembly 0.65/1.12/1.72 ms. Those operations cannot explain seconds. Native split/merge/extract/filter and scheduling together are tens of milliseconds in the traced arms, though deferred work is included in consuming stages. Field prompt setup/tokenization takes 3.43/14.01/34.96 ms.

Native startup performs an initial forward and every returned response computes another lookahead before EOS/length filtering. Actual per-row native forward appearances are 36/78/156 versus 33/71/142 returned sampled steps across 3/7/14 rows: exactly one extra forward per row. Text tokens excluding EOS are 30/66/132. The terminal unused lookahead remains part of measured work.

## Actual chunk comparison

| Cards | Plain256 complete | Plain512 complete | Observed reduction | Raw tokens identical | Valid pages at256/512 |
| --- | ---: | ---: | ---: | --- | --- |
| 1 | 4100.6 ms | 4024.3 ms | 1.86% | yes | 0/0 |
| 3 | 6660.2 ms | 6528.8 ms | 1.97% | yes | 0/0 |
| 6 | 11669.8 ms | 10973.4 ms | 5.97% | no | 0/0 |

These are one attempt per case/cell, with pressure/warmth/order variation, not a repeatable speedup estimate or quality promotion. Complete wall times include identical diagnostic source-state hashing/host transfers and final native-stream drain; model load and report IO are excluded. Trace complete times are 4.14/6.80/11.78 seconds and are diagnostic, not the speed comparison. Their extra synchronization was not treated as a serving optimization.

At six cards, title-5 changes with chunk segmentation:

```text
T5: Ridge 23L Daypack — durable everyday
T5: Ridge 23L Daypack — durable lightweight
```

Both assert unsupported durable for the selected lightweight/roomy Ridge item. Raw wording remains preserved and the strict quality result is unchanged. The one-card heading improperly contains product identity/numerals; multi-card attempts also contain truncated fields and repeated labels. No invalid attempt is counted as a usable-page speedup. All three cells retain fixed denominators of three: zero accepted pages per cell; one accepted title in the one-card case, zero accepted titles/headings/badges in the globally unfinished multi-card pages. The original manifest remains the best existing quality baseline.

## Narrow512 controls and provenance

Before comparing chunks, four actual width-three controls ran at512 under the separately declared seven-GiB guard: forked complete state versus independently rebuilt identical prefix, each repeated once on the preserved Meadow diagnostic. All branches matched pristine recurrent/convolution/KV source hashes, the source remained unchanged, seven responses per each of three lanes matched, and all twelve final-position full logprob vectors matched exactly and were finite. This is 84 sampled responses and twelve saved vectors, not a universal512/cache proof. Actual v5 eighteen-arm256 controls and non-skipped tiny native compatibility receipts were also checked against current runtime/source hashes. Prior six-GiB failures remain untouched.

Executed commit `113595b4508ce5c3b3299bfffa630b3e616a05a4`; profiler SHA-256 `ea5204c508bc7ae50af2e1245967eee6cd76a8562134a69c82fdb3da6c694f65`; installed native source SHA-256 `9d46803799f99d6fc7c35fff2d4357c79b1a25ae98cc63efeaaebeb50a7b5f33`. Resident model load 2017.4 ms, controls 16056.2 ms, entire bounded process 86869.7 ms. Global peak 6,367,123,658 bytes under 7,516,192,768-byte guard. Per-arm peak values are cumulative. [Raw report](BATCH_STAGE_RESULTS.json), [derived analysis](BATCH_STAGE_ANALYSIS.json), ignored `data/ranker/pretrained/batch-stage-profile-v1/` and NPZ control vectors preserve actual outputs/events/hashes. An independent CPU audit verifies all nine raw quality replays and additive trace sums.

## Full grid and architecture decision

[Full-grid estimate](GRID_SCALING_ESTIMATE.md) projects tens of seconds for12/24 cards with the current design; these are rough two-point extrapolations, not measured full-grid latency. [Prepared workload](FULL_GRID_WORKLOAD.json) now contains four actual public diagnostic input cases:12/24 requested cards under two preference profiles, each choosing from36 unique eligible products with genuine compatible image-caption variants. It records 28/56 required independent fields and proposed cap totals448/896. Nothing is silently removed. No12/24 native generation or quality outcome is claimed.

CPU tokenization measures 8,211/8,213 required ranking-prefix tokens and72 product/image readouts for the36-candidate input. The existing1,536-token cap rejects it. Raising a count enum alone also breaks the current two-section grouping and does not create a trained larger-context representation. The packet therefore records unsupported serving guards rather than bypassing them. It specifies model-cold, resident-cache-cold, resident-cache-warm, A→B fresh conditioning, return-to-A and catalog invalidation conditions.

Challenge targets in the packet are warm complete-grid p50≤500 ms and model-cold full completion≤2,500 ms, including every required fresh copy field and strict validation. These are explicit evaluation targets, not a user-agreed SLA or demonstrated feasibility. Decision-only latency, first-card delivery and full fresh-copy completion must be reported separately. Current9B measurements miss the complete-copy target by a large margin. Arbitrary generated strings are a different workload from typed decision-only serving; no vendor latency is treated as an equal-workload comparison.

The next bounded architecture experiment should test reusable catalog representations plus fresh profile-conditioned scoring for every candidate/product-image action, with a constrained complete-grid assignment. This directly targets the uncached1.9-second ranking pass and the8.2k-token future input, but it needs a newly trained/evaluated conditioning/readout path: current jointly contextualized head features cannot be reused unchanged across shoppers. Measure cold build, warm A, changed B, return A and catalog invalidation, with held-out ranking/selection quality and every requested grid choice present. Do not claim decision speed as completed fresh-copy latency.

For the genuine-copy portion, prefill coherent selected-plan/rules/evidence once, then fork exact complete native state into short independent field instructions; use local universal section evidence for headings. This targets the measured repeated-suffix cost without generating templates or reducing cards. First measure the existing six-card diagnostic under identical facts, caps, strict quality and full state isolation; then use the prepared full-grid protocol after count/context/head compatibility is implemented. A faster plan alone leaves several seconds of9B active decode, so500 ms complete-grid arbitrary copy likely requires an explicitly chosen serving architecture/hardware change or narrower optional-copy contract. No smaller model, paid compute, MTP migration, UI change or template-only substitution was made.
