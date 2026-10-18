# Shared immutable evidence-prefix experiment

Prefilling selected evidence once reduced the six-card independent-field attempt from 11.88 to 8.70 seconds in this fixed comparison. The reuse mechanism passed matched native state/token/distribution controls, and cached versus repeated versions of the new layout produced the same fourteen raw field outputs. The page still fails strict quality. This is a measurable reduction in duplicated work, not a usable-page speedup or a promoted generator.

## What was duplicated and changed

The exact b6d2057 baseline has 14 field jobs: 2 headings, 6 titles and 6 badges; it processes 3,437 aggregate row suffix tokens after one 1,446-token shared ranking prefix. Each of six title/badge pairs repeats exactly the same 747–782-character selected-binding tail: identity, tags, requiredobservedtags, badge support, approved titles, style preferences and allowed vocabulary. Those tails are 173–187 standalone tokens each (1,081 extra standalone tokens across the six repeated copies); standalone substring counts overlap other duplication and are not additive native work measurements. Generic framing words are repeated across card prompts as well.

The experiment keeps the same six product/image decisions, deterministic sections, facts, per-field caps, greedy sampler, native256chunking, widththreecontinuous refill and strict validator. One common userturn contains the selected evidence/rules with separately bound per-card allowed words and per-section universal tags. It contains no prewritten generated output. Fourteen short later userturns request independent fields. Both the rankcache and the final evidencecache remain immutable; each lane receives a complete private recurrent/convolution/KV fork. Tokens stay autoregressive within each lane.

| Prompt work after ranking | Logical token positions |
| --- | ---: |
| Existing baseline aggregate field suffixes | 3,437 |
| New common evidence, once | 950 |
| New minimal field suffixes, all 14 | 504 |
| Cached new layout: common + all suffixes | 1,454 |
| Same new layout repeated uncached for every field | 13,804 |

Cached logical suffix work is 57.70% less than the existing baseline. The final evidence prefix has 2,396 tokens, including the unchanged 1,446-token ranking context. Actual GPU work also includes padding, native state traffic, normalization and lookahead; token arithmetic alone is not a speed guarantee. The common append has an explicit 2,048-token bound and total experimental context 4,096; existing serving guards were not weakened. There is no truncation or field/candidate reduction.

## Actual end-to-end attempt comparison

| Arm | Full completion | Shared evidence preparation | Native field prefill/decode | Text tokens / sampled steps inclEOS | Strict valid pages | Accepted new titles / badges / headings |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 11875.3 ms | 0.0 ms | 9418.5 ms | 132 / 142 | 0/1 | 0 / 0 / 0 |
| sharedRepeated | 25778.3 ms | 0.0 ms | 23367.8 ms | 110 / 124 | 0/1 | 3 / 2 / 0 |
| sharedCached | 8699.8 ms | 1348.4 ms | 4931.9 ms | 110 / 124 | 0/1 | 3 / 2 / 0 |
| sharedCachedTrace | 8970.4 ms | 1370.2 ms | 5105.8 ms | 110 / 124 | 0/1 | 3 / 2 / 0 |

Baseline-to-cached full attempt time falls 26.74%, including 1,348.4 ms for evidence preparation. That comparison changes prompt layout, wording and generated length as well as reuse: 142 sampled responses become 124. It cannot isolate caching alone. The identical-new-input repeated-versus-cached control falls 66.25% with identical raw tokens/finishes/quality and all 14full-input hashes equal. Its uncached common-table layout is deliberately more redundant than the original baseline and is an attribution control, not a sensible serving alternative.

Each arm still ranks all six candidates and their twelve product/image actions before copy generation; structured work remains 1.93–1.96 seconds. Decision heads and deterministic assembly are not mistaken for full serving time. Full times include diagnostic complete-state hashing/host transfers and all preparation/generation/validation; model loading and report IO are excluded. Hashing costs can differ with larger source states. One run per inspected case/cell, fixed order and OS/cache warmth prevent a statistical latency claim. No model-cold full request or12/24-card native result is claimed.

## Measured stage attribution

Compared with the prior b6d2057 instrumented six-card baseline, sharedCachedTrace spends 1,370.2 ms preparing the evidence once plus 1,913.8 ms processing minimal field suffixes, versus 6,078.0 ms previously spent in per-field suffix prefill. The combined region is 3,284.0 ms, an observed 45.97% reduction. Active decode/lookahead is 2,578.8 ms versus 2,837.6 ms, with fewer generated steps and larger per-lane history. Field startup is 523.8 ms versus 561.8 ms. These method spans include Python/synchronization and deferred native cache materialization; they are not pure kernel timings.

The new traced native field region totals 5,103.9 ms exclusive, with only about 1–2 ms outside the traced roots. There are 138 native forward lane appearances for 124 responses across 14 rows, preserving one additional startup/lookahead appearance per row. Tracing preserves exact raw output, strict quality and admission/finish schedule relative to the plain cached arm; its full 8.97 seconds is diagnostic, not the speed comparison.

## Quality failure, raw output and no promotion

All three new-layout arms finish every row naturally at EOS; the baseline has four length-truncated badge rows. The new layout nevertheless omits five required labels: heading-2, badge-2, badge-3, badge-4 and badge-5. Three generated titles exactly copy approved wording; the first heading asserts compact even though that section has no universal compact fact. Those errors remain in the raw output and strict page rejection. Three accepted new titles and two accepted badge fields on a rejected page are not a fulfilled-page result.

Complete cached raw output:

```text
S1: Compact daily choice for everyday travel
Simple daily choice for everyday travel
T1: Grove 23L Daypack — compact choice
B1: compact
T2: Cedar 26L Daypack — durable and compact
Durable and compact
T3: River 26L Daypack — durable
Durable
T4: Field 14L Daypack — lightweight
NONE
T5: Ridge 23L Daypack — lightweight
NONE
T6: River 14L Daypack — durable and compact
B6: Durable Compact
```

For example, Ridge 23L changes from the baseline’s unsupported durable title to a factual lightweight title, but it exactly copies an approved title and is rejected for novelty. Correct attributes alone do not satisfy the fresh-wording mandate. The unsupported heading and missing labels are concrete blockers to promotion, not validation thresholds to relax. No label was inserted, output repaired, truncated row rescued, approved title substituted or failed field hidden.

## State correctness, memory and evidence

Before full arms, three actual width-three controls ran:fork, rebuild each lane’s common extension independently from the same immutable ranking cache, then repeat fork. Each returned 19 responses (7 heading, 7 title, 5 badge with natural EOS), 57 total. Complete final evidence hashes matched; all sources stayed unchanged. Every observed returned 248,320-vocabulary distribution was checked finite on host and hashed; hashes,rawtokens,stopIDs,ARcounts and finishes match. Full vectors were not retained, so the independent audit verifies the executed finite checks and hash receipts rather than independently recalculating unseen vectors. Current non-skipped native compatibility receipt was checked against decode/test/runtime sources.

Cached/uncached new-layout token identity holds for every field and includes all chat boundaries. Common evidence is prefetched with add_generation_prompt=False, then only exactdelta chunks of 256 are run through model.model; only complete cache states are materialized. No unused full-logit or hidden tensors were forced. All four strict raw assembly/quality results and the baseline’s exact b6d2057raw replay are independently CPU-audited. Numerical equivalence outside this inspected shape is not claimed.

Global MLX peak:6,366,069,178bytes under the explicit7,516,192,768-byte seven-GiB guard. Per-arm peaks are cumulative. Cached-arm active memory is5,272,398,348bytes versus5,139,884,556in baseline, measured while their different source states remain live; this is not an isolated per-cache or per-field memory slope. Resident load:2863.1ms; bounded process:69300.9ms. Executed commit`a368fe36d4ba3535dc29ed9152b8e1ad80bb6f12`, source SHA-256`2f02403b87fe64658b28de947ee7d1b928f471398cd666b5dfc607742a4e59a3`.

[Raw result](SHARED_EVIDENCE_RESULTS.json), [derived analysis](SHARED_EVIDENCE_ANALYSIS.json), [binding duplication audit](SHARED_EVIDENCE_DUPLICATION.json), and ignored `data/ranker/pretrained/shared-evidence-v1/` retain original outputs,token/state/distribution hashes and trace events. Previous manifests, failed experiments and externalQ001files remain untouched.

## Next bounded work

The experiment establishes that common evidence reuse can remove repeated model work while preserving identical new-layout outputs; it does not establish usable copy or near-instant storefront serving. Keep the original manifest as the best existing quality reference. The next copy gate should address exact field protocol, title novelty and section-wide grounding under the shared-cache architecture, with raw failures and full six-card completion retained. Repeating minor wording variants without these quality gates is not a performance result.

[Independent fusion design](CATALOG_FUSION_DESIGN.md) specifies a separately trained catalog/query representation and 32,786-parameter fresh user-conditioned scorer, full-grid decisions/image choices, held-out evaluation, cold/warm A→B→A and invalidation controls. This new readout needs training and evaluation; cached catalog embeddings do not preserve existing joint-head behavior by assumption. Any fast head-only result must remain separate from all required genuine-copy completion.

The [12/24-card workload](FULL_GRID_WORKLOAD.json) is byte-unchanged. All requested choices and 28/56 fresh field jobs remain in scope. Its 36-candidate requests need 8,211/8,213 ranking tokens and exceed the current ranker guard; no native full-grid quality/memory claim follows. The original near-instant challenge budgets remain unmet. No smaller model, template-only substitution, MTP migration, paid compute, UI change or external runner ideation was used.
