# Fixed paired-copy feature fit: independent CPU audit

The completed six-case comparison has **5/6 accepted original fourteen-field pages with the learned adapter versus 0/6 frozen**, including 3/3 learned test pages. The independent audit found no evidence-integrity issue in 10,616 checks. All 504 full-shape feature archives were streamed and verified against their original native tensor receipts. All twelve saved raw completions reproduce the saved strict validator results exactly. This establishes results for this small synthetic comparison; it does not establish production quality or a complete above-fold payload.

The immutable fit executed at `f5db2b3406b4da0040be497bcacfc8650bb7b8d2`. Its original `results.json` SHA-256 is `7536c474a9e48969b8204e82a23a62e78ceadfdd9d37903fef1bd42035d3690a`. The tracked copy in `semantic_copy_feature_results/v1/results.json` has identical bytes. All 504 tracked JSONL gate receipts equal their original individual gate JSON files.

## State, features and updates

- Every row passed the exact nonzero-B direct-native, auxiliary-native and frozen-feature scalar-loss/A/B-gradient comparison before any optimizer update. Numerical tolerances were not used. The extraction adapter and each case's pristine native source state remain identical across their receipts.
- The archived arrays retain the complete `[1,L,12288]` down input and `[1,L,4096]` residual/base-down outputs. Original tensor dtypes are recorded individually; float32 host storage represents their values exactly. Archive hashes, member names, shapes, dtype receipts, all finite values and native tensor hashes pass for all 504 rows.
- The feature archives and metadata occupy 1,741,824,945 bytes, under the declared 2 GiB limit. One row is loaded at a time under the declared 32 MiB host-feature limit. No collection of 36 hybrid caches is retained for randomized updates.
- The actual fit starts with the original seeded A and zero B. Exactly 504 evaluated updates cover a single seeded global permutation of all 504 distinct case/field entries, with no omitted or repeated entry. The one epoch and final-step checkpoint were fixed before evaluation.
- Only the final layer31 rank4 down-projection adapter has trainable parameters: 65,536 A/B elements. The native run verifies 927 frozen language tensors before/after fitting. Adapter reload reproduces the same-mode feature loss exactly: `0.3103650212287903`.
- Checkpoint SHA-256: `0b0aeda377a9c9ad7eb37b529c139dac84bc5acd4bb559176d40245865ffe1b2`. The checkpoint has only the expected A/B leaves and shapes. Frozen and learned evaluation arms use identical fixture decisions, field caps and pristine state hashes.

The separate learned-checkpoint probe also passes, with zero updates. At the learned A/B values, its recaptured x/h/y tensor receipts equal original archive row0 exactly; direct-native, auxiliary-native, freshly reconstructed-feature and original archived-feature loss/A/B-gradient receipts agree. Fresh source/token identity, source preservation, frozen language hashes, checkpoint identity and source immutability checks pass. That is one representative learned-parameter row, rather than a second 504-row learned-parameter recapture. The earlier inference-mode feature failures remain failures.

## Strict quality and actual fields

Each split has three fixed six-card cases, eighteen titles, six headings and forty-two total required fields. `NONE` is an actual sampled exact sentinel. It fulfils a badge field without generating visible badge wording; supported cards may also validly abstain with `NONE`.

| Split/arm | Valid pages | Accepted titles | Accepted headings | Actual/accepted non-NONE badges | Actual/accepted exact NONE | Fulfilled fields including NONE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dev frozen | 0/3 | 7/18 | 1/6 | 10/10 | 8/8 | 26/42 |
| Dev learned | 2/3 | 18/18 | 5/6 | 12/12 | 6/6 | 41/42 |
| Test frozen | 0/3 | 11/18 | 1/6 | 12/12 | 6/6 | 30/42 |
| Test learned | 3/3 | 18/18 | 6/6 | 12/12 | 6/6 | 42/42 |

Across all six cases, the learned adapter fulfils 83/84 fields: 36 titles, eleven headings, 24 visible badges and twelve exact `NONE` values. Frozen fulfils 56/84 fields, including fourteen `NONE` values; two are allowed abstentions on supported cards. Both arms correctly produce `NONE` for all twelve cards requiring it. All 168 rows finish with the native stop; none is truncated. No raw output is repaired to achieve these counts.

The single learned rejection is dev case `copy-pair-fresh-970546a93dcd488a`, `S1: Structured travel choices`. Its first section has an empty tag intersection. Its three members have `{compact, roomy, structured}`, `{organizer, packable, rolltop}` and `{adjustable, roomy, water-resistant}` respectively. The first card's supported preference does not license a section-wide structured claim. The existing validator rejects that heading and the whole page remains invalid.

One complete accepted learned test output, from `copy-pair-fresh-aa14cc65832e88f1`, is preserved verbatim below:

```text
S1: Everyday travel choices
S2: Everyday travel choices
T1: Qoriven 19L Daypack for ventilated travel
B1: ventilated choice
T2: Zelimar 26L Daypack for water-resistant travel
B2: water-resistant choice
T3: Vesudor 33L Daypack for everyday travel
B3: NONE
T4: Qoriris 13L Daypack for ventilated travel
B4: ventilated choice
T5: Qorimar 20L Daypack for water-resistant travel
B5: water-resistant choice
T6: Tavuven 27L Daypack for everyday travel
B6: NONE
```

For that same case, the frozen heading is `S1: Ventilated and Water-Resistant Options`; both frozen section headings are rejected because neither attribute is universal. Frozen `T3: Vesudor 33L Daypack — adjustable` is also rejected, while the learned fresh wording above is accepted. The learned outputs closely follow the disclosed synthetic teacher grammar and frequently repeat the generic heading. This is evidence of improved binding under that grammar, rather than evidence of broad stylistic diversity.

## Wall time, tokens and memory

The complete extraction/fit/evaluation job takes 779.4716 seconds. Learned evaluation emits 580 content tokens and 664 autoregressive response steps; frozen emits 552 content tokens and 636 steps. Each arm has 84 natural EOS response steps across six pages, recorded separately from content tokens.

The learned valid-page median is **11.3421 seconds**, with a range of **11.0989–11.7254 seconds**. Dev valid-only median is 11.1796 seconds; test valid-only median is 11.4459 seconds. All-attempt medians are 11.2354 seconds frozen and 11.3940 seconds learned. There is no measured latency improvement. Frozen has no valid-page latency sample.

These are `lockedPlanCopyMs` measurements. They include the full ranking-context prefill, common evidence preparation, native cache/hash checks, every copy field and strict assembly/validation. The fixture decisions are locked: fresh-query ranking decision work and browser paint are excluded. They are not end-to-end storefront readiness measurements. The original fourteen fields also omit visible C/body copy, so a complete above-fold claim is unsupported.

Full-fit observed MLX peak is 6,360,122,248 bytes, below the unchanged 7 GiB guard of 7,516,192,768 bytes. The learned snapshot probe peaks at 6,316,935,806 bytes. The guard covers both active allocations and observed cumulative peak. No budget was raised.

The inherited inner `pagePlan.method.headingsAndTitles` string still describes a frozen decoder in both arms. The outer arm label, checkpoint and adapter controls establish that the learned arm enables the trained final-MLP adapter; the inner method string is legacy metadata. Section grouping remains deterministic. Ranking/image quality is unlabeled, and these synthetic dev/test fixtures are now inspected evidence for future attempts.

Machine-readable evidence is in `SEMANTIC_COPY_FEATURE_AUDIT.json` and `SEMANTIC_COPY_FEATURE_FIELD_AUDIT.json`. No model execution, optimizer update, modification to the fit's frozen sources, original-report rewrite or commit was performed by the auditing lane.
