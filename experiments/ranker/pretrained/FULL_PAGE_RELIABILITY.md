# Bounded full-page reliability comparison

This single prompt experiment preserves the successful [manifest baseline](GENERATION_DIAGNOSIS.md) (7/12 evidence pages, 0/10 cache pages) and all saved outputs/checkpoints. The intervention is a locked public evidence capsule: section attributes are the intersection across every member, and each card's identity, factual tags and observed-evidence subset are explicit. Product selection is locked; the decoder may not replace a poorly ranked item with a preferred different item. Numeric catalog identity may be copied exactly, but new measurements/claims are prohibited. Titles/headings still require genuine fresh generation, not prewritten wording.

An independent section audit distinguishes real universal-claim failures from false positives. In the prior ten cache manifest attempts, four repeated A responses fail lexical `short` and missing `compact` emphasis, while six contain genuine group/binding errors. For example, Grove has water-resistant/compact, Cedar durable/compact, and River durable/organizer: no tag is universal, so a heading asserting compact/durable for all is false. The second six-card section mixes lightweight/roomy Field/Ridge with durable/compact River, again leaving no universal tags. Changed-profile B still selects River 26L despite lightweight/roomy preferences; faithful text cannot repair that ranking error.

Two separate single-product evidence headings naming actual **Field 20L Daypack** are factual but rejected by the generic section proxy's missing identity/numeric vocabulary. Those remain rejected under the unchanged validator; they are not labeled hallucinations. Safe plural/framing lexical gaps remain separate from true unsupported attributes. No factual vocabulary, novelty, truncation or page-completion check is relaxed.

The fixed roster retains all ten cache conditions plus all twelve evidence-plan cases. Fresh manifest and locked-capsule arms alternate order, use the same selected product/image IDs, frozen 9B model, trained heads, native exact-prefix layouts, cold/warm conditions, and greedy 64/128/224-token budgets for 1/3/6 cards. Every arm includes complete ranking, cache build/materialization where applicable, fresh generation and validation. The original manifest outputs must replay exactly; changed decisions abort the comparison.

```sh
.tools/pretrained-venv/bin/python -m unittest experiments.ranker.pretrained.test_full_page_reliability -v
.tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.full_page_reliability --model /absolute/verified/pinned/snapshot
```

Four offline tests verify universal intersection versus union, locked identity/support, unknown preferences, and unchanged factual/novelty rejection. Actual paired model results are recorded in `FULL_PAGE_RELIABILITY_RESULTS.json` and ignored `data/ranker/pretrained/full-page-reliability-v1/benchmark.json`. Section grouping and evidence projection remain deterministic policies, not learned layout. Cases were inspected earlier; this is a reliability/mechanism test, not untouched generalization or human conversion evidence.

A separate performance worker owns an exact-prefix/token/stage-cost experiment. The catalog-cache path computes user-first ranking state, discards it, and separately prepares catalog-first decoder state. The existing evidence path already reuses complete ranking state once. Replacing one prefix with the other changes the causal input and is not a cache-only equivalent optimization; independent recurrent/conv/KV snapshots cannot be spliced. Native batching is a separate next implementation comparison; it does not change this completed result. MTP remains an unimplemented compatibility proposal. No new runtime/model was installed.

## Measured result

The locked capsule is not promoted. All 22 manifest baseline texts replay exactly, selections/images are unchanged, and both arms stop with exact field order in all 22 attempts. The strict validator is unchanged.

| Metric | Manifest baseline | Locked evidence capsule |
| --- | --- | --- |
| Complete valid pages, all cases | 7/22 | 0/22 |
| Complete valid pages, evidence subset | 7/12 | 0/12 |
| Complete valid pages, cache subset | 0/10 | 0/10 |
| Accepted fresh titles | 34/43 | 24/43 |
| Accepted generated headings | 13/25 | 4/25 |
| Accepted supported badges | 33 | 33 |
| Full attempt latency p50/p95 | 3677/8353 ms | 4221/9526 ms |
| Valid-page latency p50/p95 | 3216/4803 ms (7) | No valid page |

Resident model loading is separate; per-attempt times include ranking, native cache preparation where applicable, generation and validation. Peak MLX allocation was 6,344,673,924 bytes against the unchanged 6,442,450,944-byte budget. These are single attempts per fixed case, not a deployment latency estimate.

The independent locked-heading audit finds 4 accepted fields, 10 factual single-product headings rejected by the generic identity/numeric proxy, 9 genuinely nonuniversal or misbound section headings, and 2 other lexical/policy rejections. For example, “The River 26L Daypack is a durable, organizer backpack.” is factual for the selected River, but the generic section proxy rejects its brand/capacity. Conversely, compact/durable claims over a mixed Grove/Cedar/River section remain false. Identity binding fixes that one changed-profile heading without fixing selection or complete-page reliability. No field is rescued or validator vocabulary expanded.

The prompt introduced a conflict between globally permitted selected identity and the section validator's generic identity-free vocabulary. It also sometimes copied approved titles. This negative result does not establish that explicit evidence is harmful in general; the capsule serialization, finite validator and inspected synthetic cases are specific limitations. The preserved manifest remains the better measured baseline. Next, independent-field native batching must compare identical per-field prompts sequentially and concurrently, and report complete valid page outcomes, token parity, cache-copy cost, early completion, memory and wall time.
