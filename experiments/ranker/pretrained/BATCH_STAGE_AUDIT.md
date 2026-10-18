# Independent batch-stage audit

CPU replay confirms all four matched 512-prefill controls and all nine full attempts completed. Every instrumented 256-prefill attempt exactly matches its uninstrumented partner in sampled tokens, completion, strict quality, admission/refill schedule and complete source-state hashes. All nine full attempts remain invalid pages under the unchanged validator. No result establishes a usable-page speedup.

The audit inspected `BATCH_STAGE_RESULTS.json` and ignored `data/ranker/pretrained/batch-stage-profile-v1` evidence. It launched no native/model work and modified only this report.

## Checks performed

- Recomputed all six published evidence hashes, the executed profiler hash and installed native generation-source hash. The published result without its added evidence map equals the raw `profile.json` exactly.
- Verified the fixed three-case summary and every complete-prefix before/after hash across nine arms.
- Replayed raw assembly and unchanged strict quality for all nine arms using the saved case inputs and rows.
- Recomputed all three plain256/trace256 exact-parity comparisons, quality equality and scheduling equality: admissions, finish order, live width, native response rounds and row autoregressive steps.
- Loaded all twelve matched-control log-probability vectors on CPU. Each has 248,320 finite entries and equals the corresponding `fork-r0` vector exactly. All four controls preserve pristine source hashes and have three exact rebuilt/forked prefixes, seven responses per row and identical tokens/finish reasons: 84 sampled responses total. These are response distributions, not an additional raw-forward alignment test.
- Recomputed category totals from event exclusive times. Each is nonnegative; the exclusive sum equals the sum of the disjoint root `_next` and insertion intervals. Parent and child durations are not added together.
- Counted each UID's appearances in startup/active forward events and matched them to actual response observations. Every UID has exactly one more forward appearance than sampled responses, preserving native startup and terminal lookahead work.

| Cards | Independent rows | Sampled responses, including EOS | Native forward lane appearances | Startup forward calls | Active forward calls | Exclusive native root total | Native timing outside traced roots |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 33 | 36 | 2 | 15 | 1,939.18 ms | 0.70 ms |
| 3 | 7 | 71 | 78 | 6 | 31 | 4,576.36 ms | 1.43 ms |
| 6 | 14 | 142 | 156 | 12 | 60 | 9,573.82 ms | 2.08 ms |

Forward lane appearances count rows processed within calls; they are not the number of whole-model calls. The extra appearance per UID is native lookahead, not another delivered output token. Actual decoded output token counts excluding stop tokens are 30 / 66 / 132 in both plain chunk settings.

## Measured bottleneck

| Cards | Suffix prefill | Last prompt token / first sample | Active decode and lookahead | Suffix share of traced native total | Plain256 external full wall | Plain512 external full wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,146.98 ms | 92.97 ms | 681.32 ms | 59.15% | 4,100.61 ms | 4,024.28 ms |
| 3 | 2,816.19 ms | 268.46 ms | 1,447.51 ms | 61.54% | 6,660.16 ms | 6,528.83 ms |
| 6 | 6,077.99 ms | 561.85 ms | 2,837.65 ms | 63.49% | 11,669.76 ms | 10,973.38 ms |

The common ranking prefix has 1,446 tokens and twelve candidate readouts in these cases; its forward/readout costs approximately 1.92–1.94 seconds per attempt. The independently timed eager prefix-fork materialization is only 0.42 / 1.40 / 3.64 ms in traced arms. Removing that fork alone is not the leading latency opportunity demonstrated here.

Suffix timing includes right-padding/finalization, already-required cache evaluation and consumption of lazy native cache-merge work. It is not a pure attention-kernel measurement. Dispatch categories also include synchronization and memory checks; small dispatch times do not prove that all deferred cache traffic is cheap.

Plain512 finishes 76.33 / 131.33 / 696.38 ms earlier than plain256 in these single pairs. Three-card chunk order reverses, but machine pressure/order effects remain uncontrolled. Chunk settings also change admission timing and, in the six-card case, actual wording. These inspected measurements cannot establish a general speed improvement or increased usable-page throughput.

## Exact 512-prefill output difference

All rows match across chunk sizes for one and three cards. Thirteen of fourteen rows match for six cards. Only `title-5` differs, at zero-based sampled token index 12; both versions stop at EOS with fourteen sampled tokens:

```text
256: T5: Ridge 23L Daypack — durable everyday
512: T5: Ridge 23L Daypack — durable lightweight
```

The changed token is 17248 (`everyday`) versus 28191 (`lightweight`). Selected Ridge 23L Daypack facts are `lightweight` and `roomy`; both versions still assert unsupported `durable`. Identical global quality results must not be described as identical factual meaning or proof of field correctness. Neither raw output was rewritten or rescued.

The six-card response-round count changes from 60 to 58. At 512, `badge-5` and `title-6` enter together; at 256 they enter separately. Final row order and finish order remain the same. The chunk comparison therefore changes native scheduling history as well as prompt segmentation; it does not isolate a single kernel operation.

## Practical architecture implication

Repeated field suffix processing is the main measured native cost. A useful next bounded experiment is to compute ranking as today, append one shared generation-instruction/selected-evidence prefix after the decision, then fork complete native state for short field-specific suffixes. This would change prompt layout explicitly and needs fresh quality, state-isolation and full-latency checks; the current invalid pages cannot justify promotion. Removing repeated instructions/evidence should be tested before investing in a prefix-copy optimization whose directly measured cost is a few milliseconds.

At three products per section, twelve cards require 28 independent fields and twenty-four cards require 56. This report measures at most fourteen fields and width three. Existing fixed budgets and ranking-prefix limits do not establish twelve/twenty-four-card feasibility. A full-grid experiment needs predeclared larger cases, candidate/head-input coverage, output budgets, strict field/page denominators and an explicit memory allowance; extrapolated timings are not measured results.

## Provenance and limits

Process wall time was 86,869.69 ms, including loading, four controls, nine attempts and report work. Matched512 controls account for 16,056.18 ms of their own diagnostic wall timer. Peak MLX allocation was 6,367,123,658 bytes under the declared 7,516,192,768-byte seven-GiB experiment guard. This separate run does not replace the preserved failed six-GiB experiment. Synchronization is explicit to the native generation stream. No unused prefill logits were evaluated by the hooks.

| Evidence | SHA-256 |
| --- | --- |
| Published `BATCH_STAGE_RESULTS.json` at audit time | `751db3762e0d842a7ec7b99cce0c2d5ef1bded9f89efaf6bb938207d143cf77e` |
| Raw `profile.json` | `b4803aa66cc47014ebd6033b1205b854d4f7fb3723b6f1cff022c091d640be52` |
| `case-plan.json` | `5cb4815e44cbeca6c2cfdbf8241d75399fd50903d7d4f02a7864ef741eba30b6` |
| `controls512.json` | `6c83959ae423d2f47e5d4e19701a7a216ef03cc0ee73f4d9574f34f06f01655b` |
| `controls512-logprobs.npz` | `cd2110a42f70d024664396cb787ac2552a3edbd796fb016b1d0009ac38551df7` |
| Executed `batch_stage_profile.py` | `ea5204c508bc7ae50af2e1245967eee6cd76a8562134a69c82fdb3da6c694f65` |
| Installed native `mlx_lm/generate.py` | `9d46803799f99d6fc7c35fff2d4357c79b1a25ae98cc63efeaaebeb50a7b5f33` |

State hashes establish preservation of the inspected complete source state. Matched controls and exact instrumented pairs do not establish universal native cache correctness. Instrumented times include synchronization/Python overhead, and shared state hashing adds observation cost to every full cell. No native runtime, precision, sampler, factuality validator or UI was changed by this audit.
