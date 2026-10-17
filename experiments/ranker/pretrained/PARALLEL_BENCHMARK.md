# Bounded independent-field native-batch comparison

The independent-field implementation ran on the pinned 9B model. Its measured comparison stopped at an exact-token parity guard on the third case; it is not promoted. The existing manifest baseline remains preserved: 7/22 complete valid pages overall, including 7/12 evidence-source pages, with 22/22 raw baseline replay in the last comparison. The locked-evidence serialization is not promoted.

`parallel_benchmark.py` freezes all twelve existing evidence-source cases, including the fixed 1/3/6-card scale cases. Each measured arm begins before the same user-first ranking prefill. Manifest whole-page generation is compared separately with factorized sequential generation and native batch widths 2/3/6. Identical factorized prompts and per-field budgets isolate concurrency; whole-page versus factorized prompts change another factor. Total output caps remain 64/128/224. Catalog-first cache conditions are excluded because substituting their state would change the ordered context.

The runner checks the exact unskipped native test name/result, runtime versions and decode/test source hashes in an actual passing tiny native compatibility receipt at `data/ranker/pretrained/parallel-decode-v1/native-compatibility.json` before constructing the pretrained backbone. The actual native receipt passed and was verified against current source/runtime hashes before the pretrained run. A metadata-only or fabricated pass must never replace the native cache tests. The expected gate covers complete recurrent/convolution/KV forks, branch independence, unequal suffixes, sequential-versus-batch token parity, early EOS and row compaction. The six-GiB process cap remains unchanged.

Each factorized arm must include state materialization, suffix prefill, native decoding, raw assembly and unchanged strict validation in complete latency. Every generated row's exact token IDs, stop reason, prompt hash and budget are compared with sequential output. Failed and unattempted arms remain in the fixed twelve-case denominator; rejected pages cannot count as usable-page speedups. The runner stops after any native failure or memory error rather than raising the allowance. Increasing width order introduces a disclosed warmth/order bias. These are inspected synthetic cases, not human conversion evidence or an untouched generalization benchmark.

Four CPU checks pass for raw token/prompt/budget/prefix comparison, nonempty unique row sets, executed-native receipt validation tied to current runtime/source hashes, and denominators retaining failed, unattempted and rejected pages:

```sh
.tools/pretrained-venv/bin/python -m unittest experiments.ranker.pretrained.test_parallel_benchmark -v
```

The following gated model command was actually executed:

```sh
.tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.parallel_benchmark --model /absolute/verified/pinned/snapshot
```

## Approval history and actual execution

Automatic review denied the worker's edit/test operation, then denied the single retry after new parent-supplied user transcript evidence. Neither attempt executed. The rejected action was to update `parallel_decode.py`, `test_parallel_decode.py` and `PARALLEL_DECODE.md` for separate title/badge/heading rows, fixed field budgets and continuous refill, then run the CPU suite and tiny native Metal cache-parity test. The retry rejection stated:

> The action modifies and tests a substantial native parallel-decoding implementation beyond the explicitly authorized scaffold, while the available quoted user remarks do not specifically authorize these code changes or native test side effects. Do not bypass this rejection through a workaround or indirect execution.

Subsequent direct user approval explicitly named those three edits and CPU/native tests/push. The next retry was approved and executed. Independent title/badge/heading fields and continuous refill are now on disk. The tiny native test ran and initially failed a cached-versus-whole-prefill numeric comparison; a matched-chunking control showed exact fork parity, while whole-versus-segmented prefill differed by 0.00149166584 in logits with the same greedy argmax. That distinction is being retained explicitly, rather than widening a tolerance. The extended actual native batch/EOS/refill gate subsequently passed, including four rows at width two, single-slot refill while another row remained active, exact sampled EOS/step parity, and unchanged source state. The bound source/runtime receipt appears in the measured result. The pinned 9B comparison then executed and stopped at a token divergence, detailed below.


## Actual bounded 9B result

`PARALLEL_BENCHMARK_RESULTS.json` preserves all fourteen executed arms, exact raw sampled tokens (including EOS), selected IDs, row prompt hashes, cache/suffix/decoder/assembly timings and all unattempted fixed cases. The twelve-case roster was not shortened after inspecting outcomes. The guard intentionally stopped later arms rather than assuming a passed tiny test guaranteed exact 9B greedy parity.

| Arm | Executed attempts | Unattempted cases | Complete valid pages among executed | Exact sequential parity | Full-attempt p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Preserved manifest | 3 | 9 | 2/3 | Not applicable; 3/3 historical replay | 2980 ms |
| Independent sequential | 3 | 9 | 0/3 | Reference | 4009 ms |
| Native width 2 | 3 | 9 | 0/3 | 3/3 | 3615 ms |
| Native width 3 | 3 | 9 | 0/3 | 2/3 | 3294 ms |
| Native width 6 | 2 | 10 | 0/2 | 2/2 | 3230 ms |

These are incomplete, inspected synthetic measurements. Width six had only three independent fields in its attempted one-card cases, so its observed concurrency never exceeded three; this is not evidence of six simultaneous streams or 3/6-card behavior. No multi-card case ran after the guard stopped. Faster rejected attempts do not establish reduced usable-page latency. The historical full manifest baseline remains 7/12 evidence pages (7/22 overall); the two valid fresh pages are a partial replay, not a new twelve-case quality estimate.

The third case's badge diverged at sampled token seven: sequential and width two selected token 85688 (`Meadow`), while width three selected 417 (` B`). Its heading and title raw tokens, including the title EOS, were identical. All prefix/row prompt hashes, per-row caps and suffix lengths matched. The badge was already malformed and reached its cap in both arms. This was the initial three-row admission, not a refill event. The saved outputs alone cannot distinguish floating-point batch numerics from a native cache/runtime issue, so neither explanation is asserted as proven.

All three attempted sequential headings reached the sixteen-token cap; whole-page assembly consequently reports a non-stop finish and the unchanged global validator rejects every generated field. Independent factual titles in the first two cases therefore cannot rescue a complete page. Heading member evidence also exposes indexes/tags without selected names; a first-case heading named River 14L despite selected Grove 23L. These output-design limitations are separate from concurrency correctness. No output was rewritten or validator relaxed.

Peak MLX allocation was 6,284,001,495 bytes against the unchanged 6,442,450,944-byte cap. No OOM occurred. The separately bounded diagnostic completed and reproduced the divergence from one pristine full prefix. Sequential and width two produced exactly equal normalized log-probability vectors at the first divergent position; Meadow exceeded B by 0.0625. Width three tied them and selected lower-ID B. Its repeated width-three distribution was exactly equal, and complete source-state hashes stayed unchanged. This establishes batch-shape-associated output sensitivity, not a proven kernel-versus-cache root cause. See `NATIVE_BATCH_DIAGNOSIS.json` for the actual probabilities, raw tokens, state hashes and full saved distribution evidence. The probe does not continue the rejected comparison or supply a usable-page speed claim.
