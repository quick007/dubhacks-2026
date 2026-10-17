# Native control stages three through five

Independent CPU audit of the saved artifacts confirms that v5 completed all eighteen controls with exact logit/response alignment, exact repeatability and a measured peak below the six-GiB experiment guard. The earlier failed observations remain preserved. This audit launched no native model work and changed no experiment source.

| Stage | Native prefill | Completed controls | Result | Recorded peak bytes | Monotonic diagnostic wall time |
| --- | ---: | ---: | --- | ---: | ---: |
| v3 | 256 | 12/18 | Stopped during `w3-fork-r0` on peak guard | 6,563,560,644 | 38,901 ms |
| v4 | 256 | 12/18 | Stopped during `w3-fork-r0` on logit/response alignment assertion | 6,279,572,052 | 35,506 ms |
| v5 | 256 | 18/18 | Functional controls and measured peak compliance passed | 6,364,654,164 | 64,026 ms |

The fixed guard is 6,442,450,944 bytes. v3 exceeded it by 121,109,700 bytes; v5 remained below it by 77,796,780 bytes (74.19 MiB). These are cumulative MLX allocation peaks within separate processes, not process RSS or an OS hard ceiling. Diagnostic observation and persistence costs make the wall times unsuitable for complete-page speed claims.

## Why the failed observations differ

v3's observer explicitly evaluated the entire model logit tensor after every forward, including multi-token prompt prefill. Native prompt processing normally evaluates cache state and leaves its unused vocabulary projection lazy. The observer therefore forced allocations that the measured production path does not require. It stopped on the first detected peak excursion, at width three before fresh-full controls. Reducing native prefill to 256 did not by itself establish memory compliance. The saved failing arm still has status `running` inside the v3 report; the enclosing `failed-stopped` status and `MemoryError` govern its outcome, and it must not count as completed.

v4 corrected prefill observation to evaluate only native cache state for multi-token calls. It stayed below the guard through the first twelve controls, then failed the exact alignment assertion at the first width-three arm. With 256-token suffix prefill, the shorter heading lane enters generation before the title/badge lanes. A global single-token forward-call index therefore identifies a different badge position. The saved `w3-fork-r0` reconstructed and actual log-probability vectors differ by as much as 11.125. This is failed observer alignment; that vector is not evidence of a same-position model/cache discrepancy.

v5 retains cache-only prefill observation and counts autoregressive forward positions separately for each native UID. A scoped `GenerationBatch` observer exposes the actual UID order before each native `_step`; the badge vector is taken at its own zero-based forward position six, then checked against response position six. This handles staggered admission and row compaction. Every saved reconstruction equals the corresponding native response log-probabilities exactly across all 248,320 vocabulary entries. No tolerance was widened.

## What v5 measured

One pinned Qwen3.5-9B four-bit model process used the same 988-token common prefix, unchanged field prompts, greedy sampling and seven sampled responses per lane. Widths one/two/three each ran forked prefix, independently rebuilt prefix and fresh-full prompt conditions twice. This produced 252 sampled responses. Independently rebuilt prefixes use the original single-call prefix segmentation; fresh-full native prefill uses 256-token segmentation. The latter is a different numerical factor. At 256, reused title/badge suffix prefill also splits into 256+13 tokens, unlike the historical 512-prefill run.

All fork/rebuild prefix hashes equal the pristine complete source hash. Source hashes remain unchanged before and after every arm. At each width/repeat, fork and independent badge raw logits and log-probabilities are exactly equal. All nine repeated configurations have exactly equal raw vectors, normalized vectors and sampled tokens. All six preceding badge tokens match across compared conditions: `[33, 16, 25, 417, 16, 25]`. Every saved vector is finite.

| Width | Cache condition | Meadow raw logit | B raw logit | Meadow minus B | Seventh sampled token |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | Fork / independent | 14.875 | 14.6875 | 0.1875 | 85688 (` Meadow`) |
| 1 | Fresh-full | 14.9375 | 14.75 | 0.1875 | 85688 (` Meadow`) |
| 2 | Fork / independent | 14.875 | 14.75 | 0.125 | 85688 (` Meadow`) |
| 2 | Fresh-full | 15 | 14.8125 | 0.1875 | 85688 (` Meadow`) |
| 3 | Fork / independent | 14.8125 | 14.75 | 0.0625 | 85688 (` Meadow`) |
| 3 | Fresh-full | 14.8125 | 14.8125 | 0 | 417 (` B`) |

Both repeats give the values above. Probe logits have bfloat16 dtype. At width three, fresh-full has an exact raw-logit tie before normalization, and the unchanged greedy sampler selects token 417. Its fork/rebuild counterpart retains a one-step 0.0625 margin. The raw vector's maximum absolute difference between fresh-full and fork is 0.4375 / 0.28125 / 0.125 at widths one / two / three, respectively. Between fresh-full width one and three it is 0.28125, with identical preceding badge context.

The controls demonstrate repeatable shape/segmentation-associated distribution sensitivity and exclude bad prefix copying/reconstruction as its necessary cause in this case. They do not identify a particular kernel, prove every native merge/padding/position operation correct, or establish universal cache correctness. All conditions share native model and generation code. The original strict parity failure stays preserved; v5 is a separately configured diagnostic, not a replacement pass for the original 512-prefill benchmark.

These probes intentionally stop at seven responses. They establish no accepted field/page count, copy-quality improvement or usable-page speedup. Complete-page quality and latency belong to separate experiments with their own fixed denominators and unchanged validation.

## Evidence checked on CPU

The audit recomputed each NPZ hash, checked every saved vector for finite values, independently compared v5 raw/log-probability arrays for fork/rebuild and repeat equality, verified exact reconstructed/actual equality, and ran the actual semantic control gate. It also verified each stage's preserved prior-report snapshot hash and v5's observed source hash against the file on disk.

| Artifact | SHA-256 |
| --- | --- |
| `data/ranker/pretrained/native-batch-diagnosis-v3/probe.json` | `d529a1c78f2b478d60eb0e7bfb194c618ecc03fe507fb3e19152c0a278df66de` |
| `data/ranker/pretrained/native-batch-diagnosis-v3/position6-arrays.npz` | `11c619397e531656ba944d42226cfa1a9743943b35ec0356b5a12c6b4f378535` |
| `data/ranker/pretrained/native-batch-diagnosis-v4/probe.json` | `d8a52a48b05f1cb312998c229c6888df9a8f6d2b0461e0a3a494994896080fa4` |
| `data/ranker/pretrained/native-batch-diagnosis-v4/position6-arrays.npz` | `bd2582e8bbdaa57b59eff88e3eda51046c114332e51bacd50d2cb63719cf9dc5` |
| `data/ranker/pretrained/native-batch-diagnosis-v5/probe.json` | `56948fe246a59cb60f02b47b6056543b702ab493ff68c9280c9495e747acdbc5` |
| `data/ranker/pretrained/native-batch-diagnosis-v5/position6-arrays.npz` | `fee4dfe4386028b9d9d0943c70583bb07c63228ce812863bb0f847da11799514` |
| Observed `native_batch_diagnosis.py` | `0e010754c957d7c8677a3463679fbe1eb098eff65dcb4117ed24fcebe78afca4` |

The matching observations are also retained under `NATIVE_BATCH_DIAGNOSIS.json.stage3`, `.stage4` and `.stage5`. Historical source hashes identify the code used for each stage; later source corrections must not be represented as having run in an earlier process.
