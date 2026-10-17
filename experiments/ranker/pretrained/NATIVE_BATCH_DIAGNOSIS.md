# Fixed-case native batching diagnosis

The original strict parity abort is reproducible. At badge row `badge-1`, zero-based sampled token index 6, sequential and width 2 choose token 85688 (` Meadow`); width 3 and its repeat choose token 417 (` B`). All six preceding badge tokens match. Heading and title raw tokens, EOS, step counts and finish reasons remain identical across all four probe arms.

| Native active width | ` Meadow` logprob | ` B` logprob | Meadow minus B | Chosen token |
| ------------------- | ----------------: | -----------: | -------------: | ------------ |
| 1                   |           -1.5625 |       -1.625 |         0.0625 | 85688        |
| 2                   |           -1.5625 |       -1.625 |         0.0625 | 85688        |
| 3                   |           -1.5625 |      -1.5625 |              0 | 417          |
| 3 repeat            |           -1.5625 |      -1.5625 |              0 | 417          |

Width 3 introduces an exact tie in the sampled log-probabilities. The unchanged greedy sampler selects token 417. All 248,320 observed vocabulary log-probabilities match exactly between sequential and width 2, and between width 3 and its repeat. Width 3 versus sequential has maximum absolute difference 0.25 and mean absolute difference 0.023826614. Every observed value is finite. This is evidence of a repeatable batch-shape-associated distribution change; it does not identify whether floating-point kernels or native cache merge/position handling caused the change.

## What actually ran

One corrected process loaded the pinned local Qwen3.5-9B 4-bit snapshot once and built a pristine 988-token user-first ranking prefix once. It forked complete native hybrid states for the unchanged three field prompts, then ran sequential, width 2, width 3 and repeated width 3. It changed no arithmetic precision, sampler, model weights, token budget, validator or native installed source. The observer subclasses `BatchGenerator` within this process, copies `Response.logprobs` to host float32 for inspection, and restores the original class afterwards. Observed normalized log-probabilities are not raw pre-normalization logits. Diagnostic observation costs make these timings unsuitable for speed comparisons.

The prefix hash matches all three preserved failing-case arm reports. Each of the nine preserved row prompt hashes, suffix lengths and budgets matches the probe; the extra cross-arm checks were also rerun on CPU after measurement. The title uses 14 autoregressive steps including EOS 248046, heading uses all 16 steps and badge uses all 12 steps. The latter two finish by length. Complete base cache state hashes, including recurrent/conv state, padding/lengths and KV offsets, remain identical before and after every arm. This rules out mutation of that pristine source state, while leaving possible errors inside derived native batch state unproven. Width 3 admits all three rows initially; the divergent step precedes compaction or any refill. Width 2 exercises late admission and stays exact in this case.

Two model process starts occurred: one failed startup with zero decode arms, then one successful corrected process with four decode arms. The corrected process exited successfully. Peak MLX memory was 6,237,453,733 bytes, below the unchanged 6,442,450,944-byte limit. Runtime: MLX 0.32.4, mlx-lm 0.32.2. An earlier startup attempt loaded the model and checked prompts but failed before decoding because `mlx_lm.generate` resolved to an exported function. The report records that failure; the separately authorized corrected retry used `importlib.import_module` and validated the native class before loading.

## Deliverables and limits

`NATIVE_BATCH_DIAGNOSIS.json` contains actual top-five log-probabilities, exact tokens/EOS/steps, all state hashes, prompt identities, memory and preserved report hashes. Full 248,320-element observations for each arm are retained under ignored `data/ranker/pretrained/native-batch-diagnosis-v1/badge-position6-logprobs.npz`, with a SHA-256 receipt. A separate startup-failure receipt and current diagnostic source hash are recorded; the only post-probe source change expanded CPU preflight identity checks to all three preserved arms. Progress and per-arm observations were saved incrementally.

`native_batch_diagnosis.py` reproduces only this fixed diagnostic. It deliberately leaves the existing decoder and strict parity gate unchanged. No tolerance relaxation, precision change, sampler change, broad comparison rerun or quality rescue was attempted. These synthetic outputs remain invalid pages; the prior aborted benchmark is preserved. A matched-state forward/kernel comparison would be needed to distinguish numerical sensitivity from an internal native cache issue. No additional model run is included or implied.


## Stage two: actual matched-shape controls, 2026-10-17

The exact previously denied v2 launch was retried once with the new model-optimization delegation and direct user authorization. Automatic review approved it. One native process completed all eighteen short controls: actual initial widths one/two/three, forked/rebuilt/fresh-full caches, each repeated once. No model worker or native test was running before launch. Two isolated agents independently reviewed state/measurement correctness and output quality without GPU work.

At every width/repeat, forked and independently rebuilt complete prefix states hash exactly like the pristine source, and the badge raw logits and normalized probabilities match exactly. All eighteen raw-logit reconstructions match their corresponding native response probabilities exactly across 248,320 vocabulary entries. Every pristine source-state hash stays unchanged. All nine distribution/token repeats are exact. The first six badge tokens match across cache conditions and widths, so the seventh-token comparisons have identical causal badge context.

| Actual initial width | Fork/rebuild Meadow raw logit | Fork/rebuild B raw logit | Fork/rebuild chosen token | Fresh-full Meadow minus B | Fresh-full chosen token |
| --- | ---: | ---: | --- | ---: | --- |
| 1 | 14.8125 | 14.75 | ` Meadow` | 0.1875 | ` Meadow` |
| 2 | 14.8125 | 14.8125 | ` B` | 0.0625 | ` Meadow` |
| 3 | 14.8125 | 14.8125 | ` B` | 0.125 | ` Meadow` |

The fork/rebuild width change turns a 0.0625 bfloat16 raw-logit margin into an exact tie, before normalization. Widths two and three have identical badge distributions here; the original width-two benchmark admitted its badge later, so that older equality with sequential is a different active shape/history. Fresh-full lanes independently show width-sensitive logits without reused prefix caches, while retaining ` Meadow` in this probe. Fresh-full prefill uses native 512-token segmentation; fork/rebuild uses the original single-call prefix segmentation, so comparisons across those modes change an explicit numerical factor.

These results exclude incorrect prefix copying/reconstruction as the necessary explanation of this fixed divergence. They provide strong evidence for deterministic batch-shape numerical sensitivity. All modes still share native model/merge/padding/generation code: they do not uniquely localize kernels or rule out every shared native state/position defect. The strict original token mismatch remains a failure, and the tiny native isolation receipt remains separate evidence rather than a universal guarantee. The report field named `allRawTokensExactlyEqual` compares whole row records, including probe statistics; the actual repeats are exactly equal in both tokens and distributions, so this naming defect changes none of these observations. The proposed follow-up gate checks tokens explicitly.

### Memory compliance failed; wider comparison stopped

The configured allocator limit stayed at 6,442,450,944 bytes. The actual recorded peak reached **6,475,156,144 bytes**, 32,705,200 bytes (31.19 MiB, 0.508%) above it, first recorded in `w3-full-r0`. The original `check_memory` checks active allocation only at boundaries; it missed this transient peak, and all eighteen arms completed before the post-run audit detected it. The allocator setting is not a hard ceiling. Functional completion must not be presented as fixed-budget compliance. No allowance was increased, no OOM was hidden, and no broader model run followed this discovery.

The controls produced 252 sampled responses (seven per lane), with 58,989 ms summed per-arm diagnostic wall time. The report's start timestamp to final artifact mtime spans about 65,092 ms including startup/common prefill and persistence; it excludes Python imports before recording and is an artifact-backed wall-clock interval rather than a monotonic process timer. These probes deliberately stop after seven responses and do not generate or validate complete pages. Instrumented diagnostic timings establish no usable-page speedup.

`NATIVE_BATCH_DIAGNOSIS.json.stage2` preserves the original v1 report plus every new arm, logits/top-five values, tokens, state hashes, comparisons and memory. Full raw/logprob/reconstructed vectors remain in ignored `data/ranker/pretrained/native-batch-diagnosis-v2/position6-arrays.npz`, with SHA-256 receipts. `NATIVE_BATCH_V2_AUDIT.json` separately records functional completion, failed peak compliance, the actual CPU preflight rejection, and timing provenance; original observations are not rewritten as a pass.

`parallel_semantic_benchmark.py` prepares the separately named fixed twelve-case manifest/sequential/width-two/three/six experiment. It preserves the old strict abort and unchanged prompts/budgets/validator, retains raw mismatch evidence, and requires actual native receipt/source/runtime plus completed matched-state controls. Its new peak-budget preflight rejects the actual v2 result **before model loading**, so it has executed zero comparison arms. Its local backbone hook also detects peak excursions after native rounds if a later compliant control ever permits execution; detection cannot prevent all transient allocations. Eleven CPU tests pass, with the separately gated native test skipped in this CPU run.

The next best performance experiment is a new, explicitly labeled smaller-prefill control (for example 256-token native prefill) with peak checks after every native call, preserving this 512-token result and the six-GiB allowance. Segmentation is an experimental factor, so it requires fresh matched-shape controls rather than relabeling old evidence. Once functional controls and measured peak both meet the bound, run the prepared semantic comparison. The independent output audit recommends a subsequent heading-prompt-only quality experiment; no copy/prompt change or validation relaxation was made here.
