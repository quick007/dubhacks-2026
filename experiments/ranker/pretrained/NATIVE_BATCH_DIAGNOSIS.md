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
