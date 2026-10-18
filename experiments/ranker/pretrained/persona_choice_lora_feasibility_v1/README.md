# Decision LoRA feasibility, preparation only

The smallest reasonable native adaptation is rank4 on `language_model.model.layers.31.mlp.down_proj`, initialized from the pristine frozen local Qwen3.5-9B4bit base, with A seeded and B exactly zero. Its 65,536 trainable floats update the existing pretrained residual path and frozen language-model projection. This uses pretrained decision logits; it does not establish better semantic generalization. The failed copy adapters and random projection heads remain preserved and are not warm starts.

No training schedule or routing vocabulary is frozen here. The separate lexical product-bound routing pilot should determine whether A–P position codes remain appropriate. The independently authored 128 TRAIN rows also need a source/label/overlap audit before constructing any new fit. All previously inspected 40/32 cases and ten history pairs are development data. Any future fresh holdout must be authored and sealed after the candidate recipe is chosen.

## What the CPU audit established

`source-audit.json` pins installed source, model metadata, and public TRAIN data. It parsed no private labels and made zero model calls. Using the current pilot's exact public catalog and prompt, the 320 public TRAIN requests in canonical/reversed order yield 640 complete prompts: **1,019–1,471 tokens**, 787,196 total tokens. Every prompt plus each of the sixteen provisional A–P continuations retained an exact one-token boundary. The final prompt token is token25 (`:`). These numbers are provisional prompt-cost evidence, not a frozen choice of routing vocabulary or model-quality result.

The locally available model is revision `8b2b98c00a6b4d291155e4890773ca8f769aee53`: 32 text layers, hidden4096, MLP intermediate12288, four-bit/group64 weights, separate frozen LM head. The installed LoRALinear implementation accepts quantized base linears and applies a low-rank delta with the source dtype/cast order. Source and metadata were hashed; fresh six-GB weight hashing was deliberately deferred until an authorized native run.

## Exact feature path to gate first

1. Render the full target-free catalog/profile prompt. Derive the cache prefix from its actual token IDs `prompt[:-1]`, never by joining separately tokenized strings. Prefill this prefix with the base adapter disabled.
2. Clone complete attention and recurrent cache state. Execute the one final prompt token `prompt[-1:]` as the **entire** native suffix, with the adapter enabled. Capture pre-down gated-MLP input `x[1,1,12288]`, residual `h[1,1,4096]`, and base projection `y[1,1,4096]` through the transformed auxiliary-output path.
3. Fit only A/B using the installed sequence `delta=(scale*((x@A)@B)).astype(x.dtype)`, `hidden=final_norm(h+(y+delta))`, then the frozen pretrained LM projection. No answer token, teacher response, or EOS enters the input. Supervise exactly the one next-token decision distribution.
4. Before training, require exact logits, scalar loss, and A/B gradients at nonzero B across native, auxiliary-native, and serialized-feature paths. Also test initial zero-B restoration. The copy code's EOS targets and implicit `lm_head` assumptions must not be reused unchecked.
5. A sliced sixteen-row quantized LM projection is an optimization hypothesis. Compare its logits/loss/A/B gradients against the full native quantized projection. Different output shape or dequantized FP32 multiplication may change numerical kernels. If exact parity fails, retain the full frozen projection and gather the sixteen routing logits afterwards. Never call a dequantized substitute exact without the gate.

The new loss must preserve multiple legitimate top-grade products: sixteen existing TRAIN rows have two grade3 choices. Prefer a normalized target distribution over those valid routing tokens rather than arbitrary catalog-order tie breaking. Full-vocabulary one-token CE and route-conditional CE answer different questions; choose and freeze one before optimizer updates, report both at evaluation, and do not alter it from development outcomes.

## Memory, cost and boundaries

The final-down adapter cannot alter earlier layers, the final layer's attention computation, or its cache. That is why its fixed input features can be reused across updates. Existing copy code contains this cache proof and exact A/B-gradient gates, but the new decision path still needs its own proof before and after fitting/reload. Require all927 frozen language tensor hashes unchanged; exact adapter reload; raw source-conditioned predictions frozen before development annotation reads; full failure/timeout/shutdown receipts; and the existing shared native-owner lock.

A single native-row archive is **80KiB** in float32 (`x+h+y`), or **50MiB for640 rows**, before headers. A/B require256KiB; FP32 gradients and Adam moments add about768KiB, excluding runtime overhead. These are shape-based projections, not native peak measurements. The audit-time host had10.30GiB available out of24GiB; that snapshot does not reserve memory or prove the7GiB guard. Keep the existing guard and first run one bounded numerical/memory gate under root's exclusive ownership.

The full prompt still needs a native forward for a new profile/query. Cached catalog prefixes might reduce it, but this proposal does not establish sub193ms or fresh copy/UI readiness. Extracting640 prompts means approximately787k input tokens before gate/control work; there is no measured extraction duration for this path. After the gate, start with a small TRAIN-only extraction/time pilot and extrapolate honestly before committing to the complete fit. More final layers or attention LoRA would change upstream features/cache and require a different full-autodiff/memory design; they are not covered by the existing feature shortcut.

## Primary sources

The [installed-version MLX LoRA implementation](https://github.com/ml-explore/mlx-lm/blob/v0.32.2/mlx_lm/tuner/lora.py) supports low-rank adapters on quantized linears. The [installed-version Qwen3.5 implementation](https://github.com/ml-explore/mlx-lm/blob/v0.32.2/mlx_lm/models/qwen3_5.py) determines the actual layer/residual/LM-head path. The [original LoRA paper](https://arxiv.org/abs/2106.09685) motivates frozen pretrained weights with trainable low-rank updates; it supplies no quality or latency guarantee for this catalog task.

Status: **feasible bounded next gate; not ready to train**. Original sources, checkpoints, features, and all failed outputs are unchanged. No GPU, browser, serving, installation, or Git action occurred.
