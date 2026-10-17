# Local copy-training capability audit

The installed stack contains a usable implementation path for a bounded QLoRA copy experiment on the existing quantized Qwen3.5-9B. Source support does not establish that backward execution fits this M5 or that training improves copy. The next concrete gate is a short, serialized forward/backward/update test with finite loss, real adapter changes, unchanged base weights, explicit memory accounting and reload equivalence. No model load, MLX operation, native/GPU execution, download, installation or training was performed for this audit.

## Installed implementation and actual targets

Inspected MLX `0.32.4`, MLX-Metal `0.32.4`, MLX-LM `0.32.2`, and the existing pinned model config plus safetensors headers at revision `8b2b98c00a6b4d291155e4890773ca8f769aee53`. The model is affine four-bit, group size 64: hidden size 4,096, intermediate size 12,288, vocabulary 248,320 and 32 decoder layers. Layers 3,7,11,15,19,23,27,31 use full attention; the other 24 use GatedDeltaNet. This matches installed `qwen3_5.py:209–226` and the pinned config rather than the source class's fallback defaults.

Installed source locations below are relative to `.tools/pretrained-venv/lib/python3.14/site-packages/`.

| Target | Real runtime parameter path on the top-level model | Dimensions, input → output |
| --- | --- | --- |
| Final MLP down projection | `language_model.model.layers.31.mlp.down_proj` | 12,288 → 4,096 |
| Final full-attention query and gate projection | `language_model.model.layers.31.self_attn.q_proj` | 4,096 → 8,192 |
| Final full-attention value projection | `language_model.model.layers.31.self_attn.v_proj` | 4,096 → 1,024 |
| Recurrent input projection, example | `language_model.model.layers.30.linear_attn.in_proj_qkv` | 4,096 → 8,192 |

Safetensors header-only inspection confirms packed U32 shapes `[4096,1536]`, `[8192,512]`, `[1024,512]` and `[8192,512]` respectively, consistent with eight four-bit values per U32. No weight payload was read or materialized. Full-attention q/k/v/o projections are defined in `models/qwen3_next.py:74–100`; MLP gate/up/down projections at 154–162. Recurrent linear adapters could target `linear_attn.in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a` or `out_proj` (`qwen3_5.py:116–130`). The generic adapter converter does not support Conv1d (`tuner/utils.py:75–86`).

`linear_to_lora_layers` accepts relative block keys, so a narrow configuration is `num_layers: 1` and `lora_parameters.keys: ["mlp.down_proj"]`, with rank 4, dropout 0 and explicit `scale`. It wraps only the final block's down projection. Expected new parameter names end in `.mlp.down_proj.lora_a` and `.lora_b`; the original packed base is nested under `.mlp.down_proj.linear`. At rank four this is 65,536 trainable values, 0.25 MiB at float32. Final q+v together would be 69,632 values, about 0.266 MiB. These are adapter payload calculations, not training-peak estimates.

The converter's absent `keys` default discovers all supported linears across blocks, and the CLI defaults to sixteen adapted layers. Set both deliberately. `num_layers=0` is unsafe as a way to disable adapters: installed `utils.py:113` uses `[-max(num_layers,0):]`, whose zero start includes every layer. The model is frozen before adapter insertion (`lora.py:232–251`); `LoRALinear.from_base` preserves the quantized base, with new float adapters (`tuner/lora.py:13–32,86–98`). Assert the exact trainable-name roster rather than infer it from configuration.

## Gradients, recurrence and checkpointing

Quantized base gradients are not needed for QLoRA. Installed `mlx/nn/layers/quantized.py:207–212,261–282` freezes the packed parameters and calls `mx.quantized_matmul`; `mlx/nn/utils.py:12–35` differentiates only trainable parameters. The version-matched MLX C++ implementation has an input-gradient path through quantized matmul and rejects gradients with respect to the packed quantized matrix. That supports propagation from frozen downstream weights to trainable adapters without expanding the entire base to float weights. It is not full-weight training of a four-bit checkpoint. [MLX v0.32.4 quantized-matmul derivative](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/primitives.cpp#L3326-L3352).

Qwen3.5 has an explicit training branch: `qwen3_5.py:183–194` calls `gated_delta_update(..., use_kernel=not self.training)`. In training mode, `gated_delta.py:639–646` uses the differentiable ops path, whose `557–602` implementation loops over every token. The recurrent state is float32 with shape `[B,32,128,128]`, two MiB at batch one (`gated_delta.py:580–590` plus the pinned dimensions). A hypothetical retained state per token is 512 MiB per recurrent layer at 256 tokens and four GiB at 2,048 tokens; these illustrate graph pressure, not measured peaks or proof that every state remains allocated. Stock `model.train()` enables this path even on frozen recurrent modules. Last-layer-only trainable adapters do not require gradients with respect to earlier frozen-layer parameters, but their forward computation still runs.

Stock `TrainingArgs` supports checkpointing, batch size, maximum sequence length and accumulation (`tuner/trainer.py:43–85`). `grad_checkpoint` wraps `mx.checkpoint` around decoder calls (`27–40`), but patches the decoder **class**, affecting all instances of that type; use an isolated process or restore the class method when finished. It rematerializes whole blocks, rather than providing a dedicated chunked recurrent backward kernel. Training switches between `.eval()` and `.train()` during validation (`178–209,285–307`). Any custom frozen-prefix eval-mode optimization must survive those transitions and have numerical/output controls; it is not supplied by the CLI.

An upstream issue contains conflicting first-hand reports of Qwen3.5 backward memory failures and successful base-M5 runs on older versions. It is evidence that hardware/configuration testing matters, not a confirmed failure or fix for this installed stack. [MLX-LM issue 1206](https://github.com/ml-explore/mlx-lm/issues/1206).

## Loss, sequence and memory plan

A conservative first experiment uses only final `mlp.down_proj` rank-four adapters, batch one, accumulation one, and a compact per-field binding sample whose **entire** prompt, target and EOS fit 128–256 tokens. Keep selected identity, facts, observed support or section-member universals intact. This is a separately declared compact training representation; the current shared inference prefix alone is about 2,400 tokens and cannot be truncated to fit. Stock batching warns and truncates overlength samples (`trainer.py:151–169`), so reject overlength records during CPU preflight instead. Stock training has no inference-style `prefill_step_size` or chunked-training control. Accumulation increases effective batch size, without reducing one example's peak. [Versioned MLX-LM memory guidance](https://github.com/ml-explore/mlx-lm/blob/v0.32.2/mlx_lm/LORA.md#memory-issues).

Prompt masking is necessary for a copy objective but does not remove prompt logits: default loss projects the whole sequence before applying its mask (`trainer.py:88–99`). One float32 vocabulary tensor for batch one and sequence 256 is 242.5 MiB, and 2,048 is 1,940 MiB, before gradients/intermediates. A custom causal loss can compute the hidden sequence with `model.model`, then apply `model.language_model.lm_head` only to positions `offset-1` onward, aligned with target token IDs `full_tokens[offset:]`, including EOS. Assert lengths and nonempty supervision explicitly. This retains the correct causal target loss while avoiding unused prompt-vocabulary projection; it still requires a measured backward test.

Stock chat dataset processing omits `enable_thinking=False` (`tuner/datasets.py:62–75`). The pinned template's generation branch distinguishes that option (`chat_template.jinja:147–152`), whereas existing inference disables thinking. Build exact inference-prefix token IDs and supervised suffix/offset explicitly; assert that the full teacher-forced token sequence starts with the inference prefix, and supervise only the fresh value plus the real stop marker. Do not rely on default tokenizer EOS assumptions or mask offsets from a different assistant prefix. Local JSONL support needs no Hugging Face `datasets` package (`datasets.py:208–222,312–321`); that optional package is absent here, but NumPy, PyYAML, Rich and Transformers are installed. No external dataset or integration is necessary. [Versioned prompt-masking contract](https://github.com/ml-explore/mlx-lm/blob/v0.32.2/mlx_lm/LORA.md#prompt-masking).

Stock training also calls `maybe_set_recommended_wired_limit` (`trainer.py:231`; `utils.py:74–83`), which sets an allocator wired-memory limit to the device recommendation. Preserve any explicitly declared experiment guard and record actual peak separately. Prior inference peaks do not establish backward-memory headroom. Keep one resident backbone, synchronize/evaluate losses, gradients, optimizer state and updated adapters each step, and stop on nonfinite values or the declared limit.

## Preserve trained-head and cache semantics

The existing decision heads were trained on base-backbone readouts. Enabling any adapter during ranking changes that representation; preserving head checkpoint bytes alone does not establish unchanged decisions. Enabling q/v adapters only after a base-prefilled ranking cache also mixes base cached values with an adapted decoder. For such targets, recompute a generation prefix under the enabled adapter after unchanged base-head ranking, and include the extra prefill in full latency.

A narrower possible exception is adapters exclusively in the final block's MLP: its output is after that block's attention-cache writes, with no later layer to consume the modified output (`qwen3_5.py:234–240,294–296`). Therefore those adapters cannot change any recurrent/KV cache tensor in this architecture. Ranking can stay adapter-off and generation can use the adapter, but verify fresh adapter-enabled prefix versus reused base prefix on exact complete-state hashes and subsequent outputs before relying on this seam. This is a source-derived feasibility argument, not an executed cache proof. Never reuse mutable training caches or splice recurrent states.

The first native training gate should verify zero-adapter/base output equivalence, exact trainable names, finite target loss/gradients, a changed LoRA-B after one update, unchanged packed base weights, and save/reload equivalence. LoRA-B begins at zero, so LoRA-A's first-step gradient may correctly be zero. Only then continue the predeclared bounded training count and evaluate fresh generation under the corrected identity/factuality validator, preserving raw output and failed cases. Neither lower training loss nor successful backward execution proves complete-page quality, learned section grouping, or a latency improvement.

## Installed-source fingerprints

| File | SHA-256 |
| --- | --- |
| `mlx_lm/models/qwen3_5.py` | `851449fd9989d6bf4b3c4fed92ddc3be5f1e781eb2246f89c54d7eff2518d9c6` |
| `mlx_lm/models/gated_delta.py` | `f69e837ea68641dbb06fdb5dcc024a9a7e1d9353f08ae58c05275253a0c39428` |
| `mlx_lm/tuner/lora.py` | `4d3a8edab111d4ddba33398ba8700203db7b61621c39e9c348fdd50e57278b45` |
| `mlx_lm/tuner/utils.py` | `f59f3c027176a3975fe60e7fbd9f3f38723c27c3f888cf9e591c103f93f68588` |
| `mlx_lm/tuner/trainer.py` | `0b6027fdb11803e951ceee52c83a81f82daa922ea8c6b7ca6888953461064e8e` |
| `mlx_lm/tuner/datasets.py` | `dcd675376d68d1813261c8389236b586b16b9a58a5ea80be8eb535bb9d0e9689` |
