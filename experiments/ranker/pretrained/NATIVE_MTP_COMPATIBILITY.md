# Native MTP compatibility audit

The installed MLX-LM 0.32.2 runtime and pinned Qwen3.5-9B four-bit artifact cannot enable same-backbone multi-token prediction with a flag. The artifact lacks MTP weights, its native loader discards MTP tensors, and its separate-draft speculative path requires a trimmable cache that this hybrid model does not provide. This is a source/metadata conclusion, not an executed inference result.

Audit scope: local installed source, package metadata, pinned config/index and safetensor JSON headers, plus primary publisher/runtime documentation inspected on 2026-10-17. No MLX import, model load, native test, external weight download, package installation or source modification was performed. Only this report was created. Existing model and benchmark evidence remains intact.

## Local compatibility

| Requirement | Actual local evidence | Outcome |
| --- | --- | --- |
| Installed runtime | Distribution metadata: `mlx==0.32.4`, `mlx-lm==0.32.2`; `mlx-vlm` absent | Current Python path is MLX-LM |
| MTP metadata | Pinned config has `mtp_num_hidden_layers: 1`, `mtp_use_dedicated_embeddings: false` | Metadata alone does not supply trained tensors/runtime support |
| MTP tensors | All 1,260 index keys and both safetensor headers contain zero `mtp`/`nextn` keys; language layers are exactly 0–31 | No local MTP module weights to activate |
| Native model implementation | `TextModel` contains the language body and one LM projection; `sanitize` drops keys containing `mtp.` | No installed Qwen3.5 MTP execution path |
| Separate-draft speculation | `speculative_generate_step` takes a `draft_model`, builds target/draft caches and rejects a non-trimmable target | Existing hybrid target is incompatible with that entry point |
| Existing continuous batching | Native server marks requests with a draft model non-batchable; project decoding uses `BatchGenerator` | Speculation is not a drop-in addition to the measured field-batching path |

The pinned directory is `.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit/snapshots/8b2b98c00a6b4d291155e4890773ca8f769aee53`. Its first shard has 1,253 tensors and a 157,678-byte JSON header; the second has seven tensors and an 818-byte header. The index lists only language body/head and vision tower prefixes. No layer beyond the 32 language layers or alternative extra prediction-head prefix appeared. Reading these headers did not read tensor payloads into a model or execute checkpoint code.

Installed source paths below are relative to `.tools/pretrained-venv/lib/python3.14/site-packages/`:

- `mlx_lm/models/qwen3_5.py:318–341`: language body followed by the ordinary LM head.
- `mlx_lm/models/qwen3_5.py:347–354`: hybrid cache construction and explicit removal of MTP weights. The [upstream v0.32.2 implementation](https://github.com/ml-explore/mlx-lm/blob/v0.32.2/mlx_lm/models/qwen3_5.py#L347-L354) has the same behavior.
- `mlx_lm/generate.py:485–546`: distinct target/draft arguments and the trimmable-cache rejection. `:597–599` rewinds both caches by trimming; `:601–609` drafts tokens autoregressively; `:628–644` verifies the proposal against the target. See [upstream v0.32.2 generation source](https://github.com/ml-explore/mlx-lm/blob/v0.32.2/mlx_lm/generate.py#L485-L664).
- `mlx_lm/server.py:449–462`: separate draft loading and `is_batchable = draft_model is None`.

## MTP and ordinary draft speculation

Qwen's native MTP uses an additional trained prediction module associated with its target model. It can avoid loading a second complete language backbone, but it still needs those learned module weights and a runtime that connects target hidden states, draft proposals and target verification. It does not make arbitrary future text independent of preceding tokens. The official [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B) documents MTP through SGLang `NEXTN` and vLLM `qwen3_next_mtp`; those are different runtime integrations, not evidence that the installed MLX-LM path supports MTP.

Ordinary draft speculative decoding instead calls a separate causal model to propose tokens, verifies a block with the target and handles any rejected suffix. Installed MLX-LM's `draft_model` API is this separate-model interface. Passing the target as its own draft would not create learned MTP functionality, would not remove the hybrid-cache rejection, and offers no established benefit. Likewise, requesting logits at multiple supplied token positions performs verification/teacher forcing; it does not by itself predict unknown future tokens.

## Hybrid-state blocker

The pinned model has 24 linear-attention layers and eight full-attention layers. Native `make_cache` uses `ArraysCache(size=2)` for each linear layer and `KVCache` for full attention. Qwen3.5 updates convolution state in slot zero and DeltaNet recurrent state in slot one (`qwen3_5.py:150–168`, `:180–198`). These overwrite a compact state, rather than retaining an append-only token history.

`ArraysCache` inherits `is_trimmable() == False` from `_BaseCache` (`cache.py:164–165`); its serialized state also includes padding and lengths (`cache.py:596–626`). Therefore the source predicts that MLX-LM's speculative entry point will raise its non-trimmable-cache error for this target before decoding. This error was not provoked in a native test during this audit.

The project's tested full-state prefix forks do not supply speculative rollback at every possible accepted draft position. A future compatible verifier must preserve or reconstruct convolution, recurrent and KV state for the exact accepted prefix. Removing the rejection check or trimming KV alone would be incorrect. A complete snapshot plus replay design could be investigated separately, but its correctness, memory overhead and verification cost are unmeasured here.

## Published artifact availability

The publisher provides [mlx-community/Qwen3.5-9B-MTP-4bit](https://huggingface.co/mlx-community/Qwen3.5-9B-MTP-4bit), a separate MTP drafter rather than a standalone chat model. Its card specifies `qwen3_5_mtp`, block size two and use alongside a compatible target through MLX-VLM. The [published file tree](https://huggingface.co/mlx-community/Qwen3.5-9B-MTP-4bit/tree/222dfd2c23fc9518d7b817e4f8e0cb0571787489) lists an approximately 137-MB safetensor file and approximately 164 MB total repository contents. These are publisher-listed download sizes, not measured resident memory. No artifact was downloaded or hash-verified locally.

The [MLX-VLM project documentation](https://github.com/Blaizzy/mlx-vlm#speculative-decoding) describes drafter-based verification and an MTP family. This is a candidate runtime direction, not a compatibility receipt for the pinned local target. The artifact card's example uses a five-bit target; equivalence with our exact four-bit revision, tokenizer, non-thinking prompt boundary and source-cache reuse remains untested. Hugging Face's generic autogenerated MLX-LM usage snippets are insufficient evidence for this specialized drafter; the publisher's specific model card and installed source govern this audit.

## Practical next step

Keep MTP disabled in the existing benchmark. Under the present no-download/no-runtime-change scope, there is no usable local same-backbone MTP experiment to launch. Continue evaluating the already supported field-batching and copy-quality changes with their preserved evidence.

If a separate MTP investigation is authorized, first pin and review a runtime revision that explicitly implements Qwen3.5 MTP plus complete hybrid-state rollback, and pin the compatible small MTP artifact. Check target/tokenizer/quantization compatibility before any execution. Then test tiny deterministic acceptance/rejection, EOS and partial-acceptance cases with exact full-state restoration; only afterwards compare the same short storefront fields, including prefill, proposal/verification work, rejected drafts, complete-page validation, latency and peak memory. Combining it with continuous field batching needs its own supported path and measurements. No speedup or unchanged-output claim follows from this audit.

## Local provenance

| Inspected file | SHA-256 |
| --- | --- |
| Pinned `config.json` | `a96942cb6a8a1d3f1d17514d81a1925d04362a6a3233b389d13012211baaa9f8` |
| Pinned `model.safetensors.index.json` | `dd023913fb87cfdae27fb11dcf695117c925833796ccac3c64117d6652d8ff1e` |
| Installed `mlx_lm/generate.py` | `9d46803799f99d6fc7c35fff2d4357c79b1a25ae98cc63efeaaebeb50a7b5f33` |
| Installed `mlx_lm/models/qwen3_5.py` | `851449fd9989d6bf4b3c4fed92ddc3be5f1e781eb2246f89c54d7eff2518d9c6` |
| Installed `mlx_lm/models/cache.py` | `c96cb169fd5e0ec4584a162dd61dc62ab08ccdca03a9850f8a491ce6809145e6` |
