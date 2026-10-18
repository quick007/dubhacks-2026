# Copy adapter control gates

Independent CPU/source review checklist for a final-layer `mlp.down_proj` adapter. This is a proposed execution gate, not a native result or a training-quality receipt. No model/native/GPU execution was performed by this reviewer. See `COPY_TRAINING_CAPABILITY_AUDIT.md` for installed implementation and source fingerprints.

## Freeze the actual experiment

- Record committed runner/data/tokenizer/model/head/validator hashes, seed, adapter rank/scale, exact module target, sample order, maximum complete sequence length, step count, optimizer settings and memory guard before observing heldout generation. Preserve the old reports and checkpoints.
- Keep all target words and target-derived metadata out of encoder messages. Freeze train/validation/heldout separation by SKU/family and fact combinations. Mark synthetic teacher grammar explicitly; successful teacher fitting is not independently verified prose quality.
- Assert six distinct selected products, two actual deterministic sections, fourteen required field rows, and unchanged per-field inference caps. Keep development results separate from prospective heldouts and retain failed/unattempted cases in denominators.

## Match the supervised boundary

- Build the exact full shared chat prompt used by the declared inference arm, including the same ranking messages if that arm includes them, separate common/field user turns, native template and `enable_thinking=False`. A shorter training representation must be named as a distribution change, not presented as identical full inference context.
- Save or hash prompt token IDs, target token IDs, full teacher-forced IDs and offset. Require `full[:offset] == inference_prompt_tokens` and `full[offset:] == fresh_target_tokens + actual_stop_tokens`; decode-check the target and terminal stop. Never guess a tokenizer EOS ID or allow an extra reasoning prefix to become a target unnoticed.
- With `inputs=full[:-1]`, logits from hidden positions `offset-1 : len(full)-1` align to `targets=full[offset:]`. Assert equal, nonzero lengths; supervise fresh text and the real stop marker only. Do not project the entire prompt vocabulary and merely call that target-only memory use.
- Reject overlength samples before native work. Do not use stock warning-and-truncation behavior to discard evidence, selected identity or supervised EOS. Do not append cached inference states to a mutable teacher-forced training graph.

## Prove the adapter changes only intended weights

- Freeze the base first, then assert the exact trainable-name set is the two arrays under `language_model.model.layers.31.mlp.down_proj`: `lora_a` and `lora_b`, with shapes `[12288,rank]` and `[rank,4096]`. Verify no embedding, LM head, recurrent projection, earlier MLP or quantization scale/bias became trainable.
- Hash every original frozen parameter array before and after training, including packed weights, quantization scales/biases, norms and recurrent parameters. Snapshot adapter arrays separately. Check finite scalar loss, supervised count, all gradients, optimizer state and updated arrays at every bounded step.
- Require a real adapter-B update from its zero initialization. A first-step zero gradient for A is expected when B starts at zero. Save/reload the adapter with the exact config and require identical adapter bytes plus same-prompt output/distribution equivalence; do not fuse/requantize the base for this control.
- If using precomputed frozen features, show that they are the exact final-MLP input/residual features for the same teacher-forced token positions. Compare custom loss and adapter gradients to a direct forward on one short identical sample with nonzero B. A feature cache must be keyed by complete prompt/target tokens, base revision, selected module and extraction code, and cannot silently reuse another field or target trajectory.
- Preserve floating-point operation order. Installed LoRA computes `base_down(x) + (scale * ((x @ A) @ B)).astype(x.dtype)`, then the block adds its attention residual `h`, then final RMSNorm runs. Cache `x`, `base_down(x)` and `h` separately for the exact feature path: `norm(h + (base_down + cast(delta)))`. Caching `y = h + base_down` first and then computing `norm(y + delta)` re-associates additions and can change low-precision rounding; omitting the delta cast changes the implementation too. Full teacher-forced extraction must include the final prompt position that predicts the first target token, not begin after that position was already consumed into a cache.

## Prove the nonzero adapter cache seam

The source argument applies only to adapters in the final block's MLP: its output follows that block's attention-cache writes and there is no later layer. Do not extend the argument to q/v projections or earlier blocks.

1. Use a recorded nonzero adapter (after an actual update, or a clearly labeled small deterministic control perturbation). Assert finite adapted logits and at least one actual base/adapted logit difference. Zero-initialized B is an insufficient seam test because it trivially reproduces the base model.
2. With the same fixed full prompt token IDs, mode, shape and chunking, build complete native cache states with the adapter disabled and enabled. Hash every recurrent/convolution state and KV tensor plus relevant cache metadata. Require exact equality, exact prefix tokens/lengths and immutable original states. Never trim or splice the hybrid cache.
3. From the base-built prefix, enable the nonzero adapter and compare a matched continuation against an independently rebuilt adapter-enabled prefix. Require the same forced suffix/input tokens, actual active width, raw response token IDs, finishes, stop tokens, AR counts and finite response-distribution digests. Repeat and independently rebuild per-lane common extensions. Confirm each digest count equals the native response count.
4. Preserve model eval mode for both cache constructions. Stock trainer mode changes and its class-wide checkpoint patch must not silently alter the continuation path; restore hooks/methods in `finally` or isolate training in its own process.
5. Rank with the adapter disabled and require exact original hidden readouts, head scores, selected product/image IDs and observed-support evidence. Keep copy output enabled only at the established seam. If the proof fails, rebuild generation context under the enabled adapter and include that complete extra prefill in latency.

The seam controls compare identical fixed inputs. Different freely generated base/adapted tokens are expected once wording changes and are not themselves evidence of cache corruption. Conversely, exact base/adapted text is not proof the adapter was active; record the nonzero logit/update evidence.

## Measure the resulting behavior

- Synchronize the actual native stream and evaluate all arrays used by each measurement. Avoid forcing unused full-prompt vocabulary logits. Record resident load, feature preparation, training/update time, serialization, full request time, actual sampled steps, raw text and observed peak separately.
- Re-run genuine fresh values after training through the corrected identity/factuality/novelty/observed-support/global-completion policy. Preserve canonical host labels separately from raw generated values and never repair old output. Report valid complete pages, strict accepted fields, raw NONE policy, truncations and semantic failures.
- Retain all fourteen fields even if one truncates. Count head/cache equivalence, native training feasibility and generated quality as separate outcomes. No speed or complete-page improvement follows merely from lower target loss, a saved adapter, or successful backward execution.

## Source-review status

At checklist creation the root adapter runner had not yet appeared. This document states required controls; it does not certify their native implementation or execution. The separate identity-policy design is also not a validator execution receipt.

Implemented the separately requested pure helper `semantic_copy_tokens.build_training_tokens(tokenizer, rank_messages, row, max_context_tokens=4096)` and five passing CPU stub tests in `test_semantic_copy_tokens.py`. The helper keeps explicit rank/common/field turns, disables thinking, encodes fresh target plus verified single-token `<|im_end|>`, checks native-wrapper or plain-tokenizer EOS declarations, includes EOS in the sampled cap, rejects embedded EOS/overflow, and returns exact causal prediction alignment and model-input-only hashes. Tests cover caller immutability, metadata exclusion, alternate EOS declaration forms, off-by-one alignment, exact cap boundaries and refusal to truncate. This certifies helper behavior on those fixtures; it is not a model-loss, real-data or native cache proof.

### Initial runner source review

Reviewed `semantic_copy_training.py` without executing or editing it. Observed snapshot SHA-256 `4c07d0c85477189aabb6c0cb56793981434b8a2bffc9e13348239ee64afdc104` has correct feature algebra: the post-attention norm tap captures residual `h`, the down-projection tap captures actual input `x` and base output `y`, and feature loss computes `norm(h + (y + cast(delta)))`. Feature extraction includes the final prompt position using `predictionStart - shared_prefix_length`, and explicitly checks the number of captured rows against target tokens. The disabled-adapter context restores the original module in `finally`. Complete native source caches are forked rather than trimmed. Base freezing is provided by the existing `FrozenBackbone` constructor.

Two source defects were found and corrected by the root during this review: the full evaluation's patched layout originally recursively called itself, and gradient comparison originally could silently drop mismatched keys. The observed snapshot uses `heading_rows` directly and requires equal two-key gradient rosters. The direct-versus-feature test uses nonzero B, preserving evidence of failures rather than silently accepting a changed loss.

Additional controls requested of the root were explicit finite prefix/continuation logits, finite updated adapter/optimizer arrays, B-specific update verification, and execution/runtime/data-file provenance. These requests are source-review followups; this document does not assert they were present in the observed snapshot or that any native control passed. The first smoke test exercises one training row and does not exercise full heldout-evaluation dispatch. Native feasibility, cache equivalence, reload behavior and generation quality require their separate measured receipts.
