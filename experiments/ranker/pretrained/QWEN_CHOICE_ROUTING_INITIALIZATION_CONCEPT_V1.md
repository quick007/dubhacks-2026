# Pretrained routing initialization: conditional follow-up concept

This is a CPU-only design note. No new model, fit, feature extraction, evaluation or live promotion is implemented. The parent owns the frozen 20-forward public-demo pilot. Its source and plan remain unchanged.

The pilot prompts Qwen with all 16 public catalog entries, a public shopper request/history and an assistant prefix ending in `Choice:`. It scores 16 unique next-token codes using the pretrained LM projection. The feature is the **raw final post-RMSNorm hidden state** at that last prompt token, without L2 normalization. The installed untied Qwen head supplies one 4,096-dimensional weight row for each code token.

## Exact relationship to preserve

For a fixed prompt order, let `h` be the 4,096-dimensional raw hidden vector and `w_p` the pretrained LM row for the code assigned to catalog position `p`. The selected logits are `z_p = w_p · h`, with the actual installed quantized-head arithmetic. Product identity comes from the prompt's explicit code-to-candidate mapping.

A–P rows are **position/code classifiers**, not pretrained product embeddings. The row for A does not become a permanent mug row: A refers to a different product in the reversed-order pilot. Reusing these rows with old whole-profile embeddings or L2-normalized query features would change the mathematical task and would not reproduce the pilot. A short profile-only forward cannot silently substitute for the full candidate-conditioned prompt.

If the pilot shows useful source-conditioned choices in both catalog orders, a small learned decision head could initialize its 16 shared routing rows from those pretrained rows. Initially, its outputs should reproduce the native routing logits before any optimizer update. A zero-initialized residual or low-rank update could then adapt TRAIN relevance, alongside a separately declared learned ownership term if warranted. Start with a TRAIN-only ablation and a fixed schedule; do not choose hyperparameters or checkpoints from the now-inspected public demos or old evaluation.

## Necessary controls before fitting

- Export only the 16 selected LM rows, if needed, to a local ignored artifact with token IDs, original tensor names/quantization settings, source hashes and dtype. The quantization `biases` tensors are dequantization offsets, not evidence of an ordinary affine output bias. Use the installed MLX dequantization API after inspection; do not invent a decoder or export the full vocabulary matrix.
- Compare the 16 dequantized-row dot products against the actual frozen LM-head outputs on saved raw hidden states. Account for native bfloat16/quantized matmul versus CPU float32 arithmetic. Record maximum logit error and whether a small top-two margin makes ordering numerically unstable. Do not claim exact parity without checking it.
- Cache TRAIN context features only for the same complete prompt construction and last-token position. Keep code mapping, catalog order, tokenizer/chat template, public context bytes and all feature keys pinned. Never append a teacher-selected code before feature extraction.
- Balance code/catalog positions in TRAIN. Two fixed orders would require 640 feature rows for the 320 original-plus-contrast TRAIN requests, unless exact cached keys already exist. Those are 640 full prompt forwards, not a few matrix operations. Group membership remains the original 80 TRAIN groups.
- Preserve source-factual candidate metadata and public constraints. TRAIN grades/ownership contrasts are private supervision; no persona-to-product table, teacher answer, expected choice or rationale enters the runtime prompt. Multiple plausible positives need an explicitly declared relevance objective rather than arbitrary single-answer conversion.
- Report untouched pretrained routing, learned initialization and any residual/ownership additions separately. Retain a random-init counterpart with equal data/steps to test whether initialization helps. Predeclare these before an independent evaluation; inspected pilot/demo outcomes are development evidence.

## What the two-order pilot can and cannot establish

Catalog-order agreement, routing probability mass and free-vocabulary top-token compliance distinguish useful semantic selection from code/position priors or forced ranking among implausible code tokens. An owned item remaining highly scored can indicate a source-conditioning failure even if its general product category is sensible. Per-person raw scores should remain visible; ten public demo profiles are insufficient for a generalization claim.

Low order agreement would motivate broader position-balanced data or a different candidate representation before adding capacity. High agreement with plausible choices would justify testing the initialization on TRAIN contrasts. Either observation remains a diagnostic; it does not prove that the raw LM head satisfies hard constraints or is safe to promote.

The 16-row CPU projection would be small, but the full catalog-conditioned Qwen prompt is 1,022–1,446 tokens in the frozen pilot. Feature extraction dominates that serving path. Measure the entire fresh prompt forward plus decision/guards, and then separately complete generated copy and browser readiness. Do not describe the small matrix projection as complete storefront latency. A later compression/distillation lane would need its own factual-quality and constraint tests.
