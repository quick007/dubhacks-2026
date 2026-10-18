# Model experiment ledger

Run names distinguish optimizer updates from numerical probes and inference.
The 9B quantized backbone weights stay frozen throughout these experiments.
Synthetic public-fact correctness is a proxy; none of these results establishes
human conversion lift or a production storefront integration.

| Run | Actual optimizer work | Checkpoint and outcome |
| --- | --- | --- |
| `semantic-copy-direct-v1` | Completed 336 updates, 65,536 final-layer `mlp.down_proj` LoRA A/B parameters | Saved `data/ranker/pretrained/semantic-copy-direct-v1/adapter.safetensors`; exact direct-loss reload. Dev titles 6→15/18; test titles 6→6/18; zero valid pages in each three-case split and model arm. Not promoted. |
| `gradient-feature-probe-v1` | Zero updates; one nonzero-adapter numerical control row | Direct native, auxiliary native and reconstructed-feature loss/A/B gradients exact. No trained checkpoint. |
| `catalog-user-fusion-paired-v1-attempt-1` | Completed 80 CPU steps for full head and 80 for independent prior | Raw-feature head had zero preference-driven product flips, including training. Saved full/prior/random checkpoints; not promoted. |
| `catalog-user-fusion-centered-v1-attempt-1` | Completed 80 CPU steps per full/prior fit | Train-only role means restore conditional image-style choices: validation 32/32, test 48/48. Product pairs remain 0/4 and 0/6 both correct. Not a complete-view result. |
| `catalog-user-semantic-factors-v1-attempt-1` | Completed 80 CPU steps per full/prior fit; full product model has 57,358 trainable parameters | Learned seven-atom sigmoid factors pass validation 8/8 and test 12/12 product choices, all paired interventions correct. Shuffled-query product controls fall to 1/8 and 4/12. Image checkpoint is byte-identical to centered-v1. Inspected synthetic replay, pending fresh transfer evaluation. |
| `semantic-copy-feature-fit-v1` | Separate prospective run: requires 504 exact extraction gates **before** any of its 504 planned globally shuffled updates | Live stage/counts are in its `data/ranker/pretrained/.../results.json`. A gate executes three loss/gradient comparisons and is never counted as an optimizer update. Only a completed fit can supply the next adapter checkpoint. |

The earlier 140/336 copy-update progress belongs to the now-completed direct
fit. A later 183/504 or similar zero-update report belongs to the separate
feature-extraction stage of the new fit. They are different runs.

The direct fit's all-attempt completion medians were dev 9.290→8.832 seconds
and test 9.648→9.468 seconds (frozen→learned). With zero valid pages, there is
no valid-page latency sample and no demonstrated usable-page speedup.
An actual held-out raw title changed from `Herilis 24L Daypack — compact` to
`Herilis 24L Daypack for structured travel`; `structured` was unsupported.

Current S/T/B copy experiments contain fourteen fields for six cards, capped at
224 sampled tokens. They omit visible per-card body copy. The distinct fixed
first-section workload requires ten fields including C1/C2/C3 (cap 208), exact
original-name/price/image metadata and a measured validated/serialized payload
receipt. First-token times and last-token lower bounds are not payload readiness.
Section grouping remains deterministic. No UI, API or shopper harness was changed.

Raw reports, checkpoint hashes, exact replay audits, token counts, wall time and
memory receipts are retained in each named result directory and tracked report
copies. Failed experiments and historical validation accounting remain intact;
new values-only attempts use strict field provenance and exact raw `NONE`.
