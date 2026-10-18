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
| `semantic-copy-feature-fit-v1` | Completed 504 exact extraction gates, then 504 real globally shuffled optimizer updates, 65,536 final-down LoRA parameters | Saved and exactly reloaded `data/ranker/pretrained/semantic-copy-feature-fit-v1/adapter.safetensors`. Dev pages 0→2/3; test pages 0→3/3; trained titles 18/18 in each split. Peak MLX 6,360,122,248 bytes. |
| `semantic-copy-learned-probe-v1` | Zero updates; one exact learned-parameter control row | Native loss/A/B gradients match both recaptured and original archived features exactly; full source cache and frozen language tensors unchanged. |

The earlier 140/336 copy-update progress belongs to the now-completed direct
fit. A later 183/504 or similar zero-update report belongs to the separate
feature-extraction stage of the new fit, which subsequently completed all 504
updates. They are different runs; both checkpoints and reports are preserved.

The new fit's complete report contains all twelve frozen/learned evaluation
arms and all 504 per-row gate receipts. All 504 stored feature archives passed
independent checksum, shape, dtype and finite-value verification. The learned
checkpoint SHA-256 is `0b0aeda377a9c9ad7eb37b529c139dac84bc5acd4bb559176d40245865ffe1b2`.
All 927 frozen language tensor hashes are unchanged. The reported fit/extraction/
evaluation wall time is 779.472 seconds and excludes earlier startup token/file
verification; whole-process startup wall time was not separately captured.
Five valid learned pages have median 11.342 seconds and range 11.099–11.725.
Frozen controls have no valid-page latency sample. These times include the full
rank context encoding but use locked fixture decisions; they are copy-stage
measurements, not newly measured ranking or complete visible-card endpoints.
The one learned dev rejection is `Structured travel choices` where the selected
section's universal attribute intersection is empty. Validation was not weakened.

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
