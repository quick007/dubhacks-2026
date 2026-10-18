# Actual-store copy v2 independent prefit audit

Status: no launch blockers found in the reviewed candidate. This is a CPU/code audit, not evidence of native training, model quality, GPU memory, or usable-payload latency.

Reviewed frozen SHA-256 values:

- `store_copy_fit_v2.py`: `7f63b9ac18f9142910694499b18c1a5427d4b2da13f0dc3be20860fb44b89968`
- `test_store_copy_fit_v2.py`: `3af9cbf42edde2731acf9d39f0c52aaa5ec931261e4b1f60be507b727d02ce2d`
- `store_copy_data_v2.py`: `ac424d601d8dc97d480d3c22ed0175de002e46281c7c35f9054f00bb32a8082e`
- `STORE_COPY_FIT_V2_PREFLIGHT_V1.json`: `cd8ead6445ac7785c90dfcfb0c15f6999e35a2da4c54530a59106db87cbb1cbd`

Independent verification:

- All 25 CPU tests passed: 12 runner and 13 data tests, using `.tools/pretrained-venv/bin/python -m unittest experiments.ranker.pretrained.test_store_copy_fit_v2 experiments.ranker.pretrained.test_store_copy_data_v2`.
- Recomputed the entire tokenizer preflight with the pinned local tokenizer. It exactly equals the frozen JSON, including all 480 train/eval field contracts, source/data/model receipts and token hashes. Every declared source hash matches current bytes. No native model execution occurred.
- Dataset freeze/check passes. TRAIN contains 16 requests and 320 fields; evaluation contains 8 requests and 160 fields. Each product occurs six times in TRAIN and three times in evaluation. Product/color states are disjoint: 32 TRAIN states and 16 evaluation states. The 288 synthetic fact swaps reject stale targets and accept the corresponding fact-updated targets.
- TRAIN maximum root/suffix lengths are 329/79 tokens; evaluation maxima are 333/78. Uncompressed TRAIN feature projection is 1,420,738,560 bytes, or 1,421,787,136 bytes including the declared reserve. These are tokenizer/shape projections, not measured native memory or compressed archive size.

Code review findings:

- Native teacher forcing extends the identical ordinary-chat source prefix and inference prompt; prediction positions align with target-only next-token loss and an explicit native EOS. Serving jobs are reconstructed from public request/bound facts and do not read teacher strings, constructors, case/split IDs, or persona metadata.
- Only TRAIN rows enter feature extraction and fitting. Evaluation target tokenization checks caps but supplies no evaluation numeric features or gradients. Final checkpoint selection is the fixed update count, without evaluation checkpoint selection.
- Each of 320 TRAIN rows must pass exact direct-native, auxiliary-native, and full-suffix feature loss/A/B-gradient comparisons under the same temporary nonzero-B control. Every control restores the exact original A and zero B, including exception paths. All 320 passing gates and exact true initialization precede any Adam update; there are 320 shuffled individual update calls with finite-state checks.
- Feature math preserves native per-tensor dtypes, full suffix shapes, LoRA scale/cast order, both residual additions, final normalization and target projection. Original archived feature bytes, metadata and per-tensor hashes are checked on load. Learned direct-native loss/gradient parity against the original archive is checked on feature index 0 only; the report explicitly does not claim 320 learned replays.
- Frozen base verification checks all 927 language tensors after extraction, fit, replay and evaluation. Zero/control and learned-mode source-cache equality are runtime assertions. Width-6 fork/rebuild/repeat controls compare tokens, EOS and complete distribution hashes; these assertions still require native execution to establish a passing result.
- Six cards, twenty fields and the 416 sampled-token cap remain intact. The first-three endpoint requires all ten first-section fields. Validation and actual UTF-8 payload serialization occur inside each continuous endpoint timer; first-section callback work remains inside full completion time. Invalid attempts have no valid-result latency sample. Ranking, images and browser paint are explicitly outside these copy-stage measurements.
- The fixed 24-arm roster exists before tokenizer verification, with eight cases per mode and fixed-denominator rates. Failure/cancellation receipts preserve progress and partial rows; deadlines cover gates, optimizer work, controls and requests. Native execution uses the shared `data/ranker/pretrained/native-owner.lock` and an exact reviewed-preflight checksum. Existing output directories are refused.

Limits: inspected synthetic labels, known product identities, held-out product/color bindings and query wording. The bounded relational grammar rejects some true English; it is not comprehensive semantic validation. Product selection, grouping and image/color binding are locked fixtures, so this experiment cannot establish learned ranking, image understanding, unseen-catalog generalization, end-to-end storefront latency or browser integration. A completed native result must be audited separately.

An initial pre-tokenizer failure-roster accounting gap was reported to the author and corrected before the hashes above were frozen. No reviewed producer, dataset, checkpoint or runtime file was edited by this audit; only this audit note was added.
