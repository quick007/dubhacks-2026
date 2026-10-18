# Actual-store copy fit v2 independent audit

Status: passed; 18,734 CPU checks, zero issues. The native run completed all 320 exact row gates, 320 optimizer updates and 24 inference arms. This audit verifies experiment execution, artifact integrity, token decoding and the frozen contract validator. Contract acceptance is not an independent semantic-factuality score.

Audited native receipt: `data/ranker/pretrained/actual-store-copy-fit-v2-attempt-1/results.json`, SHA-256 `f995a3ad9e6184ccd907b67a231de220f764d204ff914483a7a878efc8efe8ae`.

Checkpoint: `data/ranker/pretrained/actual-store-copy-fit-v2-attempt-1/adapter.safetensors`, SHA-256 `56c6c1f4199d8689f1920d871213a46d2d121fec4e5836ff70e33aaf98c7c7b8`. Actual saved A/B tensor shapes, FP32 dtypes, finite values and raw bytes match the producer's exact reload receipts. Both A and B changed from initialization.

| Mode | Contract-valid first3 | Contract-valid full6 | Headings /16 | Titles /48 | Badges /48 | Bodies /48 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen | 0/8 | 0/8 | 0 | 41 | 0 | 0 |
| Failed daypack adapter | 0/8 | 0/8 | 0 | 0 | 48 | 0 |
| New store adapter | 2/8 | 1/8 | 16 | 43 | 48 | 25 |

All arms completed. There are no failed, unattempted or in-progress arms. Each arm generated the original twenty fields for six cards at width six and the unchanged 416 sampled-token cap. Natural sampled-cap failures remain failures of the contract; no outputs were repaired, padded, templated or target-fed.

| Mode | Attempt first3 median | Attempt full6 median | Contract-valid first3 latency samples | Contract-valid full6 latency samples |
| --- | ---: | ---: | --- | --- |
| Frozen | 5.008 s | 6.225 s | None | None |
| Failed daypack adapter | 4.798 s | 6.674 s | None | None |
| New store adapter | 5.669 s | 6.836 s | 5.788 s, 5.577 s | 6.710 s |

Timings include the fresh ordinary-chat root, full first-section or full-page required fields, validation and actual UTF-8 payload serialization. They exclude ranking, images, browser paint, cold model loading and process startup. Invalid attempts have no usable-result latency sample. The new adapter's first3 valid-only median is 5.683 seconds, based on just two samples; the full6 sample is a single result. These values do not establish complete storefront latency or a speedup.

The native run wall time is 561.633 seconds from `run()` entry, including tokenizer/preflight and shard verification but excluding Python startup. Peak producer MLX allocation was 6,148,503,274 bytes, about 5.73 GiB, under the declared seven-GiB guard. Compressed TRAIN feature archives plus metadata total 685,004,406 bytes.

Verification details:

- All 320 original feature archives and metadata passed independent checksums. Each archive was decompressed one row at a time and checked for original full-suffix tensor shapes, native dtype receipts, FP32 exact host representation, finite values, native per-tensor byte hashes and complete EOS supervision.
- Every gate's actual direct-native, auxiliary-native and feature-loss/A/B-gradient receipts agree exactly. All token/source/prompt hashes match the frozen tokenizer plan. Every extracted row belongs to TRAIN and appears in the expected order; no evaluation features were extracted.
- The initial B receipt matches actual all-zero FP32 bytes. The temporary probe B receipt matches actual `1e-5` FP32 bytes, with original A preserved. All 320 gates report exact restoration before the 320 seeded individual Adam calls. Update row identity, permutation, finite loss/gradient and evaluated-update receipts were independently checked.
- The learned original-archive parity proof covers feature index zero, with matching native/auxiliary/feature loss/gradient receipts and original archived feature hashes. It does not claim 320 learned replays. Frozen927 equality and native cache invariance are authenticated producer evidence; the CPU auditor does not repeat GPU tensor/cache/gradient computation.
- All three width-six fork/rebuild/repeat control paths agree in each of the three modes, including raw tokens, EOS and full distribution hashes. Saved control files match report values.
- All 480 raw output rows were independently decoded with the pinned local tokenizer. Every completed first3/full6 payload was rebuilt through the unchanged frozen validator and compared exactly, including its raw rows, rejection lists, payload checksum and UTF-8 byte count. Fixed eight-case rates, timings, token totals and summary counts agree.
- Every reviewed producer source hash and frozen dataset hash still matches. The auditor creates separate new reports and does not modify producer files, checkpoints, feature archives or the UI.

The new adapter's most frequent contract rejections are 23 body values outside the bounded relational grammar and five titles outside its color relation grammar. Four bodies omit or alter the exact price, and four violate the sentence/length rule; one identity and one selected color are also missing or altered. Categories overlap. Some grammar rejections may still be factual English, so these counts must remain separate from the independent qualitative factuality review.

For evaluation case zero, new-store C1 is `Everywhere rain jacket is shown in Forest with an item price of $128.00.` and passes the contract. C2 is `Steel lunch box with selected Rust band has an item price of $32.00.` and fails the finite relational grammar. C3 is `Everyday round sunglasses with selected Black selected` and fails the required price and finished-sentence rules. The audit JSON preserves raw examples from all three modes.

This remains an inspected synthetic, known-product, held-out product/color-binding evaluation with locked product/image/grouping decisions. It does not establish unseen-catalog transfer, learned selection, image understanding or production integration. The adapter is a preserved experimental checkpoint, not a promotion recommendation.

Reproduce with `.tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.audit_store_copy_fit_v2 data/ranker/pretrained/actual-store-copy-fit-v2-attempt-1 --report NEW_REPORT.json --check-features --decode --require-complete`. Full checks and summaries are in `STORE_COPY_FIT_V2_INDEPENDENT_AUDIT_V1.json`; prefit reasoning remains in `STORE_COPY_FIT_V2_INDEPENDENT_PREFIT_AUDIT_V1.md`.
