# Native function v5 independent audit

The completed v5 run is a useful candidate for ranking single-function requests within the same 16-product catalog. Exact CPU replay confirms **29/32 top choices and 32/32 targets in the first six**, versus **14/32 and 29/32** for frozen v3 on these same independently authored synthetic queries. These are new wordings for known products, not unseen products, human evaluation, or broad profile personalization.

| Frozen arm | Top choice | Target in first six |
| --- | ---: | ---: |
| v5 learned | 29/32 | 32/32 |
| frozen v3 | 14/32 | 29/32 |
| random initialized head | 2/32 | 13/32 |
| equal-score prior | 2/32 | 12/32 |
| v5 with fixed query permutation | 1/32 | 13/32 |

The paired comparison has 16 improvements, one regression, 13 cases correct for both heads, and two wrong for both. Blanket, lamp, and crewneck each score one of two; the other 13 functions each score two of two. The remaining v5 misses are `v5-eval-13` (blanket, selected pouch, target rank 4), `v5-eval-14` (lamp, selected lunchbox, rank 2), and `v5-eval-23` (crewneck, selected headphones, rank 5).

The original scorer accidentally reused the last loop's target while reporting miss ranks. A bounded fix uses each case's own target. A regression with two distinct misses fails before the fix and passes after it, including reversed label order. The aggregate top-choice and first-six counts were already correct. Across all five arms, 105 diagnostic rows change; no score, ordering, selected product, model weight, or aggregate count changes.

`CORRECTED_RESULTS.json` and `CORRECTED_REPORT.json` are new receipts. The original report remains byte-identical with SHA-256 `cac2ba0ea15809b5ca6ddbc0037966d9809c56973fc51bb8ec8b2b5b9f0d77d5`. The original frozen predictions SHA is `eb8728f78539db3a9443b8ff0cb0da461a734acd6c62de621d4f8d70d30f543a`; the learned checkpoint SHA is `5d07673e51450bbbbb1803c8a44ef99721b39fd3b1ae92edb9168e27a1377d6b`. `ORIGINAL_RUN_FILES_SHA256.json` preserves all 310 original file hashes.

The independent CPU audit passed 1,317 checks. It validates the frozen dataset and archived source pins; all 144 feature records, raw keys, text/token identities, tensor hashes and shapes; the 48 reused features against their original files; all 96 new native receipts (64 TRAIN followed by 32 evaluation); and all 32 completed evaluation slots. It exactly reproduces every score and ranking in all five frozen arms from their CPU checkpoints and verifies each fresh endpoint's action binding and serialized payload hash. Original files are checked unchanged before and after replay.

Training membership, target digest and normalization means contain only the balanced 96 TRAIN queries. The source fixes 80 training steps. Checkpoint freeze precedes evaluation feature creation, and the hash-linked prediction freeze is written before evaluation labels are parsed in the archived driver. The initial random checkpoint exactly matches the deterministic initial head; learned weights differ. The source uses native batch size one and records one complete, untruncated forward per new feature. The producer reports all 927 frozen backbone tensors unchanged and its archived source performs the actual before/after comparison. This CPU audit verifies those receipts and code boundaries; it does not recreate native tensor hashes or provide a syscall-level file-access attestation.

The recorded fresh encode-through-serialized-actions median is **1,357.224 ms**. It includes feature and report writes, 16 verified catalog reads, repeated `load_snapshot` provenance/asset checks, learned scoring, guards and serialization. It excludes copy, images, UI and browser time. Evaluation encoder receipts have a 201.269 ms median; that substage is not live serving latency, and unmeasured disk-check overhead must not be subtracted to invent one. The run records a 4.889 GiB peak. All 16 canonical guard probes pass, while the separate `including-delivery-alias` diagnostic fails and remains visible.

The recommended next step is explicit v5 serving integration with a new live measurement and browser relevance check. All 48 copy bindings are already TRAIN-seen. This audit does not establish new copy quality, combined request latency, visible readiness, multiple-need preference handling, or all natural-language hard constraints. The historical notebook browser failure needs a new measured request against the v5 adapter before claiming it fixed.

Reproduce the CPU receipt audit from the repository root:

```sh
.tools/pretrained-venv/bin/python -B -m experiments.ranker.pretrained.function_v5_audit_v1.AUDIT_NATIVE_FUNCTION_V5
```

The script performs CPU checkpoint inference only, never native extraction or fitting. It refuses changes to the 310 original files and refuses replacement of differing audit receipts. Future mutable tests now use the current explicit v10 image binding, while a new test keeps stale v9 rejection required. Historical v9 preflight, native sources and receipts stay untouched. Test and archive results are recorded separately in `VERIFICATION_AND_PACKAGE.json` when complete.
