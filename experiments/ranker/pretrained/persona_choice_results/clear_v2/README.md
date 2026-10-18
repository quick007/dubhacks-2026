# Clear-v2 result: no quality promotion

The single bounded fit completed mechanically, but the learned adapter failed the quality gate and underperformed frozen base on the same authored packet.

| Canonical packet-v4 metric | Learned raw | Frozen base raw | Learned fused | Frozen base fused |
|---|---:|---:|---:|---:|
| Primary top1 /32 | 19 | 23 | 19 | 22 |
| Novel relevant top1 /32 | 24 | 28 | 24 | 27 |
| Strong positive top3 /32 | 30 | 32 | 30 | 32 |

Primary19/32 misses the20/32 threshold and the strict contribution requirement. Both pipelines passed4/4 explicit replacement cases. Known DEVELOPMENT history had8/10 owned-item priority demotions but only5/10 novel relevant top choices afterward, below8/10. Product prices/typed eligibility had zero violations; that check does not validate selected variants, images, generated copy or browser readiness. Canonical order is primary; reverse order is robustness only. Packet v4 is now inspected DEVELOPMENT. Earlier candidates used different packets, so no cross-packet improvement is claimed.

The fixed objective used256 views of audited128 TRAIN profiles for14 passes, with a .25 frozen-base soft-target CE term from640 legacy public-context views:3584 updates, no legacy authored-priority targets in the optimizer. The native lifetime took532.0196 seconds, including976 reused basis rows,264 freshly gated PUBLIC rows, fit, controls and prediction freezes. This is not request latency. Peak MLX memory was6,046,273,826 bytes under the7GiB guard.

All102 source/data pins and927 frozen tensors were preserved. Four initial zero/nonzero paired-objective controls and four learned native/original-archive/target-free controls passed exactly; checkpoint reload passed. The pure public cache contains20 X/H/Y entries without targets. Query preparation used20 fresh base forwards plus16 byte-exact reused vectors. All344 records for each of four raw/combined models were frozen before private quality labels were parsed.

`SUMMARY.json` contains the measurements and limits. `receipts/audit/ADJUDICATION.json` is the byte-exact independent decision. `MANIFEST.json` inventories copied text receipts/source and omitted originals. `native-report-subset.json` is explicitly derived; the full original report/plan, predictions, feature archives and checkpoints remain at their original local paths. No weights, NPZ, feature matrices, full feature inventories or large full plans are included. No new optimizer, native, browser or serving action was performed to build this archive.
