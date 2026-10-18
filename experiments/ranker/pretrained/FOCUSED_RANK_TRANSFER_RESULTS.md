# Focused rank transfer results

Independent CPU audit passed for the root-owned native run at commit `3ca08b274e6dedff2772ab9480621fdd1fc5b965`. The frozen focused-v2 learned head chose the unique gold product in **24/24 requests**, with **12/12 preference pairs both correct and changing selection**. There was no fitting in this run.

| Fixed arm | Unique gold /24 | Pairs both correct /12 | Pair selections change /12 | nDCG@12 | nDCG@24 |
|---|---:|---:|---:|---:|---:|
| Learned | 24 | 12 | 12 | 0.995754 | 0.995405 |
| Empirical prior | 2 | 0 | 0 | 0.579183 | 0.687479 |
| Random head | 0 | 0 | 0 | 0.357094 | 0.545477 |
| Opposing query | 0 | 0 | 12 | 0.395051 | 0.587542 |
| Public tag oracle | 24 | 12 | 12 | 1.000000 | 1.000000 |

The opposing-query arm changes selection but selects the wrong target, which distinguishes responding to a query from responding correctly. The prior and random arms do not adapt their selection across preference pairs. Learned top-one quality reaches the oracle here; its lower-rank order remains slightly imperfect.

All five arms had **zero hard guard violations**, each over 144 complete grid components (24 requests × three roster rotations × top12/top24). Every grid returned its full requested count: zero shortfalls. This audit checked 2,592 selected slots per arm, including repeated slots across the three rotations and both grid sizes. These are product-ID lists with deterministic public stock/budget/shipping guards, not complete storefront cards.

The actual learned warm endpoint consists of one fresh query native encode, feature serialization and verification, 36 verified disk-cache product loads, frozen head scoring, hard guards, and both top12/top24 readouts. Across 24 measured requests its median was **124.137 ms**, range **119.719–125.510 ms**. Component medians were native query 115.929 ms, query encode/save/verify 117.025 ms, verified catalog cache 7.029 ms, head 0.195 ms, and guards/top-k 0.049 ms. Component medians are separate statistics and need not sum to the endpoint median. Per-request accounting residuals were 0.00067–0.00187 ms.

Cold catalog encode/save/verify took **4.232 seconds**; resident model load took **2.370 seconds**. The saved run-phase timer was **10.033 seconds**, beginning after head verification/copy and FeatureStore setup. The externally saved whole-child-process timer was **13.382 seconds**, including imports, CPU preparation, full local shard verification (2.418 seconds), model load, native extraction and evaluation. No timing here measures rendered cards, image choices, generated text, browser rendering, or a full page. Peak observed MLX allocation was **4,649,075,650 bytes (4.330 GiB)**, under the 6 GiB active/peak guard.

The audit independently:

- Verified the unchanged exact packet, source inventory, source HEADS_FROZEN/raw feature freeze, three copied checkpoint files and their metadata, native model/shard receipts, current model file stats, tokenizer bytes and all 1,310 public input tokens. The audit checked the saved full-shard receipt against pinned shard hashes; it did not reread six GB of weights or load MLX.
- Loaded and validated all 60 raw `(1,4096)` features and all 120 serialized feature/receipt files, plus 60 B1 native receipts with no truncation. Recomputed 35,400 saved score/probability scalars exactly from those CPU features and frozen heads.
- Recomputed 360 arm/roster views, exact ID-bound scores after canonical row ordering, both top-k lists, all hard masks, all quality aggregates and the 12 product-pair controls against the unchanged private labels. The opposing mapping matches the frozen controls exactly.
- Confirmed the saved prediction freeze and source code's labels-after-freeze protocol, and checked all 132 run output files unchanged before/after this audit. A later audit cannot reconstruct historical file-open events; the CPU orchestration regression test separately exercises that boundary.

Packet SHA-256: `02055b7f4d4612398f6bf947eb78cab072298b6b59e4f0da9165235757fe99bd`. Native report SHA-256: `dcb86e7faf918182f10082c9285b8ad9ed331b6e6a5d0b38c6fbf7f028ca5c44`. Full checks, artifact hashes, five-arm aggregates and timer details are in `FOCUSED_RANK_TRANSFER_RESULTS.json`. Native artifacts remain unchanged at `data/ranker/pretrained/focused-rank-transfer-v1`.

This is an **inspected synthetic wider-pool replay**, not a new sealed holdout. It tests selection among 36 products and full top12/top24 lists. Its 24 requests repeat 12 preference triples across image styles, and focused atom text overlaps earlier source splits: catalog occurrence counts train/validation/test 9/3/3, query 2/10/6 (overlapping sets; do not sum as unique). The focused representation uses named positive public atoms; hard constraints are applied afterward. These results do not validate hidden persona/history learning, unseen real-store attributes, image ranking, fresh copy, conversion, or end-to-end page latency. The next bounded component check remains freshly trained same-backbone copy with unchanged factuality and complete-payload validation.
