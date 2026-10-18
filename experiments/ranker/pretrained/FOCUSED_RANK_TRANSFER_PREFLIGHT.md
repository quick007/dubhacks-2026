# Focused rank transfer preflight

CPU-only preparation completed. No native calls, fitting, or GPU work performed by this lane.

- Source: completed `data/ranker/pretrained/generalization-v2-attempt-1`; HEADS_FROZEN, six head metadata/checksums, upstream packet, raw feature freeze and all raw bytes verified. Only frozen focused fusion/prior/random heads transfer.
- Target: frozen old **inspected** `catalog_fusion/fresh_eval_v1`, 36 public products and 24 public requests. No new labels or dataset edits.
- Native budget: exactly 36 focused product B1 forwards, then 24 fresh focused query B1 forwards. Fixed 6 GiB active/peak guard, 512-token cap, 180-second extraction/load guard, 60-second two-forward pilot. Full local weight shard verification before loading MLX; no download path.
- Exact input format: source `catalog_text_v2(...,'focused')` and `query_text_v2(...,'focused')`. Actual catalog inputs 18–25 tokens, query inputs 21–25 tokens. No label, hidden persona, price, routing ID, or image-style feature enters this focused encoder. Deterministic guards consume public budget/stock/shipping afterward.
- Five fixed arms: learned, empirical prior, random, opposing public preference query, and public tag oracle. Fixed denominators: 24 requests, 12 product preference pairs, top12/top24, three roster rotations. Ranking canonicalizes product ID order before the fixed 36-row Torch head calculation and maps scores back; each rotation reruns the head and requires exact ID-bound score and readout parity. This avoids numerical changes caused solely by GEMM row order.
- Every warm measured request performs actual fresh query encode/save/verify, 36 verified cache loads, frozen learned head scoring, hard masks and both top-k readouts. Cold catalog and resident model load are separate. Query timing includes feature serialization/checksum verification; no first-token/page timing claim.
- Features and predictions are frozen before private evaluation labels open. Failures retain fixed denominators and observed completed forward/query counts; no partial promotion.

Four CPU tests passed in 0.582 seconds. Tests include a full 60-call **CPU stub** orchestration run, private-label opening boundary, exact rotation/cardinality/hard guards, corrupt checkpoint hash/metadata, nonfinite/incomplete scored roster, and wrong encoder provenance. Stub output is explicitly labeled `cpu-test-stub`; its quality numbers are not native evidence.

Frozen packet: `focused_rank_transfer_prepare_v1/packet.json`, SHA-256 `02055b7f4d4612398f6bf947eb78cab072298b6b59e4f0da9165235757fe99bd`.

Script SHA-256 `e544643f8e5a742c3867a25b2bd9e5e07a7f3a4cedf3c342347753e7c6837ccd`; tests `03112956a241292b11d9960f976fdd178584bd134e22e006c8a935dcccb3ccbd`.

The target has wider candidate cardinality, but focused public atom text repeats source semantics. Exact target text overlap occurrence counts with source train/validation/test are catalog 9/3/3 and query 2/10/6; these sets overlap and must not be summed as unique rows. The 24 requests repeat 12 preference triples across image styles. This evaluates product ranking in an inspected larger pool; it cannot establish unseen-persona, hidden-history, image-selection, generated-copy, full-grid payload, storefront conversion, or end-to-end page latency quality.

Root-only command after serialization/worker inventory review:

```sh
.tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.focused_rank_transfer run --out data/ranker/pretrained/focused-rank-transfer-v1 --packet experiments/ranker/pretrained/focused_rank_transfer_prepare_v1/packet.json --reviewed-packet-sha256 02055b7f4d4612398f6bf947eb78cab072298b6b59e4f0da9165235757fe99bd --native-owner root --confirm-serialized
```
