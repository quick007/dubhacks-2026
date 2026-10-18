All20 planned pairs pass exact rank, generated-word/slot-ID, action, validation and provenance parity. Every timed arm returned6 cards and20 accepted fresh parallel fields. All40 timed arms are retained, with two startup warmups excluded.

| Server event | Original p50 / p95 ms | Profiled p50 / p95 ms |
| --- | ---: | ---: |
| First3 serialized | 189.1490 / 198.6442 | 191.7717 / 202.8072 |
| Full6 serialized | 192.7846 / 202.3703 | 195.6975 / 207.3403 |

Paired B−A full6 delta: median 3.7815ms, nearest-rank p95 13.5610ms, range -9.5981 to 14.4371ms. Difference of global medians is a different statistic. This measures observer cost on the same fast model; it is not a generic UI latency comparison, browser paint or a speedup experiment.

Profiled exclusive median fresh backbone encode is 170.6221ms; catalog loads 4.3750ms; parallel text head forward 0.14506ms. Whole parallel generation inclusive median is 1.0270ms. The tiny head-forward number does not describe full request latency. Native encode includes its existing completion/pooling/transfer boundary; there is no isolated GPU-kernel timing.

The exclusive partition was independently recomputed from every raw parent/child interval. Inclusive times overlap. Phase medians cannot be added to obtain the median total. No observer cost is subtracted from reported request times.

Attempt1 remains a failed excluded B warmup: TokenizerWrapper encode restoration raised AttributeError, zero timed arms ran and all40 were unattempted. Its original sources, error and full927/adapter/head/export shutdown proofs remain in the inventory. Attempt2 reports all corresponding integrity checks true. Neither run is overwritten.

The lossless inventory covers 96 original JSON/source files (13,492,706 bytes), including all raw request receipts and both preflights. It includes no tensors. Compression is deferred until the browser timing window is released. The companion audit script can produce and verify a new archive with an explicit --archive path.

All40 stored logical signatures were also independently reconstructed from the original per-request receipt files, including field word IDs, finish reasons, rejections and endpoint metadata. Native receipts each record one fresh feedforward call and zero teacher strings read.
