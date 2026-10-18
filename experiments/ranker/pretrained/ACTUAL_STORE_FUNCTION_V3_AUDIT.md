# Actual store function v3 independent CPU audit

Audit passed. All 64 feature receipts, checkpoints, source and dataset freezes, 80 learned plus 80 prior optimizer steps, TRAIN-only means, full 16-product scores, eight guard probes, endpoint payload hashes and fixed denominators were independently checked. CPU replay reproduced the learned and prior checkpoint tensors and every recorded training loss exactly. No GPU was used.

The learned head achieved **12/16 top-one**, **16/16 top-six**, and **4/8 preference pairs with both choices correct**. Choice changed in 6/8 pairs. TRAIN accuracy was 32/32. All 16 catalog identities and function classes were TRAIN-seen; this is request-paraphrase adaptation on the actual known store, not unseen-catalog generalization.

| Evaluation arm | Top one /16 | Top six /16 | Both correct /8 | Choice changes /8 |
| --- | ---: | ---: | ---: | ---: |
| learned | 12 | 16 | 4 | 6 |
| prior | 1 | 6 | 0 | 0 |
| random | 0 | 6 | 0 | 4 |
| permuted-query | 2 | 5 | 0 | 8 |
| public-oracle | 16 | 16 | 8 | 8 |

The public oracle uses evaluator labels, is deterministic and is not a serving model.

The four missed evaluation requests were:

- `store-eval-03`: A travel bag for a weekend away. Chose **Everyday daypack**; target **Weekender duffel** ranked 4.
- `store-eval-05`: A zipped pouch to keep loose cables together. Chose **Pocket field notes**; target **Zip essentials pouch** ranked 4.
- `store-eval-08`: Something to carry groceries home in. Chose **Everyday daypack**; target **Market canvas tote** ranked 4.
- `store-eval-13`: Something that goes over my ears for listening to music. Chose **Everyday daypack**; target **Everyday over-ear headphones** ranked 2.

The measured median ranking endpoint was **203.702 ms** across all 16 fresh requests (range 191.238–208.786 ms). The timer spans fresh native query encoding, immutable query receipt serialization and verification, 16 verified warm catalog reads, learned scoring, hard guards and serialized top-six ranking payload. It excludes model startup, generated copy, image decisions, browser rendering and complete useful above-fold readiness. This audit replays payload hashes, not native timing.

Stock and size constraints abstain because those attributes are unknown. Budget guards use inclusive item price only and do not assert delivery affordability. All eight guard probes and explicit shortfalls replayed exactly.

Learned checkpoint SHA-256: `265c728a2e3349fa03e03ca37ca5d54814fe72407b95ed436d8f9734b602a99e`.

The source order establishes 48 TRAIN feature extractions, checkpoint freeze, then 16 fresh evaluation extractions. Evaluation labels enter only after predictions freeze; earlier file hashing is an integrity check. The failure receipt paths were separately CPU-tested before launch. The deadline is cooperative. `nativeExtractionSecondsIncludingLoad` includes fitting and endpoint work and should be read as lifecycle time.

No audit blockers remain. The four ranking misses and absent generation/UI measurements remain substantive quality and integration limitations.

Detailed checks, request text, file hashes and training/control tallies are in [ACTUAL_STORE_FUNCTION_V3_AUDIT.json](ACTUAL_STORE_FUNCTION_V3_AUDIT.json).
