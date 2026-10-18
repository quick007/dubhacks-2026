# Actual-store function ranking v3

Completed64 native B1 encodes,80 learned and80 prior optimizer updates. All16
product identities are TRAIN-seen; evaluation uses16 independently authored
request wordings. This measures known-catalog adaptation, with no unseen-product
or human-conversion claim.

| Arm | Correct top1 /16 | Target in top6 /16 | Both-correct pairs /8 |
| --- | ---: | ---: | ---: |
| learned | 12 | 16 | 4 |
| prior | 1 | 6 | 0 |
| random | 0 | 6 | 0 |
| permuted-query | 2 | 5 | 0 |
| public-oracle | 16 | 16 | 8 |

Median contiguous fresh-query→serialized ranking endpoint: 203.702ms
(range191.238–208.786ms,16requests). This includes exact native
whole-query encoding, immutable query feature serialization/reload,16 verified
warm catalog reads, learned scores, hard guards, top6 and serialization. It
excludes generated text, images, browser rendering and startup. The timer includes
progress-log I/O; values are actual measured endpoints, not summed component medians.

Resident model load2.631s; native lifecycle including CPU
fit/checkpoint writes/endpoints15.503s;
run() wall20.474s (Python startup excluded).
MLX peak4,716,428,690bytes;6GiB guard retained.

Misses: weekend travel→daypack (duffel ranked4), cable organizer→field notes
(pouch ranked4), groceries→daypack (tote ranked4), over-ear listening→daypack
(headphones ranked2). All16 desired products remain in top6. Eight deterministic
hard-guard probes pass; they are separate from the unconstrained learned cases.
No arbitrary unknown-intent abstention is established. Exact-name policy, rich
history, negative cues, multineed ranking, image selection and full UI integration
are separate work. No copy adapter is promoted.

Source commit79990df7ef83c9f605375bf3d336bf5ba0a10268; learned checkpoint SHA-256
265c728a2e3349fa03e03ca37ca5d54814fe72407b95ed436d8f9734b602a99e. All original checkpoints/features remain at
`data/ranker/pretrained/actual-store-function-v3-attempt-1`. This archive preserves
JSON/JSONL evidence only. See ACTUAL_STORE_FUNCTION_V3_AUDIT.md/.json for independent
exact80+80 CPU reproduction and feature/score/payload checks.
