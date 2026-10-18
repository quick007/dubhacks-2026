# Public context preencoding measurement candidate

V3 is an unpromoted development measurement candidate. It inherits the unchanged V2 ranking formula, query gate, hard constraints, source-bound variant selection, and fresh parallel copy. It does not improve or reclassify the failed development quality result. No native benchmark has been executed by its author.

`store_persona_adapter_runtime_v3.py` adds a bounded CPU tensor cache for the learned color branch. Its key is the **entire exact `public_context_text`**, which includes the actual query, public budget/constraints, persona, and history. Default startup preencodes the ten byte-pinned public demonstration people with the UI's exact `Browse the store` query, preserving every other public field. This happens after base/rank adapter attachment and before READY. It runs no ranking, variant head, or copy generation. Startup and shutdown use V2's existing exclusive adapter ownership and integrity controls.

Every real request defaults to `forceFreshQuery=true`: V2's last-query feature slot is cleared before and after ranking. The actual normalized request text receives one fresh pristine-base B1 encoding. Fresh route-logit projection, query-intent scoring, color selection, and all applicable copy word slots still run per request. No response words or completed rankings are cached.

Both the color context key and the unchanged public20 rank X/H/Y key include the query. A new query may therefore trigger a new full color-context encode **and** dynamic rank-prefix/final16 capture. Preencoding `Browse the store` is not query-independent persona encoding. Unique-query performance must be reported separately; this candidate does not factor or neutralize the query.

## Receipts

`startup.persona.featurePreencoding` separates one-time `encodingMs`, actual `nativeBackboneForwardCalls`, `baseEncodingCalls` (native tokenization/forward timings), ten per-person key/tensor checksums, and zero ranking/copy warm responses. `warmCacheSha256` is SHA256 of UTF-8 `json.dumps(rows, sort_keys=True, separators=(',', ':'))`. This seals that particular startup receipt, including its measured costs. The pure tensor SHA hashes compact JSON `[str(dtype), list(shape)]`, one NUL byte, and contiguous CPU tensor bytes. Tensor content is checked before reuse.

A reused startup context reports `colorFeature.kind=preencoded-public-color-context`; a subsequently reused dynamic entry reports `exact-full-text-multi-color-context`. Each carries exact full-text `cacheKeySha256`, `featureTensorSha256`, runtime source SHA, warm-cache receipt SHA and `cacheOrigin`. Existing original-public20 and fresh-native kinds retain their meanings. The cache stores at most32 tensors by default; eviction does not supply a replacement result.

`queryFeature.encodingMeasurement` and `featureEncoding.baseCalls` carry actual `nativeTokenizeMs`, `nativeForwardMs`, padded tokens, batch rows and counted forwards. These are the original backbone measurements: tokenization includes native input creation; forward includes pooling, synchronization, host copy and finite/memory checks. `elapsedMs` additionally includes ownership scopes and validation. These intervals overlap and must not be added as exclusive phases.

`colorFeature.elapsedMs` includes formatting/cache acquisition, tensor checks, the original head call and cache bookkeeping. `rankDecision.elapsedMs` is unchanged. `featureEncoding.totalRuntimeRankMs` covers runtime rank only. `runtimeIntegrityChecksMs` measures inherited source/artifact integrity checks including file I/O, not the hard-constraint sorter and not pure CPU execution. Fusion, hard-constraint ordering and action construction remain within the rank residual. Engine admission/outer checks, copy generation/validation/serialization, transport and browser readiness remain separate.

Actual B1 encoding counts and rank decision body counts are separate (`baseB1EncoderForwardCalls`, `rankDecisionNativeBodyForwards`). Warm profile reuse does not imply zero total native work. The rank public-cache hit flag is always reported independently.

## Root-only execution after exclusive worker release

The entry is `python -m experiments.ranker.pretrained.store_persona_adapter_engine_v3`, with the exact artifact arguments already supplied to V2. It retains port8767, width20 and the existing7GiB guard. Only three new controls are added:

```text
--force-fresh-query / --no-force-fresh-query
--preencode-public-profiles / --no-preencode-public-profiles
--color-cache-entries 32
```

The first two default true. Use the following fixed arms, preserving complete outputs, all six eligible cards/20 copy fields, exact request identities and full-forward counts:

1. Current/cold control: V2, or V3 `--no-preencode-public-profiles --no-force-fresh-query --color-cache-entries 1`. A repeated request may hit the original last-query cache; record it rather than comparing it as fresh-query work.
2. Preencoded profiles with fresh query: V3 `--preencode-public-profiles --force-fresh-query --color-cache-entries 32`, same ten exact public `Browse the store` requests in fixed order. Compare full ranks, scores, source variants and generated word IDs, not just latency.
3. Unique actual queries: the same warm V3 worker, distinct public query text for each request. Preserve actual rank/color misses; do not report these as preencoded persona hits.

Keep one-time initialization and profile encoding cost apart from per-request model and browser readiness. Preserve the ordinary six-card/20-field workload and record any legitimate hard-constraint shortfall. Use fresh unique output directories and the existing global native lock. No automatic launch, endpoint replacement, accepted profile or model promotion is included here.

CPU verification: 10 V3 tests plus17 unchanged V2 tests pass. They cover full-text A/B/A cache reuse and query misses, no head execution during prime, tensor corruption, bounded eviction and failed forwards, exact public UI requests, forced query behavior including failure cleanup, actual native timing transfer, startup policy propagation and inherited owned cleanup. Native numeric/output parity and latency remain pending root execution.
