# Explicit adapter candidate runtime

This is CPU-tested integration preparation, with no accepted profile, native run,
service launch or quality promotion. The original persona/native entry remains
available. The new entry is `store_persona_adapter_engine_v2`; it requires every
artifact path and checksum explicitly and uses the original asset-root GPU lock.
It binds 127.0.0.1:8767 and preserves the existing local 4313 CORS policy.

The product branch uses the trained rank4/scale20 final layer31 MLP-down LoRA and
16 product-bound pretrained lexical LM routes. It never samples answer tokens.
The exact public721-token catalog prefix is prepared with both adapters disabled.
Each dynamic public request retains its complete prompt, and only its actual last
16 tokens run through the rank adapter. Input limits are 12 observable events,
six background strings, four color preferences and 500 characters per text value;
excess input is rejected rather than clipped. No target or teacher input is accepted.

The parent measured a maximum0.125 route-logit difference between plain native
inference and the original gradient-mode TRAIN feature basis in the zero/nonzero
mechanical pilot. Therefore this runtime does not admit plain native dynamic
inference. A miss uses the frozen target-free sum16-public-route-logit gradient
capture helper, discards gradients without an optimizer, then scores the captured
X/H/Y with the same plain reconstruction used for a public20 feature-cache hit.
Only rank A/B are trainable in that transform. The exact co-located certificate
must separately prove zero/nonzero and all four learned postfit controls; a TRAIN
loss/gradient gate alone cannot unlock serving.

The loader requires `serving.json` version `persona-choice-adapter-serving.v1`,
3584 completed updates, original927 unchanged, exact helper/source/model/catalog
pins, rank4/scale20 A/B-only safetensors, pure public20 X/H/Y archives, and a
checksummed `inference-parity.json` certificate. The certificate fixes
`cachedScoringMode=plain` and `dynamicMode=gradient-public-XHY-last16`, binds the
adapter and capture/scoring helper SHAs, and contains distinct proof SHAs for four
learned controls. Public archive keys are exact whole-prompt token SHAs, with no
person/case IDs, labels, target weights, grades or teacher strings. Relative
co-located paths have no original-path fallback. Every matching archive is hashed
before reuse; all20 are verified at startup and controlled shutdown.

A separately pinned learned query-intent head consumes the exact original frozen
base4096 feature of the existing full v5 request text. The shared combination is
`(1-pSpecific)*softmax(LoRA) + pSpecific*(v5 + 0.05*softmax(LoRA))`.
There is no manual phrase whitelist. The balanced query-intent v2 preparation had
a source-validation failure. The runtime now explicitly binds the reviewed v3
repair and TRAIN manifest25fdb0c976dc3b6e01e826e7f30ea63fc65e3fbb85f56f24f4c83e8538703232.
No gate artifact currently exists, and candidate launch remains pending.

The reviewed raw2500 context model supplies learned within-product variant scores
only. Its failed product ordering is discarded. Product ordering always comes from
the new LoRA/query blend and unchanged strict budget, stock/size/color, waterproof
and exact-name guards. Returned actions bind real source variants and declared
image metadata. There is no learned image/pixel claim. This combination still
needs independent variant and live fact validation.

Rank, copy and pristine base projections share only the original frozen linear
object. A serial identity owner explicitly switches among three distinct objects,
synchronizes before/after each scope, forbids nesting, and restores the original
copy object in `finally`, including cancellation and native failures. Fast copy
uses the existing fresh parallel word head on selected public product/color facts;
no response words are cached, no persona/query text conditions those copy slots,
and no native copy/rank adapter sum occurs. Zero eligible products emit a truthful
`no-generation` event with no padded cards. Receipts preserve exact inputs,
feature reuse flags, actual forward counts, generated words, validation, failures
and serialized rank/firstThree/complete timings.

The separate combined comparison must freeze every model's predictions before
private gold is parsed. The same gate probability, v5 scores and hard guards must
apply to learned, frozen-base lexical, TRAIN-popularity and latest-owned controls,
including all input controls. A combined prediction file carries raw16 source
scores, pSpecific, v516, final16 scores, learned order and guarded order; actual
variant/image/copy fields may be included only when computed. The parent seals
`ALL_MODELS_PREDICTIONS_FROZEN.json` with `goldParsed:false`, artifact checksums,
checkpoint/packet pins and explicit adapter/queryGate/allControls stage flags.
Missing variant/image/copy receipts leave those gates pending.

CPU verification:

```sh
/Users/lukas/Documents/Projects/dubhacks-2026/.tools/pretrained-venv/bin/python \
  -m unittest experiments.ranker.pretrained.test_store_persona_adapter_v2 -v
```

The 15 tests cover the real input/catalog contract, last16/prefix/target-free
shape rules, exact feature checksums, certificate/mode/helper rejection, shared
pristine-base attachment and A/B-only loading, ownership restoration on error and
cancellation, initialization/shutdown cleanup, and empty hard-guard receipts.
They use injected CPU fixtures; they make no native parity or quality claim.
