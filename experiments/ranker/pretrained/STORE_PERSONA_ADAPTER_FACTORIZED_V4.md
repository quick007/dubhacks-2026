# Factorized public-profile serving experiment

V4 uses the same1e88 LoRA adapter, c524 learned query gate, original v5 query head, learned color-only branch and fresh parallel copy head. It trains no weights and does not change the failed quality result. **Its context conditioning differs from V2/V3.** Neither quality nor exact output parity with those versions is implied. V2/V3 and their launchers remain available unchanged.

For context features only, `profile_request(actual)` constructs exactly:

```text
query: Browse the store
itemBudget: null
hardRequirements: []
persona: exact supplied public persona
publicHistory: exact supplied observed history
```

Only query/budget/hard-requirement conditioning is removed. There is no persona ID or demonstration-ID lookup. The profile key hashes the compact canonical normalized public request. Public history order, event data and all persona strings remain unchanged. The existing lexical/color formatters determine semantic features; opaque event metadata remains in the profile key even where the original formatter does not encode it.

The original actual request goes unchanged into inherited V2 admission, fresh pristine-base B1 query encoding, v5 query scores, learned query-intent probability, the existing blend formula, budget/color/stock/size/waterproof guards and exact-name policy. The profile feature branch alone uses the neutral request. Soft color selection uses the learned neutral-context head, with actual hard color requirements still filtering eligible source variants. Selected public facts still drive fresh neural copy word slots. Query meaning reaches ranking through the query head and guards, rather than the profile LoRA prompt; that is the architectural tradeoff being measured.

## Pure feature caches and proof

Before READY, V4 encodes all ten authorized public profiles as neutral color4096 features and neutral full-prompt X/H/Y features. It uses the same literal721-token catalog prefix, pristine base prefix state, target-free sum16 gradient capture at the final16 tokens, and plain cached reconstruction already mechanically verified for V2. Each new profile capture additionally compares all16 captured native route scores with reconstructed scores **exactly** and refuses a mismatch. The capture verifies original KV/SSM state and rank adapter unchanged; original copy/rank ownership and927 shutdown checks remain active.

The bounded32-entry profile cache contains only `x`, `h`, `y`, native dtype metadata and16 public route IDs. Separate metadata contains prompt/feature checksums and a parity boolean, with no score vector. No rankings, response words, targets, grades or teacher strings are stored. Every cache hit performs a fresh LoRA/LM-head projection. Capture receipts retain matching hashes and exact-match proof flags, omitting the transient route-score vectors. No complete ranking or copy response is generated during startup.

An unknown new public history/persona captures its pure features once. Later requests with that same exact public profile can change query, budget or hard requirements while reusing those features. Every actual query still receives a fresh B1 encoding. Capacity eviction causes a new capture rather than a substituted response. Pure tensor checksums are checked before reuse and on full shutdown verification.

## Root-only native entry

Use the exact existing V2 artifact arguments with the new module:

```text
python -m experiments.ranker.pretrained.store_persona_adapter_engine_v4 [same V2 artifact arguments]
```

Do **not** append V3 control flags: V4 fixes fresh query encoding, ten-profile preencoding and32-entry cache capacity. It retains local8767, the same global native lock, width20 and existing7GiB guard. Choose a unique output inside the current code root. No accepted profile or default launcher update is included.

Startup `persona.measurementEngine` identifies V4, `wholePromptRankCacheIncludesQuery=false`, and `contextConditioning` discloses the normalized profile policy. `featurePreencoding` contains separate original color warmup receipts, ten rank capture proofs, total startup encoding cost/body forwards, pure-cache disclosure and a sealed profile warmup hash. Every request's ranking receipt includes `contextConditioning`, actual fresh-query timing, neutral profile keys, cache hit/body/projection counts and existing full readiness/copy receipts. Initial profile encoding and browser readiness must remain separate.

CPU verification: eleven V4 tests plus27 V2/V3 tests pass. Tests cover query/budget/hard-invariant profile keys, observed history/persona invalidation, original query text, full inherited-rank v5 score/order changes with different actual query features, and exact-name/budget guards, original actual pipeline forwarding, neutral color conditioning, pure-cache field rejection and corruption, once-only capture/fresh projection, failure/cancellation and bounded eviction, exact native/reconstruction admission, explicit conditioning disclosure and inherited owned shutdown. Native latency, memory pressure, source927 integrity and factual/output behavior remain pending root execution. This serving-only experiment requires its own factual/conditioning analyzer; the V3 analyzer intentionally expects actual-query-conditioned feature keys.
