# Warm public-persona browser audit

All 13 warm-run public requests reproduce the prior V3 run exactly: lexical route logits, v5 scores, blended scores, query-specific probability, complete product order, six selected variants/images, card and section text, and all 260 raw generated field records including their word-slot IDs. This demonstrates output parity on this fixed demo workload. It does not improve the failed development quality result or establish parity on arbitrary new requests.

The [JSON evidence](PERSONA_WARM_BROWSER_V1_INDEPENDENT_AUDIT.json) binds all 26 native receipts, actual POSTs, response/clone bytes, 13 paired browser receipts, startup cache records and source hashes. I independently parsed each NDJSON body and matched it to its native receipt. All 139 common source pins match the prior run; all 79 currently available declared code-source files match their recorded hashes. Warm source maps and startup metadata are unchanged before/after. Model/UI/copy-head pins agree across all 13 pairs. The active measurement runtime differs: it preencodes public context and forces fresh query encoding. Large weights and resident tensor bytes were not independently rehashed or recomputed by this CPU audit.

Ten loaded, pre-scrolled profile switches passed visible-first-three and decoded-six contracts, with no failed or omitted case. Nearest-rank results in milliseconds:

| Measured scope, fixed n=10 | p50 | p95 |
| --- | ---: | ---: |
| Fresh base query encoding | 214.7 | 283.5 |
| Cached public color-context acquisition | 0.4 | 0.7 |
| Fresh LoRA decision from cached public XHY | 21.1 | 27.3 |
| Runtime ranking, including guards/variant binding | 271.4 | 338.9 |
| Server first-three ready | 292.6 | 363.7 |
| Server full-six ready | 299.9 | 370.5 |
| Browser first-three from native fetch | 335.4 | 416.6 |
| Browser full-six from native fetch | 350.8 | 416.6 |
| Browser first-three from option click | 349.5 | 431.6 |
| Browser full-six from option click | 364.7 | 431.6 |

At n=10, p95 is the maximum; this is not stable tail evidence. The server’s full-six p50 is 299.9 ms; the browser’s 350.8 ms includes additional delivery, DOM and image/observer work. The per-case browser-minus-server residual has p50 46.4 ms for first-three and 52.9 ms for full-six. These residuals are not pure network latency or physical-paint measurements. Runtime integrity checks have p50 3.6 ms inside runtime ranking. There is **no isolated guard timer**: guards and variant binding are included in runtime ranking along with feature reads and other CPU work. Do not invent guard time by subtracting component medians. Post-ranking-to-server-full-six p50 is 20.1 ms, including copy, validation, source checks and receipt/event work; the nested head-forward/argmax p50 alone is 0.19 ms. Timers overlap and medians are not additive.

Startup spends **5,120.4 ms** encoding ten actual public color contexts, ten native base calls, before READY. Each of the ten runtime context keys, formatter-text hashes and feature-tensor hashes matches its corresponding startup record; the warm-cache inventory digest recomputes exactly. This proves recorded bindings, not an independent tensor re-encoding. All 11 Browse requests, including bare Bob, use those context features and existing whole-prompt public XHY features. Each still performs one fresh base-query forward, one fresh rank projection and one fresh neural copy feedforward. There is no cached response text or startup ranking/copy response warmup. Across all 13 timed requests, receipts count 14 base encoder forwards, 9 rank body forwards, 15 rank LM projections and 2 target-free gradient evaluations; startup’s ten encodes are additional. These are distinct counters, not a combined GPU-call count.

The caches include the actual query: color features use the entire public formatter text, and rank XHY features use the whole prompt. This is **not a query-independent history cache**. The first new Bob “work bag” query misses color and rank caches, performs two base forwards plus six rank body forwards, and reaches decoded-six completion at 3,314.6 ms from fetch. It creates the 721-token catalog prefix cache (three prefill forwards). Same-query reload hits the color cache but still misses rank XHY, performs one fresh base forward and three rank body forwards, and completes at 1,537.5 ms from fetch / 1,641.0 ms from navigation. Both select pack, duffel, tote, organizer, notebook and sweatshirt, exactly as before.

The older V3 loaded-switch p50s were 844.4/851.4 ms. This descriptive difference is **not a paired causal latency estimate**: the prior service had a different cache history, its Maya and reload queries hit the query-feature cache, and its first work-bag catalog prefix was already warm. The warm run forces fresh query encoding for all 13 requests. Startup cost is excluded from the displayed request latencies.

All 13 protocol cases pass, both functional cases pass, and no request is unattempted. The initial bare navigation still records **zero wholly useful cards** at its actual page-top capture; it has no first-three readiness. Reload also has no first-three readiness, but no initial-viewport snapshot exists for it, so no zero-card pixel observation is inferred. Its decoded-six completion is separate. Each response generates 20 accepted fields, while the current UI renders 19: the second heading remains unused. Six decoded cards are not six simultaneously wholly visible cards.

Functional scope remains limited: rapid switching observes the browser abort and final correct profile, without proving immediate GPU interruption. Per-person cart and already-completed local demo checkout receipt invalidation pass; `checkoutRaceExercised` remains false. There is no real-payment or conversion evidence.

Output quality is unchanged: public owned-repeat and soft-color misses from the [prior semantic audit](PERSONA_DEVELOPMENT_BROWSER_V2_V3_INDEPENDENT_AUDIT.md) persist. Copy is neutral source-conditioned short wording from a head trained on all 48 catalog bindings. Known-development primary remains **17/32 versus frozen base 20/32**; promotion and quality gates remain false. No future quality outputs or private labels were read. No browser, GPU, service call, Git action, raw-file or source mutation was used by this auditor.

JSON SHA-256: `ccdab3945abda4511d7dfbf3b26bf2f5a22868d6b19c4992596fc203a8b292a6`. Warm report: `ff598a333ba2d3604c5bbb3d0ef23ca03462755e851ebd658af07a1e10d9ddb7`. Prior report: `5f2135156560c83d0c82d077af39f8c78b3c924787e6809c1bc5cc256e46520f`.
