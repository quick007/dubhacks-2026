# Paired incremental browser audit

**All 20 pairs / 40 arms are valid under the frozen plan.** Independent nearest-rank recomputation gives **+217.4 ms p50 / +234.4 ms p95** for paired local-minus-generic first-three readiness. This is added latency for local ranking and fresh generated copy over the existing deterministic storefront. Both arms already put a sensible primary product first for these four broad queries; this run does not measure shopper benefit from the extra copy.

| Loaded, pre-scrolled endpoint | n | p50 | p95 | Minimum–maximum |
| --- | ---: | ---: | ---: | ---: |
| Generic first three visible cards/images |20|64.3 ms|65.1 ms|62.6–81.2 ms|
| Local first three visible cards/copy/images |20|281.1 ms|297.7 ms|264.3–298.4 ms|
| Local full six: 20 fields + decoded images |20|281.1 ms|282.4 ms|264.3–283.1 ms|
| Paired local minus generic |20|+217.4 ms|+234.4 ms|+198.9–234.9 ms|

The [JSON audit](STORE_PAIRED_INCREMENTAL_PROBE_V1_INDEPENDENT_AUDIT.json) retains every arm's values, hashes, query, order, body evidence and caveats. It records 3291 successful CPU checks plus public-query payload verification. All 82 original JSON/PNG files were rehashed unchanged. No browser, GPU, Git, runtime source or service action was performed. The active 120-session cohort was neither inspected nor pooled.

## Denominator and estimator

The run matches the repository PLAN and saved pre-run plan: four fixed public queries, five repetitions, 20 pairs, 40 arms, 10 generic-first and 10 local-first pairs. All arms were attempted, protocol-valid and nonempty; failures, unattempted arms and zero-result arms are zero. No first observation or outlier was dropped. Each arm uses a new isolated context by the pinned source's context-creation loop, with its own unscored Browse bootstrap before the scored hash dispatch. Generic makes zero POSTs throughout; local makes 20 bootstrap and 20 scored native POSTs with 40 distinct request IDs. Every native request payload contains only its public query.

Statistics were recomputed directly from all 20 numeric slots: p50 is sorted rank 10 and p95 rank 19. Paired deltas were formed within each planned pair before sorting; their p50 is not the difference of marginal p50 values. Generic's 81.2 ms maximum remains in the data even though nearest-rank p95 is 65.1 ms. With only 20 repeated observations, p95 is descriptive and does not establish a population tail. The 20 pairs represent four repeated intents, not 20 independent shoppers.

## Shared endpoint and workload differences

Both modes use the same `installPairedObserver` function and common two-animation-frame barrier after current-query DOM, full first-three card/title/image visibility and successful primary-image decode. Generic establishes current-query state using search input and status text. Local uses the fresh current-request React-commit mark, then requires its matching 10-field app mirror and actual response. The common local barrier begins from the commit observation rather than waiting for the app's two frames and adding another two; captured candidate/frame/endpoint chronology agrees in every arm. App and common observer frames overlap.

Local requires visible generated badges/bodies; generic retains canonical static titles and has no generated badges or body fields. The viewport/frame predicate is shared, but the content workloads differ. Passive response-clone overhead is included for local. Raw-body settlement occurs after endpoint timing and before the next arm.

The separate full-six endpoint requires the complete app mirror, all 20 accepted fields and six decoded images. It does not require six entire cards to be visible. It precedes first-three readiness in five local arms and follows it in one, because its predicate lacks the common first-three visibility/frame barrier. Its narrower p95 must not be substituted for first-useful readiness.

## Native, DOM, image and source evidence

For all 20 local bootstrap and 20 scored requests, the finished Playwright/CDP response and passive clone bytes match exactly by independently computed SHA-256, length and NDJSON parsing. Scored IDs differ from bootstrap IDs, both capture paths settle, and rank/firstThree/complete IDs agree. All 20 scored first-three 10-field and full-six 20-field payloads contain accepted fresh parallel-head predictions, with no earlier response cache or serving teacher lookup. Generated titles, badges and bodies match DOM text and canonical public name/color/item-price facts. DOM price nodes/variant IDs are not directly captured by this probe; canonical response facts remain distinct from that missing DOM observation.

All first-three entire cards, titles and primary images are within the 1280×960 viewport; every local copy paragraph is within it. All 20 local full-six payloads retain six cards and decoded primary images. None claims six fully visible cards. All 40 observer readings report DPR1. The 13 unique selected responsive assets match their manifest hashes and encoded dimensions; the minimum encoded-to-rendered dimension ratio is 1.00699. This supports pixel sufficiency at the measured DPR1, not high-DPR or semantic image understanding.

All 29 pinned source hashes agree with the plan, before/after run inventories and reviewed current bytes. Ready HTTP 200 health and startup metadata are unchanged. The run binds the v11 integration manifest, v5 rank checkpoint `5d07673e…377d6b` and fully catalog-trained copy checkpoint `074169e3…4889`. It supports known-catalog development performance, without unseen personalization, hard-constraint or conversion evidence. No ordinary autoregressive-LLM comparison occurred here.

Two measurement details should remain visible. `installObserver` assigns its `documentId` UUID on every reset; bootstrap and scored UUIDs differ within the same page. These are observer-instance identifiers, not independent document-lifetime proof. Fresh contexts are established by the frozen run source. Also, 16 generic arms move from scrollY 452 to 454.5 between dispatch and endpoint; local scroll positions remain fixed. There is no scripted/user post-dispatch scroll in the probe, but actual scroll position was not invariant. The cause is unestablished; actual whole-card visibility still passes.

Each whole-arm page lifetime records one `WebSocket closed without opened.` error with a Vite-client stack at `/@vite/client:225:30`. These 40 residual page errors include bootstrap lifetime and are not 40 scored native failures. No request failure was recorded.

The measured clock begins at a loaded, pre-scrolled hash dispatch after each arm's bootstrap. Browser caches are eligible from that bootstrap; the resident model and catalog features are warm, and OS/native caches are uncontrolled. These are not cold-navigation, click-to-ready, LCP or physical-paint measurements.
