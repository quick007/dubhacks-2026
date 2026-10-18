# Compact delivery and v5 browser audit

The repaired navigation capture passes **8/8 with the old rank head and 8/8 with v5**. The separate loaded-collection workload passes **4/4**, with complete first-three cards, text and decoded primary images visible at a median **269.95 ms from the model effect**. The notebook is now ranked first for the fixed notes query; the old head placed it 13th. These are four fixed known-catalog queries, not a generalization or conversion benchmark.

The [JSON evidence](STORE_COMPACT_V5_BROWSER_INDEPENDENT_AUDIT_V1.json) independently binds raw request IDs, parsed NDJSON, both response-byte capture paths, screenshots, DOM text, source facts, image pixels and provenance. Over 2400 CPU checks passed. All 64 original files across four run directories were rehashed unchanged. This review performed no browser, GPU, service, Git or runtime source actions.

## Fixed denominators and timing

The earlier compact old-rank attempt1 retains all eight slots but only four complete protocol receipts: slots 2, 4, 7, 8 lack matching captured response bodies/IDs. Its cohort remains unscored; no successful-only speed number is substituted. The repaired old-rank attempt2 and v5 navigation attempt1 are distinct runs, each four fresh contexts followed by four explicit same-context reloads.

| Median, four observations per condition | Old rank, repaired compact | v5 rank, compact |
| --- | ---: | ---: |
| Fresh navigation → first-three text/two-rAF |487.00 ms|501.35 ms|
| Fresh navigation → complete text/two-rAF |487.10 ms|515.55 ms|
| Fresh navigation → complete copy + six decoded images |511.80 ms|533.10 ms|
| Reload navigation → complete text/two-rAF |325.05 ms|345.80 ms|
| Reload navigation → copy + six decoded images |341.75 ms|362.40 ms|
| Fresh effect → copy + six decoded images |324.65 ms|310.85 ms|
| Reload effect → copy + six decoded images |262.35 ms|282.80 ms|
| Fresh product-image transfer |749,804 B|725,686 B|
| Fresh all-resource transfer |28,805,913 B|28,781,795 B|

All first observations remain included. Raw cold copy-plus-six-image navigation endpoints are 1002.8, 510.7, 497.5, 512.9 ms with the old rank head, and 1067.9, 528.4, 537.8, 484.7 ms with v5. These sequential dev runs have different rank heads, some selected assets and a later capture guard; they support no causal speedup claim. Passive Response.clone overhead is included. HTTP-cache-eligible reload image bodies are zero bytes while header-sized transfer remains. Native/OS caches are uncontrolled. No LCP, physical-paint or tail-percentile statistic was measured.

Loaded collection is a different workload: one unscored Browse-the-store bootstrap, pre-scroll before each dispatch, then four hash-query updates in one loaded document. The fully visible first-three endpoints are 269.0, 272.6, 270.9, 262.6 ms, median 269.95 ms. Complete text/two-rAF median is 254.05 ms and complete-copy/six-decoded median 271.70 ms. No post-dispatch scroll creates the endpoint. All six selected cards receive 20 accepted generated fields; only the first three entire cards are within the viewport. These observations must not be pooled with cold navigation or described as click-to-ready.

## Protocol, facts and image evidence

All 20 scored requests contain exactly one matching actual POST and fresh response ID. The finished Playwright/CDP response bytes and passive in-page clone bytes agree exactly by independent SHA-256, byte count and NDJSON parsing. All rank/firstThree/complete events match their DOM mirror IDs. Capture settlement completes before the next navigation/query; both byte paths remain evidence, rather than fabricated fallback output. The navigation documents are distinct; the four loaded queries intentionally share one document.

Every first-three 10-field and complete 20-field payload matches DOM name/title/color/badge/body text and canonical public product facts. Copy is fresh parallel-head prediction from the same fully trained catalog checkpoint, with neutral identity/color/price wording. This audit does not claim a general semantic validator, unseen catalog performance or shopper-intent copy. DOM prices and variant IDs are not directly exposed by the probe; canonical response price/variant evidence and visible screenshot sampling remain distinct.

All observed selected primary images decode. CurrentSrc files match the responsive manifest hashes and encoded WebP dimensions. The minimum encoded-to-rendered dimension ratio is 1.00699, sufficient at DPR1 for the recorded boxes, including full-sheet tiled images and clipped card areas. Navigation explicitly configures and observes DPR1. The loaded-collection probe uses the default context scale but does not record DPR, so its DPR1 interpretation remains inferred. No high-DPR/mobile claim is supported.

At 1280×960, navigation snapshots still show three complete primary images but zero visible titles/copy paragraphs and zero fully visible cards. They do not establish useful above-fold readiness. The pre-scrolled loaded-collection snapshots independently show all first-three card, title, copy and image rectangles within the viewport. Manual inspection of the saved notebook collection screenshot confirms its first-ranked notebook, full titles, color/price copy and actions. Below-fold cards remain present; no two-card workload reduction occurred.

## Provenance and practical relevance

Both completed navigation reports record identical before/after hashes for all 30 tracked files, unchanged native startup metadata, ready HTTP 200 health and the frozen v10 integration manifest. They precede checkout-v11 changes. Old-rank response provenance binds checkpoint `265c728a…a99e`; v5 health and every event bind actual checkpoint `5d07673e51450bbbbb1803c8a44ef99721b39fd3b1ae92edb9168e27a1377d6b` and serving export `17a0a15a2da6e2fc90e651e1f5bb2c237d61241795ec0cca548a4f3ce319d703`. The local export and checkpoint bytes were independently hashed. Export metadata discloses known16 single-function adaptation, unchanged frozen 927 training hashes and no teacher/evaluation-file lookup at serving. Copy checkpoint remains `074169e3…4889`.

The historical navigation-v2 source `fd0614…8b31` was recovered and independently matched to the old-rank plan; v5 uses `32804c…e1c7`. Recomputing the diff shows only the later guard rejecting a settled capture whose outcomes count differs from pending captures. The guard runs after the navigation endpoint clock; endpoint math and navigation actions are unchanged. It still changes evidence validity/dispatch behavior, so the versions remain explicitly separate. Exact recovered source and recovery-receipt hashes are bound in JSON.

V5 first choices are lamp for bedside light, bottle for drinking water, notebook for notes, and tote for the broad carrying-bag query. The old head's notebook 13th/organizer-first miss is preserved; its daypack-first broad-bag result was also reasonable. This supports a concrete improvement on the notes query, not a broad four-query accuracy estimate. Secondary six-card relevance, budget constraints, negation sensitivity and shopper outcomes were not established here.

Each navigation run retains eight `WebSocket closed without opened.` page errors. Unlike the older receipts, their stacks identify the observed origin as `http://127.0.0.1:4312/@vite/client:225:30`; no deeper cause is inferred. Loaded collection retains one such bootstrap error and zero scored-query page errors. Native protocol/image checks pass independently of these residuals.

The collection report has only its smaller five-file source inventory and no separate full source-after/health-after receipts. V5 event provenance is verified, but this cannot be described as equivalent to navigation's 30-file before/after closure. The later checkout cohort and trace configuration require their own evidence; this audit makes no trace-capture or runtime-trace-status claim.
