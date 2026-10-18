# Full personalized-content browser baseline

The already measured navigation-to-personalized-content median is **497.80 ms
with a fresh browser context**, and **348.85 ms on a cache-eligible reload**.
Each condition has four fixed requests. This endpoint requires all20 generated
fields on six selected cards to be committed, followed by the existing two-rAF
paint opportunity, and all six primary images to complete and decode. It is
not the isolated193.50 ms backend timer.

| Browser condition, resident model | Navigation to20 text fields committed +2rAF | Navigation to text plus6 primary images decoded | Image bodies transferred |
|---|---:|---:|---:|
| New isolated context,4 requests | 481.15 ms median,473.5–820.6 ms | 497.80 ms median,490.1–831.0 ms | 20.82–21.30 MB per request |
| Same-context explicit reload,4 requests | 332.50 ms median,328.5–338.3 ms | 348.85 ms median,345.9–354.8 ms | zero recorded body bytes;8.7–9KB headers |

Network: unthrottled loopback HTTP, headed Chrome155.0.8059.82,1280×960,
resident M5/Qwen model,16 public products/48 color bindings, original assets at
ba2b942. Measurements ran2026-10-18 07:19:05–07:19:11 UTC. Static Qwen catalog
and fact vectors were already computed. Every scored request performs a fresh
Qwen query encoding and a fresh learned parallel text prediction; no response
text cache is used. Each request has a unique actual POST, response ID and new
document. All20 field/DOM/provenance checks pass. The first navigation outlier
is retained, not excluded as warmup. Full raw values are in the
[baseline JSON](STORE_FULL_PERSONALIZED_CONTENT_BASELINE_V1.json).

The page retains16 catalog cards, of which six receive personalized copy. The
initial hero puts card titles and copy below the viewport: this navigation
endpoint is content/decoded-image readiness, not useful visible-card readiness.
No LCP, physical paint, every below-fold image, whole-page screenshot completion
or non-loopback network time is measured. Eight navigation trials retain an
unattributed WebSocket error. Both notebook trials rank the pouch first and
notebook13th, so correct rendered copy is not a preference-success claim.

A distinct loaded/pre-scrolled collection workload verifies all three complete
cards, title/body/badge geometry, decoded primary images and exact response/DOM
matches. Four actual new queries take253.2,238.9,223.9 and237.0ms from query
effect to that endpoint: median237.95ms. All four have factual visible cards;
only three top-one choices are reasonably relevant because notebook still
fails. These are warm loaded-page timings, with passive-response-clone observer
overhead included; they are never pooled with navigation samples. No scored
page/request errors occur, but the unscored bootstrap retains a Vite WebSocket
error. Independent [navigation](CANDIDATE_BROWSER_PROBE_V1_INDEPENDENT_AUDIT_V1.md)
and [visible collection](CANDIDATE_COLLECTION_PROBE_V1_INDEPENDENT_AUDIT_V1.md)
audits preserve the two earlier protocol failures and their excluded slots.

Responsive lower-resolution WebP derivatives are being integrated next. All
original image files remain unchanged. Report optimized bytes and the same
browser endpoints separately; this baseline must remain available regardless
of the result.
