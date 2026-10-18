# Responsive-image browser audit

The responsive run reduces median cold product-image transfer **96.47%**, from 21.27 MB to 0.750 MB. It does **not** show a cold-navigation latency improvement: navigation through complete copy plus six decoded images rises from 497.8 ms to 523.4 ms. Same-context reload median falls from 348.85 ms to 322.6 ms. These are four observations per condition, run sequentially rather than randomized; they do not establish a causal latency advantage.

This CPU-only audit independently reparsed all eight responsive receipts and all eight baseline receipts, exact NDJSON bodies, DOM snapshots, resource timings, screenshots and source metadata. The [JSON audit](CANDIDATE_BROWSER_PROBE_V2_RESPONSIVE_INDEPENDENT_AUDIT_V1.json) records 1821 successful checks, per-attempt values and evidence hashes. All 37 original evidence files remained unchanged. No browser, GPU, service, Git or runtime source action was performed.

| Metric, fixed four per condition | Baseline v1 attempt2 | Responsive v2 attempt1 |
| --- | ---: | ---: |
| Fresh-context navigation → copy + six decoded, median |497.80 ms|523.40 ms|
| Fresh-context effect → copy + six decoded, median |316.85 ms|310.50 ms|
| Reload navigation → copy + six decoded, median |348.85 ms|322.60 ms|
| Reload effect → copy + six decoded, median |269.05 ms|239.85 ms|
| Fresh-context product-image transfer, median |21,269,174 B|749,804 B|
| Fresh-context all resource transfer, median |49,291,384 B|29,509,769 B|

The endpoint includes React commit/two animation callbacks and successful decode/observation of all six selected primary images. Navigation and hook-effect clocks are reported separately. It does not measure physical paint, click response or all gallery images. First-three and full-six image endpoints coincide in this run. Raw responsive navigation endpoints are 1392.1, 526.2, 497.6, 520.6 ms fresh and 364.0, 314.8, 316.0, 329.2 ms reload; the first 1392.1 ms observation remains in all statistics. Its DOMContentLoaded endpoint is 706.4 ms versus roughly 148–179 ms on later fresh navigations, consistent with initial dev module-loading cost. Attribution specifically to transforms is an inference, not an isolated measurement.

The 0.70–0.77 MB per fresh attempt describes product-image transfer, not the whole page. The development page still transfers about 29.46–29.53 MB of resources: the first responsive receipt includes 21.62 MB for the icon module, 3.13 MB React DOM and a 0.72 MB imported responsive-image manifest. That manifest import appears in the captured network evidence. A later compact resolver is outside this run. Reload product-image bodies are zero bytes with header-sized resource transfers; call these cache-eligible reloads, not universally warm or zero-network sessions. Native query/model caches and OS caches were uncontrolled.

All 16 navigations contain exactly one matching actual POST and HTTP 200 NDJSON response with verified byte hash, parsed rank/firstThree/complete events, fresh unique request and document IDs, settled body capture, matching response/mirror IDs and exact rendered text. All eight responsive first-three 10-field and complete 20-field receipts pass, as do the baseline receipts. Full raw copySource objects are richer than the frontend summaries: raw equality is false while retained parsed fields agree, as expected. All 16 page-error arrays retain `Error: WebSocket closed without opened.`; its origin is unknown and the summary's error arrays omit it. No native request, parse or image-decode failure was observed.

## Image pixels, crop and viewport

All 48 responsive primary-image observations decode. All 48 currentSrc selections map exactly to their baseline selected source URLs. The ten unique selected derivative files and original assets match their source-bound manifest hashes. Encoded WebP headers confirm 1152×384 pixels, or 1152×385 for the lunchbox sheet. Gallery tiles show about 381.33 CSS pixels each; full-sheet rendered boxes reach about 1144.01 CSS pixels for tiled mug/daypack views. The minimum encoded-to-rendered dimension ratio is 1.00699, so the selected files cover the measured boxes at DPR1 without pixel undersampling. Browser naturalWidth 1143/1144 is density-adjusted; it is not the encoded file width.

DPR itself was not recorded. The pinned probe creates contexts without a deviceScaleFactor override, so DPR1 is inferred from Playwright defaults. This is not evidence for high-DPR screens or other viewport sizes. Each of the 48 card rectangles and clipped-primary-image rectangles exactly matches baseline geometry; no additional clipping was introduced by delivery. Manual inspection of responsive screenshots 1 and 7 against baseline 1 shows the same recognizable lamp/mug/towel framing and bag/lunchbox/tote crops, with expected softness from lower resolution/lossy encoding. This is three-screen visual sampling, not exhaustive perceptual QA of below-fold content.

At 1280×960 and scrollY 0, all eight optimized snapshots contain 16 DOM cards, three fully visible primary images, zero fully visible cards, zero visible product titles and zero visible copy paragraphs. Six selected images decoding does not establish complete useful above-fold readiness. Below-fold cards remain present with copy; the workload was not reduced to two cards.

## Version and usefulness limits

Both runs pin the same probe, performance mirror, ShopView, ProductCard and hook hashes; only the recorded ProductPhoto hash changes. The responsive hash is `aba27a310adfc4214b787ef7e1f48377e76dd578883a4496328fea20c1541ac9`. The initial exact-source snapshot gap was subsequently resolved by the independently verified recovery below. Plans still omit native-engine and full transitive source pins, so complete environment reconstruction remains limited.

All raw responses still use rank checkpoint `265c728a…a99e` and TRAIN-ALL parallel-copy checkpoint `074169e3…4889`. Ranking v5 is absent. Selected six-card orders agree between corresponding runs. The light and bottle queries rank the matching product first; daypack is reasonable for the broad bag query. The notebook query still ranks the organizer first and the notebook 13th, outside the six generated-copy cards. All generated bodies remain neutral declared color/price facts; fresh generation on this fully trained catalog does not establish heldout or preference-copy generalization.

No trace was captured or measured. These receipts do not establish whether a separate trace integration was configured or enabled in development. They support the stated browser endpoints and image-byte savings only; later resolver, ranking and trace configurations need their own receipts.

## Source recovery addendum

The [v2 browser archive](store_candidate_browser_results/v2/README.md) now contains both exact historical ProductPhoto files under `verified-source/<sha256>/ProductPhoto.tsx` in its text tar, with readable copies under `recovered-source/`. Independently extracted tar bytes equal the recovered files and original plan hashes: baseline 1066 bytes, SHA `5f39422337a129d66ef964c8f5de5971f7265d4414eddf911066cb2377f89010`; responsive 1548 bytes, SHA `aba27a310adfc4214b787ef7e1f48377e76dd578883a4496328fea20c1541ac9`. The baseline receipt records recovery from commit `ba2b942c83e9e710414b290442acbf0bcd394bd1`. The responsive recovery was independently recomputed from its preserved input snapshot using exactly one `./product-responsive-runtime` → `./product-responsive-source` import substitution. No Git command was executed by this audit.

This resolves the specific ProductPhoto source-availability gap without inserting reconstructed files into original run directories. All 37 original baseline/responsive evidence files were rehashed unchanged against the archive inventory. Timing, image, quality and sample-size conclusions above remain unchanged. Full native-engine/transitive-source closure is still outside the archive's claim. Exact source-recovery and archive hashes are recorded in the JSON addendum.
