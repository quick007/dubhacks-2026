# Model experiment ledger

## Search navigation polish, 2026-10-18 13:29 UTC

A real UX bug was confirmed: search submission scrolled to the hero. At the
960px viewport the first card title started around886.5px and Add controls ended
at1112.5px, below the screen despite completed inference. A post-render query-route
effect in use-route.ts now reveals the existing collection. No styling, model
weights, inference source, public request schema or serving process was changed.

Same native91663 and launcher91659 throughout. Fixed ten-search before/after:
visible3 0/10→10/10; after p50/p95 332.4/395.4ms, decoded full6 349.4/413.6ms.
All ten actual requests, score vectors, ranked IDs/variants and neural copy are
exactly equal. Endpoint timeouts/idle intervals differ, so do not claim inference
acceleration. Original raw failures and all old receipts remain unchanged. The
initial no-query hero remains unchanged. Model quality/promotion caveats remain.

76 storefront tests and typecheck/build passed. Fresh actual presentation render
passed50 checks across all5 slides with strict Tiger source and no model calls.
README URLs4313/4312/4310 return200; native is ready with queue0. Evidence:
store_search_navigation_polish_v1/{report,live-check}.json and before/after PNGs;
store_cohort_presentation_search_polish_v1/report.json. The running launcher and
all checkpoints remain preserved.


## Current integration checkpoint, 2026-10-18 13:15 UTC

Left ready for recording: launcher91659, native91663/8767, persona UI92022/4313.
The final active lifetime is exec10284, data/personalized-development/
2026-10-18T13-13-52.310Z-825e4301-40d8-4cf9-b36d-7c045131c1e5.
See persona_factorized_results/v1/live-service-receipt.json for fresh ready/queue0,
HTTP200 storefront/presentation and actual source:tiger readback. This active
lifetime has not been shut down; its prior main lifetime has the clean927 proof.
Only stop launcher91659 gracefully before another native worker.

The explicit development entry `pnpm dev:personalized:demo` now selects V4 on
http://127.0.0.1:4313. It uses MAIN code and copied MAIN runtime assets, local
simulated checkout4331, public background/history and fresh query computation.
`dev:personalized:warm` preserves V3; `dev:personalized:uncached` preserves V2.
The accepted-only `dev:personalized` stays closed. Model promotion is false.

V4 technical/factual acceptance passed:13 primary browser requests,10 profile
three-card/six-card observations,2 functional arms,10 distinct query observations
and1 new-history HTTP probe. All23 browser actual requests/native receipts were
independently joined. Main-only launch then passed13 actual browser checks over
2 real requests with changed query/order and same public profile key. Its
graceful owned cleanup was unforced, with all927/rank/copy/cache/source proofs.
Failed CDP receipt capture is preserved; a new response-clone probe version passed
without modifying the storefront/model. See store_integrated_personalized_factorized_launch_v1.

Loaded profile first3 useful-content p50/p95:335.5/649.0ms; full6 decoded copy
p50:351.2ms. Ten distinct queries full6 p50/p95:413.5/463.4ms, but first3 useful
visibility0/10 because actual search scroll leaves copy below the fold. Initial
hero also exposes0 complete useful cards. Do not substitute rank/head timing
for these complete content measurements. Startup profile encoding18.753s; new
history capture remains about2s. Physical paint is not measured. Pure persona/
history features are cached, scores/ranks/response words are not; every actual
query is freshly encoded B1 and hard constraints use the original request.

V4 changes context conditioning; prior independent packet scores do not measure
its quality. Public facts/images passed but four profiles repeat owned products,
palette/function misses and a common30/60 product-slot trio remain. The first
adapter failed17/32 versus base20(raw)/19(fused); bounded clear-data follow-up
failed19/32 versus base23(raw)/22(fused) on a DIFFERENT packet. Both are now
development and neither is promoted. All checkpoints/failed runs are retained.
No further fit is planned before the parent recording boundary.

Main integrated CPU checks passed:51 runtime/analyzer/admission Python tests,
52 launcher/profile Node tests,23 browser-protocol Node tests. Earlier unchanged
UI acceptance includes76 storefront tests/typecheck/build,11 generic real browser
checks and50 strict-Tiger presentation checks. The hosted120-session cohort is
the older frozen v5 model, never a V4 result; no new conversion benefit is claimed.

Read PERSONA_FACTORIZED_V4_PUBLIC_INDEPENDENT_AUDIT_V1.md and
persona_factorized_results/v1/README.md for measured scopes and retained failures.
The final live demo lifetime is recorded separately from these retired evidence
lifetimes. Stop its launcher gracefully before any new GPU work. Keep strict
Tiger4311, checkout4331, website4310 and generic storefront4312 unchanged.
All following sections are preserved chronological history.


## 2026-10-18 resumed overnight checkpoint

The user resumed the model work after the earlier pause. The explicit
catalog-trained parallel decoder completed32 fresh public-query requests with
six cards and20 valid fields each: median first-three193.428ms, full193.498ms,
including fresh Qwen query encoding, learned ranking, guards, learned word-slot
predictions, validation and JSON. Startup/browser/images are separate. All8
constraint probes passed; offline relevance30/32top-one and31/32top-six.
This is a curated known-catalog demo: all48 product/color bindings were trained.
Independent audit reproduced528/528 supervised fields without runtime teacher
lookup. Earlier heldout card fields0/126 and three failed research variants are
preserved. Attempt1's timed32 all passed but its later guard audit failed;
attempt2 fixes only the expected conflicting-color error handling and repeats
all32 requests. No unseen-generalization or conversion claim.

Fresh AR width20/root reuse completed a separate explicit8GiB paired experiment
at4571.540ms versus6402.946ms ordinary width10, but both0/3 full useful pages.
Repaired generated-earlier cache completed32 hits at219.627ms median, all below
300ms, but0/32 full useful pages. Original failures and all927 unchanged tensor
receipts remain. See [native candidate report](STORE_NATIVE_CANDIDATES_V1_RESULTS.md),
[TRAIN-ALL audit](STORE_PARALLEL_TEXT_TRAIN_ALL_DEMO_V1_AUDIT.md), and
[CPU research results](STORE_PARALLEL_TEXT_CPU_V2_RESULTS.md). Actual browser
integration of the parallel head is now being checked; native193ms is not
presented as full storefront readiness.


Run names distinguish optimizer updates from numerical probes and inference.
The 9B quantized backbone weights stay frozen throughout these experiments.
Synthetic public-fact correctness is a proxy; none of these results establishes
human conversion lift or a production storefront integration.

| Run | Actual optimizer work | Checkpoint and outcome |
| --- | --- | --- |
| `semantic-copy-direct-v1` | Completed 336 updates, 65,536 final-layer `mlp.down_proj` LoRA A/B parameters | Saved `data/ranker/pretrained/semantic-copy-direct-v1/adapter.safetensors`; exact direct-loss reload. Dev titles 6→15/18; test titles 6→6/18; zero valid pages in each three-case split and model arm. Not promoted. |
| `gradient-feature-probe-v1` | Zero updates; one nonzero-adapter numerical control row | Direct native, auxiliary native and reconstructed-feature loss/A/B gradients exact. No trained checkpoint. |
| `catalog-user-fusion-paired-v1-attempt-1` | Completed 80 CPU steps for full head and 80 for independent prior | Raw-feature head had zero preference-driven product flips, including training. Saved full/prior/random checkpoints; not promoted. |
| `catalog-user-fusion-centered-v1-attempt-1` | Completed 80 CPU steps per full/prior fit | Train-only role means restore conditional image-style choices: validation 32/32, test 48/48. Product pairs remain 0/4 and 0/6 both correct. Not a complete-view result. |
| `catalog-user-semantic-factors-v1-attempt-1` | Completed 80 CPU steps per full/prior fit; full product model has 57,358 trainable parameters | Learned seven-atom sigmoid factors pass validation 8/8 and test 12/12 product choices, all paired interventions correct. Shuffled-query product controls fall to 1/8 and 4/12. Image checkpoint is byte-identical to centered-v1. Inspected synthetic replay, pending fresh transfer evaluation. |
| `semantic-copy-feature-fit-v1` | Completed 504 exact extraction gates, then 504 real globally shuffled optimizer updates, 65,536 final-down LoRA parameters | Saved and exactly reloaded `data/ranker/pretrained/semantic-copy-feature-fit-v1/adapter.safetensors`. Dev pages 0→2/3; test pages 0→3/3; trained titles 18/18 in each split. Peak MLX 6,360,122,248 bytes. |
| `semantic-copy-learned-probe-v1` | Zero updates; one exact learned-parameter control row | Native loss/A/B gradients match both recaptured and original archived features exactly; full source cache and frozen language tensors unchanged. |

The earlier 140/336 copy-update progress belongs to the now-completed direct
fit. A later 183/504 or similar zero-update report belongs to the separate
feature-extraction stage of the new fit, which subsequently completed all 504
updates. They are different runs; both checkpoints and reports are preserved.

The new fit's complete report contains all twelve frozen/learned evaluation
arms and all 504 per-row gate receipts. All 504 stored feature archives passed
independent checksum, shape, dtype and finite-value verification. The learned
checkpoint SHA-256 is `0b0aeda377a9c9ad7eb37b529c139dac84bc5acd4bb559176d40245865ffe1b2`.
All 927 frozen language tensor hashes are unchanged. The reported fit/extraction/
evaluation wall time is 779.472 seconds and excludes earlier startup token/file
verification; whole-process startup wall time was not separately captured.
Five valid learned pages have median 11.342 seconds and range 11.099–11.725.
Frozen controls have no valid-page latency sample. These times include the full
rank context encoding but use locked fixture decisions; they are copy-stage
measurements, not newly measured ranking or complete visible-card endpoints.
The one learned dev rejection is `Structured travel choices` where the selected
section's universal attribute intersection is empty. Validation was not weakened.

The direct fit's all-attempt completion medians were dev 9.290→8.832 seconds
and test 9.648→9.468 seconds (frozen→learned). With zero valid pages, there is
no valid-page latency sample and no demonstrated usable-page speedup.
An actual held-out raw title changed from `Herilis 24L Daypack — compact` to
`Herilis 24L Daypack for structured travel`; `structured` was unsupported.

Current S/T/B copy experiments contain fourteen fields for six cards, capped at
224 sampled tokens. They omit visible per-card body copy. The distinct fixed
first-section workload requires ten fields including C1/C2/C3 (cap 208), exact
original-name/price/image metadata and a measured validated/serialized payload
receipt. First-token times and last-token lower bounds are not payload readiness.
Section grouping remains deterministic. No UI, API or shopper harness was changed.

Raw reports, checkpoint hashes, exact replay audits, token counts, wall time and
memory receipts are retained in each named result directory and tracked report
copies. Failed experiments and historical validation accounting remain intact;
new values-only attempts use strict field provenance and exact raw `NONE`.


## Completed transfer and complete-card diagnostics

The first-section ten-field endpoint completed all 36 frozen/learned arms at
native widths 1/3/6. Independently rebuilt, exact-forked and repeat cache controls
pass at every matched width in both modes, including raw token IDs and complete
log-probability hashes. Across-width text equality is not the quality criterion.
No arbitrary hybrid-cache trim/splice was used. Strict complete-view acceptance
is **0/36**. All 108 body rows stop naturally within 32 tokens (maximum 23), so
this failure is not body truncation. Unsupported cross-product attributes and
lexical failures dominate; original raw evidence and conservative policy remain.
Accepted bodies are 3/54 frozen and 6/54 learned. For example, generated rolltop
was unsupported for structured-only Neridor, and compressible/reflective were
unsupported for Tavulan. Ordinary lexical rejections and invented mechanisms
remain distinguished in the qualitative audit.

Attempt medians at widths 1/3/6 are frozen 12.113/9.823/8.804 seconds and learned
10.943/9.742/7.890 seconds. There are no accepted-payload latency or speedup
samples. Wall time is 426.624 seconds and cumulative MLX peak 6,756,602,444 bytes
under the fixed seven-GiB guard. See above_fold_copy/BENCHMARK_AUDIT.md,
BENCHMARK_QUALITY_NOTES.md and results/v1/ for exact field/token/payload/control
receipts. This adapter had no body-copy supervision and changed prompt layout.

The separately frozen fresh-catalog evaluation used 36 new backpack SKUs,
72 variants, 24 new three-tag requests and 96 actual B1 native encodes. There
was no fitting. Learned product correctness is **2/24**, matching prior and
opposing-query controls; product pairs are 0/12 both correct and only 1/12
preference flips. The public-fact oracle is 24/24. Perfect inspected-split replay
therefore did not establish transfer. All 144 grid components per arm pass
assembly guards, which is distinct from ranking quality or complete pages.
Inherited image choices are 639/864 with 207/432 both-correct paired actions.
The independent audit verified 96 artifacts, 288 score rotations, 240 readouts
and 720 assembled grids. See CATALOG_FUSION_FRESH_EVAL_ATTEMPT_1_AUDIT.md and
catalog_fusion/results/fresh-eval-v1-attempt-1/ for preserved raw evidence.

Actual model-resident fresh-query encoding plus 72-row verified warm catalog,
scoring and both 12/24 grids has median 210.816 ms. Cold catalog preparation is
21.018 seconds, model load 2.855 seconds, complete native run wall 33.012 seconds,
and peak MLX 4,856,720,708 bytes under six GiB. These exclude generated copy and
browser rendering; they establish neither page quality nor whole-page latency.

## Actual storefront alignment and next bounded experiment

Read-only teammate inspection pinned origin/feat/design-dash at
65b5896b28d4901b95a691fd68436e79ad264e61. Its mock presentation has six jackets
and ten public typed tasks. The separate actual storefront still shows fixed
mug/pack/notebook products and has no learned inference hook. Main incorporates
the teammate VISION.md merge; no design branch was merged and no UI/API/shopper
harness was changed. The model-only jacket_alignment_v1 snapshot preserves
multi-valued size/color/waterproof attributes and excludes persona metadata,
scripted outcomes and source IDs from predictive text.

Every public jacket task has only one or two eligible products. Images, stock
and shipping are absent. Current fixed-three-card/image-required backpack
contracts cannot honestly represent those tasks. The next bounded experiment
should freeze an attribute-preserving jacket input and variable-card,
image-abstaining output contract with explicit shortfalls, then compare real
learned copy/ranking on separately frozen tasks and catalogs. Keep item-price
budget semantics separate from delivered cost and waterproof distinct from
water-resistant. Do not fit these ten inspected tasks and call their inherited
held-out names a fresh evaluation. Grouping remains deterministic. No checkpoint
from the failed transfer or complete-card arms is promoted.


## Corrected-input generalization v2, completed

The explicitly approved restart adds versioned typed jacket inputs and honest
variable-card plans without rewriting legacy encoder/validator/checkpoint bytes.
All21 targeted CPU tests pass; all10 pinned public tasks reproduce15 returned
cards of30 requested,15 disclosed shortfalls and55 independent unexecuted copy
jobs. Image selection abstains, stock/shipping remain unknown, and hard masks
precede caller scores. See JACKET_MODEL_CONTRACT.md and the independently checked
jacket_alignment_results/v1/report.json. This is a model-side contract, not native
jacket ranking/copy or a live serving integration.

A real corrected-data v2 comparison completed at execution5ca10cb:192B1 native
encodes, four fixed80-step factor fits (320 actual steps) and two random heads.
The frozen9B weights were unchanged; TRAIN-only role means use21queries and21
products per representation. Private ranking targets are opened only after all
six head checkpoints freeze. Exactly matched supervision/schedule compares full
public text against a new compact native encoding of only declared positive
preferences/product attributes. Both remain real native hidden features, not
one-hot oracle features. No image variants were invented.

Focused validation and test each achieve21/21 best-set choices, nDCG3=1 and
94/94,115/115 both-correct disjoint-best-set pairs. Full text achieves8/21 and
7/21; its test prior gets10/21. Focused test prior10/21, random4/21 and
opposing-query11/21 distinguish the learned conditional result. All14 novel
pair/triple queries per held-out split pass; singleton meanings/texts necessarily
overlap and some focused catalog rows match TRAIN singleton text exactly. This
is nuisance invariance plus combination transfer on synthetic public attributes,
not evidence of unseen singleton meanings, natural preference learning or human
conversion. Pair metrics reuse21queries and are correlated, not115independent
observations. Both modes share newly balanced data, so gains over older runs
cannot isolate the coverage correction from the representation correction.

Actual test native query-encode medians are180.804ms full and124.990ms focused;
full context60–75tokens, focused15–25. These are encoding measurements from one
fixed-order run, not end-to-end rank/grid/copy/page speedups or latency tails.
Model load3.009s, native lifecycle33.669s, CPU fit/freezes.125s, evaluation.281s,
wall including startup37.041s, peak MLX4,706,802,514bytes under6GiB. No cache or
memory limit was hit. Full raw tokens/counts/timings/probabilities/scores,
checkpoints/fit traces and immutablefeature/checkpoint receipts are retained in
catalog_fusion/results/generalization-v2-attempt-1 and the original data run.

The independent audit verifies192native artifacts,384sealed feature files,
six checkpoint reloads,320steps,630score vectors,52,920probability scalars and
420atom confusion tables. See CATALOG_FUSION_GENERALIZATION_V2_ATTEMPT_1_AUDIT.md.
No v2 checkpoint is promoted to jacket, image, generated-copy or full-page use.
The next transfer experiment must use current storefront facts and a new public
task/label freeze, then validate fresh copy and complete-request latency.

Teammate main advanced during this restart. A normal non-force merge at e172f60
preserves their16-product storefront/color/gallery work and the model fix. That
catalog is separate from the pinned six-jacket scenario; versioned actual-store
input mapping is being recorded separately. MEMORY_MODEL_ASSESSMENT.md records
real24GiBhardware/headroom/swap and larger-model planning. The6/7GiBguards are
experiment limits, not user hardware restrictions. No apps were closed and no
weights were downloaded: current corrected-data quality improved on the same9B.


The actual-store import is now completed and independently checked:16products,
48color variants,195gallery records,32separate previews and81hashed local assets.
`store_catalog_v1.py` preserves all external crop/action bindings while excluding
routing IDs and persona briefs from model text. The additional `store_model_bridge`
checks current bytes of15selected input-producing sources before/after preparing
all195rows; historical authenticity alone does not establish active freshness.
The actual CPU receipt is store_model_results/v1/report.json; no native action,
rank, image choice or new copy is claimed. Three freshness/action tests supplement
14import tests. See ACTUAL_STORE_MODEL_INPUTS.md for capabilities and public task
compatibility limits. The old jacket snapshot remains unchanged.

The independent data review finds old labels teacher-consistent, but the old
TRAINquerydistribution was28singletons while validation/test usedonlypairs;
all84oldcatalogproducts hadexactlytwo tags. V2balances bothTRAINroles at21rows,
seven each cardinality1/2/3, eachatom6positives/15negatives. Seeded nuisance
assignment is not empirical decorrelation: its six-product testcatalog has
price-versus-tag-cardinality Pearsonr=.837. This narrow result needs new catalog
replications and explicit nuisance counterfactuals before broad claims. Focused
v2also omits free task wording/history from its positive-preference vector; it
is not a replacement for observed-profile/history reasoning. Unknown/negative/
contradictorypreferences and hiddenpersona inference were not trained/evaluated.
Hard public constraints remain outside learned scores. No larger model was
substituted to disguise these data/coverage limitations.


## Focused ranking wider-pool development replay, completed

Execution3ca08b2 reuses the exact frozen focused-v2 learned checkpoint
(c1adca343738d5d00f606587bdb03c5bb15f08c8e97751d3858e4fd621a50bd3).
No fitting occurred. Sixty real B1 forwards cover36products and24requests.
Learned choices are24/24 correct and12/12 preference pairs both correct with
changed selection, versus prior2/24 and random0/24. All144 learned grid
components obey hard guards. This is the previously inspected synthetic
fresh_eval_v1 development replay that informed representation/data changes,
not a new sealed holdout or actual-store/native-copy certification.

The actual fresh-query plus36verified warm catalog reads, learned head, guards
and both top12/top24 readouts median is124.137ms. Cold catalog build4.232s,
resident model load2.370s, whole child process13.382s, peak4.649GB. Images,
generated copy and browser/full-page rendering are excluded. Public positive
preference/tag preparation, hard guards, tie ordering and grouping remain
deterministic; trained4096-to-seven atom projections provide learned scores.

FOCUSED_RANK_TRANSFER_RESULTS.md/.json independently audit all60featurevectors,
checkpoint receipts,360roster views and unchanged raw outputs. The original
features remain in data/ranker/pretrained/focused-rank-transfer-v1. Raw predictions,
receipts and all three small checkpoints are preserved byte-for-byte under
focused_rank_transfer_results/v1 with COPY_RECEIPT.json. The concurrent
short-copy-fit-v1 experiment is still running; no copy result is implied here.


## Clean replacement: reconciliation and actual-store runner repair

At restart main was1ca67c8. An elevated read-only process inventory on2026-10-18
found no Python/model worker and neither prior PID32633 nor32623. All uncommitted
handoff/copy evidence and checkpoints were retained. No apps were closed. Host
has24GiB physical RAM,4.30GiB swapped at inventory; this run keeps the existing
6GiB native guard and does not infer usable headroom from free pages alone.

The independent short-copy V2 audit verifies240 exact gates and240 actual updates,
480 generated field rows,927 unchanged frozen language tensors (producer receipts),
and0/24 valid first3 and0/24 valid full6 payloads. Native wall972.864s, MLX peak
6,626,986,154bytes. No adapter promotion or valid-payload speedup. Original1.844GB
feature archive and both adapter checkpoints remain local. The existing508-file
copy under short_copy_results/v1 is preserved; weight copies are excluded from
this scoped commit.

Actual-store adaptation retains the original32 TRAIN/16 independently worded
EVAL requests and all16 TRAIN-seen products. Original source and v1/v2 packets
remain archived. The repaired v3 packet uses48 TRAIN/catalog encodes,80 learned
updates plus80 prior updates and exact head reload, then16 fresh evaluation
encodes while Qwen remains resident. Each evaluation timer spans native encode,
immutable query feature save/reload,16 verified catalog reads, learned scoring,
hard guards, top6 and actual serialization. It excludes generated copy, images
and UI paint. Post-call/final cooperative elapsed guards reject returned late
work; they cannot interrupt a hung native call. FAILED receipts preserve fixed
64-forward/32-TRAIN/16-EVAL denominators and partial progress, including setup
and postprocessing failures.12 adaptation CPU tests pass; combined prior suite
passed36 before adding3 receipt tests. Native v3 experiment is queued, not yet
claimed completed here.


## Actual16-store function adaptation v3, completed

Execution79990df:64 real B1 whole-input forwards,80 learned +80 independent
prior updates, exact head reload and TRAIN-only means. Learned evaluation12/16
top1,16/16top6,4/8both-correct preferencepairs; prior1/16, random0/16,
permuted-query2/16top1. TRAIN32/32top1. All16products are TRAIN-seen; independent
wording evaluation is now inspected. No unseen-catalog generalization claim.

Contiguous fresh-query→verified16warmcatalog→learnedhead→guards/top6→serialized
ranking median203.702ms (16requests). Copy/images/browserpaint/startup excluded.
Model load2.631s, native lifecycle15.503s including.074s CPUfit, run()wall20.474s,
MLX peak4,716,428,690bytes under6GiB. Eight deterministic guardprobes pass.
Independent audit reproduces all80+80losses/finaltensors exactly and verifies64
features/allscorearms/all16endpoint hashes. Details ACTUAL_STORE_FUNCTION_V3_RESULTS.md
and ACTUAL_STORE_FUNCTION_V3_AUDIT.md/.json; JSONevidence archive
actual_store_function_results/v3. Weights/features remain local unchanged.


## Actual-store six-request copy transfer, completed

The preserved failed-daypack adapter on actual learned top6 decisions gives0/6
validfirst3 and0/6validfull6. Invalid-attempt median4,979.665ms/6,227.023ms;
no valid-payload latency samples. Nativewall100.889s, peak6,127,417,756bytes
under7GiB. All6 width6 fork/rebuild/repeat controls exact;927 base tensors unchanged.
Titles7/36,bodies6/36,headings7/12 lexical/protocol/native-EOS accepted; badges36/36
exactNONE. Independent CPUtoken decoding/reassembly reproduces120fields/12payloads.
Known lexical relational loopholes remain disclosed. No promotion, no native
training in this diagnostic, no full-ranking+copy or browser measurement.
See ACTUAL_STORE_COPY_TRANSFER_V1_RESULTS.md and preserved JSONevidence under
actual_store_copy_transfer_results/v1. Next actual-store v2copy dataset changes
supervision/prompt together and must compare frozen/failed/new under matched inputs.


Resident actualstore action/seam diagnostic completed24 nativequeries. All16
replayed score/order rows exactly match actualstore-v3. All4copy-adapter-disabled
rows match frozen scores/actions exactly;927languagehashes unchanged. Fourreal
native policychecks pass (exactname,budget,colorvariant,stockunknown),5CPUtests
pass. Median174.942ms includes freshquery,16verifiedcatalogloads,guards,source
freshness and actualactionserialization; excludesgeneratedcopy/UIpaint. Separate
endpoint/run; no203.702→174.942speedupclaim. See ACTUAL_STORE_RANK_SESSION_V1_RESULTS.md.


## Actual-store source-conditioned copy v2, frozen before native work

Preparation is complete:16 TRAIN requests/320 fields and8 evaluation requests/
160 fields. Every product occurs6/3 times; TRAIN uses2 colors three times each,
evaluation uses its third declared color. All32 TRAIN and16 evaluation product/
color states are disjoint.96 unique TRAIN titles and96 bodies cover six
constructions per product. Badges now generate factual selected-color phrases;
category slogans cannot be repurposed as attributes. A new bounded relational
validator handles exact identity/color/item-price claims and decimal prices;
it is not comprehensive semantic proof and does not rewrite old validator results.
All288 CPU fact swaps reject stale targets and accept changed-fact targets.

25 combined runner/data CPU tests pass. Exact ordinary-chat tokenizer preflight
passes: root≤329 TRAIN/333 EVAL tokens, suffix≤79/78; no truncation; projected
uncompressed TRAIN host features1,420,738,560bytes (under5GiB). Final preflight
SHAcd8ead6445ac7785c90dfcfb0c15f6999e35a2da4c54530a59106db87cbb1cbd.
Source7f63b9ac18f9142910694499b18c1a5427d4b2da13f0dc3be20860fb44b89968.
Native plan:320 exact nonzero-control gates, restoring true initial zeroB after
each;320 seeded shuffled Adam updates from frozen base;927 frozen tensor hashes;
exact checkpoint reload and original-archive learned parity;24 fixed width6 arms
(8 requests × frozen/failed-daypack/new-store). All20 fields remain sampled,
including10 complete first3 fields; no serving target/template/fallback or2-card
substitution. Copy-stage endpoints include fresh root and validation/serialization,
exclude ranking and UI/browser paint. Native execution is queued at this receipt.


## Integrated local demo checkpoint (2026-10-18; supersedes queued work above)

The 320-update actual-store copy v2 fit completed all gates/updates/24 evaluation
arms. Checkpoint SHA256
56c6c1f4199d8689f1920d871213a46d2d121fec4e5836ff70e33aaf98c7c7b8.
Strict new-adapter payload acceptance is 2/8 first3 and 1/8 full6; the independent
contextual audit is 6/8 first3 and 6/8 full6. These are different endpoints.
All927 frozen tensors, cache and checkpoint reload controls pass.

Broader v4 ranking was completed and not promoted: semantic14/21 hit1,10/21
full target-prefix sets,1/8 pairs; six explicit-avoidance failures. Training
data lacks several positive+negative/multi-need combinations. V3 remains the
demo ranking candidate, with its known12/16 replay top1 and16/16 top6.

The actual-query hybrid v2 benchmark completed: strict1/6 first3,0/6 full6,
zero joint strict useful first3. Query-to-unrelated-product copy leakage and
malformed price strings were independently identified. Preserve these failed
results. Optional wider-copy/960-step continuation candidates are prepared and
CPU-tested but have NOT run. User priority is the usable integrated demo.

Opt-in React integration and a resident localhost service now exist. Default
storefront styling, catalog, gallery and cart remain intact. Learned v3 query
encoding orders all eligible16 products; selected top6 get fresh native v2
copy, progressively first3 then full6. Shopper text affects ranking; generation
uses selected public facts and a neutral browsing request. This is explicit
provenance, not a claim of shopper-conditioned copy wording. The prospective
practical screen withholds individual unsafe fields; unchanged strict v2
judgments are also archived. There are no generated-template/teacher fallbacks.
Public catalog execution is exactly equal to the training export, with only
reviewed optional ProductCard rendering bytes excepted by a versioned manifest.

First smoke run:9 requests,7 full-six. Warm resident width10 medians are
215.343ms rank,5169.288ms first3 and5751.707ms full6, HTTP receipt to serialized
events; browser paint/images/startup excluded. Seven full-six payloads accept
132/140 fields and withhold8. Bottle->bottle and bedside light->lamp, but
notebook->pouch is a real ranking miss. Exact Trail mug under$20 returns empty;
waterproof jacket returns one actual eligible card with4/4 fresh fields.
Chrome visibly rendered the streamed bottle and notebook results. Old source
model hashes927 and checkpoint remain unchanged after graceful shutdown. Peak
MLX6,519,340,896B under7GiB guard. Read store_demo_results/v1 and final demo
report for the later browser instrumentation measurement and launch state.
No conversion lift, unseen generalization, learned image selection, or
ranking-only full-page latency is claimed.


## Completed native comparison and restored serving checkpoint (2026-10-18)

Supersedes the earlier NOTRUN/pending-manager status above. The human cleared
the opaque sidebar item; the authorized native comparison ran successfully.
No current approval blocker exists. Read STORE_AR_COMPARISON_V1_RESULTS.md and
the independent audit; all native results, controls and source snapshots are
preserved in store_ar_comparison_results/v1, with originals left under data/.

Three queries, AB/BA/AB, six cards20fresh fields: conventional median first3/full6
attempt11.350/21.199s, hybrid5.555/5.937s. Both frozen usable proxies0/3. A's18
correct numerical bodyprices written dollars were falsely rejected by gates;
independent raw title/body judgment is3/3A,0/3B, without changing observed
readiness. Top1 relevance3/3A,2/3B; B misses notebook and withholds ambiguous or
unfinished bodies.3.57x full-attempt ratio is not a fully useful page speedup.
All927frozen hashes+adapter unchanged at shutdown; peak6,592,642,354B.

Service restored PID81217, output data/ranker/pretrained/store-demo-v5-post-comparison-1;
healthready zeroqueue927tensors, storefront4312HTTP200. Previous v4 lifetime
gracefully stopped with integrityreceipt; v3 disconnected without shutdown
receipt, preserved honestly. Shopper task01a14d29-68ba-71a3-b36b-3be096ea4020 owns
all new runner/packet/results files and received timing release. It implemented
real Sol screenshot adapter through bundled official CLI/existing login;
smokes pending at this checkpoint. Model task must not stage its files.
Tiger Data task01a14d35-a146-72b4-9cb9-11b0ed52d022 owns async experiment-results
DB adapter/routes, not serving, store DB or presentation UI; interfaces shared.
No credentials inspected or transmitted.


## Evaluation loop complete; overnight queue prepared (2026-10-18)

Successful repaired Sol pairv2: generic12actions72.553s, local8actions44.817s,
both independently verified simulated checkouts and protocol/evidence/budget pass.
One pair only,18slotsunattempted at checkpoint; no lift claim. Localbrowser
first3paintopportunity5450.7ms10/10fields; full6 6384ms19/20fields. Native
receipt archived store_demo_results/v5: rank204.920ms,first3 5423.797ms,
full6 6367.897ms. All required first-section fields finished and source factual;
only bottle directly matches narrow need. No physicalpaint/image timing.

Latest user instruction: majorarchitecture/performance work OVERNIGHT, 'notnow'.
Servicev5 unchanged healthywidth10,927sourcehashes,queue0. Shopper taskreceived
exclusive window for remaining9pairs/18sessions using unchangedcommittedsettings;
no GPUprofile/training/servicechanges may overlap. No new ownedworker launched.

CPU preparations now ready: profilev2+exclusive driver+finalpreflightv4, separately
versioned equivalentdollar usefulnessscorerv2 (oldoutputs/gates/timesunchanged),
generated-earlier copycachehelper (no nativewarming),718,810paramparallelwordhead
with34TRAIN/14heldoutbindings (no nativefeatures/quality), primarysourceDFlash/
short-generation feasibilityreport, audited32query/16function developmentfrontier.
66 combinedCPUtests pass. ENGINEEREDheadsmoke is arithmetic-only andpoorquality,
not a pretrained result. Read STORE_OVERNIGHT_PERFORMANCE_QUEUE_V1.md.
Low100s-ms personalizedcontent remains atarget, notmeasurement.

Tiger task ownsadapter/API. Initial real hostedreportuploadwasdenied before
execution; parent then confirmed explicituserapproval andDBowner resumedone
scopedupload/readback. Modeltask made no export/readcredentials; outcome pending
DBownerconfirmation at thischeckpoint. Preserve all separateownercommits.


## Full shopper suite and completed component profile

Shopper owner released allGPU/browserrequests after20actualreceipts. Fullsuite
fixeddenom10each: generic8expected(6fakecheckouts+2correctabstentions),local7
(5+2). Preserved pair04genericwalltimeout/localinterruption,03localtimeout,
08bothincorrectabstentions. Read auditedfullsuite summary; requestweighted
readinessincludeszero-copycases, not10comparablefreshcopy samples. No export
of the newfullsuite wasperformedbymodeltask.

Root then ran bounded nativecomponentprofile on same3developmentqueries,
6cards20fields416cap. Firstlaunch failedbeforepreflight/model becauselauncher
resolvedvenvsymlinktosystemPython; failurelogpreserved. Correctedattempt2
complete,3/3fulltoken/binding/EOSparity, no nativefailedattempts. Ordinaryfull6
median5580.099ms, instrumented5945.541ms (+365.442ms/6.55%observeroverhead).
Instrumentedmedians AR2517.889ms, suffixprefill1969.822ms, cacheoperations
562.505ms, root494.417ms, rowclone3.424ms. Independentread-onlyauditpass,
all927+adapterunchanged atshutdown, peak6,622,274,410B under7GiB. No new
speedoptimization orusefulcopylatencyclaimed. Read STORE_DEMO_COMPONENT_PROFILE_V1_RESULTS.md.

Unchangedservice restoredPID4452, data/ranker/pretrained/store-demo-v6-post-profile-1,
healthready zeroqueue927tensors. v5gracefulshutdownintegrityconfirmed.
Architectureexperiments remainqueued per 'notnow'; no newtraining/download/
install/deployment. Allsource/nativeevidencearchived store_demo_performance_results/v1.

## Integrated persona demo and completed follow-up, 2026-10-18 12:56 UTC

Canonical main now includes the ten-person public persona/history UI, real Tiger
public-history readback, actual local checkout, accepted-only launcher, explicit
failed-quality development launcher, and measured fresh-query profile caching.
Main 6955302 is pushed. No persona model has passed the promotion gate.

The first final31 rank4 LoRA fit completed 3,584 optimizer updates and all 927
frozen tensor/adapter/cache/reload checks, but independent quality was 17/32
primary versus frozen-base 20/32 (fused 17 versus 19). A complete audit of the
legacy 320 authored priority labels found ambiguous priorities despite valid
source history facts. The one bounded follow-up used only the independently
audited 128 clear TRAIN profiles for supervised loss and legacy contexts for
frozen-base distillation. Its 3,584 updates and native parity/integrity checks
passed, but a separately frozen 32-case packet scored 19/32 versus base 23/32
(fused 19 versus 22). These are different packets; 17-to-19 is not an improvement
comparison. Both packets are now development. All failures/checkpoints remain.
See persona_choice_results/{v1,clear_v2} and the independent adjudications.

The V3 warm-context candidate preserves V2 outputs exactly. It preencoded ten
public color contexts in 5.12 seconds, then performed a fresh Qwen query forward,
rank projection and copy predictions on every request. Ten loaded, pre-scrolled
profile switches reached three visible cards with decoded images at p50 335.4ms,
p95 416.6ms; full six-card decoded copy completed at p50 350.8ms. Startup and
physical paint are excluded. The initial hero exposes no complete useful card.
New work-bag query completion was 3.315s and reload 1.538s because full-context
feature keys still contain the query. All 13 protocol and two functional checks
passed; separate public relevance review still found ownership/preference misses.

Main `pnpm dev:personalized:demo` was actually launched, completed a real six-card
HTTP request with a fresh query forward, then gracefully stopped native82756 and
owned UI83106 unforced. All 927/copy/rank/cache/source shutdown proofs passed.
See store_integrated_personalized_launch_v1/report.json. The old uncached entry
is preserved as `pnpm dev:personalized:uncached`; `pnpm dev:personalized` remains
accepted-only and deliberately closed. Local weights are not in Git.

Main generic storefront also passed 11 actual browser checks: 16 products, real
search/cart, local fake checkout POST202 + fresh GET + matching visible ID, return
navigation, zero model requests and zero page errors. Earlier probe/HMR failures
are retained separately. Correct direct startup uses explicit PORT=4312.

One serving-only V4 experiment is now being measured separately. It neutralizes
query/budget/hard requirements only for profile-context feature conditioning;
actual fresh queries and actual hard guards still use the original request. It
preencodes pure color and X/H/Y tensors, verifies native/cache logits exactly
before storing, and caches no scores/rankings/response words. This is a different
function, with unvalidated quality; old packet quality and V3 parity do not apply.
No more training is planned in this pass. V2/V3 and their evidence remain intact.

Current non-model services: main website4310, generic storefront4312, strict
Tiger experiment API4311, local checkout4331. All hosted cohort proofs and old
READY/snapshots/packets remain immutable. Native timing workers and persona4313
are root-owned and may change only at an explicit graceful boundary.
