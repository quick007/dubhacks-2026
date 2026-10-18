# Model engineering handoff

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


Worktree: `/Users/lukas/Documents/Projects/dubhacks-2026`, remote
`quick007/dubhacks-2026`, branch `main` only. Preserve teammate changes, all
checkpoints, raw runs, reports and prepared candidate sources. No UI, API,
shopper-harness or browser work is needed for the next model experiments.

## Verified learned ranking result and its limits

`focused-rank-transfer-v1` achieved24/24 top-one choices and12/12 both-correct
preference pairs with changed selections. Prior2/24, random0/24. All144 learned
grid components obey hard guards. This replays the previously inspected
`catalog_fusion/fresh_eval_v1`36-product/24-request dataset. Its earlier failure
informed the new data and representation: this is a development replay, not a
new sealed holdout, unseen-catalog guarantee or human-conversion measurement.

Checkpoint: `data/ranker/pretrained/generalization-v2-attempt-1/focused-fusion.pt`,
SHA-256 `c1adca343738d5d00f606587bdb03c5bb15f08c8e97751d3858e4fd621a50bd3`.
A byte-identical copy is tracked in `focused_rank_transfer_results/v1`.
Training used21 public TRAIN query rows and21 public TRAIN product rows,
balanced seven singleton/pair/triple examples in each role,80 fixed CPU steps,
TRAIN-only role means, and public preference/tag binary supervision. Two learned
4096-to-seven biased sigmoid projections contain57,358 parameters; predicted
query/catalog atom probabilities supply product scores. Actual features come
from the frozen quantized9B backbone. Public positive-atom text preparation, hard
guards, canonical ordering/tie readout and section grouping are deterministic.
The compact representation omits free task/history/negative-preference semantics.

The124.137ms median includes one fresh query native encode, serialization and
verification,36 verified warm catalog feature reads, head scoring, hard guards
and both top12/top24 readouts. It excludes4.232s cold catalog construction,
2.370s resident model load, images, generated copy and browser rendering. Whole
child process13.382s; peak4,649,075,650bytes. See
`FOCUSED_RANK_TRANSFER_RESULTS.md/.json` and the copied raw results/checkpoints
under `focused_rank_transfer_results/v1`; original60featurevectors remain at
`data/ranker/pretrained/focused-rank-transfer-v1/features`.

## Factual-copy experiment at migration boundary

The prior run `data/ranker/pretrained/short-copy-fit-v1` is completed.
Its workers32633/32623 were absent at clean-restart process inventory.
Execution source2baf1e2; frozen preflight `SHORT_COPY_PREFLIGHT_V2.json`,
SHA-25656b62d8cbc469aa66595cdf6a6a8c65b620b5548a21596a24ce85bbefff319e2.
The clean replacement task explicitly authorizes subsequent local experiments.

All240 exact native/auxiliary loss and A/B-gradient gates passed before fitting;
all240 fixed Adam updates completed.65,536 trainable FP32 A/B parameters affect
only layer31 last MLP down projection; all927 base language tensors are unchanged.
Exact checkpoint reload and learned archived/native gradient parity passed.
New adapter: `data/ranker/pretrained/short-copy-fit-v1/adapter.safetensors`,
SHA-25685ce8215add1d4ce561441d6972c8e2958dbb73966bd89f53595ef65b7fddb57.
It warm-started the immutable prior checkpoint
`data/ranker/pretrained/semantic-copy-feature-fit-v1/adapter.safetensors`,
SHA-2560b0aeda377a9c9ad7eb37b529c139dac84bc5acd4bb559176d40245865ffe1b2.
All240 feature archives+metadata total1,844,548,247bytes under5GiB, one full host
feature row resident at a time. Keep the original archive in place.

TRAIN12 public cases/240fields include fresh C supervision and independent
S/T/B/C fields. Six previously inspected dev/test cases compare prior/new at
width3/6, fixed24arms. Native controls compare reused/rebuilt/repeated full
hybrid caches at matching widths in frozen/prior/new modes. Each first-three
endpoint requires ten complete fields and actual validation+serialization; full
six requires all20fields. These are locked-decision copy-stage measurements,
not ranking/full-request/browser timings. Token/strict factual validators are
unchanged, no templates/fallback/teacher text is used at inference.

## Current actual16-product integration

`store_catalog_v1` imports the real16products,48colorvariants,195galleryaction
records,32previews and81hashed local assets. `store_model_bridge` rejects stale
input-producing sources before/after candidate preparation. All17 import and
freshness CPU tests passed at this boundary. Public name/category/itemprice/
searchTerms/colors and declared image metadata are preserved; pixels are not
examined. Stock/sizes unknown. Shipping is cart-subtotal policy, separate from
item budgets; a hidden shipping quote is not free delivery.

The old seven-factor representation has only organizer supported among these
products:15products collapse to the exact same empty-atom text. Do not claim
actualstore success from the old24/24 synthetic replay. New candidate
`store_function_adaptation.py` prepares FULL publictext features with a separate
16-function learned head,32 public TRAIN requests and16 independently authored
evaluation requests in eight preference-only pairs. All16 catalog identities
are intentionally TRAIN-seen: this measures known-catalog request-wording
adaptation only. Evaluation packet is frozen before vectors/fitting; see
`store_function_eval_v1.json` (SHA55eb12fc85b4d87289bdea71dad1b4c8b5cb55f31d762c06c39a47131868e377)
and `STORE_FUNCTION_DATA_LABEL_AUDIT.md`. Native64B1 and fixed80-step fitting
remain unexecuted at this handoff.

`store_copy_diagnostic.py` is a separate actual-store copy candidate with strict
public-fact lexical/protocol checks, actual action bindings, genuine native
wording, complete first3/full6 timings and no runtime fallback. Its ordinary
chat prefix differs from the short-copy training distribution, so quality must
be measured, not assumed. It does not establish learned image selection or
comprehensive semantic factuality. No native store-copy inference has run.

`fused_result_benchmark.py` is the prepared six-request synthetic combined
ranking/image/copy diagnostic. It requires exact cached-vs-cacheNone feature
compatibility and full native cache/adapter seam proofs before timing fresh
requests. It is unexecuted and must stop on strict mismatch. Six cards from36
synthetic products are separate from actual16-store integration.

## Resources and next work

Apple M5,24GiB physical; `.tools/pretrained-venv/bin/python`, MLX0.32.4,
mlx-lm0.32.2,torch2.14.1,transformers5.20.1. Existing local frozen model:
`.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit/snapshots/8b2b98c00a6b4d291155e4890773ca8f769aee53`.
No new weights were downloaded.6/7GiB native guards are experiment settings,
not the24GiB hardware limit. No apps were closed. Heavy GPU work remains serial.
Native hybrid state is24recurrent/convolution ArraysCache +8KV histories; exact
whole snapshots/deepcopies only, no arbitrary trim/splice.

Next task should review final copy evidence and candidate CPU tests, then run
the actual16-store64B1/fixed80-step adaptation once, measure real fresh-query
ranking and guard readouts, and run the bounded store-copy diagnostic only after
its source/provenance controls pass. Keep all failures and original checkpoints,
use ordinary scoped main commits/pushes, and preserve the actual quality/latency
boundary. Do not promote an adapter from faster invalid attempts. Perceptron
remains deferred.


## Clean replacement checkpoint (supersedes queued statements above)

Main167295f is pushed. All old checkpoints and candidates remain intact.
Actual16-store v3 completed64 native inputs and80+80 CPU head updates:
12/16 top1,16/16top6,4/8 both-correct pairs. Independent audit exactly reproduces
all optimizer steps/final tensors. Median203.702ms contiguous serialized ranking
endpoint excludes copy/browser. All8 separate guard probes pass. New resident
rank/action seam completes24 native queries:16 score/order replays exact,
4 copy-adapter-disabled score/action rows exact and4 policy probes pass
(exactname, budget exclusion, Cobalt binding, unknown-stock abstention). It is
model-side integration; React/API files were not changed.

Actual-store old-adapter copy transfer completed6 requests:0/6 valid first3 and
0/6 full6; invalid-attempt medians4.980/6.227s. Matched-width controls and927 frozen
hashes pass. All120 fields decode/reassemble exactly on CPU. No promotion.

The new balanced source-conditioned v2 copy data/runner is frozen and tokenizer-
preflighted (25 CPU tests); native320-gate/320-update/24-arm run is next.
Final preflight STORE_COPY_FIT_V2_PREFLIGHT_V1.json SHA
cd8ead6445ac7785c90dfcfb0c15f6999e35a2da4c54530a59106db87cbb1cbd.
Read latest ledger/reports for progress; preserve incomplete/failed runs.


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


## Final demo and evaluation preparation checkpoint

Main987685c pushed; opt-in Chrome storefront is live at
http://127.0.0.1:4312/?model=local with service8766, current output
data/ranker/pretrained/store-demo-v3. Do not launch another native GPU owner
while it holds native-owner.lock. Three measured warm browser scenarios:
rank opportunity276.9/342.6/291.2ms; first3opportunity5572.5/5000.3/5571.4ms;
full6opportunity6152.2/5733.5/5940.5ms. Paint opportunity is DOM commit+two
frames, not images or physical paint. Each full6accepts18/20 native fields;
all5latest hybrid attempts fail accepted-title+body-per-card proxy. Metadata
cards/cart still function; no full-useful-fresh-page latency claim.

User explicitly says full personalized-content latency is the performance
target. Width10independent field batching is live, not20 serial strings.
Median fresh neutral root529.5ms; combinedfork/suffix/prefill/decode loop
5074.8ms; copywhole5637.6ms. Root hash constant, notyetcached. Detailed split
currentlyunknown; newprofilecandidate measures it without changingalgorithm.
Larger tuning reservedforlater; integration+evaluationloop immediate.

Conventional singleAR comparator5publiccases tokenpreflighted,14CPUtests;
initialnativecomparison NOTRUN. Pendingmanager0190482d6a3340a9a61b094f52da8621:1
reported byparent withoutexactcommand/target/reason; noinspectiontoolavailable.
Parentread_thread lookup returnedno turns. Clarificationrequested asynchronously;
do notbypassapendingaction. CPU/browserframeworkwork continues independently.

Currentdashboard10personas/20runs are scripted six-jacketmock, notactualstore
Chromeagents. ReallegacyPlaywright onlyclicksfirstAddtocart/fakecheckout and
blocks8766; itdoesnot scorepreferences. Newstandalonecontract+10actualstore
casepacket+prepare/validateCLI beingbuiltcompatiblywithoutlegacy/UIedits.
A fork-none independentChrome genericshopper pair02 hasbeenlaunched withonly
publicquery/privatehumanneeds,nogoldIDs; preserveactualresultsandfixed20run
denominator. No realpurchases. Richmerchantprofilecontextnotyetwired; current
localhook sendsqueryonlyplusserviceoptionaltypedguards,neverhiddenpersonatruth.


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

## Resumed overnight: useful known-catalog parallel text

The later user restart superseded the earlier pause. Work resumed with serialized
native jobs and CPU preparation/audits. Main pushed through `3cc5fa7`; weights
and feature tensors remain local. All previous checkpoints/failures preserved.

The explicit ALL48 catalog demo head at
`data/ranker/pretrained/store-parallel-text-train-all-demo-v1-attempt-2/head.pt`
has SHA `074169e31bf5f6b99cc5bc35c9b2e64d82396d5225b3a67cc2cf5f613fcc4889`.
Fixed320 CPU steps, 528/528 supervised outputs correct, independent teacher-blocked
reload confirms real learned outputs. It reads cached frozen public Qwen feature
vectors, canonical name/color/price tokens and a fixed plain style; serving
loads no teacher, gold or TRAIN output file. All48 bindings were trained; this
is a curated demo, not heldout generalization. Original34/14 split had0/126
heldout card fields, and three CPU research variants also failed. All archived
losslessly under `store_parallel_text_results/v1/`, without weights.

Successful fresh native run:
`data/ranker/pretrained/store-parallel-text-train-all-fresh-v1-attempt-2`.
All32 fresh queries return six cards and20 valid fields; median first3/full6
193.428/193.498ms. Offline primary-function top1 30/32, top6 31/32; notebook
and blanket misses remain. All8 separate guard probes passed. Attempt1 is still
failed after its32 valid timings because its guard audit mishandled a safe
conflicting-color error. Both complete shutdowns verify all927 frozen tensors
and native adapter unchanged. Full source/raw archives and independent audits
are committed. Startup/browser/images remain separate endpoints.

Fresh AR8GiB experiment completed three paired queries: width20/root reuse
4571.540ms versus ordinary width10 6402.946ms; both0/3 useful full pages. Earlier
width20/7GiB profile exceeded budget and remains failed. Repaired native-generated
earlier-copy cache run completed32 hits, median219.627ms, all under300ms, but
0/32 full useful pages and9/32 first3 title/body. Its original eight warming
costs, invalid fields, first failed nested-deadline run and genuine new miss/hit
proof remain archived under `store_native_candidate_results/v1/`. Cache was not
promoted based on speed alone. Read STORE_NATIVE_CANDIDATES_V1_RESULTS.md.

Actual candidate service is currently PID30860, sole native worker on8766,
`data/ranker/pretrained/store-candidate-demo-v1-attempt-1`, parallel mode,
reviewed frontend integrationv6. First HTTP after startup full20/20 fields456.811ms.
Live budget/color/conflict checks preserve zero-generation abstention and
partial-group heading withholding, no padding. New frontend copySource flags
visibly separate fresh learned catalog-trained text, fresh native AR, earlier
cache, mixed work and no-generation. Original AR CLI remains unchanged and
available with explicit reviewed manifest. Tracing owner added lazy opt-in
hooks to main.tsx at a coordinated boundary; main is pinned in v6. They own
those hooks/dependencies/docs, model task does not stage them.

Browser integration is in progress with no active shopper-agent batch. First
probe's four new-context requests passed, but four same-URL repeats did not
navigate or send new POSTs; PROTOCOL_AUDIT excludes them, raw evidence retained.
Repaired attempt2 performs actual reload and requires unique document/request
IDs:8/8 actual POSTs and all20 fields match DOM/provenance. First3/full6 decoded
image endpoints from query effect were fresh-context618.2/294.4/326.2/307.5ms,
reload270.0/268.1/267.1/272.5ms. The default hero leaves card text below the
viewport (zero fully visible cards), so these are DOM/image readiness, not
useful above-fold or physical-paint claims. Each page also retains a WebSocket
console error, with cause unproven. Separate pre-scrolled collection-view probe
and independent browser audit are next. Preserve these raw outputs.

Queued useful work: learned public-fact pointer decoder research for heldout
color composition (CPU-only two fixed arms), compact fresh-AR diagnostic16
requests after browser timing releases. No DFlash install/download or paid
compute. Root owns native lock/jobs; all other model agents CPU only.

## Latest morning boundary: balanced rank and responsive delivery

Main now includes86a4b51 (responsive assets, compact runtime, lifecycle guards,
v5 rank adapter/data/audit),45e2ff5 (browser/pointer evidence), and teammate
checkout/telemetry consumer commits through55daca6. All checkpoints, original
images, failed attempts and local candidate outputs remain preserved. No weights
or feature tensors were committed.

V5 head5d07673e51450bbbbb1803c8a44ef99721b39fd3b1ae92edb9168e27a1377d6b;
serving export17a0a15a2da6e2fc90e651e1f5bb2c237d61241795ec0cca548a4f3ce319d703.
96 native forwards +48 reused =144 vectors; all927 tensors unchanged. Independent
1,317-check replay:29/32 top1 and32/32 top6 versus old14/32 and29/32. Corrected
miss-rank diagnostics are separate; all310 original run files unchanged. Copy
head074169e3... and public feature2624a947... remain the ALL48 TRAIN demo.

Latest independent browser audit verifies8/8 navigation and4/4 fully visible
first3 views. V5 notebook ranks1st instead of13th on the fixed development query.
Loaded visibility/decode median269.95ms; cold/reload full-content+6images
533.10/362.40ms. All20 fields factual,16 catalog cards retained. No physical
paint, high-DPR generalization, comparative shopper effect or causal speedup is
claimed. Partial raw capture attempt4/8 remains unscored, with exact historical
source recovered. Read STORE_COMPACT_V5_BROWSER_INDEPENDENT_AUDIT_V1.md.

Both temporary service PIDs84158/87933 exited gracefully with927/adapter/copy
assets/image sources unchanged; v5 shutdown also verifies head/export. V11
manifest7e4b06fcaecce868d79454587dcbf118cb1d4a3989db039c0db08df8a52a6839
pins final checkout frontend main5c70bb1... . Frontend90191 on4312 proxies
local-only authoritative checkout4331/PID89743. Generic real Sol development
smoke confirmed POST202+GET200/disk/DOM receipt. Harness is preparing60 pairs
and holds Chrome until root READY. No private cohort goals/gold are shared.

Root is finishing bounded component profiling and a20-pair generic incremental
latency probe before freezing/restoring the cohort service. Profile attempt1
failed in excluded observer warmup (delegating tokenizer restoration), with all
40 timed arms unattempted and clean927/checkpoint shutdown. Attempt2 uses the
same unchanged model and fixes the observer only. Do not infer timings from a
failed run. Model/worker changes are forbidden during the frozen shopper cohort;
future prefix/catalog-cache candidates run only in agreed isolated windows.

The automatic telemetry pipeline and new context-only consumer are separate
from query supervision: no shopper queries or real customer rows exist in the
verified dataset. Four explicit synthetic positive choices pass actual
fit_inputs with clearly fake4096 CPU protocol vectors; no optimizer/native
embedding/live training occurred. Guides apps/api/SHOPPING_TELEMETRY.md and
apps/api/SHOPPING_MODEL_INPUT.md. Parent cloud thread cannot be reached by local
app messaging (thread not found); live commentary/shared main and harness relay
carry status.

## Frozen cohort and isolated integration, 2026-10-18 08:47 UTC

The bounded profile and 20-pair browser comparison are complete. Original full6
p50/p95 was192.785/202.370ms; the instrumented observer adds paired median3.782ms.
Fresh Qwen encoding dominates at170.622ms exclusive median. All40 profile arms
passed exact-output parity. Browser generic/local first-useful p50/p95 is
64.3/65.1ms versus281.1/297.7ms; paired additional delay217.4/234.4ms. All40
browser slots passed, with cached-image eligibility and loopback scope explicit.

READY.json SHA b679afab96666a0fc96536d6abfe5d08f7f6cbc33765cba44ed8b75721c1ebfb
released native4200/8766, storefront90191/4312 and checkout89743/4331. The
60-pair cohort started08:42:32Z, freezes670 source/assets/checkpoint files and
keeps all assigned failures in denominators. No other native or browser jobs
may run until the harness releases its timing window. Do not modify READY or
frozen files, even for unrelated improvements. Shutdown integrity is pending
until this service lifetime ends.

Latest upstream presentation update675ef4c was fast-forwarded before freeze;
it changes apps/web only. It did not remove the actual storefront's older
Demo store banner or cart labels. These are being corrected in the separate
worktree /Users/lukas/Documents/Projects/dubhacks-persona-integration on
codex/persona-integration, preserving the existing visual design. User now
authorizes a compact bottom-right person switcher, separately authored public
history/backgrounds, meaningful synthetic activity and actual candidate
training/evaluation. Synthetic and real provenance must remain separate;
private cohort goals/gold never enter model inputs.

Morning acceptance requires coherent switching/query/cart/navigation, real
session export through Tiger into an actually trained and held-out-evaluated
candidate, verified cohort results bound into the existing presentation website,
package/build/test checks, concise run instructions and normal remote main
verification. Integrate candidate work only after the cohort finishes. Preserve
failed candidates and retain the current model unless measured evidence merits
promotion. The earlier CPU bridge with fake protocol vectors is not training.

## Isolated persona preparation, 2026-10-18 09:40 UTC

The active cohort remains frozen; 92/120 receipts are saved with two retained
timeouts, one per arm. Wait for the harness's explicit ALLBROWSERIDLE signal,
then gracefully stop owned native4200 and verify its shutdown927 proof before
starting another native job. Keep checkout4331 alive for final independent GETs.

The isolated branch now preserves the persona UI, authored public seed, real
Tiger readback, TRAIN-only purchase contrasts, weak first fit, independent
audits and guarded launcher. Latest local commit574d05a; nothing merged into
the frozen storefront. The first CPU context head fit TRAIN159/160 primary
choices but only1/40 development primary choices and0/10 history demotions;
it is explicitly not promoted. All source archives and checkpoints remain local.

The next predeclared candidate uses432 genuine frozen-Qwen public-context
forwards, then250 fixed CPU updates over320 TRAIN rows. Preparation v4 SHA
71c9cabc533083e726cfd5938fed9e2cacf88e4ef841b102b4dc3a73ba619c35
and its source/data pins are verified. The fresh32 independent packet remains
unscored. Root promotion gate SHA89e243b2444eea4516b420491c1c70a18a5b115a686d1fc0f1324f4cd061aa4f
freezes20/32 primary,24/32 novel-relevant and28/32 strong top-three minimums,
baseline contribution, factual checks and separate live-store acceptance.
Runtime portability tests pass: a moved checkpoint uses its adjacent exact-hash
public context cache, with no fallback to the original worktree path.

## Completed cohort and next representation, 2026-10-18 10:25 UTC

All120 main shopper sessions and8 separate controls are preserved. Main intent
success is generic58/60 versus local57/60; suitable checkouts46/48 versus47/48.
The $96 merchandise difference came from one generic decision timeout. No
conversion benefit was demonstrated. REPORT.md in store_shopper_cohort_v2
contains fixed denominators, paired intervals, visibility mismatch, readiness
scope and the31 transient last-safe observations that subsequently completed.
Actual Tiger report store-shopper-checkout-cohort-v2-final has envelope SHA
31afcba97aa08096b805960f6b6d5f8871fe5fde258467e9ff6ac451d9394172.
The default five-slide website now consumes that report, with50 real-Chrome
acceptance checks passed and screenshots/source pins saved separately.

The harness released all browsers at10:04:43Z. Root gracefully retired native4200,
verified its process absence and all927 frozen tensors/checkpoint/source pins;
store_fast_model_snapshot_retirement_v1.json records the evidence. Checkout4331
and strict Tiger receiver4311 remain live. Original READY/snapshots are immutable.

The new genuine432-context extraction completed234.24s with5.53GB peak and all927
tensors unchanged. Its250-update head failed independent32:4 primary,7 novel
relevant,15 strong top-three. Integrity and facts passed. The32 cases are now
development replay; do not promote or reuse them as unseen evidence. Independent
TRAIN replay found original requests45/160 primary, contrasts145/160, severe
feature anisotropy and remaining underfit. Three preregistered CPU follow-ups
completed3750 total updates (raw2500, centered250/1000), each fitting TRAIN320/320
and known history diagnostics10/10. All still scored only8/32 development primary.
Sources, failed checkpoints, means, all predictions and freeze receipts remain in
the isolated persona worktree; no runtime accepted profile has been created.

A separate20-forward public-demo diagnostic is testing the pretrained LM answer
logits on complete catalog-conditioned prompts in canonical/reversed orders.
This uses no gold, optimizer or autoregressive decode and is not a promotion
claim. Any new candidate needs separately frozen independent quality evidence
and actual serving/browser acceptance before becoming the normal store model.

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
