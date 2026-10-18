# Local model demo

## Current recording checkpoint — 2026-10-18

The canonical 35-product/105-variant development demo is running on loopback8767,
with storefront4313 and local-only checkout4331. Use `pnpm dev:personalized:demo`
for a new owned lifetime; do not start it over a ready session. Canonical live URLs:
http://shop.dubhacks.localhost:1356/ and http://dubhacks.localhost:1356/.

Binding: `store_expanded_integration_v2/manifest.json`, SHA
`971f30d146415abffac4b0aee59827b04d7c06458a38ae9bcd0cd22d46539088`.
Native output: `data/personalized-development/2026-10-18T15-54-04.966Z-42330f4f-d52a-40b2-be98-b199c08847fb/native`.
Original16 LoRA routes supply family priors; new35 catalog vectors and105 copy-fact
vectors are genuine frozen-Qwen features. The new copy head passed1575/1575
supervised fields after400CPU updates; all105 bindings are TRAIN. No new-ID LoRA,
unseen quality, promotion or human conversion claim is made. Thirty actual
profile/search checks passed eligibility/budgets with no unrelated padding.

Warm backend10six-card samples:85.5ms median. Loaded-collection visible/decoded
three cards:138.8ms median across10actualprofile switches. Fullsix-image completion
not established. Two fresh-query HTTP samples:266.8/310.9ms backend; the first
novel browser trial's stale-mirror audit failure/cancellation is preserved. Startup,
initial navigation and physical paint are excluded. Exact query input features may
be reused; every request recomputes learned scores, actions and word slots.
See `expanded_store_results/v1/CURRENT_LIVE_SUMMARY.json` and
`STORE_EXPANDED_ENGINE_V1.md`. Keep the current worker running for recording.


The current fast known-catalog candidate uses `store_candidate_image_service_v1`,
the 320-step parallel copy head, public Qwen fact features and reviewed v11
integration manifest. Its optional explicit v5 rank-head export uses the verified
original v3 catalog features without disguising the new run's 96 native forwards
as the original run's 64. The independent ranking audit records 29/32 first
choices and 32/32 targets in six, versus 14/32 and 29/32 for v3 on the same
independently authored queries. This is known-catalog paraphrase evaluation.
See the [root setup](../../../README.md#enable-the-local-model) and
[fresh-query audit](STORE_PARALLEL_TEXT_TRAIN_ALL_FRESH_V1_AUDIT.md). All 48
product/color bindings were supervised; its short learned word-slot decoder
uses no runtime teacher lookup. The shopper query affects the Qwen ranker,
while text is conditioned on the selected catalog facts and plain style.

Copy events declare `copySource`: `fresh-parallel-head`, `fresh-native`,
`generated-earlier`, `mixed-native-cache` or `no-generation`. The frontend checks
consistent generation flags, labels catalog-trained text, and never calls
earlier-generated cache text fresh. Zero eligible products emit no-generation;
short groups with fewer than three products withhold their untrained heading
without padding the roster. All cards retain the original name, exact price,
real variant and existing gallery/cart action. The separate generated-cache
mode is experimental and retains the old AR copy failures.

Requests use the original single-worker HTTP queue, budgets, cancellation and
frozen-tensor checks. The image entry additionally pins its serving helper,
rank adapter and asset-verification scripts for the entire lifetime. It checks
those pins before emitting each payload. Failed startup preserves the original
error and attempts frozen-hash cleanup; native stream teardown still runs if
hashing or receipt writes fail. Shutdown records separate rank/export checks.

Responsive WebP delivery retains all original images and catalog facts. Its
3,125-byte runtime manifest carries geometry only; full source/derivative hashes
stay in offline audit manifests. Freeze/startup verify all assets. The API mock
fixture transition to fieldwork-study is an explicit auxiliary-source exception:
its private persona goals never enter model query or product inputs.

Use the current command in the root setup guide. Choose a new output directory
per lifetime. The following AR protocol and old commands describe historical
comparisons; their older integration receipts do not cover the current frontend.

The opt-in storefront path uses the completed actual-store v3 ranking head and
320-update copy v2 adapter on one resident local Qwen backbone. It performs a
fresh native query encoding and generates every displayed title, badge, body
and section heading natively. There is no template or teacher-text fallback.

After the reviewed frontend integration manifest is frozen, run from repo root:

```sh
.tools/pretrained-venv/bin/python -B -m experiments.ranker.pretrained.store_demo_service --width 10 --output data/ranker/pretrained/store-demo-new-session
```

Choose a new output directory for each service lifetime to preserve startup/shutdown receipts.
The currently verified lifetime uses `data/ranker/pretrained/store-demo-v3`.
Open `http://127.0.0.1:4312/?model=local` with the existing storefront dev server running.
The service binds `127.0.0.1:8766`. One dedicated thread owns the native worker
lock for the process lifetime. Native requests run serially. Width 10 is the
default; explicit width 6 or 20 is supported. Active and peak allocations must
remain within the inspected 7 GiB guard. No download or training is performed.

`GET /health` returns HTTP 503 during startup or after an error, and HTTP 200
with `ready:true` when the resident engine is available. Startup hashes all 927
frozen source tensors and verifies the completed adapter checkpoint. Graceful
termination checks the frozen hashes and checkpoint again.

`POST /personalize` accepts a bounded JSON body:

```json
{"query":"a bottle for water under $40","itemBudget":null,"hardRequirements":[]}
```

Optional hard requirements have `{attribute,value}` string pairs. Supported
attributes are `color`, `category`, `stock`, `size` and positive `waterproof`.
Stock and size are unknown in this catalog and therefore produce no eligible
items. Colors bind to a real matching declared variant. No unsuitable products
are added to fill a six-card target.

Typed `itemBudget` is inclusive and covers item price only. When absent, an
explicit dollar ceiling after `under`, `below`, `less than`, `up to`, `max`,
`maximum` or `budget` is recognized. `under/below/less than` are exclusive.
Recognized budget text can accompany a whole exact product name; an ineligible
exact product causes an empty selection. Delivered-cost budgets return a clear
unsupported error. Simple explicit `in stock` and `size M`-style phrases become
hard constraints. An unnegated literal `waterproof` requires exact positive
public source evidence. Other free text remains a learned ranking preference;
this is not a general natural-language constraint parser.

The response is newline-delimited JSON with a `type` discriminator:

- `rank`: all eligible `productIds` in learned order, `selectedProductIds` with
  up to six cards, a `variants` map, metadata `sections`, `shortfall`,
  `eligibleCount`, `timing.rankMs` and provenance.
- `firstThree`: generated copy for the first `min(3, selected count)` cards.
- `complete`: generated copy for every selected card, with at most six total.
- `error`: `requestId` and `message` after a streaming request fails.

Each copy event contains `sections`, `acceptedFields`, `withheldFields`,
`requiredFields`, validation reasons and timing. Cards retain `productId`,
`variantId`, `originalName`, `itemPrice`, `selectedColor`, `catalogCategory` and
the bound image receipt. Generated `title`, `badge` and `bodyCopy` can be null;
section headings can also be null. An empty eligible roster emits `rank` then
`complete` with empty sections and zero field counts.

The public shopper query affects ranking only. Fresh copy is conditioned on
the selected public identity, category, price and color plus the neutral public
request `Show a clear collection of store products.` This avoids transferring
one shopper use claim to every unrelated product. Runtime source checks bind
the learned catalog to the reviewed actual storefront. Images remain the
existing deterministic storefront gallery policy; image/pixel learning is not
claimed.

The prospective demo safety gate allows natural framing outside the narrow
pilot grammar while checking complete identity, color, numerical price claims,
native completion and obvious unsupported use/availability claims. Whole exact
catalog category phrases are supported; their individual words are not treated
as an arbitrary bag of attributes. Unsafe fields are withheld as null without
repair. This lightweight screen is not a comprehensive semantic truth proof.
The old strict benchmark and all its results remain unchanged. Each receipt
also records the unchanged strict v2 per-field judgments separately.

Receipts are preserved under
`data/ranker/pretrained/store-demo-v1/requests/<uuid>/receipt.json`, including
exact public input, head input, scores, raw native text and tokens, guards,
checkpoint provenance, practical/strict field judgments and timing.
`rankMs`, `first3readyMs` and `fullReadyMs` start at HTTP request receipt and
include queue delay, fresh encoding, generation, validation and JSON
serialization as applicable. They exclude resident startup, browser paint and
image transfer. Client disconnects cancel stale work between native batches.

CPU/HTTP contract checks (fake worker; no native model load):

```sh
.tools/pretrained-venv/bin/python -B -m unittest experiments.ranker.pretrained.test_store_demo_engine -v
```

Browser instrumentation records request effect entry to React DOM commit plus
two animation frames. Read the last12 completed standard User Timing measures
from `html[data-local-model-readiness]`. These are after-paint opportunities,
not physical screen paint, image readiness or proof of viewport visibility.
The detail includes server field counts and native timing. Full copy completion
can contain withheld null fields; do not describe it as all20 generated fields
being acceptable. `engineInitializationMs` includes engine verification/load/
hashing but excludes process imports and lock wait; it is not an OS-cold metric.
