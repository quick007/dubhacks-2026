# Local model demo

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
