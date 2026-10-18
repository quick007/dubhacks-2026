# Actual storefront local demo integration

The opt-in storefront is running at `http://127.0.0.1:4312/?model=local`,
using one resident localhost Qwen worker at port8766. Main987685c is pushed.
The default storefront, CSS, existing gallery and cart remain intact. The
trained v3 head orders all16 eligible products; top6 receive independently
fresh adapted titles, badges, body text and two section headings. Selected
public facts condition copy; the shopper query conditions ranking only.
Images, hard guards, price displays and cart actions use deterministic catalog
metadata. Generated fields that fail the prospective practical screen stay
null; original catalog text remains available. No copy templates or teacher
answer fallback are served.

The final demo uses rank SHA
265c728a2e3349fa03e03ca37ca5d54814fe72407b95ed436d8f9734b602a99e
and copy SHA
56c6c1f4199d8689f1920d871213a46d2d121fec4e5836ff70e33aaf98c7c7b8.
Versioned integration manifests verify the execution export exactly matches
the actual16-product training catalog and pin all integrated frontend bytes.

## Measured readiness

All figures below use the M5/24GiB, the same warm resident Qwen3.5-9B4bit
backbone, width10, fresh query encoding and fresh native copy. These three
latest Chrome requests ran sequentially with no competing native worker.
A full page reload preceded the first measured request; images/browser caches
were not cleared and OS-cold latency was not measured.

| Public shopper query | Server rank | Server first3 | Server full6 | Browser rank opportunity | Browser first3 opportunity | Browser full6 opportunity | Fresh fields | Top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| a light for my bedside table |224.5ms|5.517s|6.097s|276.9ms|5.573s|6.152s|9/10 then18/20|lamp|
| a bottle for drinking water |246.3ms|4.943s|5.686s|342.6ms|5.000s|5.734s|9/10 then18/20|bottle|
| a notebook for quick notes |224.6ms|5.517s|5.901s|291.2ms|5.571s|5.941s|8/10 then18/20|pouch, incorrect|

Browser timing starts at hook effect entry before subscription/fetch, then
ends after React DOM commit and two animation frame callbacks. Median across
these three is291.2ms rank,5.571s first3 and5.941s full6. This measures a
paint opportunity, not physical screen paint, image completion or viewport
visibility. It is not a complete useful above-fold readiness proof: fields
were withheld and product text may be below the viewport. Server timing starts
at HTTP receipt and ends after generation/validation/serialization; it includes
queue delay but excludes startup, images and browser. Complete means decoding
finished for the requested six cards, not that all20 native fields were valid.

All five final native requests completed at18/20 accepted fields; two fields
per request were withheld. Earlier smoke requests and the independent
qualitative audit are preserved under `store_demo_results/v1` and
`STORE_DEMO_V1_QUALITATIVE_AUDIT.md`. That audit found seven ambiguous lamp
collection-price bodies, now prospectively withheld with a targeted regression
test. Strict v2 judgments remain separate and unchanged. These are inspected
smoke/development queries, not unseen generalization. The notebook failure is
retained rather than repaired by a keyword ranking fallback.

Final engine initialization was10.065s, including source/checkpoint verification,
model load,927 frozen tensor hashes and catalog-head construction. The narrower
backbone load was1.072s. Process imports, lock waiting and OS-cold effects are
excluded. The initial lifetime peak was6,519,340,896 MLX bytes under7GiB; both
completed prior lifetimes verified927 frozen hashes and adapter unchanged on
graceful shutdown. Final lifetime is intentionally still serving and has no
shutdown integrity claim yet.

## Functional checks

Chrome visibly showed bottle and light ranking/copy and the notebook miss.
Switching the bottle to Cream removed the Graphite-generated copy; Add to cart
then showed the catalog-backed Cream variant at$36. No order was placed.
Exact `Trail mug under $20` returned zero eligible products and a six-card
shortfall. Waterproof returned the one genuinely eligible jacket and a
five-card shortfall, with4/4 freshly generated fields; these smaller workloads
are not six-card latency evidence.

34 storefront CPU tests and33 model/integration/transparency CPU/HTTP tests
pass. Typecheck and production build pass. All workspace locked dependencies
were restored; manifests and lockfile are unchanged. No deployment, paid
compute or new credentials. All checkpoints/large archives remain local.

## Remaining work

A conventional single autoregressive decision+20-field comparator is CPU-
prepared and token-preflighted. No native comparison has run and no speedup or
quality-winner claim is supported. Its execution is held while the parent-
reported approval manager0190482d6a3340a9a61b094f52da8621:1 remains pending; the
exact underlying approval action/reason was not supplied to this child. No
related operation was retried or bypassed. The resident service also owns the
shared native lock and must be shut down gracefully before any comparison run.

The transparency packet contains8TRAIN/4EVAL public-only examples with external
labels and7 tests. It is explicitly untrained; existing v3 training contains
no non-null budgets, so hard budget compliance is deterministic guard behavior,
not learned budget reasoning. Frontend visible provenance is abbreviated;
complete source/neutral-copy/guard disclosures are in docs and receipts.
No human conversion lift, learned image selection, all-fresh acceptance or
ranking-only full-page latency is claimed.
