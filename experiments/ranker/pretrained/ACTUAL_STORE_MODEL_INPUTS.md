# Current-store model candidates

The actual storefront merged from teammate main is distinct from both the old
backpack experiments and six-jacket API scenario. `store_catalog_v1.py` captures
its reviewed pure runtime exports using already-installed Node+tsx, then pins
source blobs, canonical outputs and all local assets. No dependency was installed
and no UI/API/shopper file was edited.

The frozen source commit is e172f605. The import contains16products,48color
variants,65base image records,195color-gallery action records,32optional color
previews and81unique asset files. Model input canonical SHA-256 is
205572ed535b6f40cd56ede51208c348ad05998ad4d205f193a8b5bfe3e79fb7.
`store_catalog_v1/IMPORT_RECEIPT.json` records exact source/runtime/asset evidence.

`catalog_text` retains declared product names, category, item price, search terms,
color and gallery label/alt/context/presentation metadata. It excludes action IDs,
asset URLs, swatches, crop coordinates and synthetic scenario IDs. All routing
metadata remains in a separate index-bound mapping. These image descriptions
are fixture declarations; no pixels were fed to a model or visually verified.
Gallery actions preserve product/color/image identity plus asset, position and
tile metadata. Preview images are separate from gallery actions.

## Active-model freshness gate

The frozen loader authenticates a historical snapshot. To prepare candidates for
the current store, use the additional fail-stale bridge:

```sh
PYTHONDONTWRITEBYTECODE=1 .tools/pretrained-venv/bin/python -B -m experiments.ranker.pretrained.store_model_bridge --out /tmp/current-store-model-new-output
```

The output directory must not exist. `store_model_bridge.py` loads the pinned
snapshot, verifies current bytes of all15selected sources before and after text
preparation, and returns195candidate rows with exact external action bindings.
A changed product/price/color/gallery/input source fails rather than silently
reusing the archived snapshot. Asset byte/Git-blob checks remain active as well.
The first actual preparation is preserved at `store_model_results/v1/report.json`.
No score, image choice, generated copy or native call is fabricated by preparation.

A real feature cache must additionally bind the active bridge/representation,
model weights, tokenizer, runtime/batch shape and any head checkpoint hashes.
The CPU receipt does not invent those unexecuted cache keys or predictions.

## Known factual boundaries

Stock and sizes are unknown. Item price is not delivered price. Shipping is a
cart-subtotal rule: visible quotes charge5at50, become free strictly above50,
and hidden quotes remain omitted. A hidden displayed subtotal establishes no
actual charged shipping fact. Category/search words are public declarations,
not independently verified material/performance attributes.

The ten public API tasks are kept separately with explicit compatibility-not-
established provenance; they include medium/large/small requirements unsupported
by current-store size evidence. They are not silently rebound to new products.
Illustrative shopper-persona goals/candidate briefs are excluded; no hidden bio,
traits, synthetic split, purchase paths or outcomes enter predictive text.

Validation:14importer CPU tests plus3active-source/action-binding tests pass.
An independent audit reproduces the permanent snapshot, all195encodings and81
asset checks. This corrects model-input/catalog alignment and provides a tested
model-side entrypoint. It does not wire a serving endpoint or demonstrate
learned ranking, color/image choice, generated-copy quality or browser latency.
The successful seven-atom v2 head is a separate synthetic backpack component;
its vocabulary and perfect score do not certify this diverse16-product catalog.
