# Actual jacket catalog alignment, prospective v1

This is a CPU-only import and model-side adapter design for teammate branch
`origin/feat/design-dash`, pinned to commit
`65b5896b28d4901b95a691fd68436e79ad264e61`. It does not serve a store, run a
model, train an adapter, change a validator or establish jacket-domain quality.
Existing backpack results remain backpack evidence.

## Frozen inputs and boundaries

`model_inputs.json` contains the six source products' exact `name`, `price`, and
multi-value `attributes`, plus ten public tasks' `query`, optional `budget`, and
typed requirements. Requirement labels are public task display text. Product,
persona and requirement IDs stay in `external_mapping.json`; array indices bind
public facts to those external identities. No persona name, bio, traits, goal,
split, history, scripted path, purchase, optimal-step count or outcome enters
model inputs. The preparer reads the two source JSON blobs to select these
allowlisted fields; it does not copy their other contents into the snapshot.

`IMPORT_RECEIPT.json` freezes branch/commit, source paths, exact source byte
SHA-256 and Git blob IDs, preparer hash, and canonical output hashes. Canonical
hashing is UTF-8 JSON with sorted keys, compact separators and unescaped Unicode.
Model-input hash:
`dfe8123ad96da8a86a664961bcf4a706a142feacf8b52c390dfc0742b97cb1e1`.
The preparation script refuses to overwrite any frozen output. Any new source
revision or policy requires another version directory.

## Actual eligible product counts

`guard_receipt.json` checks every task against every product. All `hard`
requirements must match exact membership in the corresponding attribute list.
Preferences do not remove products. An explicit budget is an inclusive item-price
ceiling, matching the branch's evaluator at
`apps/api/internal/evaluator/evaluator.go:58-74`. The free-text word “under” is
not independently parsed into a different bound. Query text does not introduce
additional hard requirements through an unreviewed parser.

| External task identity | Eligible products, source order | Count | Shortfall to three |
| --- | --- | ---: | ---: |
| dev-commuter | ridgeline | 1 | 2 |
| dev-hiker | harbor, coastline | 2 | 1 |
| dev-student | trail, summit | 2 | 1 |
| dev-searcher | ridgeline | 1 | 2 |
| dev-gifter | harbor, ridgeline | 2 | 1 |
| ho-angler | harbor, ridgeline | 2 | 1 |
| ho-cyclist | alpine | 1 | 2 |
| ho-walker | coastline | 1 | 2 |
| ho-hurried | coastline, ridgeline | 2 | 1 |
| ho-parent | ridgeline | 1 | 2 |

These are deterministic factual eligibility results, not learned rankings. No
ordering among eligible products is inferred. Every task has fewer than three
eligible products. The existing fixed three-card above-fold contract cannot
represent this catalog honestly without a prospective variable-card contract.
Never fill a shortfall with a hard-constraint failure, duplicate or fictional
product. These ten known fixture tasks are an alignment set; their inherited
“held-out” names do not make a future experiment an unseen generalization test.

## Domain adapter requirements

Keep `attributes` as a typed map of arrays. A product with sizes `[S,L]` fails
size `M`; do not flatten it to a scalar or infer inventory. Preserve `waterproof`
as its exact source attribute. It is not the old `water-resistant` atom. Neither
direction of synonym substitution is authorized. Stock, shipping, delivery,
material, warmth, breathability and sustainability are unknown. Item-price
eligibility establishes no delivered-price or in-stock guarantee.

The existing `boundary.py:7-8` drops nested attributes, and its variant input
loop at `:52-64` requires `presentationVariants`. The current seven-atom fusion
head uses the backpack vocabulary (`catalog_fusion/semantic_factors.py:15-22`,
with the declared atoms in `paired_data.py`). Passing these jackets through the
old boundary either loses the decisive facts or lacks required variants. A new
versioned product encoder must preserve `name`, item price and all attributes,
and a public task encoder must preserve typed hard/preference distinctions.
Product features must be independent of the task if catalog caching is claimed.
IDs, persona metadata and scripted outcomes remain outside the predictive text.

The source has no image asset, image key, description or variant. The import
therefore declares image selection **unsupported**. Do not manufacture an image
description, single default image variant or style merely to satisfy the old
roster. Product scoring must support product-only rows in a future version;
image-action capability must abstain explicitly. The existing above-fold payload
requires `imageCaptionId`/`imageKey` and must not be reported as satisfied by this
text-only snapshot (`above_fold_copy/layout.py:83-91`).

A future model-side `jacket-text-plan.v1` should bind selected eligible product
indices through the external mapping and carry `requestedCardCount`, actual
card count, explicit shortfall and unsupported image capability. Hard guards
remain independent of model scores. A learned preference score may only rank
the already eligible products, with a declared neutral tie rule. No such learned
score or generated copy is produced by this import.

Generation should use only observable task and complete facts for the selected
eligible products. Titles must retain selected source identity under a new
prospective jacket identity policy. Badges and body claims must cite exact
attributes or item price; no lexical transfer from backpack support lists.
Section headings can state only facts true for every member. Grouping remains
deterministic. Absent evidence should lead to omission or a permitted explicit
NONE value, never unsupported factual completion. Freeze a jacket-specific
factual lexicon, identity rules, raw-value codec and variable-card validator in
new files before evaluating any copy. Legacy validation and results stay intact.

## Current integration gaps on the pinned branch

Remote references below are all relative to the pinned commit, not local main.

| Source and lines | Observed contract or behavior | Consequence |
| --- | --- | --- |
| `packages/contracts/src/index.ts:6-35` | Typed task requirements and optional task on a fixture persona | Suitable public source; bio/traits/goal are separate and excluded here. |
| `packages/contracts/src/index.ts:164-172,199-204` | Run request is fixture IDs plus adapter; catalog response contains experiment/personas/variants | No dynamic public-task/catalog request and no product catalog API. |
| `apps/api/app.go:105-139,141-201` | Existing run routes and strict fixture-ID request, no inference route | No endpoint accepts or returns a model page plan. |
| `apps/api/internal/scenario/scenario.go:13-38` | Product is ID/name/price/attributes; path includes predetermined purchase or reason | Product facts exist, but shopper path outcomes are simulation fixtures. |
| `apps/api/internal/runner/runner.go:52-95` | Mock replays scenario path selected by persona/variant; purchase comes from fixture | Mock completion is not learned personalization or measured buyer behavior. |
| `apps/api/internal/scenario/frame.go:13-14,38-67` | SVG stand-in, placeholder rectangles, optional existing attributes | These frames are not a generated live store or real image selection. |
| `apps/api/queue.go:19-23,137-152` | Eight run workers, two browser slots; mock/browser adapters, deterministic evaluator/optimizer | Parallel shopper sessions do not provide native model batching or an MLX serving hook. |
| `apps/api/internal/optimizer/optimizer.go:13-37` | Development abandonment rule proposes only `showAttributes:true` | This is a factual UI proposal rule, not trained page-copy optimization. |
| `apps/storefront/src/config.ts:10-63` | Only headline/shipping/order parsed; fixed mug/pack/notebook products | The live demo differs from the jacket scenario; new showAttributes is not consumed. |
| `apps/storefront/src/main.tsx:7-15,42-85` | CSS product shapes, fixed body/category/section text, configurable hero headline | No product-image action, generated card title/body/badge/section plan or model response hook. |
| `apps/runner/src/playwright.ts:47-60` | Passes headline/shipping/order and clicks first add-to-cart | Browser execution does not send the task, jacket facts or showAttributes and does not verify jacket requirements. |
| `apps/api/internal/model/model.go:8-22` | Mock string completion adapter | No structured same-backbone inference implementation is wired. |

The recommendation helper at `apps/api/internal/recommendation/recommendation.go:9-33`
sorts by price, history IDs or popularity. It does not implement the typed jacket
requirements, image selection, section planning or generated copy.

## Next bounded model-only experiment

First freeze the new attribute-preserving input representation and variable-card,
image-abstaining output/validator; use this snapshot as a CPU integration fixture.
Publish exact deterministic guards and all shortfalls alongside any learned
scores. A subsequent separately authorized native experiment can compare a
frozen-backbone baseline and a prospectively trained jacket-domain adapter on
new, separately frozen tasks/catalog combinations. The existing backpack adapter
may be an explicitly labeled transfer arm, but it has no established jacket
quality. Do not fit to the ten alignment tasks and call the inherited held-out
names a fresh test.

Any performance result must include actual query encoding, CPU scoring/guards,
fresh copy-evidence prefill and all required generated fields, with warm catalog
cache preparation disclosed separately. A locked-selection copy experiment must
remain labeled copy-stage timing. No native run, token preflight, fit, image
capability, real browser integration or end-to-end result is claimed here.

Validation performed: four stdlib-only CPU tests cover multi-value membership,
strict waterproof/case distinctions, soft preferences/inclusive budget, frozen
canonical hashes, excluded metadata and all ten shortfalls. No existing source,
UI/API/harness file or runtime artifact was edited.
