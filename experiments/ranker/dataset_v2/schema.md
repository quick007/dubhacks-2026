# Dataset v2 schema

The layout preserves the v1 public/supervision/private separation, while requests and catalog objects use `schemaVersion: 2`. Every record is synthetic. Monetary amounts share one fictional currency; total cost includes shipping. Timestamps are UTC ISO 8601 strings ending in `Z`.

## Public request

Top-level fields are `schemaVersion`, `synthetic`, `recordId`, `userId`, `asOf`, `context`, `history`, `candidateProductIds`, `presentationCompatibility`, `missingFields`, and `provenance`.

- `context`: `{task: string, statedBudget: number|null, statedRequirements: string[], statedPreferences: string[], presentationSurface: "compact-card"|"detail-panel", presentationPreferences: string[]}`. Tasks and event descriptions use natural language. Requirement and preference strings are catalog tags. Presentation preferences are `detail-view` or `in-use-view`, or empty when unobserved.
- `history`: `[{at, type, productId, dwellSeconds?:number|null, source, confidence, description:string, presentationStyle?:string}]`. All timestamps satisfy `at <= asOf`. A presentation signal is included only when observed. Source is synthetic first-party or low-confidence shared-session activity.
- `candidateProductIds`: exactly four unique product IDs. They link to catalog facts and must never enter predictive text/features as semantic identifiers.
- `presentationCompatibility`: `{productId:{variantId:boolean}}`, derived only from the public surface and variant compatibility rules. It does not expose latent preferences or suitability labels. Apply product budget/stock/requirement guards separately.
- `missingFields`: dot paths for unknown/empty observations. Unknown values remain null, absent or empty, never inferred from private truth.
- `provenance`: `{source:"fictional-generator", generatorVersion}`; audit metadata, never predictive input.

`recordId` and `userId` are linkage only. Family, split, latent attributes, future actions and target grades never appear in public requests.

## Catalog

`catalog.json` contains `{schemaVersion:2, synthetic:true, products:[Product]}`. A product has the v1 fields `id,title,category,description,price,shippingCost,inStock,tags,presentationVariants`.

Each of its two variants has:

| Field                | Meaning                                                              |
| -------------------- | -------------------------------------------------------------------- |
| `id`                 | Opaque linkage key; never a predictive feature                       |
| `title`              | Factual preapproved title alternative                                |
| `imageKey`           | Logical `fictional-descriptor/...` key; no image pixels/file implied |
| `facts`              | Capacity and catalog tags supporting the title/descriptor            |
| `imageDescription`   | Plain-language description of a hypothetical approved image          |
| `presentationStyle`  | `detail-view` or `in-use-view`                                       |
| `compatibleSurfaces` | Detail views allow both surfaces; in-use views allow `detail-panel`  |

## Separate supervision

`labels/train.jsonl` and `labels/validation.jsonl` have:

```text
{
  recordId,
  relevance: {productId: 0|1|2|3},
  bestProductId: string|null,
  presentationScores: {productId: {variantId: 0|1|3}},
  bestPresentationAction: {productId, variantId}|null
}
```

Relevance is 3 for the highest latent utility among suitable products, 2 for the next, 1 for remaining suitable products and 0 for unsuitable ones. Presentation scores are zero when the product is unsuitable or the variant is incompatible; otherwise 3 for the latent preferred style and 1 for the other style. The best action combines the best suitable product with its highest-scoring compatible variant. Product ranking and presentation selection remain distinct targets.

Targets are never prompt/features. There is no `labels/test.jsonl`.

## Private truth and diagnostics

`private/truth.jsonl` includes the supervision fields plus `synthetic`, `userId`, `personaFamilyId`, `split`, `preferenceCombination`, `hiddenPreferences:{tagWeights,pricePenalty}`, `hiddenConstraints:{totalBudget,requiredTags,avoidedTags,requiresInStock}`, `hiddenPresentationPreferences:{preferredStyle,rubric}`, `observationMode` and `futureOutcome:{at,intendedChoice}`. Future actions are hypothetical and after cutoff. Training and feature preparation must not read this file.

`public/diagnostics.jsonl` contains six additional public requests in three controlled preference pairs. `private/diagnostic-truth.jsonl` contains their separate targets and private preferences/constraints. `manifest.diagnosticPairs` maps each pair's two record IDs. These diagnostics are excluded from all training, selection and primary test populations.

## Manifest

`manifest.json` records generator version/seed, counts, family assignments with preference combinations, product assignments, explicit user assignments, `trainStageRecordIds:{initial,adaptation}`, diagnostic pairs, position diagnostics and SHA-256 file hashes. It is metadata, never predictive input.

The family plan assigns 12/3/6 exact combinations to training/validation/test before generating users and SKUs. There are 42 main users and 84 unique SKUs. Each training stage contains six complete families. Best-product positions are stratified; variant positions are randomized and their imbalance is disclosed. IDs and complete product fact sets do not overlap across splits; individual attributes and the shopping domain deliberately do.
