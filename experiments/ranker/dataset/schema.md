# Dataset schema v1

All timestamps are UTC ISO 8601 strings ending in `Z`. Prices and shipping costs are fictional amounts in the same single currency unit. All records have synthetic provenance; no trait describes a real person.

## `catalog.json`

Object with `schemaVersion: 1`, `synthetic: true`, and `products`. Each product has:

| Field                                    | Type / meaning                                                                        |
| ---------------------------------------- | ------------------------------------------------------------------------------------- |
| `id`, `title`, `category`, `description` | Factual fixture strings                                                               |
| `price`, `shippingCost`                  | Nonnegative numeric amounts; total cost is their sum                                  |
| `inStock`                                | Boolean                                                                               |
| `tags`                                   | Factual attribute strings                                                             |
| `presentationVariants`                   | `{id, title, imageKey, facts: string[]}` alternatives; facts only repeat catalog tags |

## `public/{train,validation,test}.jsonl`

One request object per line. This is the entire merchant-observable record, with linkage metadata kept out of predictive inputs.

| Field                        | Type / meaning                                                                               |
| ---------------------------- | -------------------------------------------------------------------------------------------- |
| `schemaVersion`, `synthetic` | `1`, `true`                                                                                  |
| `recordId`, `userId`         | Opaque stable join identifiers, never model features                                         |
| `asOf`                       | Observation cutoff                                                                           |
| `context.task`               | Shopping goal in plain language                                                              |
| `context.statedBudget`       | Numeric maximum total spend, or `null` when unknown                                          |
| `context.statedRequirements` | Exact catalog tags explicitly required; `[]` means none observed, not no hidden requirements |
| `context.statedPreferences`  | Observed soft preference tags; may be empty                                                  |
| `history`                    | Event objects `{at, type, productId, dwellSeconds?, source, confidence}`                     |
| `history[].type`             | `view`, `save`, or `purchase`; fictional past signal, not future target                      |
| `history[].dwellSeconds`     | Number, explicit `null`, or omitted; do not fill from private truth                          |
| `history[].source`           | `synthetic-first-party-event` or `synthetic-shared-session`                                  |
| `history[].confidence`       | `0.85` or `0.35`; source ambiguity rather than a private-preference signal                   |
| `candidateProductIds`        | Six unique product IDs, all in the task category                                             |
| `missingFields`              | Dot paths documenting unobserved values or empty signals                                     |
| `provenance`                 | `{source: "fictional-generator", generatorVersion, observedThrough}`; audit metadata         |

Every event satisfies `at <= asOf`. Feature builders must independently enforce this cutoff. Unknown budgets remain `null`; unknown requirements/preferences remain empty. No family/split IDs, latent preferences, future actions, relevance labels, or best-product target are present. Event counts are deliberately bounded at three to fit compact candidate contexts.

## `labels/{train,validation}.jsonl`

Separate supervision: `{recordId, relevance: {productId: 0|1|2|3}, bestProductId: string|null}`. Join by identifier **after** selecting permitted features. Test labels are intentionally absent. Grades use hidden utility and hard constraints; the single best feasible product has grade 3. A null best product means none is suitable under hidden truth.

## `private/truth.jsonl`

Independent synthetic grading file with `recordId`, `userId`, `personaFamilyId`, `split`, `synthetic: true`, `relevance`, `bestProductId`, and:

- `hiddenPreferences: {tagWeights: {tag: number}, pricePenalty: 0.006}`.
- `hiddenConstraints: {totalBudget: number, requiredTags: string[], avoidedTags: [], requiresInStock: true}`.
- `observationMode`: `uncertain-twin-a`, `uncertain-twin-b`, `sparse`, `noisy`, `incomplete`, or `established`.
- `counterfactualTwinGroup`: family ID for the ambiguous twin pair, otherwise `null`.
- `labelProvenance`: explicit synthetic hidden-utility description.
- `futureOutcome: {at, intendedChoice: string|null}`: hypothetical action after cutoff, never a model input or a real observed purchase.

Family/user membership is disjoint across splits. Twin pairs stay in the same family/split. Their observable model inputs match exactly while hidden best products differ. Perfect observed-context inference is therefore impossible. Private budget/requirement violations must be reported separately from violations of merchant-known constraints.

## `manifest.json`

Records `schemaVersion`, `synthetic`, `seed`, `generatorVersion`, `familyAssignments`, counts, `sha256` for every data file, `publicFileHashes`, `cutoffPolicy`, `splitPolicy`, provenance and limitations. `trainStageRecordIds` lists 120 initial-fit and 120 adaptation-fit training records from disjoint sets of 20 families each; this supports training stages without reading private truth. These assignments never become features. The manifest contains family/split metadata for auditing; it is never model input. Generation has no time-dependent timestamp, ensuring reproducibility.
