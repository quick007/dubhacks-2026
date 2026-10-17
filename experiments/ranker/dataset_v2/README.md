# Small product-and-presentation fixture v2

This independent synthetic fixture supports a bounded pretrained-backbone/parallel-decision experiment. It preserves v1 unchanged. It contains fictional observed customer context, richer private preference truth, factual product alternatives and separate supervision. It does not contain real customers, sensitive targeting traits, actual image pixels or real conversion outcomes.

```sh
python3 -m experiments.ranker.dataset_v2.generate --output data/ranker/datasets/v2 --seed 2026
python3 -m unittest experiments.ranker.dataset_v2.test_generate -v
```

Python 3.10+ and the standard library are sufficient. Generated output stays in ignored `data/ranker/datasets/v2/`.

## Populations and boundaries

| Population | Cases | Families | Unique candidate SKUs |
| ---------- | ----: | -------: | --------------------: |
| Train      |    24 |       12 |                    48 |
| Validation |     6 |        3 |                    12 |
| Test       |    12 |        6 |                    24 |

Each family has two distinct fictional users: one clearer request/history and one sparse or noisy observation. Every request has four products with two presentation variants each. Training is further divided into six initial-fit and six adaptation-fit families, 12 records per stage. `manifest.trainStageRecordIds` permits that partition without reading private truth.

Families and exact triples of product focus, intended use and presentation preference are assigned to splits before generating products or users. User identities, SKU identities, complete product fact sets and those preference triples do not overlap across splits. Familiar individual attributes intentionally appear across splits: this tests unseen combinations and SKUs, not generalization to an entirely unknown shopping domain. The 42 cases and one backpack category are a workflow proof, not a representative benchmark.

`public/*.jsonl` contains only observable requests; `labels/train.jsonl` and `labels/validation.jsonl` contain separate targets. There is no public test-label file. `private/truth.jsonl` carries latent preferences, hard constraints, presentation rubric, future hypothetical actions and test targets. Neither training nor feature extraction may read private files. Local directories are a workflow boundary, not an operating-system access-control mechanism.

Predictive inputs may use context, cutoff-valid natural-language event descriptions and factual candidate/variant content. They must exclude IDs, provenance, manifest metadata, split/family assignments, input row order and target fields. IDs are opaque hashes used only for joining; prices use independent random cents, never construction ordinals. Candidate order is randomly permuted with stratified best-product positions: 6 per position in training and 3 per position in test. Catalog, row and variant order are also seeded/shuffled. Variant target positions are **not balanced**: the manifest discloses counts so evaluation can include a position-only baseline and an order-swap check.

## Presentation is a separate synthetic decision

A product's approved variants pair a factual title with either a close-up/detail descriptor or an in-use descriptor. `imageKey` names a hypothetical fixture descriptor, not an image file. A compact card permits only detail views; a detail panel permits both. These compatibility rules are public and independent of latent preferences.

Product relevance comes from latent tag utility, total price, stock and requirements. Presentation scores separately compare compatible text descriptors against an invented latent preferred style. Scores are 3 for that preferred style, 1 for another compatible style and 0 for incompatible or unsuitable choices. Private labels cannot prove that a real image would produce the predicted response. Free-form titles, image synthesis and merchant-page changes are outside this fixture.

The dataset includes hidden budgets/requirements that are not always observed, low-confidence shared-session views, null and omitted dwell times, new users, known over-budget candidates, unavailable candidates and requirement violations. At least one candidate remains suitable in each main case. Missing observations are never filled from private truth.

## Counterfactual diagnostic set

`public/diagnostics.jsonl` contains six additional requests in three pairs. Each pair fixes catalog, candidate order, history, cutoff and presentation conditions while changing explicit product preferences; its private best product changes. Targets live only in `private/diagnostic-truth.jsonl`. These are separate intervention checks, excluded from training, model selection and final test metrics. They reuse selected held-out catalogs and are not additional independent evidence of generalization. Pair IDs are listed in the manifest for auditing.

The generator records seed, version, file hashes, user/family/product assignments, stage IDs and position diagnostics. Generation is byte-for-byte reproducible without network access or the system clock. See [schema.md](schema.md) for exact extension fields.
