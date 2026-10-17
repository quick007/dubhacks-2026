# Synthetic ranking fixture

These are fictional customer records and synthetic utility labels. They do not contain real people, personal identifiers, protected traits, or observed conversion outcomes. This fixture establishes a reproducible training and grading workflow; it cannot establish human preference accuracy or revenue lift.

Run from the repository root with Python 3.10 or later; no Python packages are required:

```sh
python3 -m experiments.ranker.dataset.generate --output data/ranker/datasets/v1 --seed 2026
python3 -m unittest experiments.ranker.dataset.test_generate
```

The output is ignored local data. The generator owns only that dataset directory. Its immutable generator version, seed, per-file SHA-256 hashes, and exact family assignments are recorded in `manifest.json`. Regenerating with the same seed is byte-for-byte deterministic; no system clock or external service enters generation.

The fixture contains 60 families with six records each, split **before** example generation into 40 training, 10 validation, and 10 held-out test families. Each user appears once and belongs to exactly one family. Family identity is metadata rather than a model feature. `manifest.trainStageRecordIds` also partitions the training families into 20 initial-fit and 20 adaptation-fit families, with 120 records each, so staged training never needs to read private truth. The product catalog contains 18 independent fixture products: six each for backpacks, headphones, and desk lamps. Each request includes the six candidates in its category. This fixture does not change the demo shop catalog or design.

Each family contains two cold-start users with identical observable context and different hidden best products, plus sparse, noisy, incomplete, and more established histories. Lower-confidence shared-session views can disagree with the user's hidden preferences. Histories include explicit nulls and omitted dwell fields. Budgets and requirements may be absent from observable context even when hidden truth contains them. Known budget includes shipping. Some requests have no suitable item and require abstention.

## Read boundaries

- **Model inputs:** public request context, cutoff-valid history, and factual candidate catalog fields only. Never use `recordId`, `userId`, file location, row order, split, provenance, or family identifiers as predictive features.
- **Training targets:** join `labels/train.jsonl` by `recordId` for supervision; labels are not prompt or feature fields. Validation labels may select bounded hyperparameters. There is intentionally no public test-label file.
- **Independent grading:** `private/truth.jsonl` only. This file contains hidden preferences, constraints, future hypothetical actions, and all split labels. Training and prompt building must never read it. Local directories are a workflow boundary, not access-control enforcement; code review and tests must preserve that separation.
- **Presentation:** catalog variants are preapproved factual alternatives. Their `imageKey` values are logical fixture references, not generated images or existing storefront assets. This fixture does not provide presentation-action supervision or claim to train a presentation policy.

Hidden utility weights catalog tags, subtracts a small price penalty, and excludes unavailable items and hidden budget/requirement violations. Grades are 3 for the best feasible item, 2 for the next two, 1 for other feasible items, and 0 for unsuitable items. Labels represent this artificial objective, not purchases. The simulator's clean tag ontology and limited catalog make the task substantially easier than real ecommerce; missing observations and conflicting twins prevent perfect inference even within the fixture.

See [schema.md](schema.md) for exact fields and the parent [CONTRACT.md](../CONTRACT.md) for the shared model boundary.
