# One product semantic-factor experiment

This is preparation for a separate learned product scorer. No genuine semantic-factor fit has run. The approved method supersedes the initial linear/product-CE proposal recorded in the independent centered audit: use **sigmoid probabilities with public-attribute BCE only**, with equal query/catalog role weights. Product cross-entropy is a post-freeze evaluation diagnostic, not a training term.

The centered run learned image-style interventions on the synthetic teacher, but no held-out product pair had both choices correct. Its training query projections have 99.107% saturation while retaining nonzero variance. Singleton training preferences and unseen two-tag evaluation preferences also differ. A separate seven-atom factorization tests explicit semantic alignment/composition without a shared rank-four tanh bottleneck. It jointly changes the representation and adds public attribute supervision, so any gain cannot identify either component alone.

Each role has a biased 4096→7 linear decoder followed by sigmoid. The vocabulary is the fixed seven public tags in `paired_data.ATOMS`. There are **57,358 trainable factor parameters**. For each candidate variant, the product score is the dot product of the predicted query and catalog atom probabilities; the existing readout averages variant scores per product and applies the same guards. IDs only select receipt members and are absent from model inputs.

Training uses exactly the already verified **28 TRAIN query vectors and 42 unique TRAIN catalog-variant vectors**, each once per role. Targets are their public `statedPreferences` and product `tags`, respectively. The loss is mean query BCE plus mean catalog BCE; neither role is multiplied by its row count. All seven atoms have positive coverage. The existing centered query/catalog means remain frozen, with statistics SHA-256 `fb00e986642aac467a60286d4ca1a6ceebaef9ac048587341be0810ce621bc53`. No held-out vectors or ranking labels contribute to fitting or means. Even TRAIN ranking labels are opened only after checkpoint freeze for evaluation.

Use one fixed seed 728, 80 full-batch epochs, AdamW learning rate 0.03, weight decay 0.001, one CPU thread and the existing 60-second cap per arm. There are no parameter-search CLI switches. The independent prior fits a fresh catalog decoder using catalog BCE only; its query factor is the empirical TRAIN preference frequency, not the current request. It has 28,679 trainable parameters. Its query decoder parameters are frozen/unused. Seeded random factors use the same initialization as the full model before fitting. Query permutation cycles within each split. As in the existing evaluator, that permutation changes the input to both the new product head and the fixed image head; it does not change image weights.

The existing centered image checkpoint is copied **byte for byte** and validated against SHA-256 `e292b8e1ae0498774f7e1eee8f2a765e42be0743df57dcc45d6afe480682c582`. Its 32,786 parameters are frozen and excluded from the new factor parameter count. Full/prior/random product controls share this same image checkpoint. No new image learning is claimed. Every semantic checkpoint has a distinct format and includes the factor vocabulary, frozen means/statistics, exact TRAIN roster/target receipts, parameter count, source image identity and metadata checksum. Existing checkpoints are never overwritten.

Preparation emits a plan without fitting. Actual execution must match the reviewed plan checksum and exact source/input/statistics snapshot. Full/prior/random product checkpoints and the fixed image copy must validate before any evaluation ranking targets, held-out vectors or pair controls are read. Native catalog/request caches open read-only and validate current content/keys/checksums. Evaluation reports raw and guarded ranks, exact pair both-correct/flip counts, fixed-threshold atom precision/recall and predicted probability matrix rank. A separately labeled public-tag-overlap oracle is a nonlearned diagnostic ceiling, never a serving model.

All features already exist: this method needs **zero native forwards and zero cache writes**. A broad two-tag training expansion with disjoint held-out combinations would be constrained: only compact+laptop-sleeve and lightweight+organizer have full-matching eligible old-train products among the eleven unused pairs. This experiment instead adds atomic public supervision without changing catalog or training membership.

The risk is learning this synthetic seven-atom grammar. Old catalog and validation/test outcomes have already been inspected and replayed. This is neither an untouched test nor evidence of hidden user preference, conversion lift, learned titles/badges/sections, or complete above-fold/full-storefront quality. CPU cached-feature timing excludes native query extraction and copy/page generation. Review source and the frozen plan before running the fixed fit.

Preparation command:

```sh
.tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.catalog_fusion.semantic_runner \
  --prepare-only --out experiments/ranker/pretrained/catalog_fusion/semantic_v1_prepare
```

Review `semantic_v1_prepare/plan.json`, then use its SHA-256 with a new output directory. No fit was run as part of this preparation.
