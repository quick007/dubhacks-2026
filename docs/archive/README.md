# Historical documentation index

Experiment reports retain their original paths and results. Model manifests and
audits reference their exact paths and, in some cases, their bytes. The project
vision and earlier root latency proposal are collected under `docs/`.

## Model experiments and data boundaries

| Material | What it covers |
|---|---|
| [Engineered ranking baseline](../../experiments/ranker/README.md) | Earlier synthetic ranker, datasets and serving scripts |
| [Pretrained experiment README](../../experiments/ranker/pretrained/README.md) | Original frozen-backbone decision/fusion proof; its integration-status statements predate the resident demo |
| [Experiment ledger](../../experiments/ranker/pretrained/MODEL_EXPERIMENT_LEDGER.md) | Recorded experiments and limitations across successive contracts |
| [Model handoff](../../experiments/ranker/pretrained/MODEL_TASK_HANDOFF.md) | Preserved checkpoint; use the current service guide for live commands |
| [Actual-store source boundary](../../experiments/ranker/pretrained/ACTUAL_STORE_MODEL_INPUTS.md) | Public catalog projection and source/action mapping |
| [Focused ranking replay](../../experiments/ranker/pretrained/FOCUSED_RANK_TRANSFER_RESULTS.md) | Inspected synthetic development replay, not actual-store generalization |
| [Failed short-copy audit](../../experiments/ranker/pretrained/SHORT_COPY_FIT_V1_INDEPENDENT_AUDIT_V2.md) | Preserved failed adapter results and strict payload counts |
| [Actual-store copy preparation](../../experiments/ranker/pretrained/STORE_COPY_V2_PREPARATION_AUDIT.md) | Source-conditioned copy data and CPU preparation |
| [Copy qualitative review](../../experiments/ranker/pretrained/ACTUAL_STORE_COPY_V2_QUALITATIVE_AUDIT.md) | Contextual review separated from unchanged strict scores |
| [Earlier latency proposal](COPY_LATENCY_MINIMUM_GATE.md) | Bounded proposal and older workload; not a current speedup claim |

The [resident service guide](../../experiments/ranker/pretrained/STORE_DEMO_SERVICE.md)
and [root README](../../README.md) describe the integrated path. The experiment
directory also retains raw receipts, source freezes, failed attempts and
component-specific audits alongside these entry points.

## Product and presentation plans

- [Project vision](../project/VISION.md): intended product and evaluation direction,
  with implementation snapshots that may precede current code.
- [Case-file demo design](../superpowers/specs/2026-10-17-case-file-demo-design.md)
  and [implementation plan](../superpowers/plans/2026-10-17-case-file-demo.md).
- [Slide-deck design](../superpowers/specs/2026-10-17-slide-deck-design.md).
- [Storefront image library](../../apps/storefront/README.md) and generation
  prompts under `apps/storefront/public/products/`: fixture media provenance,
  not measured visual preferences or shopper labels.
- [Tiger Cloud setup](../tiger-cloud.md): optional historical machine note,
  unrelated to the current SQLite requirement.

Source reports, numerical receipts, checkpoints and instruction files retain their
original locations.
