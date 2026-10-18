# Frozen paired copy supervision proposal

Prepared data only, version `semantic-copy-pairs-v1`, seed 19075. No model,
tokenizer, native inference, fitting or checkpoint selection ran in this lane.
`manifest.json` pins the generator, inputs, payloads and current validators.
The package is frozen before the next native attempt. Teacher answers are
synthetic supervision, never runtime fallback text.

The proposed later fit is one fixed epoch of 504 individual field updates, final
step selection, with all language weights and decision heads frozen except the
already scoped adapter. That is a proposal recorded in the manifest, not an
executed run. The owning training lane must verify exact prompt/token alignment,
target-plus-EOS cap fit and complete context fit before any model update. Keep
the existing field caps and strict validators unchanged; no target trimming.

Training has 18 paired families × two observable singleton preferences × 14
fields. Each pair reuses the same original six-product training catalog, image
IDs, product order and grouping. At least one shared product supports both
preference atoms, and its title and badge targets must change when the observed
preference changes. Only request task and stated preference change in model
evidence; IDs distinguish records outside the rendered public evidence. Every
attribute atom has positive title and badge examples, including compact,
foldable and quilted, which had no positive targets in the first fit.

Fresh evaluation consists of three development and three test requests, each
with six new products. Public preference sets have two tags, distinct from
training singletons and all two-tag sets in the hashed inspected inventory.
Both preference tags receive positive supported targets on the page; every page
also requires two `NONE` badges. Name anchors, IDs, image IDs, full product-tag
triples and full preference sets are disjoint from training and the inspected
inventory, and disjoint across fresh development/test. Each fresh split contains
three headings with a positive universal intersection and three with an empty
intersection. Selection and image quality have no oracle labels.

For a later fixed frozen-versus-learned comparison, retain three cases per fresh
split and report whole-page acceptance, generated/accepted `NONE` separately,
supported-title raw mentions versus accepted alignment, universal headings,
identity errors, exact truncation rows, full-completion time, native memory and
valid-only latency. Keep all cases and the strict global truncation rule. The
new data are not evidence that these metrics improve.

Old development/test fixtures are already inspected. `inspected_replays.json`
pins their original files and IDs for an optional explicitly labeled replay;
never merge those denominators with fresh cases or call them new holdouts.
Do not use fresh test targets for fitting, prompt tuning, early stopping or
checkpoint selection. Once inspected for a future run, that test is diagnostic
evidence and must not be relabeled as fresh for a later attempt.
