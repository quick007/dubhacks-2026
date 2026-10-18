# Frozen synthetic copy supervision

`semantic-copy-teacher-v1` is a small audited teacher-grammar dataset, frozen
before any model extraction or training. It is not human-validated production
data, model output, or evidence of improved generation. No model was loaded and
no native execution or training was performed while creating or auditing it.

| Split | Requests | Cards per request | Individual fields | Products | Teacher pages passing strict checks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 12 | 6 | 168 | 72 | 12 |
| Development | 3 | 6 | 42 | 18 | 3 |
| Evaluation only | 3 | 6 | 42 | 18 | 3 |

Every request has two three-card sections and fourteen targets: two headings,
six titles, six badges. Caps remain heading 16, title 24, badge 8, summed 224.
These are intended generation caps, not measured tokenizer lengths. Check the
chosen tokenizer, assistant terminator and training sequence limit separately;
no token-fit or model-quality result follows from this audit.

## Files and training interface

`manifest.json` records generator, six payload-file and initial validator hashes.
`AUDIT.json` is the original machine audit. `generator.py --write` refuses to
overwrite frozen files. `--check` regenerates only in memory, verifies exact
payloads/hashes and re-runs the audit with the current imported validator.
The generator and six JSONL payloads were not changed for the new identity rule.

Consume `fields.train.jsonl` for training and `fields.dev.jsonl` for development.
Reserve `fields.test.jsonl` for final evaluation: no training, prompt/target
selection, early stopping or model optimization against it. Each JSONL record is
one independent generation, with this interface:

```text
schemaVersion: 1
datasetVersion: "semantic-copy-teacher-v1"
split, caseId, personaFamilyId: bookkeeping; never model input
rowId: "heading-1" | "title-1" | "badge-1" ...
field: trusted scheduler label "S1" | "T1" | "B1" ...
kind: "heading" | "title" | "badge"
sectionIndex: 1-based for headings OR cardIndex: 1-based for title/badge
maxTokens: 16 | 24 | 8
messages: [{role: "user", content: shared public instruction/evidence},
           {role: "user", content: one scheduler-owned field request}]
messagesSha256: hash of canonical messages JSON
targetValue: assistant supervision, one raw value, no S/T/B label
targetSource: "audited synthetic teacher grammar"
inferenceMetadataIsNotModelInput: true
```

Only `messages` are the model input. Apply the chosen tokenizer's actual chat
template to the two messages and train against the assistant `targetValue` with
the proper terminator; do not serialize the entire field record into a prompt.
The two adjacent user messages are intentional, matching the shared evidence
then independent scheduler request. Preserve their content when measuring a
training/inference comparison. Loss-mask prompt tokens as appropriate to the
training implementation; target values are never available at inference.

`cases.{split}.jsonl` supports page evaluation. Each case has `request`, `catalog`,
`pendingPagePlan`, `publicEvidence`, and fourteen `targets`. `request` uses the
existing observable request schema. `catalog` maps product IDs to public product
records and two text-only image variants. `pendingPagePlan` fixes all six
products, images and grouping before copy generation; no generated page is
stored. `targets` repeat the trusted field metadata and value. Other keys are
`schemaVersion`, `datasetVersion`, `synthetic`, `split`, `caseId`,
`personaFamilyId`, `personaFamilySignature`, `selectionSource`, `targetSource`.

For page evaluation, bind each raw generated value to its trusted field and
unchanged caps, preserve native stop reasons, then use the existing scheduler
assembly and strict validator. Teacher targets here were combined into canonical
`S/T/B: value` lines with a synthetic `stop` for grammar validation only. That
does not establish native stopping, actual output tokens or usable pages.

## Coverage and leakage audit

Products, image IDs, fictional name anchors, persona-family signatures, full
observed-preference pairs and complete product-tag triples are disjoint across
splits. All 108 product triples and names are globally unique. This preserves
held-out products and combinations while individual vocabulary attributes and
ordinary framing remain shared. Families are defined only by explicit public
preferences and presentation style, not hidden persona truths.

There are no latent labels/history in model input. Public messages contain
explicit requests, selected names, actual tags, approved titles, observed support,
section membership and universal tags. Opaque IDs, family IDs, split and target
metadata stay outside `messages`. No exact non-`NONE` target string appears in
its own input. No normalized non-`NONE` target is shared across splits. `NONE`
intentionally repeats as policy syntax.

The three case constructions in each split cover a supported common section
paired with an empty intersection, support/NONE mixed in both empty-intersection
sections, and positive universal sections with supported versus unobserved
common attributes. Each case has three supported non-`NONE` badges and three
required `NONE` badges. Overall there are 54 supported badges, 54 required
`NONE`, 18 headings with universal attributes and 18 with empty intersections.
All 108 titles preserve their complete selected name, capacity and product type,
are six words, differ from approved titles, and mention observed support when
present. Headings satisfy 3–5 words; supported badges satisfy 2–5 words.

One training-only example is `Reniven 11L Daypack for mesh travel`, grounded in
the selected `Reniven 11L Daypack` and explicit observed `mesh` support. An
unsupported card receives actual `NONE`, not an invented substitute badge.
This grammar is deliberately narrow; its targets are supervision, never a
fallback inventory of inference answers.

Limitations: the fictional names are syllabic; all products are daypacks; there
is no history; image choice/order is a frozen fixture with no ranking oracle.
The two preference tags are public, but only the first intersects catalog tags
in this version. That position shortcut is a coverage limitation. Supported
`NONE` is legal but is not a teacher target; wrong names, malformed protocols,
truncation, adversarial catalogs and real user language are not represented as
negative training examples. There are only three dev and three test pages and
no human judgments. A teacher-page pass cannot validate learned ranking, image
quality, factual reasoning beyond the lexical policy, or production performance.

## New identity policy and saved-output replay

`identity-policy-audit.json` and `identity_audit.py` record a separate CPU audit
against `public-name-or-unique-mentioned-facts-v1`, including exact current
source hashes. All 72/18/18 train/dev/test teacher titles pass identity; all
12/3/3 teacher pages still pass the full current validator. This supplements the
original frozen audit; it does not rewrite its initial validator hash.

The same audit reassembled all six saved scheduler experiment arms from
`SCHEDULER_IDENTITY_RESULTS.json`. It did not regenerate, retry or edit outputs.
All six pages were already rejected and remain rejected. Accepted title counts
for diagnostic/generated-labels stay 3, diagnostic/scheduler stay 5,
NONE-control/generated-labels stay 1, explicit-preferences/generated-labels stay
0, explicit-preferences/scheduler stay 2. NONE-control/scheduler falls 6 to 5.
Its actual `T6` value, `Meet the compact everyday travel choice`, has no selected
`Aster 18L Daypack` identity and now receives
`missing_selected_product_identity`. The name-bearing
`Flint water-resistant packable daypack` remains eligible under the new name
anchor contract, despite omitting capacity; teacher targets follow the stronger
complete-identity instruction. Other factuality, novelty, support, protocol and
global finish checks still apply separately.

Reproduce with the repository CPU environment:

```sh
PYTHONDONTWRITEBYTECODE=1 .tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.semantic_copy_data.generator --check
PYTHONDONTWRITEBYTECODE=1 .tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.semantic_copy_data.identity_audit
```

The replay script writes only its own supplemental JSON in this package. It
checks that audited source and saved-result hashes stay unchanged during replay.
