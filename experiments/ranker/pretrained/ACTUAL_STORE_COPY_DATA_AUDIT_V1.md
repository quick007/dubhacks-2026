# Source-conditioned copy data audit

CPU/read-only audit, 2026-10-18. No model load, native generation, training,
checkpoint change, runtime edit, UI edit or commit was performed. Existing
outputs and failed checkpoints remain intact. Repository `AGENTS.md` was read;
there is no `.agents` directory in this checkout and no nested experiments
`AGENTS.md` was found.

The next useful copy experiment should use **actual-store bound facts and the
exact prospective serving prompt**, with balanced source-conditioned targets.
The failed daypack adapter does not establish a backbone-capacity limit. Its
body targets are narrow, imbalanced, and unlike the current-store contract.
Keep a frozen-backbone baseline and the failed adapter as separate controls;
do not promote faster invalid outputs.

## Independently reproduced old target coverage

Inputs: `short_copy_data.py`, the frozen `short_copy_v1` cases/fields, strict
copy layout/validators, and `SHORT_COPY_FIT_V1_INDEPENDENT_AUDIT_V2.json/.md`.
The following counts were recomputed from JSONL files with standard-library
Python, without a tokenizer or model import.

| TRAIN quantity | Count |
|---|---:|
| Requests / fields / body targets | 12 / 240 / 72 |
| Unique body strings | 16 |
| Compact body target | 12 |
| Compressible body target | 13 |
| Cards without any supported observed preference | 40 / 72 |
| Distinct tags present in supported-observation evidence | 18 |
| Distinct tags actually chosen as the body target | 16 |
| Body targets starting with grammatically incorrect `A adjustable`, `A expandable`, or `A organizer` | 13 |

All body targets follow `A {tag} daypack for everyday travel.`. `padded` and
`rolltop` occur in supported evidence but never as the chosen TRAIN body target;
they are chosen in evaluation bodies. This is not a validator failure or corrupt
teacher label: teachers satisfy the old strict rules. It is a coverage gap in
the actual supervised output distribution. The manifest's all-18 evidence
coverage must not be read as all-18 C-target coverage.

The generator picks `support[0]`, otherwise `productFacts.tags[0]`. All 40
unsupported-observation cards use the latter path. Product tags are alphabetic
lists, creating a systematic default toward earlier tag names. Compact and
compressible alone account for 25/72 body labels (34.7%). A single pass over
240 mixed S/T/B/C fields contains only 72 C examples. It does not isolate whether
data balancing, additional C exposure, adapter scope, or capacity is limiting.

The existing V2 audit is consistent with a source-conditioning failure: one
new-C development output repeats `A compact daypack for everyday travel.` in
C1 through C5; selected C1/C2 products lack compact and C1/C2/C3 miss their
independently required observed support. The new-C all-body-check acceptance is
13/36 at each width, not the looser lexical 20/36. All 24 first-three and all
24 full-six payload attempts remain invalid. No usable-result latency exists.

## Current-store source and validator findings

`store_copy_diagnostic.layout()` builds an ordinary native chat root and binds
only name/category/itemPrice/selectedColor per card. The old fit used a raw
focused prefix, synthetic daypack identities/tags, support-dependent badges,
and bodies without product identities. The new contract instead requires the
complete product name in each body, always-NONE badges and generic headings.
This changes domain, available facts, target behavior and prefix together.
An inference-only result is a transfer diagnostic, not a test of in-distribution
copy adaptation. Store function-head training is ranking supervision and adds
no copy targets.

The first three source categories are slogans (`For the slow mornings`, `Pack a
smaller plan`, `Keep a good thought`). Their component tokens are not independent
product attributes. The remaining declared categories supply narrow facts such
as `Waterproof hooded shell` and `Cotton canvas carryall`; they do not establish
comfort, capacities, sizes, availability or shipping.

The gate explicitly disclaims comprehensive semantic proof, correctly rejects
unbound colors and many invented words, and enforces full-field/native receipts.
However, its allowed-word-set check permits unsupported relationships among
allowed tokens and rejects some factual equivalent formatting. Reproduced with
`bind_case` on actual products 0..5 and `field_rejections('C1', text, bound)`:

| Text for actual Trail mug / Sage / $24 | Gate result |
|---|---|
| `This Trail mug is in Sage for $24.` | Accepted |
| `Trail mug is a slow mug.` | Accepted; slogan token becomes an attribute |
| `Trail mug is the slow mornings.` | Accepted; incoherent source-token recombination |
| `Trail mug $24 and 24.` | Accepted; no coherent sentence/price relationship |
| `This Trail mug is $24.00.` | Rejected: unsupported `24.00` and multiple-sentence check |

The decimal rejection comes from string-token equality against integer `24`
and `text.count('.') > 1`, which counts the decimal point as punctuation.
`For your you` and `Select your selected` also pass the generic heading gate.
These are independent prospective validator gaps, not a reason to reinterpret
the old strict experiment as successful. Preserve the old validator/results.

The written rule asks for the exact product name, but the gate normalizes case
and punctuation for identity checks. State whether normalized identity or exact
spelling is intended before freezing a new contract. The current gate checks
lexical framing beyond a source value, not meaningful wording novelty.

## Recommended bounded successor

1. Run the prepared current-store baseline only as a bounded transfer diagnostic
   after its source/native controls pass. Preserve every raw value, failed field
   and six-case denominator. A frozen baseline and failed-adapter arm should use
   identical ordinary-chat inputs, field caps, selections and width. Copy-stage
   latency must remain separate from a contiguous ranking-plus-copy endpoint.
2. Freeze a new versioned actual-store copy dataset with the exact serving
   messages and native tokenization. A small complete pilot is 16 TRAIN requests
   (320 fields, 96 cards) and eight independent evaluation requests (160 fields,
   48 cards). Balance each product at six TRAIN / three EVAL card occurrences.
   TRAIN uses two declared colors per product, three times each; EVAL holds out
   the third declared color for every product. Rotate held-out colors by product
   so the split is not simply the first/last list position. Keep six cards/two
   sections; do not pad shortfalls or replace native copy with target strings.
3. Author several concise factual target constructions per product, rather than
   one overwhelmingly repeated body string. Require product identity plus one
   bound relational fact, usually selected color; category assertions must retain
   their supported relationship. Use correct articles and natural headings.
   Name/category/color/price must stay independently bound. Teacher examples may
   contain reusable constructions, but they are supervision only, never a serving
   template or a reference answer inserted into the inference prompt.
4. Audit target—not merely source-evidence—coverage before fitting: product,
   selected color, field kind, claim type, construction, length/EOS cap and
   support opportunity. Validate all targets with the new frozen contract and
   independently inspect unsupported claims, grammar, repeated-source collapse,
   and semantic equivalence. Keep target/rationale/case/split metadata external.
   Public sparse requests may affect ranking/framing; hidden personas never
   enter model input, labels masquerading as observed evidence, or copy facts.
5. Add a separately labeled synthetic counterfactual audit before promotion:
   swap only selectedColor, then only itemPrice or product identity, while holding
   request wording fixed. The native output must follow the changed bound fact;
   an unchanged generic body receives a factual-coverage failure, even if every
   word is allowed. Fabricated prices/names are synthetic diagnostic cases and
   must not be reported as actual storefront facts or served to shoppers.
6. Fix/version the prospective gate's decimal handling and slogan-as-attribute
   loophole; add relational claim checks or an independent semantic audit.
   A lexical-only result should continue to be labeled lexical/protocol-valid.
   Do not loosen old validators to recover old payload counts. With the model
   still required to generate every fresh field, count field-specific failures,
   complete valid first-three/full-six payloads, duplicate outputs, fact-swap
   response, EOS/caps, actual sampled tokens and serialized request readiness.
7. Use one fixed training schedule and final checkpoint, with evaluation closed
   until fitting finishes. Compare the frozen base, immutable failed adapter and
   new source-conditioned adapter. Start from the frozen base for the primary
   new arm so old daypack collapse is not silently inherited. A warm-start arm is
   a distinct optional experiment. Broaden adapter placement or increase capacity
   only after target coverage and source-conditioning failures are quantified.

This pilot tests known product identities with held-out color bindings and
independent request wording. It does not test unseen products, unseen color
meanings, hidden-history inference, learned image selection or human conversion.
Complete UI readiness additionally requires actual integration/render evidence;
serialized model payload readiness is the honest intermediate endpoint.

## Remaining constraints

No native current-store copy output was available in the inspected files, so
the proposed data/prompt changes have no measured gain yet. Current-store
runtime action IDs/crops remain externally bound and first-declared color/image
selection remains deterministic and untrained. No source edits or denied actions
occurred in this audit lane.
