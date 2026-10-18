# Actual-store source-conditioned copy v2 preparation

The new offline pilot is frozen and CPU-verified. It corrects the old copy
dataset's body-target imbalance and uses the actual-store ordinary-chat
generation interface. **No native tokenization, generation, fit, model load,
serving hook or UI edit has run in this preparation lane.**

New owned files are `store_copy_data_v2.py`, `test_store_copy_data_v2.py`, and
`store_copy_v2/{cases,fields}.{train,eval}.jsonl` plus `manifest.json`. Old
diagnostics, validators, datasets, checkpoints and results were not edited.
Files remain uncommitted for the parent to review and commit together with the
runner. Existing uncommitted handoff/results were preserved.

## Frozen inputs and coverage

| Quantity | TRAIN | Evaluation |
|---|---:|---:|
| Independently worded public requests | 16 | 8 |
| Complete six-card / two-section fixtures | 16 | 8 |
| Card occurrences | 96 | 48 |
| Occurrences of every actual product | 6 | 3 |
| Fields | 320 | 160 |
| Heading / title / badge / body fields | 32 / 96 / 96 / 96 | 16 / 48 / 48 / 48 |
| Unique title / body targets | 96 / 96 | 48 / 48 |
| Unique badge targets | 66 | 33 |

TRAIN uses two declared colors per product, each three times. Evaluation uses
the third declared color three times. Held-out color index is product index
modulo three, so all three list positions are represented. There are 32 TRAIN
and 16 evaluation product/color states with zero overlap. Every product receives
each of six title and body constructions in TRAIN exactly once; globally each
construction has 16 examples. Each of three badge constructions has 32 TRAIN
and 16 evaluation examples. Eight heading constructions are balanced.

All body targets independently state the exact name, selected color and item
price. All title targets state the exact name and selected color. Source
categories remain supplied metadata, but this bounded contract makes no
category-derived assertions, preventing slogan words from becoming attributes.
All offline teacher fields pass both complete first-three and full-six assembly.
The same 20-field roster and caps are retained: S16 / T24 / B8 / C32,
208 sampled tokens for first-three and 416 for full-six, including native EOS.
Whether every teacher plus EOS fits the actual tokenizer remains for preflight.

The parent explicitly requested fresh factual badges. The v2 contract therefore
uses selected color as typed badge evidence and trains `In {color}`,
`{color} selected`, and `{color} choice`. A supplied color requires a color badge;
`NONE` is allowed only when badge-color evidence is explicitly absent. Every
actual-store fixture supplies a declared color. This policy differs from the
old inference diagnostic's always-`NONE` policy and is versioned independently.

Selections are balanced locked copy fixtures, not learned rank or color choices.
External variant/image action mappings use the selected declared color and
its first declared image. No pixel understanding or learned image selection is
claimed. Budgets are null and requirements empty; this is not a new learned
constraint-compliance test. Stock, sizes and shipping remain unknown.

## Contract and inference boundary

`layout(bound)` returns one ordinary native-chat common user message and 20
independent field-user jobs. It is the sole prompt renderer for both teacher
preparation and prospective native inference. Teacher text, construction IDs,
case/split labels, product/action IDs and private persona metadata do not enter
these messages. Source names/colors/prices are legitimate bound facts.

`assemble(bound, rows, cards=3|6, require_native=True)` validates complete field
membership, per-field kinds, exact native EOS/cap receipts and typed factual
relations, then serializes the original raw values without repair or fallback.
It returns the payload, SHA-256, byte count, raw rows and field-specific failures.
Readiness ends at actual serialized model payload, not browser paint.

The v2 validator recognizes a deliberately bounded relational language after
replacing exact bound spans with typed identity/color/price symbols. It has no
target lookup and emits no wording. It rejects the reproduced slogan failures
`Trail mug is a slow mug.` and `Trail mug is the slow mornings.`, incoherent
`Trail mug $24 and 24.`, unsupported shipping/performance additions, unbound
colors and altered prices. Generic heading grammar rejects `For your you`.
Decimal prices are compared numerically using Decimal, and decimal points are
removed as numeric spans before checking sentence punctuation: $24, $24.0 and
$24.00 represent the same declared integer price. The new gate does not weaken
old validators or change old experiment counts.

This gate intentionally rejects some true English alternatives. Passing it
means bounded typed-relational/protocol acceptance, not arbitrary-English
semantic factuality, human-quality approval or independent wording novelty.
Native values must still be newly generated; the offline `target_values`
constructor is not called by either inference prompt rendering or assembly.
A test replaces the constructor with a failing function and confirms those
inference paths remain independent.

## CPU evidence

Executed successfully:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest experiments.ranker.pretrained.test_store_copy_data_v2
Ran 13 tests; OK
PYTHONDONTWRITEBYTECODE=1 python3 -B -m experiments.ranker.pretrained.store_copy_data_v2 freeze --data experiments/ranker/pretrained/store_copy_v2
PYTHONDONTWRITEBYTECODE=1 python3 -B -m experiments.ranker.pretrained.store_copy_data_v2 check --data experiments/ranker/pretrained/store_copy_v2
```

The tests exercise complete endpoints and native-receipt rejection, action
binding, exact serving/teacher prompt parity, target/private-field exclusion,
all target coverage, numeric formatting, slogan and unsupported-claim rejection,
color badges, immutable freeze/load checks and preservation of existing outputs.
Preparation checks current storefront-source freshness before and after freeze.

CPU fact-swap audit uses only held-out source-bound teacher fields:

| Changed fact | Stale target rejected | Correctly updated target accepted |
|---|---:|---:|
| Product identity | 96 / 96 | 96 / 96 |
| Selected color | 144 / 144 | 144 / 144 |
| Item price | 48 / 48 | 48 / 48 |

These 288 tests establish validator sensitivity and offline source binding.
Changed facts are explicitly synthetic diagnostics and are never passed off as
storefront truth. They do not establish native model source conditioning.

Frozen SHA-256 receipts:

```text
store_copy_data_v2.py
ac424d601d8dc97d480d3c22ed0175de002e46281c7c35f9054f00bb32a8082e
test_store_copy_data_v2.py
130ed9059e12c1dc6d932c40827f3ba1b074b34850629de75f17fc32cd5b40a4
store_copy_v2/manifest.json
0491d1347f493f58a21a48ae762251b1e6056280ac0d97e3d7accce15703ffa2
```

The manifest pins all four JSONL files, input snapshot/mapping/receipt, renderer
source and required store modules. Dataset files occupy approximately 296 KiB.

## Next native evidence and limits

The runner must tokenize every exact message/target and reject, rather than
truncate, any context/suffix/target/EOS cap violation. It must preserve the fixed
schedule and all failed attempts, compare matched frozen/old/new adapters,
verify frozen tensors/checkpoint reload/native cache controls, and measure
valid first-three/full-six serialized endpoints separately. Real model fact-swap
responses remain unmeasured.

The evaluation measures known product identities, held-out product/color bindings
and synthetic independently authored request wording. Color meanings come from
the pretrained backbone and are not claimed as newly learned unseen concepts.
There is no unseen-product, sealed human-quality, hidden-persona/history,
conversion, learned grouping/image or integrated browser-latency claim. No action
was denied, pending approval or left with an uncertain execution outcome.
