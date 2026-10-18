# Parallel word head: overnight preparation

Status: CPU packet and numerical plumbing only. No native feature extraction,
Qwen training, live-serving change, browser experiment or promotion occurred in
this lane. Existing autoregressive copy and its failed/accepted receipts remain
unchanged.

The candidate keeps the same frozen Qwen3.5-9B backbone for fresh-query ranking
and offline public catalog features. A small trained output head predicts every
word slot for all selected fields in one feedforward operation. It does not
consume previous generated words, teacher strings or output templates at
inference. This is a bounded vocabulary nonautoregressive generator, not a
replacement tiny backbone or selection from a table of complete descriptions.

## Data and output contract

The packet in `store_parallel_text_v1/` contains 48 separate public product-color
feature texts. Each has only name, category, item price and selected color. IDs,
asset paths, private profiles, split labels and target prose do not enter Qwen.
Feature text is `Public product facts: ` followed by the existing canonical JSON
encoding of those four facts. It needs new color-specific features: the existing
16 product-level catalog vectors are not claimed to represent these 48 inputs.

TRAIN has 34 product-color bindings. Evaluation holds out 14 combinations, one
per eligible product; all product identities occur in TRAIN. Sunglasses and
towel retain all three colors in TRAIN. A deterministic split based only on
public word coverage could not hold out a color for those products while
preserving positive supervision for each heldout word. The audit records this
exception. There is no label overlap between heldout and training bindings and
no evaluation word without positive TRAIN supervision. The 107-word lexicon
comes from TRAIN teachers plus public catalog name/color words, never heldout
teacher text.

There are 402 training field rows and 210 evaluation field rows. Three explicit
public style settings control ordinary connective wording. The six-card views
contain both section headings and title/badge/body for every card: 20 fields,
48 training views and 42 evaluation views. These are dependent recombinations
of 34/14 source bindings, not 42 independent unseen shoppers or products.

Teachers use at most nine lexical words/punctuation tokens. Each field has 14
parallel categorical slots including a learned EOS. Missing EOS, PAD before
EOS, wrong selected color, wrong item price and unsupported words fail closed.
No rejected output is repaired from facts or replaced with teacher prose.
Titles retain complete identity/color; short bodies state selected color and
item price. Canonical product names must remain subtitles and canonical prices
must remain UI metadata in any later explicit integration. This short-body
contract intentionally differs from copy-v2's requirement to repeat the
complete name in each body; old scores must not be recomputed under it.

The prospective safety screen checks bounded lexical grounding and ordinary
color/price relations, not arbitrary English truth or comprehensive grammar.
Independent review of accepted predictions remains necessary. Copy is
personalized through the selected public products and explicit style setting;
this packet does not train query-conditioned intent claims or broad prose.

## Model and bounded next native experiment

`store_parallel_text_head_v1.py` provides a CPU-only 4096-to-128 source projection,
public field/style/section condition projection and one 14-by-107 output layer.
The decoder predicts all slots independently, then stops at its first predicted
EOS. Slot disagreements, repetition and memorization are real risks. The
vocabulary cannot express new words for an unregistered catalog item. A new
fact without its exact verified feature key is refused, with no fallback.

Root owns any native window. After the shopper task is idle and serving shuts
down cleanly, the shortest complete experiment is:

1. Verify the frozen backbone, disable the copy adapter, and encode each of the
   48 `public-feature-contract.json` records once, B1, whole input, no truncation.
   Use the current frozen backbone's last-valid-token, float32 unit-L2 output.
   Preserve per-row token/timing/memory receipts and frozen tensor integrity.
2. Build a feature bundle with `make_feature_bundle(records, values, provenance)`.
   The provenance requires native-frozen-qwen, the pinned revision, 48 forwards,
   zero truncations and copyAdapterEnabled=false. It is checked together with
   exact text/key order, matrix shape, unit norms and tensor checksum.
3. Call `fit(packet, bundle, steps=160, width=128)` once with its declared final
   step. TRAIN alone supplies gradients; evaluation occurs only after fitting.
   No native optimizer update or autoregressive decoder is used. Save weights
   under `data/ranker/pretrained/`, not the committed experiment packet.
4. Call `evaluate_cases` on all 42 heldout six-card views. Save raw slot IDs,
   unmodified text, EOS/grounding failures and first-three/full-six field
   acceptance. Review the 14 heldout binding predictions independently. Report
   exact teacher match separately from grounded alternative wording.
5. Measure a real fresh-query ranking call plus this head, all serialization and
   validation contiguously. Also measure browser first-three readiness and full
   six completion if the candidate later earns an explicit integration arm.
   Query encoding, catalog verification, React and images are absent from the
   standalone CPU head timer. A low-hundreds end-to-end result is not established.

This candidate eliminates the separate copy-root prefill and repeated language
model decoding in principle. It cannot eliminate the roughly 200 ms fresh-query
ranking path by itself. Its head-only latency, native feature quality and useful
whole-page speed remain unmeasured. Preserve an accurate cold/offline-feature
cost versus warm-catalog inference distinction.

## CPU verification

Run:

```sh
.tools/pretrained-venv/bin/python -m unittest \
  experiments.ranker.pretrained.test_store_parallel_text_data_v1 \
  experiments.ranker.pretrained.test_store_parallel_text_head_v1
```

Fifteen tests passed, including input/label separation, complete 20-field
rosters, source counterfactual rejection, exact binding/checksum checks, learned
EOS behavior, one-pass generation, caller RNG preservation and a numerical fit
on engineered features. No MLX/model import occurs.

`engineered-fixture-smoke.json` preserves a 24-step, width-32 arithmetic smoke.
Its loss fell from 4.66291 to 2.59325. It achieved zero exact heldout teacher
matches and accepted 18/210 fields under the lexical screen. These deliberately
non-Qwen vectors only verify training and receipt plumbing; neither their loss,
quality nor their runtime establishes native feasibility. The initial smoke is
preserved without claiming a successful generator.
