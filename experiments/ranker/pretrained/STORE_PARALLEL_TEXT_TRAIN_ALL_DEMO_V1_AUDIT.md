# All-catalog parallel-text demo: independent quality audit

The separate **320-step TRAIN-ALL candidate passes the bounded known-catalog copy checks**. All528 archived training outputs are factual and readable:144 titles,144 badges,144 short bodies and96 generic headings. All32 previously selected six-card sets × three styles produce20 usable fields: **96/96 CPU replay views,1920/1920 fields**. This is a curated catalog demo with all48 product/color bindings supervised; it has **zero heldout bindings**.

Candidate: `data/ranker/pretrained/store-parallel-text-train-all-demo-v1-attempt-2/`. Verified head SHA-256: `074169e31bf5f6b99cc5bc35c9b2e64d82396d5225b3a67cc2cf5f613fcc4889`. The [JSON evidence](STORE_PARALLEL_TEXT_TRAIN_ALL_DEMO_V1_AUDIT.json) records every training prediction, all96 view receipts/hashes, export checks, counterexamples and source provenance. The original [v1 research failure](STORE_PARALLEL_TEXT_V1_NATIVE_AUDIT.md) and the160-step demo's two imperfect bodies remain preserved.

## Facts and short-caption utility

All144 titles bind the correct original product and selected color. All144 badges state the selected color. All144 bodies state that color and the exact public item price. No unsupported benefit, changed material, wrong fact, meaningful repetition or incoherence was found. Canonical names/prices/actions remain metadata, so short bodies need not repeat identity or satisfy the old long-copy grammar.

For the same Sage mug at$24, the three explicit styles produce:

| Style | Title | Badge | Body |
| --- | --- | --- | --- |
| plain | Trail mug in Sage | Sage selected | Shown in Sage at $24.00. |
| browse | Sage Trail mug | In Sage | Color: Sage. Item: $24.00. |
| compact | Trail mug, Sage | Sage displayed | Sage option at $24.00. |

Each of the144 `(kind, binding)` groups has three distinct style outputs while retaining its facts. These are useful concise captions and consistent card formatting. Punctuation fragments such as “Color: Sage. Item: $24.00.” are acceptable here; forcing complete long sentences would not add factual utility. The repertoire remains narrow, neutral and deterministic for the same facts/style. It does not train shopper-intent prose or open-ended phrase variation.

The160-step predecessor's failed bodies were “Color: Slate blue Item Item: $68.00.” and “Slate blue option $64.00 $64.00.” Their facts were compatible but their repetition was unsuitable. The320-step checkpoint resolves both. Increasing steps was justified by those supervised TRAIN failures, not by selecting against heldout quality. Neither result overwrites the original14-binding heldout experiment.

## Independent inference and export check

I loaded the saved head and verified native feature bundle **on CPU**, then generated from the public export alone. Teacher construction and data building were patched to raise; label-path reads through `Path.read_text` were blocked during generation. No teacher call or label-read attempt occurred. All432 registered card fields and all1920 prior-view fields reproduced the archived slot IDs exactly. This checks actual neural computation, beyond trusting a `runtimeTeacherLookup:false` flag. No fitting, Qwen call or GPU execution occurred in this audit.

The inspected head applies learned feature/condition projections and independent word-slot classifiers, then stops at its first learned EOS. Detokenization only joins whitespace/punctuation. Exact training-phrase recovery does not make this a runtime template lookup. It is still a107-word, known-catalog neural generator trained on three offline teacher families; these results do not establish open prose or generalization.

The serving export has48 exact public source records containing only name, category, item price and selected color, plus hashes and feature-bundle metadata. Offline label and replay files contain training references but are not serving inputs. I verified the head, public-export and feature-bundle hashes. The strict serving-row projection rejects injected teacher and hidden-persona fields.

## Fact-change invalidation

Actual CPU adapter/gate counterexamples passed:

- Changing Sage to Ivory or$24 to$25 while keeping the old source text/key is rejected.
- Recomputing a new$25 source key without its exact extracted feature is rejected with no fallback synthesis.
- A complete registered Ivory source record/key is accepted; all registered variant outputs were audited above.
- The old “Shown in Sage at $24.00.” caption is rejected against changed Ivory or$25 facts.

Adapter-only negative checks used explicit zero-vector shape fixtures; they are not feature-quality evidence. Actual neural reproduction separately used the verified native cached vectors. Unknown-price rejection proves invalidation, not learned price extrapolation. New prices/products need a new verified source binding and appropriate quality evidence.

## Scope for a demo integration decision

The copy-quality prerequisite passes for this registered, fully supervised catalog. The96 replay views reuse old selected cards and measure **no fresh query encoding, ranking, browser readiness or conversion**. They cannot inherit the earlier197 ms attempt median. Ranking misses in the old card sets are not repaired by better copy. A live candidate still needs its own measured fresh-query/browser receipts and exact source freshness checks. This audit did not promote the head or change serving, gates, historical metrics or the shopper evaluation packet.
