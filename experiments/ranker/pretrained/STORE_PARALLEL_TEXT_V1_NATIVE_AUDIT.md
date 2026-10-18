# Parallel-text v1: independent native quality audit

The run demonstrates **one complete factual six-card/20-field backend result at202.09 ms**, while **31/32 frontier requests lack at least one usable generated field**. The successful case uses six TRAIN-seen product/color bindings. Heldout color composition fails: **0/126 heldout card fields are useful**, beyond a mere wording mismatch. This supports a promising speed mechanism with a substantial variant-grounding failure, not broad generalization or a ready storefront.

This exhaustive offline model judgment covers **all640 frontier raw fields**, their accepted/null serialized values, and **all210 heldout label rows**. I also verified all840 heldout-view predictions project consistently onto those210 rows. The [JSON audit](STORE_PARALLEL_TEXT_V1_NATIVE_AUDIT.json) preserves every raw text, bound public fact, slot sequence, judgment and source hash. This is not a human evaluation or a browser-paint measurement. No GPU, new fit, serving change, gate change or historical regrading occurred.

## Frontier: the gates withheld real color errors

| Field kind | Fields | Factual, readable and accepted | Withheld |
| --- | --- | --- | --- |
| Generic section | 64 | 64 | 0 |
| Title | 192 | 126 | 66 |
| Badge | 192 | 126 | 66 |
| Short body | 192 | 126 | 66 |
| Total | 640 | **442** | **198** |

All198 withheld fields describe the wrong selected color. They occur on66 card appearances: Sage mug11, Olive-band lunch box16, Navy headphones12, Forest jacket16 and Rust throw11, each with three erroneous fields. These are not harmless grammar differences. For example, [request000](../../../data/ranker/pretrained/store-parallel-text-v1-attempt-1/request-000.json) binds the mug to Sage but generates “Trail mug in Cobalt,” “Cobalt selected” and “Shown in Cobalt at $24.00.” The color is wrong even though Cobalt is another legitimate mug variant. All three values are null in the serialized card; canonical Sage metadata remains correct.

All192 frontier body prices match the selected product, and all192 titles retain its complete canonical identity. Eleven withheld throw badges also repeat “selected”: “Moss selected selected” for a Rust selection. No accepted field contained a new unsupported benefit, wrong price/identity/color or meaningful fluency problem in this review. No withheld field became useful merely by allowing concise captions. Frozen and contextual complete-view counts therefore agree: **10/32 first3; 1/32 full6**. All32 retain six cards and20 generated fields; unsuccessful attempts remain in the denominator.

## The successful202 ms result is real but narrow

[request011](../../../data/ranker/pretrained/store-parallel-text-v1-attempt-1/request-011.json), query “a canvas shopping carryall,” has20 readable, source-correct fields and correct top1 tote selection. Its six cards are:

| Canonical product | Selected color | Item price |
| --- | --- | --- |
| Market canvas tote | Olive | $42 |
| Weekender duffel | Charcoal | $84 |
| Zip essentials pouch | Ochre | $26 |
| Everyday canvas sneakers | Off-white | $96 |
| Everyday daypack | Sand | $68 |
| Soft cotton crewneck | Oatmeal | $72 |

For example, “Market canvas tote in Olive,” “Olive selected” and “Shown in Olive at $42.00.” are useful factual text. Every other card uses the same readable pattern with its own correct facts. “Your selected collection” and “More selected choices” make no unsupported shared-use claim. Repeating a consistent short format across cards is not a generation loop. All six bindings occur in TRAIN. The other five cards are broad store fill, not five equally strong shopping-carryall alternatives.

The receipt's full serialization time is202.0916 ms; the fixed32-attempt full-completion median is197.0462 ms. The latter is an **attempt** median, not32 useful views. Timers include fresh Qwen query encoding/ranking, guards, cached-fact head generation, validation, freshness and JSON. They exclude startup, offline48-feature extraction, fitting, network, React and images. No browser readiness or conversion conclusion follows.

## Heldout: genuine factual failure, not long-grammar enforcement

The210 heldout label rows contain84 generic sections and42 each of titles, badges and bodies. **Only the84 sections pass**, all readable generic headings. Every one of126 card fields lacks or contradicts its selected color. Of42 bodies,35 state the correct price, six state a wrong price and one omits price. For a Slate-blue daypack at$68, the plain output is “Shown in Graphite at $89.00.” For a Forest cotton crewneck at$72, one output is “Shown in Moss at $58.00.” These are substantive fact errors.

Three daypack titles are incomplete or incoherent, including “Everyday selected in in.” Three crewneck titles add unverified canvas/woven material wording, such as “Woven canvas crewneck in Oatmeal.” Six heldout fields have repetition. Nine titles lack the full canonical string; that alone is not treated as false identity, since an adapted title such as “Everyday duffel” could be useful with its canonical subtitle. Their wrong colors still make them unusable. Short bodies such as “Cobalt option at $24.00.” are grammatically adequate captions; they fail because the selected color is Sage.

The model exactly recovers402/402 TRAIN teachers, versus84/210 evaluation teachers consisting solely of sections. Exact recovery is a mechanical metric. Independent reading finds the heldout failure remains when useful alternative short wording is allowed. The earlier [positional coverage audit](STORE_PARALLEL_TEXT_V1_INDEPENDENT_AUDIT.md) identifies one training-data obstacle; widespread color failures also affect fields without that obstacle. These results do not identify a single cause or prove the frozen features lack color information.

## Ranking and generation mechanism

Offline public-function top1 relevance is **30/32**, scored after inference against the separate labels. “A notebook for writing ideas” selects the organizer pouch and omits the notebook from six cards. “A cotton blanket to use as a throw” selects the towel first, with the throw later in six. These misses remain distinct from copy factuality. This is the curated known-store frontier, with previously disclosed training-wording overlaps and no budget/stock/size or broad-personalization evidence.

Source inspection and receipts support a **conditional neural generator**, not runtime teacher lookup. The request-only path constructs public fact/kind/style rows, strips outer field metadata, projects cached frozen-Qwen features through the learned head, argmaxes14 independent word slots, stops at the first learned EOS and joins whitespace/punctuation. It does not insert facts or repair text during detokenization. I verified all48 extraction texts against the public feature contract and all examined raw strings against their recorded slots, plus source/checkpoint/bundle hashes. I did not rerun logits or load tensor weights.

Matching training teacher phrases after training is not a runtime template fallback. It is also not evidence of open prose: this is a107-word learned vocabulary over a known catalog with three explicit style settings. Copy is neutral and conditioned on selected public facts, not shopper-intent prose or hidden persona truth. Canonical names/prices/actions remain metadata. Warmup and eight style-counterfactual receipts are outside the32-request frontier denominator; no failed output was promoted or silently replaced.
