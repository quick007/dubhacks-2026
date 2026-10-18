# Complete-card endpoint semantic review

This review reads the completed `above-fold-copy-v1` run. The independent replay in `BENCHMARK_AUDIT.json` reproduces all saved validation, payload and raw-row records exactly. It also verifies the actual tokenizer decoding/EOS IDs, actual serialized payload bytes/hash, complete metadata binding, source/input/checkpoint hashes and all matched-width control files. The original evidence is unchanged. No model, GPU, browser or UI work was performed by this audit.

All 36 fixed attempts executed. Every split/model/width combination has **0/3 complete accepted views**, hence 0/36 overall and no accepted payload-ready latency. Width 6 reduces completion time for these invalid attempts; it establishes no usable-view speedup. Each model/width evaluates the same six inspected requests with the same ten fields and 208 sampled-token cap. Full-page generation, ranking and browser paint remain outside this endpoint.

## Actual field and timing evidence

Counts below aggregate three widths, so each model has 18 view attempts and 54 title/badge/body rows. These are repeated shape evaluations of six requests, not 54 independent product generalization cases.

| Model | Accepted titles / 54 | Badges including NONE / 54 | Headings / 18 | Bodies / 54 | Complete views / 18 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen | 23 | 54 | 12 | 3 | 0 |
| Paired STB adapter, no C supervision | 40 | 42 | 11 | 6 | 0 |

Learned test titles pass 9/9 at every width, while learned dev titles pass 4/9 at widths 1 and 6 and 5/9 at width 3. The changed ten-field task therefore does not inherit the previous fourteen-field replay's successful dev behavior. Every title and body with supported observed tags mentions at least one supported tag in its raw text; mentioning the right preference alone is insufficient when other content is wrong.

| Model | Width | Completion median across all six attempts, ms | Accepted-ready samples |
| --- | ---: | ---: | ---: |
| Frozen | 1 | 12112.6 | 0 |
| Frozen | 3 | 9822.9 | 0 |
| Frozen | 6 | 8804.5 | 0 |
| Paired STB adapter | 1 | 10942.6 | 0 |
| Paired STB adapter | 3 | 9741.5 | 0 |
| Paired STB adapter | 6 | 7890.3 | 0 |

Native run wall is 426623.8 ms, with 359612.7 ms summed measured attempts, 1370.2 ms resident load, 3448 emitted text tokens and 3805 sampled row AR steps. The three length-finished title rows account for the text/AR difference being 357 rather than 360 EOS tokens. Row AR steps count returned samples, not all speculative/native forward work. Peak is cumulative across controls/arms: 6756602444 bytes, below the 7516192768-byte guard. No per-arm isolated peak or latency-tail estimate is established.

## Why body copy fails

All 108 body rows return actual EOS within their 32-sample budgets: frozen 15–23 samples, learned 5–17 samples. None has a raw body truncation or copied-catalog-body rejection. The failure mechanism is principally content, plus a conservative lexical policy and one coupled title-stop case.

Diagnostic grouping below is exclusive: first detect an attribute from the frozen cases' tag universe absent from the selected product, or an explicit forbidden term; then a title-stop coupling; then remaining lexical-only rejections. This grouping does not alter policy acceptance. A “lexical-only” result means the validator's only reported category is its lexicon: it does not prove that an unlisted mechanism is supported.

| Body result, repeated shapes | Frozen / 54 | Paired STB adapter / 54 |
| --- | ---: | ---: |
| Unsupported selected attribute or forbidden term | 29 | 21 |
| Remaining lexicon-only rejection | 22 | 24 |
| Body stopped, but associated title did not stop | 0 | 3 |
| Accepted | 3 | 6 |

Concrete failures:

- `Neridor 28L Daypack` has `compact`, `roomy`, `structured`; frozen C1 says **“Structured 28L daypack with rolltop for everyday travel.”** The learned C1 says **“Structured rolltop travel choices”**. `rolltop` is a requested attribute supported by a different card. This is a selected-product factual error despite correct `structured` preference alignment.
- `Tavulan 23L Daypack` has `organizer`, `padded`, `quilted`, with no supported preference. Frozen C2 says **“Tavulan 23L Daypack offers compressible and reflective everyday travel.”** Neither `compressible` nor `reflective` belongs to this card.
- All three roomy test products lack `padded`; frozen and learned C rows nevertheless add padded comfort/travel wording. Example: **“This roomy 31L daypack offers padded travel choices.”** Correct `roomy` does not justify `padded`.
- `Vesudor 33L Daypack` has `adjustable`, `compact`, `organizer`, with no supported preference. Learned C3 says **“This 33L Vesudor daypack offers ventilated travel choices.”** `ventilated` belongs to another card.
- Learned C1 **“This 16L Vesuris daypack is compressible for everyday travel”** stops normally at 16 samples including EOS and uses only selected facts. Its associated T1 repeats **“for everyday travel”** until the 24-sample cap, so the unchanged title/body validator rejects that body through its joint finish-reason gate. The native C receipt is not truncated; this is a distinct accounting/validation dependency, preserved without rescue.

Some rejections include ordinary wording that could be reconsidered in a separately versioned policy, such as `offers`, `features`, `use` or `choices`. For example learned **“Ventilated travel choices”** uses a supported selected attribute but fails solely on `choices`; it is also a heading-like fragment rather than the requested sentence. Frozen **“This Qorilan 20L Daypack features adjustable straps for everyday travel.”** fails on `features,straps`; `features` is grammatical framing, but `straps` additionally assumes a mechanism not supplied by the synthetic catalog. It must not be counted as a harmless lexical false positive. No lexical relaxation was applied, and these mixed cases cannot explain away the selected-attribute errors above.

Actual accepted examples include learned **“This 20L daypack is adjustable for everyday travel”** and **“Packable 30L daypack with organizer for everyday travel”**, plus frozen **“This Zelidor 14L Daypack is a lightweight, mesh, packable backpack.”** These are genuine native values, distinct from teacher targets and catalog descriptions; they are isolated fields, not complete accepted views.

## Protocol, identity and binding

Frozen often returns an exact approved title, such as **“Walinor 13L Daypack — adjustable”**, so novelty rejects it. Learned dev T1 repeats the selected identity followed by travel wording until its 24-sample cap in all three shapes. Learned dev B1 returns the exact raw string `"adjustable\n"` in all three shapes; strict wire checks correctly reject it rather than accepting its first line. Raw required NONE is correct 18/18 for each model across widths; accepted required NONE is 18/18 frozen and 12/18 learned because the unchanged global core finish rule rejects the affected dev page's fields. Neither NONE nor the malformed badge is rewritten.

All 36 completed payloads bind the ordered three selected product IDs, selected image IDs/keys, exact original-name subtitles, prices and shipping costs to trusted catalog metadata. This proves joined metadata identity, not that model-generated body facts are correct. Generated titles retain strict v1 identity; the prospective subtitle-relaxed policy was not applied. Image keys are synthetic text-only assets; pixel/render quality is unmeasured.

Matched controls use exactly one, three and six real simultaneous rows under each model mode. Width 6 includes heading/title/badge/body jobs. Each independent lane rebuilds the full root; sampled IDs, stop receipts, per-step full-distribution hashes and root hashes match fork/repeat at the same width. The recorded distributions are hashes of normalized logprobs, not saved raw logits or top-two margins. Cross-width wording differences are semantic-comparison evidence and do not contradict the same-width controls.

## Bounded next experiment

Train explicit C rows with the already audited public-evidence teacher extension, while retaining the current 504-row STB fit and this failed transfer run as separate evidence. Keep selected-card facts distinct from global request preferences, include unsupported-preference counterexamples and enforce genuine raw model output under unchanged strict validation. Do not insert teacher grammar as a runtime fallback or tune against these inspected outputs and describe them as sealed holdouts.

The existing feature-training helper's 256-token trainable suffix gate does not cover the longest approximately 601-token card-fact suffix here. A practical prospective layout is to append public card facts to that mode's actual root cache first, snapshot the entire hybrid state exactly, and train/generate through a short field suffix. It needs its own exact final-down adapted cache/continuation proof. Reusing one per-card fact cache for T/B/C also creates unequal prefix roots across cards; current equal-root batching controls do not establish native merging correctness for that change. New matched controls are required before any latency claim. No such implementation or native experiment is performed by this audit.

The six cases were inspected during preparation, the data use an audited synthetic teacher grammar, and no human usefulness or conversion labels exist. These results do not establish production generalization or conversion lift.
