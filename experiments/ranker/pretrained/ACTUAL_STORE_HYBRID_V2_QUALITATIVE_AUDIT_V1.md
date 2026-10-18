# Actual-store hybrid v2 contextual audit

I reviewed all 120 generated fields in the six completed width-six hybrid requests against each public request, selected product facts and structured payload. This is an independent offline LLM judgment, not human review, a new validator or conversion evidence. Existing outputs, strict acceptance rates and benchmark code are unchanged.

The review finds 114 definitely usable fields, five substantive problems and one ambiguous recommendation requiring verification. The unchanged finite grammar accepts 96/120 fields. Correct factual captions and reasonable request-compatible wording can be useful even when they fall outside that grammar; duplicated-decimal prices and incorrect use attribution are separate failures.

| Measure | Count |
| --- | ---: |
| Strict first3 / full6 | 1/6 / 0/6 |
| Contextually usable complete first3 / full6 | 4/6 / 0/6 definite |
| Full6 pending only the ambiguous kitchen-table recommendation | 1/6 |
| Intended public-function target ranked first | 4/6 |
| Intended target retrieved in first3 / full6 | 4/6 / 6/6 |
| Contextually usable whole first3 with target retrieved | 2/6 |
| Individual target-card text usable with bound metadata | 6/6 |

The strictly valid first3 request is the weekend-travel request, where its intended duffel appears fourth. It therefore contributes zero to the benchmark's strict joint first3 usefulness proxy. The separate contextual joint count includes the drinking-cup and carrying-water requests: all required above-fold text is usable in context and the intended product ranks first. Their observed model-side first3 completion times were 5.475 and 5.812 seconds. This is a retrospective contextual assessment of existing payloads, not a newly measured runtime validation endpoint.

| Case / preference | Target position | Whole first3 contextual judgment | Full6 blocker |
| --- | ---: | --- | --- |
| 00: kitchen-table drinking cup | 1 | Usable; generic heading is awkward and terse | Lunch box called the shopper's kitchen-table choice: ambiguous recommendation, set aside |
| 01: carry water when heading out | 1 | Usable | Lunch box described as the choice for carrying water: unsupported use attribution |
| 02: backpack for everyday things | 1 | Unusable displayed price | Bottle price is `$36.00.00` |
| 03: weekend travel bag | 4 | Text usable, intended target below fold | Pouch called a `perfect travel companion`: unsupported fit/quality recommendation |
| 04: write a quick note | 1 | Unusable wrong-product use relationship | Lunch box called `your quick note` |
| 05: zipped cable pouch | 4 | Text usable, intended target below fold | Crewneck price is `$72.00.00` |

The kitchen-table recommendation is not a proven false physical claim: a lunch box could be placed on a kitchen table. The concern is that the request specifically asks for a drinking cup, while the generated lunch-box body presents it as the shopper's choice in that context. It remains a separate uncertain judgment; accepting it would make one full6 payload contextually usable, without changing the strict zero result.

Several bodies are short product/color captions such as `Everyday round sunglasses with Tortoise selected`. I count these as understandable alongside the supplied original name, price and image, despite their strict sentence/price rejection. I do not count the malformed monetary values as repaired by the correct structured price. The pouch title rendered as a full factual price sentence is awkward but understandable. No direct wrong product identity or selected color was found; the material issues are price formatting and request context assigned to an unsuitable product.

All six first3 and six full6 payloads were independently reassembled through the unchanged validator and matched exactly. All 36 product/variant/image actions, original names, selected colors and structured prices match their bound records. Every field's raw quote, bound facts, rationale, strict rejection and exact source JSON pointer are preserved in `ACTUAL_STORE_HYBRID_V2_QUALITATIVE_AUDIT_V1.json`.

The attempt medians are 5.741 seconds for first3 and 6.967 seconds for full6. They include fresh native query encoding, learned ranking, binding, generation, validation and serialization; browser paint and image transfer are excluded. These six inspected development-replay requests do not establish unseen-catalog generalization, preference satisfaction beyond the public-function target proxy, a live storefront result or human usefulness.

The failure pattern warrants better source-conditioned training: the model sometimes assigns the request's purpose to a distractor card, while the ranking can also bury the intended product at position four. Wider decoding may improve scheduling latency but cannot establish that these quality problems are resolved.

The separate width10/20 code review found no launch blockers at `store_copy_wide_v2.py` SHA-256 `a6631ff3b9e930b6c5a74465326fb8eba05aba6a736690c18b577718d98d096c` and `store_hybrid_wide_benchmark_v2.py` SHA-256 `e1d621979def79032ceb4bef64ab36e0bf42a92155c5116dc46c41da6ce4d049`. All eleven CPU tests passed. The code retains all six cards, twenty fields and 416 sampled-token cap; uses independently matched native controls at each width; preserves the true caller ranking-to-copy timer; and retains the shared worker lock, deadlines and caller-inspected seven/eight-GiB guard. No native wider-width run was executed by this reviewer.
