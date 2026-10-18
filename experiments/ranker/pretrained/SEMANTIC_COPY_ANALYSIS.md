# Direct semantic-copy independent CPU analysis

Run status: `completed`. Recorded training steps: 336. All twelve fixed evaluation arms completed: True. Original artifacts were read only. Every completed arm was reassembled from unchanged raw values, its actual product/image/grouping plan with copy cleared, and frozen request/catalog; saved and current quality agree exactly.

| Split / arm | Pages / 3 | Titles / 18 | Badges incl. NONE / 18 | Headings / 6 | Completed / missing | All median ms | Valid median ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev / frozen | 0 | 6 | 18 | 0 | 3 / 0 | 9289.8 | unavailable |
| dev / learned | 0 | 15 | 18 | 4 | 3 / 0 | 8832.5 | unavailable |
| test / frozen | 0 | 6 | 18 | 0 | 3 / 0 | 9647.8 | unavailable |
| test / learned | 0 | 6 | 12 | 3 | 3 / 0 | 9467.8 | unavailable |

Completed paired selections compared: 6; product/image/grouping mismatches: 0.

Total run wall: 297413.7 ms; resident load: 2099.8 ms; summed evaluation completion wall: 111246.2 ms. Timers overlap; do not add these phases. Memory receipts and complete raw field/fact diagnostics are in the companion JSON.

| Split / arm | Generated badges | Accepted NONE | Raw required NONE correct / executed | Strict correct-NONE pages / evaluable | Supported title accepted+aligned / executed | Tokens | AR steps | Peak bytes* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev / frozen | 9 | 9 | 9 / 9 | 3 / 3 | 5 / 9 | 264 | 306 | 6342247208 |
| dev / learned | 9 | 9 | 9 / 9 | 3 / 3 | 6 / 9 | 271 | 313 | 6342247208 |
| test / frozen | 9 | 9 | 9 / 9 | 3 / 3 | 5 / 9 | 257 | 299 | 6342247208 |
| test / learned | 6 | 6 | 9 / 9 | 2 / 3 | 0 / 9 | 273 | 314 | 6342247208 |

*Native peaks are cumulative across training and previous arms.

## Actual saved examples

For each completed arm, show the first rejected field of each kind; if none fail, show its first accepted title. No wording is repaired. The JSON retains every field, selected identity, tags, observed support, universal tags, token count and stop reason.

### dev / copy-cfd57bc88dbf0ea9 / frozen

Page accepted: False; 89 output tokens, 103 AR steps, 9289.8 ms.

- S1 (rejected, stop, 8/16 tokens): "Roomy Daypacks for Everyday Travel". Facts: {"memberIdentities": ["Belimar 14L Daypack", "Kelamar 21L Daypack", "Faloven 36L Daypack"], "universalTags": []}
- T3 (rejected, stop, 10/24 tokens): "Faloven 36L Daypack — mesh". Facts: {"selectedIdentity": "Faloven 36L Daypack", "productId": "copy-p-28e4a7c47954b206", "imageCaptionId": "copy-v-0362dc3e9bfa8031", "tags": ["mesh", "quilted", "rolltop"], "observedSupport": [], "approvedTitles": ["Faloven 36L Daypack", "Faloven 36L Daypack", "Faloven 36L Daypack — mesh"]}

Strict rejections: ["S1:unsupported-or-missing-heading", "S2:unsupported-or-missing-heading", "T3:unsupported-missing-nonnovel-or-unresponsive-title", "T4:unsupported-missing-nonnovel-or-unresponsive-title", "T5:unsupported-missing-nonnovel-or-unresponsive-title", "T6:unsupported-missing-nonnovel-or-unresponsive-title", "T6:other_product_identity"].

### dev / copy-cfd57bc88dbf0ea9 / learned

Page accepted: False; 92 output tokens, 106 AR steps, 8832.5 ms.

- S1 (rejected, stop, 6/16 tokens): "Travel for everyday roomy needs". Facts: {"memberIdentities": ["Belimar 14L Daypack", "Kelamar 21L Daypack", "Faloven 36L Daypack"], "universalTags": []}

Strict rejections: ["S1:unsupported-or-missing-heading"].

### dev / copy-999f53226561e5ad / learned

Page accepted: False; 90 output tokens, 104 AR steps, 8629.5 ms.

- S1 (rejected, stop, 4/16 tokens): "Reflective travel choice". Facts: {"memberIdentities": ["Lumivia 27L Daypack", "Pavivia 20L Daypack", "Sorivia 12L Daypack"], "universalTags": []}
- T3 (rejected, stop, 11/24 tokens): "Sorivia 12L Daypack for everyday travel". Facts: {"selectedIdentity": "Sorivia 12L Daypack", "productId": "copy-p-e696bbd4ebe8dbdb", "imageCaptionId": "copy-v-b4354d4f7ae6769f", "tags": ["padded", "reflective", "water-resistant"], "observedSupport": ["reflective"], "approvedTitles": ["Sorivia 12L Daypack", "Sorivia 12L Daypack", "Sorivia 12L Daypack — padded"]}

Strict rejections: ["S1:unsupported-or-missing-heading", "T3:unsupported-missing-nonnovel-or-unresponsive-title"].

### dev / copy-999f53226561e5ad / frozen

Page accepted: False; 91 output tokens, 105 AR steps, 8974.6 ms.

- S1 (rejected, stop, 8/16 tokens): "Reflective Daypacks for Everyday Travel". Facts: {"memberIdentities": ["Lumivia 27L Daypack", "Pavivia 20L Daypack", "Sorivia 12L Daypack"], "universalTags": []}
- T1 (rejected, stop, 12/24 tokens): "Lumivia 27L Daypack with reflective mesh". Facts: {"selectedIdentity": "Lumivia 27L Daypack", "productId": "copy-p-c1d9e834667cca62", "imageCaptionId": "copy-v-f1a86209ee028382", "tags": ["quilted", "reflective", "roomy"], "observedSupport": ["reflective"], "approvedTitles": ["Lumivia 27L Daypack", "Lumivia 27L Daypack", "Lumivia 27L Daypack — quilted"]}

Strict rejections: ["S1:unsupported-or-missing-heading", "S2:unsupported-or-missing-heading", "T1:unsupported-missing-nonnovel-or-unresponsive-title", "T2:unsupported-missing-nonnovel-or-unresponsive-title", "T4:unsupported-missing-nonnovel-or-unresponsive-title", "T5:unsupported-missing-nonnovel-or-unresponsive-title"].

### dev / copy-8edaa97153cbab7e / frozen

Page accepted: False; 84 output tokens, 98 AR steps, 9571.7 ms.

- S1 (rejected, stop, 5/16 tokens): "Durable Everyday Travel Essentials". Facts: {"memberIdentities": ["Lumilis 11L Daypack", "Renilan 25L Daypack", "Merimar 18L Daypack"], "universalTags": ["durable"]}
- T2 (rejected, stop, 10/24 tokens): "Renilan 25L Daypack — durable". Facts: {"selectedIdentity": "Renilan 25L Daypack", "productId": "copy-p-978262c329dd94aa", "imageCaptionId": "copy-v-08e363071d243a00", "tags": ["durable", "lightweight", "water-resistant"], "observedSupport": [], "approvedTitles": ["Renilan 25L Daypack", "Renilan 25L Daypack", "Renilan 25L Daypack — durable"]}

Strict rejections: ["S1:unsupported-or-missing-heading", "S2:unsupported-or-missing-heading", "T2:unsupported-missing-nonnovel-or-unresponsive-title", "T3:unsupported-missing-nonnovel-or-unresponsive-title", "T4:unsupported-missing-nonnovel-or-unresponsive-title", "T5:unsupported-missing-nonnovel-or-unresponsive-title"].

### dev / copy-8edaa97153cbab7e / learned

Page accepted: False; 89 output tokens, 103 AR steps, 9072.4 ms.

- T4 (rejected, stop, 12/24 tokens): "Ivoren 19L Daypack for everyday travel". Facts: {"selectedIdentity": "Ivoren 19L Daypack", "productId": "copy-p-347984b76ba4dbe0", "imageCaptionId": "copy-v-d43926c9feefe1a8", "tags": ["packable", "padded", "water-resistant"], "observedSupport": ["padded"], "approvedTitles": ["Ivoren 19L Daypack", "Ivoren 19L Daypack", "Ivoren 19L Daypack — packable"]}

Strict rejections: ["T4:unsupported-missing-nonnovel-or-unresponsive-title", "T5:unsupported-missing-nonnovel-or-unresponsive-title"].

### test / copy-acc8677838207254 / frozen

Page accepted: False; 80 output tokens, 94 AR steps, 9647.8 ms.

- S1 (rejected, stop, 4/16 tokens): "Compact structured everyday travel". Facts: {"memberIdentities": ["Herilis 24L Daypack", "Evidor 38L Daypack", "Soriren 31L Daypack"], "universalTags": ["compact"]}
- T1 (rejected, stop, 10/24 tokens): "Herilis 24L Daypack — compact". Facts: {"selectedIdentity": "Herilis 24L Daypack", "productId": "copy-p-564bf3c0048c9938", "imageCaptionId": "copy-v-c0cc8f7516483bb0", "tags": ["compact", "foldable", "ventilated"], "observedSupport": ["compact"], "approvedTitles": ["Herilis 24L Daypack", "Herilis 24L Daypack", "Herilis 24L Daypack — compact"]}

Strict rejections: ["S1:unsupported-or-missing-heading", "S2:unsupported-or-missing-heading", "T1:unsupported-missing-nonnovel-or-unresponsive-title", "T2:unsupported-missing-nonnovel-or-unresponsive-title", "T3:unsupported-missing-nonnovel-or-unresponsive-title", "T4:unsupported-missing-nonnovel-or-unresponsive-title", "T5:unsupported-missing-nonnovel-or-unresponsive-title", "T6:unsupported-missing-nonnovel-or-unresponsive-title", "T6:other_product_identity"].

### test / copy-acc8677838207254 / learned

Page accepted: False; 92 output tokens, 105 AR steps, 9533.7 ms.

- S1 (rejected, stop, 3/16 tokens): "Compact travel choice". Facts: {"memberIdentities": ["Herilis 24L Daypack", "Evidor 38L Daypack", "Soriren 31L Daypack"], "universalTags": ["compact"]}
- T1 (rejected, stop, 11/24 tokens): "Herilis 24L Daypack for structured travel". Facts: {"selectedIdentity": "Herilis 24L Daypack", "productId": "copy-p-564bf3c0048c9938", "imageCaptionId": "copy-v-c0cc8f7516483bb0", "tags": ["compact", "foldable", "ventilated"], "observedSupport": ["compact"], "approvedTitles": ["Herilis 24L Daypack", "Herilis 24L Daypack", "Herilis 24L Daypack — compact"]}
- B1 (rejected, stop, 2/8 tokens): "compact travel". Facts: {"selectedIdentity": "Herilis 24L Daypack", "productId": "copy-p-564bf3c0048c9938", "imageCaptionId": "copy-v-c0cc8f7516483bb0", "tags": ["compact", "foldable", "ventilated"], "observedSupport": ["compact"], "approvedTitles": ["Herilis 24L Daypack", "Herilis 24L Daypack", "Herilis 24L Daypack — compact"]}

Strict rejections: ["unfinished-or-markup-output", "S1:unsupported-or-missing-heading", "S2:unsupported-or-missing-heading", "T1:unsupported-missing-nonnovel-or-unresponsive-title", "B1:unsupported-or-unresponsive-badge", "T2:unsupported-missing-nonnovel-or-unresponsive-title", "B2:unsupported-or-unresponsive-badge", "T3:unsupported-missing-nonnovel-or-unresponsive-title", "B3:unsupported-or-unresponsive-badge", "T4:unsupported-missing-nonnovel-or-unresponsive-title", "B4:NONE-required", "T5:unsupported-missing-nonnovel-or-unresponsive-title", "B5:NONE-required", "T6:unsupported-missing-nonnovel-or-unresponsive-title", "T6:other_product_identity", "B6:NONE-required"].

### test / copy-cfb48f005368948a / learned

Page accepted: False; 95 output tokens, 109 AR steps, 9467.8 ms.

- S1 (rejected, stop, 5/16 tokens): "Quilted travel choice". Facts: {"memberIdentities": ["Lumimar 37L Daypack", "Herilan 22L Daypack", "Avevia 16L Daypack"], "universalTags": []}
- T1 (rejected, stop, 12/24 tokens): "Lumimar 37L Daypack for everyday travel". Facts: {"selectedIdentity": "Lumimar 37L Daypack", "productId": "copy-p-613610e208cd39d2", "imageCaptionId": "copy-v-109d86d9d397f0d0", "tags": ["durable", "organizer", "quilted"], "observedSupport": ["quilted"], "approvedTitles": ["Lumimar 37L Daypack", "Lumimar 37L Daypack", "Lumimar 37L Daypack — durable"]}

Strict rejections: ["S1:unsupported-or-missing-heading", "T1:unsupported-missing-nonnovel-or-unresponsive-title", "T3:unsupported-missing-nonnovel-or-unresponsive-title", "T4:unsupported-missing-nonnovel-or-unresponsive-title"].

### test / copy-cfb48f005368948a / frozen

Page accepted: False; 90 output tokens, 104 AR steps, 8977.3 ms.

- S1 (rejected, stop, 6/16 tokens): "Quilted Compact Everyday Carry". Facts: {"memberIdentities": ["Lumimar 37L Daypack", "Herilan 22L Daypack", "Avevia 16L Daypack"], "universalTags": []}
- T2 (rejected, stop, 10/24 tokens): "Herilan 22L Daypack — adjustable". Facts: {"selectedIdentity": "Herilan 22L Daypack", "productId": "copy-p-3f9c629812d252dd", "imageCaptionId": "copy-v-fd95a3f1a0e57238", "tags": ["adjustable", "durable", "structured"], "observedSupport": [], "approvedTitles": ["Herilan 22L Daypack", "Herilan 22L Daypack", "Herilan 22L Daypack — adjustable"]}

Strict rejections: ["S1:unsupported-or-missing-heading", "S2:unsupported-or-missing-heading", "T2:unsupported-missing-nonnovel-or-unresponsive-title", "T5:unsupported-missing-nonnovel-or-unresponsive-title", "T6:unsupported-missing-nonnovel-or-unresponsive-title"].

### test / copy-e9929c454b3e51c8 / frozen

Page accepted: False; 87 output tokens, 101 AR steps, 9927.5 ms.

- S1 (rejected, stop, 5/16 tokens): "Compact Card Foldable Selection". Facts: {"memberIdentities": ["Ivovia 21L Daypack", "Darodor 14L Daypack", "Kelador 28L Daypack"], "universalTags": []}
- T2 (rejected, stop, 11/24 tokens): "Darodor 14L Daypack — foldable". Facts: {"selectedIdentity": "Darodor 14L Daypack", "productId": "copy-p-c22c7696d671e069", "imageCaptionId": "copy-v-98b69abe014bfede", "tags": ["foldable", "rolltop", "roomy"], "observedSupport": ["foldable"], "approvedTitles": ["Darodor 14L Daypack", "Darodor 14L Daypack", "Darodor 14L Daypack — foldable"]}

Strict rejections: ["S1:unsupported-or-missing-heading", "S2:unsupported-or-missing-heading", "T2:unsupported-missing-nonnovel-or-unresponsive-title", "T3:unsupported-missing-nonnovel-or-unresponsive-title", "T5:unsupported-missing-nonnovel-or-unresponsive-title"].

### test / copy-e9929c454b3e51c8 / learned

Page accepted: False; 86 output tokens, 100 AR steps, 9321.5 ms.

- T2 (rejected, stop, 11/24 tokens): "Darodor 14L Daypack for everyday travel". Facts: {"selectedIdentity": "Darodor 14L Daypack", "productId": "copy-p-c22c7696d671e069", "imageCaptionId": "copy-v-98b69abe014bfede", "tags": ["foldable", "rolltop", "roomy"], "observedSupport": ["foldable"], "approvedTitles": ["Darodor 14L Daypack", "Darodor 14L Daypack", "Darodor 14L Daypack — foldable"]}

Strict rejections: ["T2:unsupported-missing-nonnovel-or-unresponsive-title", "T4:unsupported-missing-nonnovel-or-unresponsive-title", "T6:unsupported-missing-nonnovel-or-unresponsive-title"].

## Limits

- Small fictional teacher-grammar split; no production or human-quality claim.
- Strict acceptance is lexical policy plus identity; requested word ranges are separately diagnostic.
- Ranking/image quality has no oracle; paired selections are checked only for equality.
- Cumulative native peak memory includes training/previous arms; not isolated per-arm allocation.
- Missing arms remain in fixed three-case denominators; unavailable latency is not zero.
- Valid-only latency has no value when no pages pass; all-completion latency is not usable-page throughput.
- Run wall includes load, controls, training, evaluation and serialization; phase timers overlap.

## Independent reading of the actual outputs

Do not promote this adapter. All twelve arms completed, but both arms accept
0/3 development pages and 0/3 test pages. Development titles improve 6/18 to
15/18 and headings 0/6 to 4/6. Test titles remain 6/18, headings improve 0/6 to
3/6, and accepted badges including `NONE` fall 18/18 to 12/18. Those field gains
do not establish usable-page improvement or a usable-page speedup. All valid-only
latency samples are empty. Completion medians are 9.290→8.832 seconds on
development and 9.648→9.468 seconds on test, with only three cases per cell.

The held-out support failure extends beyond the globally rejected page.
Development supported titles mention the observed attribute in 7/9 frozen raw
values versus 6/9 learned values; accepted-and-aligned counts are 5/9 versus
6/9. Test supported titles mention it in 9/9 frozen raw values versus **0/9
learned values**; accepted-and-aligned counts fall 5/9 to 0/9. On stopped learned
test case `copy-cfb48f005368948a`, `Lumimar 37L Daypack for everyday travel`
preserves name and size and is novel, but omits its observed `quilted` support.
The other two supported titles in that case do the same. On stopped case
`copy-e9929c454b3e51c8`, all three supported titles omit `foldable`. These are
observed responsiveness regressions under the frozen contract, not artifacts
of global truncation.

The exact length failure is learned test case `copy-acc8677838207254`, field
`B2`, raw `Evidor 38L Day`: 8 output tokens, 8 AR steps, cap 8,
`finishReason=length`. It neither names the required observed `compact`
attribute nor completes a useful badge. Every other row stops. The global
finish rule rejects all fourteen fields, including three raw required `NONE`
values which are correct as raw values. Thus learned test raw required `NONE`
remains 9/9, strict accepted required `NONE` falls 9/9 to 6/9, and strict
correct-NONE pages fall 3/3 to 2/3. Do not rescue that page or count isolated
fields as accepted.

The same case contains independent content errors: T1–T3 use `for structured
travel` although their selected products have observed `compact` support and
no `structured` tag. T6 says `Evidor 38L Daypack for structured travel` for
selected `Olilen 32L Daypack`, an actual other-product name and size error.
Scheduler-owned row labels prevent transport ambiguity, but do not guarantee
that the generated value describes the selected product. No generated-label
protocol rejections occur in any of the twelve arms.

Headings still leak a single member's attribute to an entire section. Learned
development `Travel for everyday roomy needs` and learned test `Quilted travel
choice` both describe sections whose universal-tag intersection is empty.
Some accepted headings are semantically weak: `Travel for everyday travel`
repeats itself, including both accepted headings in the last learned test case.
Lexical acceptance does not measure usefulness or fluency. There is also a
lexical-policy limitation in the frozen arm: development headings `Durable
Everyday Travel Essentials` and `Padded Everyday Travel Essentials` use actual
universal attributes but fail solely because `essentials` is outside the
allowed framing vocabulary. This observation does not change their acceptance
or loosen the policy.

The split is harder than new names and combinations alone. Training supported
targets cover twelve attributes; development support targets are `padded`,
`reflective`, `roomy`, and test support targets are `compact`, `foldable`,
`quilted`. Those positive target-attribute sets are disjoint. Individual words
can occur in training public catalog evidence or unsupported preferences, but
the held-out attributes have no positive supported title/badge targets in
training. Combined with the disclosed first-preference shortcut, narrow teacher
grammar and only twelve training requests, this limits what the adapter has
learned about support binding. It is a plausible mechanism, not a causal proof;
no test-informed prompt/target changes or retraining were performed by this
audit, and these inspected test cases must not be relabeled as fresh holdouts.

The run takes 297.414 seconds including 2.100 seconds resident load and 111.246
seconds summed evaluation completion time. All 336 step timers sum to 106.411
seconds; their median is 303.794 ms. These timers exclude other training/control
work and overlap the run timer. Total measured cumulative native peak is
6,342,247,208 bytes, below the declared 7-GiB guard; it cannot distinguish
isolated arm memory. Frozen/learned development output totals are 264/271 tokens
and 306/313 AR steps; frozen/learned test totals are 257/273 tokens and 299/314
AR steps. Generation did not shrink the requested six-card grid or 224-token
summed cap.

Provenance was checked after replay: current `semantic_copy_direct.py` and
`semantic_copy_training.py` hashes match the native source receipts, and the
original `results.json` hash still matches the pre/post replay hash in the
analysis JSON. All six matched pairs retain exactly the same selected products,
image IDs and grouping. The original result, frozen cases and validators were
read only. The seven analysis CPU regressions passed before the inference run;
the analysis command ran once after final completion.
