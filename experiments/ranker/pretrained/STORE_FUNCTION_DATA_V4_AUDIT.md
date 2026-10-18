# Actual-store ranking data v4

The frozen CPU package contains **64 TRAIN requests and 24 evaluation requests** for the actual 16-product store. It expands function-only paraphrases into sparse observed-search profiles, simultaneous needs, negative-only choices, exact-name searches and hard constraints. Twelve CPU contract tests pass; no native inference, optimizer update or storefront change was performed.

| Split | Single/current needs | Sparse searches | Two needs | Negative-only choices | Exact-name overrides | Hard constraints | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TRAIN | 16 | 16 | 16 | 8 | 2 | 6 | 64 |
| Evaluation | 4 current/profile preference pairs = 8 cases | included in those pairs | 8 | 4 diagnostic cases included in the pairs | 3 | 5 | 24 |

The evaluation contains eight preference pairs. Each pair holds query, budget and hard requirements fixed and has disjoint supported target sets. Four pairs require both members of a two-product need set; their eight target sets do not occur among the 16 TRAIN two-product combinations. All product identities and function classes remain TRAIN-seen. The other four pairs exercise observed searches, profile avoidances, current avoidances and clothing preferences. Four evaluation cases isolate negative cues; eight TRAIN examples isolate negative cues by listing the same two options and changing only what to avoid.

Observed profiles are synthetic public search strings and declared preferences/avoidances. They are not real merchant logs. Empty fields remain empty; the package never invents persona truth, age, gender, profession or inferred taste. The exact public schema rejects extra fields. Encoder text contains only public request/product facts; case IDs, splits, families, targets, rationale and supporting facts stay outside the encoder. Labels retain both desired products and the subset eligible after guards, with two equally relevant labels for two equal needs.

Current explicit names override stale searches in five cases. Inclusive item-price budgets, exact declared colors, unknown sizes and unknown stock produce explicit eligibility labels. Four TRAIN and three evaluation cases require empty results. Delivery is excluded from item budgets. Unknown attributes do not become factual promises.

The v3 evaluation has already been inspected and remains **development replay**. V4 wording is newly authored from current public facts, but it is visible to its author/reviewers and is not a sealed benchmark. None of these evaluation requests has been run through a model. Evaluation labels must not fit weights, select a checkpoint, select a prompt, set thresholds or define new aliases. Any reuse after model inspection should be labelled development replay.

The existing v3 single-label query softmax cannot directly consume this package. A bounded follow-up can retain the catalog function head, use independent query sigmoid scores for multiple desired functions, and train a multi-positive objective on TRAIN labels. Desired semantic relevance and deterministic eligibility should remain distinct; unknown stock/size and budget masking should not be learned from invented facts. Explicit avoided products provide hard negatives; unrequested products are not universally unusable. A query-product compatibility scorer is another option. This remains a known-catalog model in either form.

Abstention thresholds need a TRAIN-only calibration partition or cross-validation. Do not fill six cards when only one need matches or every desired product fails a hard guard. A future serving adapter must explicitly map the new request schema; no runtime code was changed.

The CPU evaluator requires outputs for every fixed case. It reports any-relevant hit@1, recovery of the entire target set within its own cardinality, relevant retrieval at six, constraint safety, explicit-avoidance exposure, empty-result abstention and both target sets recovered per preference pair. A fixture that returns only one correct item gets 21/21 hit@1 but recovers only 13/21 complete target sets and 4/8 pairs. The label oracle recovers 21/21 target sets and 8/8 pairs; these are evaluator fixtures, not learned quality.

CPU tokenizer preflight with the pinned tokenizer found 93–120 tokens per request and 45–65 per product, with no chat template. All fit the existing 512-token cap without truncation. Encoding each product and request once would require 104 vectors: 16 catalog, 64 TRAIN and 24 evaluation. No vectors were generated.

The initial redundant-positive/negative draft remains intact at `store_function_v4_draft_01/`, including exact builder/test/source bytes and verified old hashes. The final dataset is `store_function_v4/`; its `FREEZE.json` authenticates the builder, tests, catalog bridge and six data/audit files.

Deliverables: [builder](store_function_data_v4.py), [tests](test_store_function_data_v4.py), [frozen dataset](store_function_v4/FREEZE.json), and [machine-readable audit](STORE_FUNCTION_DATA_V4_AUDIT.json). No GPU work, runtime edits, commits or model-quality claims were made.
