# Lexical LoRA attempt 1: independent quality rejection

The corrected evaluator completed with exit 2, the declared code for completed offline adjudication without promotion. The candidate fails the unchanged gate. These 32 authored synthetic cases now count as development evidence.

| Canonical result | Raw adapted decision | Query-gated pipeline | Minimum |
|---|---:|---:|---:|
| Primary top 1 | 17/32 | 17/32 | 20/32 |
| Novel grade ≥2 top 1 | 23/32 | 24/32 | 24/32 |
| Grade ≥2 within top 3 | 27/32 | 28/32 | 28/32 |
| Explicit replacements | 3/4 | 3/4 | 3/4 |
| Product-budget/typed-eligibility violations | 0 | 0 | 0 |

The frozen-base lexical control scores **20/32 raw and 19/32 fused primary**, exceeding the adapted model's 17/32. This fails the required positive contribution over the pretrained baseline. Other raw primary controls score 8 owned-only, 5 shuffled-profile, 2 TRAIN-popularity and 0 latest-owned-repeat; their fused versions score 6, 4, 3 and 2. Purchase-history-removed is report-only and scores 7/32 in both pipelines.

The known DEVELOPMENT history diagnostic also fails: raw has 6/10 purchased-item priority demotions and 7/10 novel relevant choices afterwards; fused has 7/10 and 7/10. The fixed minima are 8/10 for each. No deterministic ownership suppression was introduced by this audit.

Reversed catalog order scores raw 21/32 primary, 25/32 novel and 28/32 top-three coverage; fused scores 20/32, 25/32 and 27/32. The preregistered canonical order remains primary. Selecting reversed order after seeing these outcomes would change the evaluation and is not permitted by this gate.

Mechanical validation succeeded: 1,240 extraction gates and serialized feature hashes, 3,584 updates, both learned adapter arrays, exact reload, all 927 frozen tensor hashes, source pins, cleanup, four learned native/inference controls, and raw/control freezes passed admission. The separate query replay passed all 344 requests across four models (1,376 rows), with 292 query vectors, matching the same adapter and combined freeze. That establishes saved math/provenance consistency, not recommendation usefulness.

The original frozen V1 auditor passed admission and parsed private labels, then failed on its derived metric-view case IDs before completing a quality report. Original source, freeze and attempt 1 were preserved. A separately frozen V2 changes only derived `caseId` normalization alongside the version/self-freeze filename; another reviewer checked the exact diff and regression tests before execution. No model, prediction, label, denominator or threshold changed. The correction receipt records the first label parse explicitly.

Actual output: `data/ranker/pretrained/persona-choice-independent-adjudication-v2-attempt-1/ADJUDICATION.json`, SHA `107fe4e5e330d4176ba8f0bca7ee6b4329f2d705e06b2ca84ebf18f9c410254c`. The machine-readable summary beside this document joins the admission and mechanical evidence.

Selected variants/images, copy factuality, ten-person demo coherence, specific-query regression, and useful visible first-three/full-six browser readiness remain separate serving checks. Zero product-eligibility violations cannot certify those fields. No accepted/default profile or serving promotion is supported by this result.
