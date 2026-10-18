# Independent saved-evidence adjudication

This CPU audit is prepared before new candidate quality outcomes. It does not load MLX, run a model, optimize weights, launch a browser, or read private quality JSON during preparation. The author also authored the frozen synthetic evaluation labels; this is independent implementation and source checking, not a blind human evaluation.

The exact root gate is `PERSONA_CHOICE_COMBINED_PROMOTION_GATE_V2.json`, SHA `73fa0d52c5ef4fe983834f610ef00cf89a899cf420cdb81008dd5a7909298c6e`. The evaluation packet is `persona_choice_independent_eval_v3/MANIFEST.json`, SHA `2a0801c73440dfbcea50f73c0c1b2de3a96abc8c975caf8d54e92010a84fe312`. The separate private repair audit receipt is not a replacement dataset manifest. No thresholds change.

The script first checks all adapter evidence: 1,240 fixed feature rows and their serialized hashes and measured exact gates; 927 before/after tensor hashes and cleanup; the 896 TRAIN-row join; the seeded four-pass 3,584-update schedule; both changed, finite F32 adapter arrays, true-zero initial B and checkpoint/reload receipts; four fixed learned native/inference controls against original archives; the public cache's 20 entries containing only X/H/Y arrays; and every raw nonfit prediction/control hash. It reconstructs TRAIN-popularity, latest-owned-repeat and frozen-base product-order bindings independently. It checks evaluation requests against the frozen public packet.

It then checks the combined global freeze and recomputes every supplied blend. All four model sets must use the same probability and v5 vector for each request, across canonical/reversed catalog order and across public input ablations that retain the same query and guards. The combined source vector must match the corresponding previously frozen raw vector. Input controls must exactly implement the predeclared transformations.

Only after those checks does the script durably write `ADMISSION_PASSED_BEFORE_GOLD.json` and parse the two private evaluation label files. It recalculates canonical 32-case counts, 16 matched group results, all five strict baseline comparisons, separate four-case replacement compliance, source product budgets and typed eligibility, reversed-order robustness, and the previously inspected ten-pair DEVELOPMENT history diagnostic. The core 20/24/28 minima, zero product/typed-constraint violations, strict primary contribution over all five controls, replacement minimum 3/4, and DEV history minima 8/8 are fixed. Empty/failed outcomes keep their denominator; missing artifacts or rows fail admission.

## Combined producer contract

`ALL_MODELS_PREDICTIONS_FROZEN.json` must contain:

```text
goldParsed: false
independentPacketManifestSha256: <exact packet identity above>
adapterCheckpointSha256: <same adapter as raw freeze>
gateCheckpointSha256: <hash also present in sha256 pins>
stagesCompleted: {adapter:true, queryGate:true, allControls:true}
sha256: {<absolute artifact/source paths>: <hashes>}
models:
  learned: {file:<relative path>, sha256:<hash>}
  frozen-base-lexical-decision: {file:..., sha256:...}
  TRAIN-popularity: {file:..., sha256:...}
  latest-owned-repeat: {file:..., sha256:...}
```

Each model file contains all 344 nonfit rows, or the same exact 264 independent evaluation/control rows. Each row retains `profileId`, `userGroupId`, `caseId` (`profileId + '-' + orderLabel`), `orderLabel`, `sourceProfileId`, `controlName`, canonical-catalog `scores16`, `rawSourceScores16`, `pSpecific`, `v5Scores16`, `fusionPolicy.gateCheckpointSha256`, `rawLearnedOrder`, and `order`. The learned set includes all three public input controls. `shuffled-profile` is the public record name corresponding to root gate `shuffled-public-profile`. Raw LoRA compares raw baselines; combined comparisons blend every control through the same query gate and v5 branch.

## Limits and remaining acceptance

The full quantized native LM projection is not independently re-executed on CPU. Native loss/logit/A/B-gradient and inference-mode equality are checked through actual hashed measured receipts and the fixed policy certificate. Producer `goldParsed:false` declarations establish the admission contract; they are not an external process-attestation service. Self-consistent replacement of all evidence is outside a file-checksum audit's threat model.

Separate independent query-head/vector provenance replay and the 32-query × 10-profile development regression remain pending supplementary review. Selected variants, preference-with-offered-evidence accuracy, actual images, copy factuality, ten-person demo coherence and live first-three/full-six readiness remain pending serving evidence. Product eligibility alone cannot certify those fields. The script never grants overall/default promotion. It always records `overallPromotionPass:false`; exit 2 means completed offline adjudication with acceptance still pending, and exit 1 means admission/audit failure. Threshold success, when present, is explicitly limited to offline core/replacement/history evidence.

## Commands

```sh
/Users/lukas/Documents/Projects/dubhacks-2026/.tools/pretrained-venv/bin/python -B -m unittest experiments.ranker.pretrained.test_persona_choice_combined_audit_v1
/Users/lukas/Documents/Projects/dubhacks-2026/.tools/pretrained-venv/bin/python -B -m experiments.ranker.pretrained.audit_persona_choice_combined_v1 --adapter-run /absolute/new-adapter-attempt --combined-run /absolute/new-combined-attempt --out /absolute/unused-audit-directory
```

Preparation validation: 16 fake CPU tests passed. They include fixed denominators, whole-group/label joins, empty shortfalls, observed-ownership consistency, replacement semantics, all-rank budget/typed-color checks, strict query budget precedence, malformed JSON/artifact paths, full 264-row fusion verification, missing-stage rejection, correctly rehashed but arithmetically wrong blends, correctly recomputed blends using a different v5 control vector, ten-pair history joins, and a failure-path assertion that private labels are never opened before admission. No native quality result is claimed.
