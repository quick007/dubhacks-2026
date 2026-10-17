# Bounded corrective fit and grounded evidence-plan comparison

The preserved baseline is [RESULTS.json](RESULTS.json); [DIAGNOSIS.json](DIAGNOSIS.json) records the CPU audit before this increment. No previous result or checkpoint is replaced.

One fixed corrective fit uses the same actual cached 4096-dimensional 9B readouts and all 24 training requests together. It changes pointwise product regression to per-request best-product cross-entropy, makes presentation cross-entropy conditional on compatible informative products, and trains the router with inverse-frequency weights (NONE/SELECT/GENERATE = 96/72/24). Seed, 80 epochs and learning rate 0.03 are fixed in source before evaluation. Replay/objective/weighting changes are bundled, so no single-cause improvement is claimed. Decoder weights remain frozen.

The evidence-plan interface projects trained product selection/ranking and per-product image-caption decisions onto exact catalog fact IDs and relevant high-confidence observed-save reason IDs. It is a deterministic grounding layer, not a trained evidence or layout planner. Opaque IDs remain outside predictive decoder text; verified semantic bindings condition new language. Private persona fields and future outcomes are excluded. Missing history/preferences remain unknown.

Plan and no-plan arms use the same resident backbone, corrected heads, greedy sampling and fixed 64/128/224-token budgets for 1/3/6 cards. Each arm performs a fresh complete prefill and consumes its complete native recurrent/convolution/KV state exactly once. Arm order alternates. No cache trimming, splicing, independent field decoding or speculation is introduced. Titles, optional supported badges and section headings are actual generated fields with independent fail-closed validation; no template is counted as generation.

The roster is the eight prior fixed cases, one explicit no-history/no-preference NONE control, and 1/3/6-card scaling cases from a fixed six-candidate held-out catalog pool. Every case is retained. Scale cases use stated budget 200 and no hard requirements to allow six available cards; those altered controls are not part of the original private-rubric ranking benchmark. Full timing includes every heading/card generation, failures and validation. One sample per scale is not a p95 measurement. Product uniqueness, attribute coverage, pairwise tag redundancy and explicit preference coverage are reported alongside factual validation and private held-out grading.

```sh
.tools/pretrained-venv/bin/python -m unittest experiments.ranker.pretrained.test_evidence_plan -v
.tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.evidence_plan train
.tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.evidence_plan benchmark --model /absolute/pinned/local/snapshot
```

CPU training has completed: ranking/presentation loss 2.080→0.1634; weighted router loss 1.098→0.2229; real gradients, changed weights and exact reloads verified. **Real comparison results are pending** at this increment. Passing training loss or stub tests does not establish quality or serving speed.

[SchemaFill](https://arxiv.org/abs/2610.07086) already investigates slot-parallel speculative tool-call generation with target verification. This experiment tests whether shared grounded planning improves our small personalization workload. It does not establish novelty or implement SchemaFill. Parallel title/badge decoding is deferred until evidence justifies it and native state semantics/memory support it. The prior roughly three-second single-product result does not meet the near-traditional-page latency target.
