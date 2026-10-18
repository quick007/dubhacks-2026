# Learned query-intent gate v2: frozen TRAIN plan

This gate supplies a continuous `pSpecific` from the existing frozen Qwen v5 query feature. It adds a 4,097-parameter binary decision layer; the Qwen backbone and decision/copy models remain the main model. Runtime should reuse the already computed **base-adapter-disabled** 4,096-dimensional query vector before the learned v5 ranking projection. There is no additional model forward, profile feature, query whitelist or per-person rule in this gate.

The feature text is exactly `request_text(ranking_input(checked_context_request(public_request, actual_catalog)[0]))`, including budget and typed hard requirements. The literal JSON query is a generic wrapper; the actual shopper query appears in `preferences[0]`. Classifying that generic wrapper alone would discard intent. Background/history never enter this feature text.

## Data audit and rejected v1

The only new public sources are the frozen 320 original-plus-contrast TRAIN sessions and 128 public TRAIN expansion sessions. They provide 63 distinct broad query phrases. The preserved v5 native packet supplies 96 distinct specific TRAIN phrases. Its 96 feature records reconstruct the runtime text byte for byte and have verified unit-L2 native vectors.

The first plan deduplicated full requests into 118 broad and 96 specific texts. Its classes were perfectly confounded: every broad text had a numeric budget, while every specific text had no budget; typed colors occurred only in broad examples. That plan and source remain unchanged, with `QUERY_INTENT_GATE_V1_TRAIN_REJECTION.json`. No extraction or fit was performed for v1.

V2 crosses **each of the 159 TRAIN phrases with the same eight contexts**: budget `{null,50,100,200}` times hard color `{none,Navy}`. The actual runtime parser reproduces all requested contexts without adding a query-derived guard or changing a budget. Every context cell contains 63 broad rows and 96 specific rows, so class-conditional normalized budget/color distributions match exactly.

There are 1,272 unique runtime texts: 504 broad and 768 specific. Exactly 96 text keys reuse the verified v5 TRAIN vectors; the remaining 1,176 need fresh native features. Token counts are63–84. These are 159 phrases with repeated matched contexts, not1,272 independent shopper queries. Intent labels remain broad/specific even when a budget/color combination leaves no eligible product; the label concerns the request's specificity, not fulfillment or product relevance.

The broad class means an authored, under-specified addition request that benefits from public context. Some broad phrases mention activities such as reading, writing or travel. The specific class comes from v5's explicit product/function requests. Exact phrase/text classes are disjoint, but the semantic boundary is graded. The classifier therefore emits a soft probability, and quality must be assessed in the full blended pipeline.

## Frozen model and controls

The fixed architecture is a CPU float32 linear4096→1 plus sigmoid. Recipe: seed20261019,250 full-batch AdamW updates, learning rate.03, weight decay.001, one CPU thread, loss equal to half the mean BCE of each class. Only the final checkpoint is retained as the candidate; no validation, threshold tuning or early stopping uses evaluation outcomes. Runtime consumes the continuous probability.

`store_query_intent_gate_v2.py` provides `load_intent_head(path, expected_sha256, expected_source_manifest_sha256=...) -> (head, metadata)` and `specific_probability(head, raw_query_feature) -> float`. Shape, finiteness, source manifest and unit-L2 feature contract are checked. No head checkpoint exists until the parent performs native extraction and the declared fit.

The frozen plan is `query_intent_gate_prepare_v2/plan.json`, SHA`eb9e662fb27ad5c65745d3291bdc7691113575aacd6a0dcecb7f4de53a754d07`. Native extraction refuses serving listeners and requires the shared main native-owner lock through927-tensor verification and shutdown, a7GiB process guard, a300-second lifecycle cap and a new output directory. It loads the frozen base model, adds no adapter and preserves all reused features. Feature tensors and head weights stay local rather than entering Git.

After extraction, the parent can run `store_query_intent_gate_v2 fit --features <native-run> --out <new-fit-run>`. The separate frozen `store_query_intent_diagnostics_v2` reports class-balanced TRAIN accuracy/counts, budget/color-conditioned accuracy, per-query probability range across all eight contexts, broad/null-budget cases and specific/numeric-budget+guard cases. A fixed.5 cutoff is descriptive reporting only. It never chooses a threshold or changes training. Six gate and two diagnostics CPU tests pass; no actual model fit was performed by this lane.

## Required evaluation boundary

Freeze the intent checkpoint before encoding/scoring future public query cases. Freeze those predictions before reading private pipeline-quality labels. The original32 v5 specific queries under profile constraints are development regression tests. The separately authored future public broad cases and replacements must be judged through both raw LoRA and the actual continuous-query-gated pipeline, with identical catalog/constraint guards and source-factual output contracts. Raw LoRA quality or binary TRAIN accuracy alone is insufficient for promotion. Controls need the same query gate and preserved query/guards; a control that bypasses the v5 branch measures a different pipeline.

This preparation read no future public32 packet, future private quality labels or checkout outcome. No GPU, browser, database mutation, Git operation, feature extraction or candidate fit was executed by this lane.
