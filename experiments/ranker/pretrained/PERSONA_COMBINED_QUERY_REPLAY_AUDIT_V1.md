# Query, v5 and combined-score replay supplement

This separate CPU audit leaves the frozen quality adjudicator and active trainer pins unchanged. It contains no optimizer, native-model load, browser or private quality-label reader. Its fake tensor tests do not establish actual model quality.

Before candidate outcomes, it fixes query plan `8db50d9063df26f1662c705bee79e7fe73ddf42c58b30d13c52ec5897591961a` and query TRAIN plan `25fdb0c976dc3b6e01e826e7f30ea63fc65e3fbb85f56f24f4c83e8538703232`. Actual artifacts are not available at preparation time. The input must already have a completed all-model prediction freeze. The script checks its artifact hashes and requires the separate mechanical query TRAIN audit's completed 1,272-row, 250-update replay certificate for the same head. It does not rerun that optimizer.

The supplement verifies the completed 292-forward, no-adapter native query receipts, all 927 before/after hashes, shutdown, the exact 664 public-row roster and 292-vector bundle. It reconstructs whole query text and tokenizer IDs from the public request; history/persona cannot silently enter this query path. It independently computes the linear head sigmoid and v5 mean-center/L2/linear/softmax/catalog-query matrix product from saved state tensors and the original sixteen verified Qwen catalog vectors. It requires exact saved probability and v5 vector equality. It replays all four models' 344 combined rows (1,376 total), checking float32 blend arithmetic, public identities, raw-source vectors, ranking guards and exact-name policy, catalog ordering, and flags that withhold any variant/image/copy claim.

The 320 historical query-only rows contribute their actual query feature provenance to this mechanical replay. They do not contain LoRA context decisions. This script explicitly leaves the 32-query × 10-profile behavioral regression pending and never claims that merely encoding those queries passed the regression.

## Certificate for the pending HTTP probe

The script writes `audit.json` and, after successful replay, `INDEPENDENT_QUERY_VALUES.json`. The HTTP probe should pin the entire `audit.json` digest and require:

```text
version: persona-combined-query-replay-audit.v1
status: passed
gateCheckpointSha256: <actual selected gate>
v5HeadSha256: <actual selected v5 head>
adapterCheckpointSha256: <same adapter as all-model freeze>
queryBundleSha256: <actual292-vector bundle>
combinedFreezeSha256: <same combined model freeze as quality adjudication>
mechanicalQueryAuditSha256: <completed mechanical TRAIN audit>
rows: 344
modelRowsChecked: 1376
uniqueQueryVectorsReplayed: 292
exactQueryProbabilityAndV5Parity: true
actualFloat32BlendAndGuardsExact: true
privateQualityLabelsRead: false
optimizerUpdates: 0
nativeCalls: 0
overallPromotionPass: false
servingPending: true
specific320BehavioralRegressionComputed: false
inputSha256: {absolute saved artifact paths: hashes}
sourceSha256: {absolute source paths: hashes}
```

Recheck the input/source pins when consuming the certificate. A pass covers saved mechanical score consistency. It grants no default promotion or HTTP/browser readiness. Quality admission, mechanical query audit, this supplement and the actual selected runtime artifacts must all identify the same models before a serving probe can start. Variant/source-image/copy correctness, ten-person coherence and complete visible first-three/full-six readiness require actual serving evidence.

## Commands

```sh
/Users/lukas/Documents/Projects/dubhacks-2026/.tools/pretrained-venv/bin/python -B -m unittest experiments.ranker.pretrained.test_persona_combined_query_replay_v1
/Users/lukas/Documents/Projects/dubhacks-2026/.tools/pretrained-venv/bin/python -B -m experiments.ranker.pretrained.audit_persona_combined_query_replay_v1 --combined-run /absolute/combined-attempt --query-run /absolute/query-attempt --gate-head /absolute/gate-fit/head.pt --v5-export /absolute/v5-export.json --mechanical-query-audit /absolute/mechanical-audit/audit.json --out /absolute/unused-supplement-attempt
```

Seven CPU tests passed: independent sigmoid/shape/L2 boundaries; exact agreement with the original v5 head math on explicit fake tensors; mechanical-certificate counter/head/hash tampering; full 344-row replay; incorrect probability/v5/identity/source claim; wrong blend/guard after rehashing; and failed admission retaining serving-pending without reading quality labels. Actual execution must wait for the frozen saved artifacts.
