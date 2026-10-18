Short-copy v1 is a frozen synthetic supervision packet for a new raw focused-prefix copy experiment. It contains 12 new public TRAIN requests on the existing TRAIN-only catalogs (240 fields), plus the 3 already-inspected dev and 3 already-inspected test cases (60 fields each). Evaluation cases are never training rows.

Each case contains the exact untemplated v2 focused query text, one shared native user message with the full observable request/history and copy rules, and 20 independent field jobs. Each job has one native user message binding all selected public facts and its field rule. No opaque case/product identifiers, hidden persona, target strings, or previous generated field values are model inputs. Public product identity is retained for factual copy. Full context is preserved; no token truncation is used.

The field order is `S1,T1,B1,C1,T2,B2,C2,T3,B3,C3,S2,T4,B4,C4,T5,B5,C5,T6,B6,C6`. The first 10 fields complete the first 3 cards (208 output tokens maximum including native EOS); all 20 complete 6 cards in 2 sections (416 maximum). The unchanged strict three-card validator checks each true section, with a metadata-only index projection for the second section. A valid first section does not imply full-six completion. Native receipt checks remain mandatory for model output.

All 18 copy support atoms have positive TRAIN body targets; 72 TRAIN body targets and 40 exact `NONE` badges are present. Section headings use only a fact shared by every section member, or neutral travel framing when the intersection is empty. Every body stands on its own and mentions an observed supported atom when one is available. Teacher strings are training targets; they must never be returned as serving fallbacks.

`TOKEN_PREFLIGHT.json` records the local tokenizer/proxy checks: raw roots 18–20 tokens, common append 333–336, complete suffix 113–219, target including EOS 2–15. The 240 training feature tensors project to 3,831,644,160 raw float32 host bytes before archive compression. A runner must declare and enforce its own storage/memory budgets rather than reuse the old 2 GiB archive limit implicitly.

Run CPU checks from the repository root:

```sh
.tools/pretrained-venv/bin/python -m unittest experiments.ranker.pretrained.test_short_copy_data -q
.tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.short_copy_data check
```

The runner-facing APIs are `load_dataset(DATA) -> manifest,casesBySplit,fieldsBySplit`, `build_layout(plan,request,catalog) -> commonMessage,fieldRows,evidence`, and `validate_rows(...,require_native_receipts=True)`. Raw query features are prepared separately with the new `fused_prefix` helper; append `commonCopyMessage` once before forking field branches. The new runner must verify frozen source/data hashes, native full-cache equivalence, adapter mode compatibility, exact nonzero A/B gradient parity and source preservation, and the reloaded learned checkpoint before reporting measured outputs.

No model was executed, no fit ran, and no latency or learned quality was measured while preparing this packet. Native gates and actual fitting remain root-owned. Locked product/image decisions make this a copy-stage experiment; it establishes no ranking, learned grouping, browser rendering, conversion, or jacket-domain result. The seven-atom ranking model is a separate component from these eighteen-atom copy fixtures.
