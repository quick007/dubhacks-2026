# Practical warm performance: independent saved-evidence audit

The saved warm measurements and parity checks pass. This is a technical development result; the adapter's prior quality gate failed and V4 factorized conditioning remains unvalidated.

| Endpoint | Reference p50 / p95 (ms) | Warm p50 / p95 (ms) | Samples per arm |
|---|---:|---:|---:|
| Backend complete | 288.65 / 307.00 | 66.89 / 71.31 | 10 |
| HTTP exchange | 303.80 / 321.77 | 81.68 / 85.51 | 10 |
| Browser visible three cards | 335.75 / 360.99 | 118.00 / 120.70 | 10 |
| Browser full six cards | 344.85 / 376.37 | 120.60 / 135.76 | 10 |

Quantiles were independently recalculated by linear interpolation at `(n−1)q`. The ten backend pairs contain five AB and five BA orders. Four additional paired hard controls are excluded from the timing denominator. Twenty browser requests were scored; 21 bootstrap/reset posts were explicitly unscored.

The audit joined all 48 scored server receipts to the raw HTTP/browser events, rehashed receipt/source evidence, checked the full 4096 query feature bytes and all 16 route/v5/fused scores, and compared complete ordering, actions, variants, guard receipts, generated fields and raw word IDs. All comparisons are exact. All 254 selected-card records match the public catalog's names, categories, prices, variant colors and primary image actions. Budget/Graphite, exact-name, empty-budget and unsupported-stock/size controls pass without padding.

Every warm public-profile request reports zero native query forwards, one fresh rank projection and one fresh parallel copy-head forward. Reference requests perform one fresh query forward. Neutral profile/color features are shared preencoded inputs in both arms. Each request still computes decisions, fusion, variants and copy. The combined speedup comes from exact query-feature reuse plus selected-row LM projection; these samples do not isolate their individual contributions.

Startup encodes eight distinct exact query/budget/guard keys for ten public default Browse profiles in **1371.86 ms**, with eight native B1 forwards. Key reconstruction confirms literal queries, budgets and effective hard guards are preserved. Separate neutral-profile preparation costs **17326.55 ms**; unused rejected-prefix preparation costs **7087.12 ms** and selected-projection preparation **6.92 ms**. The base constructor's **11845.14 ms** timer has a partial scope and is not complete READY latency. These timers have distinct scopes and should not be blindly summed.

Browser measurements use a loaded, prescrolled collection with HTTP/image caches eligible in both arms. Three cards are fully visible, unobscured and decoded, with ten generated fields matching the DOM. Full six has 20 generated fields and 19 rendered fields; the second section heading is backend-only. There are no scored page/request errors. These are observer timings from fetch, not physical paint. The initial viewport has zero useful fully visible cards. Novel cold-query and unknown-profile latency are not established by this workload.

No same-lifetime shutdown receipt existed when this audit ran. Startup records 927 frozen tensors; final frozen-model/adapter teardown integrity remains pending a later shutdown receipt. This audit does not run a model, HTTP request, browser, optimizer or Git operation.

Reproduce the CPU-only audit:

```sh
/Users/lukas/Documents/Projects/dubhacks-2026/.tools/pretrained-venv/bin/python -B \
  experiments/ranker/pretrained/audit_persona_practical_results_v1.py
```

The JSON receipt pins both original reports, raw backend bodies, all 48 server receipts, public catalog files and source inventories. Original experiments remain unchanged.
