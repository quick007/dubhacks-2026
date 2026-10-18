# Conventional LLM comparator preparation

This compares a conventional single autoregressive LLM response against the
resident hybrid demo. It is prepared for later authorized native execution.
No comparator GPU execution has occurred. The resident service owns the native
lock, and the parent-reported approval is still unresolved. The preparation CLI
cannot launch native execution or stop the resident service.

The primary comparator uses the same local frozen Qwen3.5-9B four-bit backbone,
warm on the same M5, with a fresh request prompt containing all 16 public
catalog products, prices, categories, search terms and declared colors. It sees
the identical public shopper request used by the paired hybrid receipt. It
chooses six distinct products and emits two three-card sections in one ordinary
greedy autoregressive JSON response. The response includes product decisions
and all 20 fresh heading/title/badge/body fields. Model loading is excluded;
complete request preparation, prefill, generation, guards, validation and JSON
serialization are included. There is no template fallback, retry, speculative
decoding, cached generated answer or reduced two-card workload.

The baseline can expose a complete first section as soon as its actual JSON
object closes. The parser does not insert braces, strip Markdown fences,
recover substrings or repair text. First-token time is a diagnostic only.
First-three readiness requires all three selected cards and their ten fields;
full-six readiness requires six cards and all 20 fields. Malformed output,
ineligible or duplicate selections, exhaustion and timeouts retain raw text,
tokens and failure elapsed time. Any valid earlier first-section readiness is
retained even if the later output fails.

Both paths apply the same deterministic public hard guards and bind the same
first satisfying declared color variant. Baseline decisions remain in their
generated order; the comparator does not reorder, substitute or pad products.
Copy uses the same prospective practical source-fact screen, with unchanged
strict v2 text judgments reported separately. A completed JSON string replaces
the hybrid's independent-field EOS boundary: no per-field native EOS receipt
is invented for the conventional stream. A full useful completion also requires
the actual whole native stream to end with EOS.

The prepared operational `usableCopyProxy` requires an accepted fresh title
and body for every displayed card, along with complete hard-eligible actions;
headings and badges may abstain. Report this proxy and all-field acceptance
separately from independent human judgments of relevance and usable copy. The
hybrid receipts must be scored with the same proxy for comparison. No quality
or latency winner can be claimed from preflight or first-token timing.

Important comparison tradeoffs:

- The primary conventional arm is frozen base; the hybrid uses a trained
  ranking head and the 320-update copy adapter. This is a system comparison,
  not an isolated parallel-decoding speedup. An optional installed-copy-adapter
  arm must be labelled separately and never pooled with frozen-base results;
  that adapter was trained for separate bound fields, not joint JSON selection.
- Hybrid query preferences affect selection while each copy prompt sees only
  bound source facts and a neutral browsing request. The baseline receives the
  same neutral-copy instruction but sees the shopper query and full catalog
  in one context. This structural difference is disclosed rather than hidden.
- Baseline color and gallery binding remain deterministic metadata policies,
  matching the hybrid; no learned image or pixel claim is made.
- Baseline has a global cap of 1024 generated tokens including JSON structure
  and selection names. Hybrid has up to 416 independently bounded copy tokens
  plus its separate rank payload. Report actual tokens and cap exhaustion;
  these are equal functional output workloads, not equal token counts.
- Shortfall cases with fewer than six eligible products are a separate labelled
  workload. Their smaller card counts cannot support a six-card latency claim.
- Existing hybrid demo HTTP timing can include queue/transport dispatch; the
  callable comparator starts at a resident function request. Report both scopes
  honestly. A controlled follow-up can call both under the same resident timer
  once native execution is approved and the service is stopped gracefully.

CPU preparation accepts only public `{caseId,request}` cases, rejects hidden
label fields, records catalog/source hashes, and optionally pairs exact-input
hybrid receipt hashes. `--hybrid-glob` includes every matched receipt, including
repeated requests, without choosing the latest or fastest one. Paired readout
uses the archived as-served fields and nulls; it does not retrospectively apply
a changed gate. It records explicit card/field counts, title/body proxy passes,
all-field acceptance and complete readiness times separately. Pair current
`store-demo-v3` receipts with the prospective ambiguous-price-scope safeguard;
the older v1 receipts are preserved in preflight v1 and are not pooled with v3.
Source hashes cover the current engine, baseline, backbone and copy modules;
checkpoint provenance records the verified copy file and completed-fit pinned
backbone hashes. This CPU preflight does not freshly rehash the large shards.
Tokenization is local and CPU-only:

```sh
.tools/pretrained-venv/bin/python -B -m experiments.ranker.pretrained.store_autoregressive_baseline \
  --requests /path/to/public-requests.json --out /path/to/new-preflight \
  --local-tokenizer --hybrid-glob 'data/ranker/pretrained/store-demo-v3/requests/*/receipt.json'
```

The later native owner may call `run_resident_case` with a verified resident
backbone, the public snapshot, an unused output directory and an explicit model
mode. It must first resolve the pending approval, stop the service gracefully,
verify its shutdown receipt, acquire the shared native lock, and serialize
timing jobs. The callable neither loads weights nor acquires/bypasses that lock.

CPU tests:

```sh
.tools/pretrained-venv/bin/python -B -m unittest experiments.ranker.pretrained.test_store_autoregressive_baseline -v
```
