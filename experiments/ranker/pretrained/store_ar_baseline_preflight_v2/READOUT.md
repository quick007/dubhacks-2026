# Comparator preparation and paired hybrid readout

Native comparator execution has **not** occurred. CPU tokenization and all
14 parser/guard/preflight tests pass. The preparation CLI cannot start GPU
execution. The reported native approval remains unresolved; the active demo
service is left running and its ownership lock is untouched.

Preflight SHA-256:
`382f1326a61a143aaa30bfe31c95667fc403fda81e96d443a5956a0e906ee1ac`

Five public cases are frozen: water, bedside light and notebook each require
six cards and 20 copy fields; waterproof requires one card/four fields; exact
Trail mug under $20 requires zero cards. The complete 16-product prompts are
1334, 1335, 1334, 1333 and 1338 tokens respectively. Each fits the 3072-token
prompt cap plus 1024-token ordinary AR output cap within the 4096 context cap.
Nothing was truncated and no product was removed from the prompt.

All five matched receipts under `store-demo-v3/requests/*/receipt.json` are
included: two water, two bedside-light and one notebook request. Older v1
outputs remain preserved in preflight v1 and are not pooled here. The v3
readout uses the actual archived as-served nulls and field counts, including
the prospective safeguard against ambiguous collection/item-price wording;
it does not retrospectively repair or re-gate fields.

| Paired hybrid endpoint | Attempts | Cards per attempt | Accepted fields | All-field passes | Title/body usable-copy proxy | Median attempted readiness |
|---|---:|---:|---:|---:|---:|---:|
| First section | 5 | 3 | 44/50 | 0/5 | 0/5 | 5.432 s |
| Full payload | 5 | 6 | 90/100 | 0/5 | 0/5 | 6.001 s |

The proxy requires accepted fresh titles and bodies for every displayed card,
plus a complete hard-eligible action roster. Badges and headings may abstain.
Each observed first section and full payload withholds at least one body, so
no usable-proxy latency is reported. These attempted readiness times include
copy and serialization; they are not proof of complete useful generated copy.
The metadata-backed storefront can still work while those fields abstain.
This proxy is deliberately separate from an independent human judgment that a
selection remains directionally useful for the hackathon.

The prepared primary conventional comparator uses frozen base Qwen, emits its
own product decisions and all fresh fields in one ordinary AR JSON response,
and applies the same public hard guards and practical/strict text judgments.
It preserves malformed output, raw tokens, field abstentions and failure time.
First-token timing cannot be reported as first-three readiness. The optional
copy-adapter AR arm is a separate ablation because the hybrid has adapted
ranking/copy modules while the frozen-base conventional arm does not.

The preflight records current engine, backbone and copy source hashes,
checkpoint/report hashes, pinned completed-fit backbone-file receipts and all
paired receipt hashes. The large backbone shards were not freshly hashed in
this CPU-only preflight. Later native execution still requires the verified
resident model, resolved approval, a graceful service shutdown and exclusive
native ownership. No speedup or conventional-baseline quality result exists yet.
