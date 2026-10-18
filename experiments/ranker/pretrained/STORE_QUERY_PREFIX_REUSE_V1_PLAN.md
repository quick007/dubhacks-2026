# Frozen-base query prefix reuse candidate

Prepared only. No native calls, model changes, live encoder patches or measured
speed claims. Root queues an isolated run after the v5 cohort resource window.

`store_query_prefix_reuse_v1.py` caches only the fixed public formatter header,
budget-scope constant and opening request key. It withholds one token at the
literal boundary, fixed before examining request wording. The local tokenizer
produces 26 reusable tokens; all 32 public v5 evaluation requests extend those
exact IDs. Each inference still formats and tokenizes its complete fresh request,
checks actual prefix IDs, and encodes every remaining token. No request output or
query feature is cached. Budgets, constraints and preference text remain fresh.

Root supplies the same resident Qwen backbone and explicitly attached copy
adapter. Preparation and each query temporarily disable that adapter, preserve
eval mode, and call `model.model` without the vocabulary head. Each B1 suffix
forks all 32 cache entries using installed `from_state(deepcopy(full.state))`:
24 `ArraysCache` entries retain both convolution and recurrent arrays plus
padding/length metadata; eight `KVCache` entries retain keys, values and offsets.
No recurrent trimming or rollback is required. The installed Qwen implementation
returns normalized hidden states and accepts existing recurrent state, so the
same last-token float32 cast, L2 normalization, NumPy conversion and Torch clone
as `FrozenBackbone.encode` can be applied to the last suffix token.

The runtime supports the required state representation. Numerical parity across
the split remains unmeasured: GDN/attention kernels can choose different sequence
paths and accumulation order. Native controls must distinguish bitwise equality
from the predeclared numerical gate (cosine ≥0.99999, max feature error ≤0.001),
and require unchanged all16 ordering, guards, exact-name policy, selected actions
and score error ≤0.001 for every one of the 32 fixed public requests. These
thresholds are fixed before native evaluation; no labels or teacher targets enter
the gate. This remains development replay, not unseen generalization. Additional
public budget/color/stock/size/exact-name cases should be reported separately.

Asset binding hashes actual model/config, tokenizer, source, catalog-feature and
head files at preparation and again after controls. It binds the verified digest
of all 927 frozen parameters. Timed queries check asset size/mtime, model identity,
base-adapter mode and exact prefix tokens; full SHA checks are outside request
timing and are not described as cryptographic per-request verification. Full
source cache hashes before/after each suffix prove the root stayed unchanged.

All overhead is exposed: complete-request tokenization, asset stats, source cache
proofs, full state clone/materialization, suffix forward/normalization and host
Torch copy. A feature-ready timestamp before final proof is separate from total
encode time including proof. Prefix preparation and its one native call are
reported separately. Copying roughly 48 MiB of recurrent state for only 26 saved
tokens may cancel any gain; no below-193 ms result is projected as a measurement.

Root-managed API sketch (no loader or native lock inside this helper):

```python
# Under root's shared native-owner lock, after the cohort window:
binding = freeze_assets(
    {"model": model_files, "tokenizer": tokenizer_files,
     "source": source_files_including_installed_cache_and_qwen,
     "catalogFeatures": verified_catalog_tensor_and_receipt_files,
     "head": [v5_export, v5_head]},
    frozen927_hashes_sha256=hashed(verified_original_927_hash_map),
    model_revision=engine.bb.revision,
)
encoder = QueryPrefixReuse.prepare(
    engine.bb, binding, copy_adapter=engine.adapter,
    native_owner="root", confirm_serialized=True,
)
# Read only frozen public-cases.json; filter its32 evaluation rows and project
# each to {caseId,request}. Never open evaluation-labels.json.
result = development_controls(
    encoder, fixed32_public_rows,
    lambda feature, request: score_function_head(engine.session, catalog16, feature, request),
    out=NEW_CONTROL_DIRECTORY,
)
```

The 32-case control requires 64 fresh request forwards plus one prefix prepare.
It compares full cache=None and reused-prefix encoding in alternating order and
preserves failed/unattempted arms. Full asset hashes, all927 parameter hashes and
the root state must still match afterward. Interpret speed only if every gate
passes; then independently measure the full serving/browser path. The existing
v5 cohort and pinned model remain unchanged throughout this preparation.
