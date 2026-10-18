# Compact fresh AR candidate

Prepared only: no native execution or training. All six cards and twenty fields
are required; candidate caps total128 sampled tokens. Generated titles2–4 words,
bodies2–5, badges1–3 and headings2–4 replace the old output contract explicitly.
Original identities remain metadata. Only selected public name/color/price and
public plain/warm style condition wording; neutral query strings schedule the
fixture and are not generation inputs. No teacher string or fallback is emitted.

`store_compact_copy_v1/` contains32 public cases, separate evaluator facts and
CPU tokenizer receipts. All48 product/color states occur four times. Real
tokenizer bounds: common root88, fact prefix≤118, final suffix≤15 tokens; badge
color evidence fits with EOS. `token-preflight.json` is the current exact
receipt; `*.prepare-*.json` files preserve superseded CPU preparation receipts.
Use a fresh preflight if any listed source changes.

The default `diagnostic8` plan uses predeclared case indices0,1,8,9,16,17,24,25:
four paired styles, all16 products, frozen/adapted modes and fact-prefix reuse
only,16 fixed attempts. `full32` runs both common-root and fact-prefix paths,
128 attempts. Neither plan drops failures. Exact same-width fork/rebuild/repeat
controls run first; cross-path numerical equality is not assumed. One common
root plus a two-entry fact-prefix LRU bounds cached native states. Every ready
endpoint includes actual validation/UTF8 serialization. It excludes ranking,
startup, images and browser paint; common-root preparation is disclosed
separately. Source-cache proof after readiness also has its own wall receipt.

Root owns GPU scheduling and supplies a resident `FrozenBackbone` under7 GiB
with an attached `LoRALinear.from_base(..., r=4, scale=20, dropout=0)` at the
final down projection. The API loads the exact completed320-update actual-store
adapter; frozen mode temporarily restores its base linear. It checks927 frozen
tensors and adapter bytes. No training, model construction or service edit occurs
inside this API. Example inside the root's resident native process:

```python
from experiments.ranker.pretrained.store_compact_copy_native_v1 import run_native_comparison

run_native_comparison(
    bb, adapter,
    'data/ranker/pretrained/actual-store-copy-fit-v2-attempt-1/adapter.safetensors',
    '56c6c1f4199d8689f1920d871213a46d2d121fec4e5836ff70e33aaf98c7c7b8',
    'data/ranker/pretrained/store-compact-copy-v1-diagnostic8-attempt-1',
    'experiments/ranker/pretrained/store_compact_copy_v1/token-preflight.json',
    native_owner='root', confirm_serialized=True,
    reviewed_preflight_sha256=REVIEWED_PREFLIGHT_SHA256,
    plan='diagnostic8',
)
```

Do not hold another native-owner lock around this call: the API acquires the
shared lock itself. Serialize it with all live-engine GPU work. Use a new output
directory for each diagnostic/full attempt. The gate is a bounded lexical fact
screen, not comprehensive semantic proof; independent blind judgments of useful,
grammatical and factually supported text remain required.

CPU checks: `.tools/pretrained-venv/bin/python -m unittest
experiments.ranker.pretrained.test_store_compact_copy_v1`. The older strict
catalog-source test can still reject the known reviewed optional ProductCard
renderer exception; current engine binding uses the separate v3 export-equivalence
manifest. That older rejection does not establish a new dirty renderer change.
