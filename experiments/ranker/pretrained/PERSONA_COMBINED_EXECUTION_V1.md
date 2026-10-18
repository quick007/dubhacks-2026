# Combined prediction preparation

No native execution, service launch, private evaluation parse or quality promotion
was performed by this CPU preparation. All existing LoRA1240/3584 sources and
native output remain untouched. The independent adjudicator remains separate.

The frozen public query plan is
`data/ranker/pretrained/persona-combined-query-prepare-v2/plan.json`, SHA
`8db50d9063df26f1662c705bee79e7fe73ddf42c58b30d13c52ec5897591961a`.
It projects all344 nonfit order-specific public rows from full plan44a7bb... plus
all320 historical public requests from plan39c4d288.... TRAIN rows, targetWeights,
loss controls and other original feature fields are excluded from its rows.
The encoding uses the actual checked public request -> ranking_input ->
request_text chain, including budget and hard-constraint normalization.

There are36 unique queries for the344 combined rows and256 unique historical
queries, with zero shared texts:292 required pristine-base B1 forwards total.
All292 whole-query preflights are64–82 tokens, under the512 cap. Catalog order,
background and history do not enter this original v5 query feature; the complete
actual public context remains in the separately trained LoRA decision path.
Historical features are query-only evidence. They cannot stand in for the missing
historical LoRA X/H/Y or claim320 combined regression cases passed.

After the parent releases the current native owner and completes the gate, the
parent may execute this explicit command from the integration worktree. This
preparation has not executed it:

```sh
/Users/lukas/Documents/Projects/dubhacks-2026/.tools/pretrained-venv/bin/python \
  -m experiments.ranker.pretrained.store_persona_combined_query_v1 \
  --run --owner root \
  --plan /Users/lukas/Documents/Projects/dubhacks-persona-integration/data/ranker/pretrained/persona-combined-query-prepare-v2/plan.json \
  --plan-sha256 8db50d9063df26f1662c705bee79e7fe73ddf42c58b30d13c52ec5897591961a \
  --out /Users/lukas/Documents/Projects/dubhacks-persona-integration/data/ranker/pretrained/persona-combined-query-v1-attempt-1
```

The extractor refuses existing output directories, non-root native callers,
serving listeners8766/8767 and a held original asset-root native lock. It fresh
hashes the original local weights, installs no adapter, freezes927 source tensors,
requires whole exact token IDs and finite unit-L2 query4096, records attempted,
physical and saved counts separately, applies7GiB active/peak and600s deadline,
and verifies all927 plus source pins before releasing the lock. Failed rosters
and partial bundles are preserved; the writer rejects partial proof. The default
CLI without `--run` only checks CPU pins and returns counts.

`store_persona_combined_predictions_v1` is CPU-only. It requires explicit raw run,
full plan+SHA, completed query run+planSHA, reviewed query-intent v3 head+SHA,
v5 serving export+SHA, original16 catalog feature root, independent PUBLIC packet
manifest+SHA and unique output directory. The query gate TRAIN manifest is fixed
at25fdb0c976dc3b6e01e826e7f30ea63fc65e3fbb85f56f24f4c83e8538703232.
Its expected completed raw run is
`data/ranker/pretrained/persona-choice-train-v1-attempt-1`; no completed artifacts
are presumed during preparation. Gate-head output path and checksum are pending
actual native extraction and fixed CPU fit. The writer accepts no private gold
path and imports no evaluator or native model.

Each of learned, frozen-base-lexical-decision, TRAIN-popularity and
latest-owned-repeat contributes its original frozen canonical16 scores. Every
model is blended with the SAME learned pSpecific and SAME v5 scores for each
actual query, using the runtime float32 formula, then the SAME existing hard
budget/color/stock/size/waterproof/exact-name guards. Canonical/reversed share
query scores; observed-profile input controls retain their original source query.
No source winner is selected after inspecting outcomes.

Each model file contains the exact fixed344 public rows, including all264
independent evaluation/control rows. Each row preserves source identity/split/
control/catalog order, canonical public-requestSHA, rawSourceScores16, pSpecific,
v5Scores16, final scores16, fusion policy+gateSHA, raw stable order and guarded
order. `copyGenerated:false` and `variantsOrImagesScored:false` are explicit.
No variant/image/copy quality is manufactured from product ordering.

`ALL_MODELS_PREDICTIONS_FROZEN.json` seals all four model file SHAs, the complete
case list, independent packet/adapter/gate SHAs, absolute source/artifact pins,
`goldParsed:false` and `stagesCompleted:{adapter:true,queryGate:true,allControls:true}`.
`QUERY_REPLAY.json` separately preserves pSpecific/v5 by exact query-textSHA and
native vector/gate proof pins. These are prediction/provenance receipts, not
quality results. The independently frozen auditor decides quality only after
admitting the raw and combined freezes; variant/live/320-LoRA gates remain pending.

CPU tests use injected fixtures, never device/model/socket work:

```sh
/Users/lukas/Documents/Projects/dubhacks-2026/.tools/pretrained-venv/bin/python \
  -m unittest experiments.ranker.pretrained.test_store_persona_combined_v1 \
  experiments.ranker.pretrained.test_store_persona_adapter_v2 -v
```

They cover full344 projection, exact runtime text and normalized budgets, no
implicit native execution, all-model shared fusion/identity guards, source roster
and checksum rejection, completed927 query proof admission, and a complete fake
344-row/four-model writer freeze using the actual trainer baseline key names.
Fake proof fixtures validate orchestration only and make no native quality claim.

CPU repair preservation: all15 original frozen files/plan receipts were copied
byte-for-byte under `persona_adapter_cpu_preserved_v1/source` before editing.
The original CPU freeze and query-prepare-v1 bytes remain unchanged. Query v2
public rows, text, tokens, order, model and recipe equal v1 exactly; only the
repaired ownership contract source checksum differs.
