# Model engineering handoff

Worktree: `/Users/lukas/Documents/Projects/dubhacks-2026`, remote
`quick007/dubhacks-2026`, branch `main` only. Preserve teammate changes, all
checkpoints, raw runs, reports and prepared candidate sources. No UI, API,
shopper-harness or browser work is needed for the next model experiments.

## Verified learned ranking result and its limits

`focused-rank-transfer-v1` achieved24/24 top-one choices and12/12 both-correct
preference pairs with changed selections. Prior2/24, random0/24. All144 learned
grid components obey hard guards. This replays the previously inspected
`catalog_fusion/fresh_eval_v1`36-product/24-request dataset. Its earlier failure
informed the new data and representation: this is a development replay, not a
new sealed holdout, unseen-catalog guarantee or human-conversion measurement.

Checkpoint: `data/ranker/pretrained/generalization-v2-attempt-1/focused-fusion.pt`,
SHA-256 `c1adca343738d5d00f606587bdb03c5bb15f08c8e97751d3858e4fd621a50bd3`.
A byte-identical copy is tracked in `focused_rank_transfer_results/v1`.
Training used21 public TRAIN query rows and21 public TRAIN product rows,
balanced seven singleton/pair/triple examples in each role,80 fixed CPU steps,
TRAIN-only role means, and public preference/tag binary supervision. Two learned
4096-to-seven biased sigmoid projections contain57,358 parameters; predicted
query/catalog atom probabilities supply product scores. Actual features come
from the frozen quantized9B backbone. Public positive-atom text preparation, hard
guards, canonical ordering/tie readout and section grouping are deterministic.
The compact representation omits free task/history/negative-preference semantics.

The124.137ms median includes one fresh query native encode, serialization and
verification,36 verified warm catalog feature reads, head scoring, hard guards
and both top12/top24 readouts. It excludes4.232s cold catalog construction,
2.370s resident model load, images, generated copy and browser rendering. Whole
child process13.382s; peak4,649,075,650bytes. See
`FOCUSED_RANK_TRANSFER_RESULTS.md/.json` and the copied raw results/checkpoints
under `focused_rank_transfer_results/v1`; original60featurevectors remain at
`data/ranker/pretrained/focused-rank-transfer-v1/features`.

## Factual-copy experiment at migration boundary

The prior run `data/ranker/pretrained/short-copy-fit-v1` is completed.
Its workers32633/32623 were absent at clean-restart process inventory.
Execution source2baf1e2; frozen preflight `SHORT_COPY_PREFLIGHT_V2.json`,
SHA-25656b62d8cbc469aa66595cdf6a6a8c65b620b5548a21596a24ce85bbefff319e2.
The clean replacement task explicitly authorizes subsequent local experiments.

All240 exact native/auxiliary loss and A/B-gradient gates passed before fitting;
all240 fixed Adam updates completed.65,536 trainable FP32 A/B parameters affect
only layer31 last MLP down projection; all927 base language tensors are unchanged.
Exact checkpoint reload and learned archived/native gradient parity passed.
New adapter: `data/ranker/pretrained/short-copy-fit-v1/adapter.safetensors`,
SHA-25685ce8215add1d4ce561441d6972c8e2958dbb73966bd89f53595ef65b7fddb57.
It warm-started the immutable prior checkpoint
`data/ranker/pretrained/semantic-copy-feature-fit-v1/adapter.safetensors`,
SHA-2560b0aeda377a9c9ad7eb37b529c139dac84bc5acd4bb559176d40245865ffe1b2.
All240 feature archives+metadata total1,844,548,247bytes under5GiB, one full host
feature row resident at a time. Keep the original archive in place.

TRAIN12 public cases/240fields include fresh C supervision and independent
S/T/B/C fields. Six previously inspected dev/test cases compare prior/new at
width3/6, fixed24arms. Native controls compare reused/rebuilt/repeated full
hybrid caches at matching widths in frozen/prior/new modes. Each first-three
endpoint requires ten complete fields and actual validation+serialization; full
six requires all20fields. These are locked-decision copy-stage measurements,
not ranking/full-request/browser timings. Token/strict factual validators are
unchanged, no templates/fallback/teacher text is used at inference.

## Current actual16-product integration

`store_catalog_v1` imports the real16products,48colorvariants,195galleryaction
records,32previews and81hashed local assets. `store_model_bridge` rejects stale
input-producing sources before/after candidate preparation. All17 import and
freshness CPU tests passed at this boundary. Public name/category/itemprice/
searchTerms/colors and declared image metadata are preserved; pixels are not
examined. Stock/sizes unknown. Shipping is cart-subtotal policy, separate from
item budgets; a hidden shipping quote is not free delivery.

The old seven-factor representation has only organizer supported among these
products:15products collapse to the exact same empty-atom text. Do not claim
actualstore success from the old24/24 synthetic replay. New candidate
`store_function_adaptation.py` prepares FULL publictext features with a separate
16-function learned head,32 public TRAIN requests and16 independently authored
evaluation requests in eight preference-only pairs. All16 catalog identities
are intentionally TRAIN-seen: this measures known-catalog request-wording
adaptation only. Evaluation packet is frozen before vectors/fitting; see
`store_function_eval_v1.json` (SHA55eb12fc85b4d87289bdea71dad1b4c8b5cb55f31d762c06c39a47131868e377)
and `STORE_FUNCTION_DATA_LABEL_AUDIT.md`. Native64B1 and fixed80-step fitting
remain unexecuted at this handoff.

`store_copy_diagnostic.py` is a separate actual-store copy candidate with strict
public-fact lexical/protocol checks, actual action bindings, genuine native
wording, complete first3/full6 timings and no runtime fallback. Its ordinary
chat prefix differs from the short-copy training distribution, so quality must
be measured, not assumed. It does not establish learned image selection or
comprehensive semantic factuality. No native store-copy inference has run.

`fused_result_benchmark.py` is the prepared six-request synthetic combined
ranking/image/copy diagnostic. It requires exact cached-vs-cacheNone feature
compatibility and full native cache/adapter seam proofs before timing fresh
requests. It is unexecuted and must stop on strict mismatch. Six cards from36
synthetic products are separate from actual16-store integration.

## Resources and next work

Apple M5,24GiB physical; `.tools/pretrained-venv/bin/python`, MLX0.32.4,
mlx-lm0.32.2,torch2.14.1,transformers5.20.1. Existing local frozen model:
`.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit/snapshots/8b2b98c00a6b4d291155e4890773ca8f769aee53`.
No new weights were downloaded.6/7GiB native guards are experiment settings,
not the24GiB hardware limit. No apps were closed. Heavy GPU work remains serial.
Native hybrid state is24recurrent/convolution ArraysCache +8KV histories; exact
whole snapshots/deepcopies only, no arbitrary trim/splice.

Next task should review final copy evidence and candidate CPU tests, then run
the actual16-store64B1/fixed80-step adaptation once, measure real fresh-query
ranking and guard readouts, and run the bounded store-copy diagnostic only after
its source/provenance controls pass. Keep all failures and original checkpoints,
use ordinary scoped main commits/pushes, and preserve the actual quality/latency
boundary. Do not promote an adapter from faster invalid attempts. Perceptron
remains deferred.


## Clean replacement checkpoint (supersedes queued statements above)

Main167295f is pushed. All old checkpoints and candidates remain intact.
Actual16-store v3 completed64 native inputs and80+80 CPU head updates:
12/16 top1,16/16top6,4/8 both-correct pairs. Independent audit exactly reproduces
all optimizer steps/final tensors. Median203.702ms contiguous serialized ranking
endpoint excludes copy/browser. All8 separate guard probes pass. New resident
rank/action seam completes24 native queries:16 score/order replays exact,
4 copy-adapter-disabled score/action rows exact and4 policy probes pass
(exactname, budget exclusion, Cobalt binding, unknown-stock abstention). It is
model-side integration; React/API files were not changed.

Actual-store old-adapter copy transfer completed6 requests:0/6 valid first3 and
0/6 full6; invalid-attempt medians4.980/6.227s. Matched-width controls and927 frozen
hashes pass. All120 fields decode/reassemble exactly on CPU. No promotion.

The new balanced source-conditioned v2 copy data/runner is frozen and tokenizer-
preflighted (25 CPU tests); native320-gate/320-update/24-arm run is next.
Final preflight STORE_COPY_FIT_V2_PREFLIGHT_V1.json SHA
cd8ead6445ac7785c90dfcfb0c15f6999e35a2da4c54530a59106db87cbb1cbd.
Read latest ledger/reports for progress; preserve incomplete/failed runs.
