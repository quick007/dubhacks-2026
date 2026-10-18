# Fast fresh generation feasibility v1

CPU/source research only, 2026-10-18. No GPU execution, package installation,
model download, checkpoint change or live-engine/UI change was performed.

The main architectural recommendation is to generate compact text from the
same fresh Qwen decision features with a small learned parallel decoder. The
parent's `parallel_text_head` lane owns that experiment; this report adds no
duplicate implementation. Retain a much shorter fresh autoregressive Qwen run
as its quality and latency control. Repeated 9B decoding can become faster, but
the evidence does not support a 100 ms complete fresh-text endpoint. The current
fresh ranking path itself already takes about 200–225 ms. A few hundred
milliseconds is a testable target for a shared-feature decoder, not a result.

## What the existing evidence establishes

The five saved width10 requests in `STORE_DEMO_LATENCY_DIAGNOSIS_V1.json` have
these separate medians: rank 224.553 ms, common neutral root 529.481 ms, combined
fork/suffix-prefill/decode 5,074.830 ms, copy 5,637.590 ms, full server endpoint
6,001.410 ms. They use twenty fields, a 416-token cap, median173 actual sampled
tokens including EOS, and median32 generation rounds. These medians are not an
additive trace. That diagnosis explicitly has no isolated current suffix/decode
GPU breakdown. Earlier `BATCH_STAGE_PROFILE.md` traced a different workload;
its suffix/decode proportions must not be transferred as current measurements.

Caching the byte-identical common root removes work represented by the measured
529 ms stage, subject to a new native correctness/timing test. It cannot remove
the remaining seconds. Width20 changes parallel field scheduling, not the causal
dependence of tokens inside each field. First-three readiness and all-six
completion must still include their entire respective required field rosters.

The local 9B safetensor headers contain 5,038,041,600 bytes of language tensors
and 912,020,960 bytes of vision tensors. The text-only MLX loader discards vision.
Embedding storage is 572,129,280 bytes; the separate full-vocabulary output head
is also 572,129,280 bytes. These are on-disk tensor sizes, not active-memory
measurements. Excluding embedding storage gives a rough 4.466 GB weight stream
per full decode pass. At Apple's advertised M5 bandwidth of153 GB/s, that is
about29.2 ms of ideal weight traffic alone. This is a simplifying estimate,
not a hardware lower-bound proof or measured token latency. Five such passes
already imply146 ms before prefill, recurrent/KV traffic, compute, scheduling,
ranking or rendering. Apple's GPU accelerator claims are not evidence that
this installed MLX quantized path reaches their peak. [Apple M5 specification](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/).

## What is available locally

The repository Hugging Face cache has one complete pretrained model: the pinned
Qwen3.5-9B-4bit snapshot with two weight shards totaling5,950,221,072 bytes.
Qwen3-8B has no complete snapshot weights and two incomplete blobs; those are
preserved. The conventional home Hugging Face cache is absent. The local
`offline-tiny` artifact is a1,313-parameter engineered ranking model with a
fixed lexical presentation heuristic, not a pretrained text generator.
No complete0.8B/2B generator or draft model is available for an offline test.
Directory disk-usage figures undercount the9B cache's shared weight symlinks;
the inspection followed snapshot paths and read only safetensor headers.

## Stock speculation, MTP and DFlash

Installed MLX-LM0.32.2 `speculative_generate_step` rejects target caches that
cannot be trimmed. Qwen3.5 constructs `ArraysCache` for its24 linear-attention
layers and `KVCache` for eight full-attention layers. `ArraysCache` inherits the
false trimmability default. Thus stock draft decoding is not a switch that works
for this target. It also requires the same tokenizer and is separate from the
current continuous `BatchGenerator` path. [Pinned generation source](https://raw.githubusercontent.com/ml-explore/mlx-lm/v0.32.2/mlx_lm/generate.py),
[pinned cache source](https://raw.githubusercontent.com/ml-explore/mlx-lm/v0.32.2/mlx_lm/models/cache.py).

The local config declares one MTP layer, but header inspection found zero
`mtp` tensors. Installed Qwen3.5 `sanitize` explicitly removes `mtp.` weights;
there is no native MTP execution path in that module. A config key therefore
does not establish usable local MTP. [Pinned Qwen3.5 source](https://raw.githubusercontent.com/ml-explore/mlx-lm/v0.32.2/mlx_lm/models/qwen3_5.py).

Speculative decoding can preserve a target distribution through verification.
Medusa and multi-token prediction add trained heads and specialized verification;
their published gains do not establish storefront latency or frozen-head
compatibility here. Independent word-slot generation is a different quality
tradeoff and must not be called lossless speculative decoding.
[Speculative decoding](https://arxiv.org/abs/2211.17192),
[Medusa](https://arxiv.org/abs/2401.10774),
[multi-token prediction](https://arxiv.org/abs/2404.19737).

DFlash is a credible separate overnight candidate. Its authors list an MLX
Qwen3.5 backend, a9B-target drafter, and quantized verification blocks of at most5.
The drafter is not a standalone generator. [Author repository](https://github.com/z-lab/dflash),
[9B draft model card](https://huggingface.co/z-lab/Qwen3.5-9B-DFlash),
[paper](https://arxiv.org/abs/2602.06036).

The inspected author `model_mlx.py` captures and rolls back GatedDeltaNet
recurrent/convolution state, addressing a limitation of stock MLX-LM. Its current
stream path handles one sequence, creates fresh target/draft caches, and exposes
no reusable-prefix-cache argument. It does not provide our twenty-lane scheduler.
It hooks target layers and patches the GatedDeltaNet class under a process lock.
That requires an isolated process and exact frozen-target/adapter and cache
checks before any integration. The draft is bound to the target embedding/head;
our adapted target's acceptance rate remains unknown. [Author MLX source](https://raw.githubusercontent.com/z-lab/dflash/main/dflash/model_mlx.py).

Its local dependency extras pin MLX0.32.0/MLX-LM0.31.3, different from installed
0.32.4/0.32.2. The loader first loads draft weights then quantizes them, so
initialization peak must be checked too. Do not install those pins over the
working environment. The inspected9B model-card benchmark uses a B200,
thinking-enabled outputs up to4096 tokens; no corresponding base-M5 short-copy
result is established. Parent-reported M5 Max long-output figures are also not
this workload's evidence. [Dependency source](https://raw.githubusercontent.com/z-lab/dflash/main/pyproject.toml),
[loading/benchmark source](https://raw.githubusercontent.com/z-lab/dflash/main/dflash/benchmark.py).

## Concrete next bounded comparison

After the parent's unchanged-field width20/common-root profile, compare the
current twenty-field workload to a declared compact-copy candidate. Keep six
actual cards, two headings, six fresh titles, six fresh badges and six fresh
bodies. Preserve source product/variant/price identity and exact-name behavior;
original names may remain separate metadata/subtitles. Candidate titles are
2–4 words, bodies2–5, badges a supported short fact or abstention, headings2–4.
Proposed sampled caps are8/8/4/4 tokens for title/body/badge/heading:128 total,
including whatever EOS work the implementation requires. Caps are a preflight
bound, not successful text truncation. This is an explicit new output contract
and training distribution, never a reinterpretation of older validation.

Use sixteen fixed six-card requests covering all16 products × three color
variants × two public query/style conditions:96 card conditions. Rotate which
three cards are above the fold. Keep all products eligible in the copy-only
comparison; separate integrated guard probes test strict/inclusive budgets,
exact names, changed colors and unknown stock, with legitimate shortfalls.
Have a separate TRAIN corpus with broader supported constructions; freeze it
and independently audit it before fitting. Do not tune exact wording to these
sixteen requests or use an evaluator answer as input. Repeated inspected cases
remain development replay. Compare three fixed repeats per condition:96 timed
requests for the two workload arms, plus separately reported controls/setup.

Cache only immutable native prefixes. Key by model/adapter/tokenizer/prompt
revision, exact public facts and token/position/cache-format identity. Append
fresh public query/style after the cached portion. Rebuild the complete prompt
as an independent reference; compare same-width token/EOS/full-distribution
receipts and preserve source cache hashes. A→B→A and changed color/price/source
must test both conditioning and invalidation. Never splice a changed query into
an existing recurrent state.

Start with one reusable common root. Per-variant fact-prefix reuse can then
share facts among title/body/badge suffixes, but all48 GPU roots are expensive:
the config implies48 MiB recurrent state per root before convolution/KV state.
At an80-token bf16 prefix, a rough full-cache estimate is51.625 MiB/root,
or2.420 GiB for48. These are allocation estimates, not measured memory.
Use a bounded hot-root LRU or separately tested host serialization, not an
unmeasured48-root GPU cache under the existing7 GiB guard. Measure misses and
construction cost as well as hits. Prior prefix hashes do not validate a changed
chat layout or batch shape automatically.

Report end-to-end server p50/p95, complete first-three and full-six readiness,
every stage's synchronized cost, cache hit/miss/build costs, native memory,
sampled tokens/forwards and all fixed failures. Browser paint opportunity and
image readiness require later separate measurement. Quality reports should
separate structural validity from blind semantic judgments: unsupported facts,
identity/color/price errors, useful grammatical copy and appropriate response
to changed public context. No exact teacher-string criterion. All generated
fields and withheld fields remain visible; metadata-backed cards do not become
successful complete fresh-copy payloads.

## Optional student and overnight order

If compact9B AR remains too slow, a0.8B text generator conditioned on the9B
selected public facts is a clean optional architecture comparison. Qwen's
official0.8B model has24 layers,1024 hidden units and tied output embeddings;
the2B option has24 layers and2048 hidden units. Neither is cached locally.
Disable thinking explicitly and distill diverse source-supported short text,
not an evaluator answer key. This adds a second backbone and cannot be described
as the same9B generating its text. [Official0.8B card](https://huggingface.co/Qwen/Qwen3.5-0.8B),
[official2B card](https://huggingface.co/Qwen/Qwen3.5-2B).

Pure4-bit parameter arithmetic suggests roughly0.4/1.0 GB before scales,
unquantized tensors and vision storage; it is not a download-size or RAM promise.
Ideal weight traffic would be roughly2.6/6.5 ms per pass at153 GB/s. Actual
small-model kernels, prefill and quality may dominate. No calibrated latency
projection is available. Keep the9B ranking cost and extra model's memory/time
in the integrated endpoint rather than subtracting them from the comparison.

Overnight order: finish profiled common-root/width20 controls; run compact AR
and the separate shared-feature learned text-head comparison; only then consider
an isolated DFlash test with blocks2/3/5 and0.8B-versus2B student acquisition.
No such install, acquisition, training, native benchmark or scheduled work was
started here. For DFlash, first use one fixed short source-bound field with the
actual adapted target, compare greedy raw IDs/EOS to unchanged AR, force rejection
and rollback controls, and record full prefill/first-token/last-token wall time.
Only passing that gate justifies a multi-field scheduler experiment. Target
distribution preservation would still preserve any factual errors in the target.

The next decision is empirical: does a compact parallel learned decoder retain
enough factual, preference-sensitive text quality near the existing query-encode
cost? If it fails, report that limit. A100 ms full endpoint is not currently
supported; caching previously generated text can be fast but must be labeled
earlier-generated cache delivery rather than fresh request-time generation.
