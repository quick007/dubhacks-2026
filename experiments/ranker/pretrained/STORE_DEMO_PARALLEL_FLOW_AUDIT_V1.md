# Serving parallelism audit

The current service genuinely batches independent copy fields. It does **not**
generate twenty complete strings one after another. It also does not generate
all text in one parallel model step: each active field remains autoregressive.
A few-millisecond full fresh-copy claim is unsupported by this implementation
and its measured M5 results.

This is a CPU source audit. No new GPU work, service changes or live engine
edits were made. The separate profile candidate is prepared but has not run
natively. The pending native approval and existing worker ownership remain
untouched.

## Actual request flow

1. A fresh public query is encoded once by the frozen Qwen backbone with the
   copy adapter explicitly disabled. The trained CPU head scores against 16
   verified cached catalog features. Public hard guards and deterministic
   variant binding select up to six products. This is the fast ranking result.
2. The same resident backbone, with the installed copy adapter enabled, builds
   a **new empty** copy cache and prefills the common neutral copy message.
   Current v3 requests use the identical 328-token root/hash, processed as
   chunks of 256 and 72 tokens, with a materialization/synchronization boundary
   after each chunk. The query rank cache cannot be substituted here: its
   tokens, context and adapter mode differ.
3. The twenty jobs are ordered as ten first-section fields followed by ten
   second-section fields. Initial width 10 admits the first ten together.
   For each admission, Python clones the complete pristine root cache and
   immediately materializes/synchronizes that clone. This clone loop is
   sequential. It does not repeat the root transformer forward twenty times.
4. Each cloned cache receives its own ordinary-chat field suffix. Titles and
   bodies repeat full bound public facts; badges repeat selected-color
   evidence; headings receive only their generic section number. The v3
   six-card requests contain 905–911 suffix tokens across twenty jobs. Each
   suffix, about 31–57 tokens, is genuinely prefetched by the native model.
   The caller passes existing caches to `BatchGenerator.insert`, so suffix
   prefills extend the shared root rather than rebuilding it from empty.
5. Native `BatchGenerator` performs continuous token batching up to width 10.
   Each call generates one next token for the active generation batch, and
   when slots are free also advances newly admitted prompt suffixes. Within
   the installed `_next`, existing generation work is scheduled **before**
   prompt prefill, but the call returns only after both phases have been
   scheduled and required host-visible work has completed. An explicit
   generation-stream synchronization follows every `next_generated` call.
6. Short headings/badges finish before bodies. Freed slots are immediately
   refilled with second-section jobs even while first-section bodies are still
   decoding. This is genuine overlap of task progress on one GPU; it is not
   independent simultaneous GPUs. Below-fold suffix prefills can therefore
   add work on the first-section critical path. Their exact share is currently
   unmeasured.
7. Once all ten first-section fields end, CPU validation and serialization
   emit that section. The same generator continues until all twenty fields
   end. Unsafe fields become null. No template or teacher output is supplied.
   Cache-state hashes verify the pristine common source before/after decode;
   these checks also copy native state to CPU and are part of observed serving
   overhead. All 927 model-parameter hashes occur at startup/shutdown, not on
   each request.

The five retained v3 requests observed 28 or 32 continuous decode rounds,
171–178 actual sampled tokens across all fields, and a 416-token summed cap.
The 173-token median is not a 173-step sequential stream: tokens from several
fields are produced in the same round. It is also not a one-step text decoder.

The earlier `store_copy_wide_v2.decode_complete_rows` uses the same complete
cache forks, field suffixes, continuous refills, temperature-zero sampling and
first-section callback. The resident demo preserves that architecture while
allowing real one-through-six-card workloads. The source audit found no
accidental fallback to one-field-at-a-time native generation and no lost root
reuse within a request. Static root reuse **between** requests was never
implemented. No isolated native measurements establish whether the new
serial clone loop, suffix prefill, decode kernels or continuous refill overhead
dominates the remaining time.

## Prepared profiling candidate

`store_demo_decode_profile_v1.py` copies the live algorithm without changing
row order, sampling, caps, cache source, admission policy or callbacks. It pins
the live decoder function SHA-256:

`2b6157efbc60473f79d1f718aedb159601ed0b237706ba7e2d963766c60a2a74`

The candidate records separate host-wall timers for cache clone/materialization,
suffix boundary preparation, insertion, `next_generated`, explicit stream
synchronization, memory guards, first-section callback, construction and close.
It also records every admission group's fields and suffix lengths, whether
below-fold jobs enter before first-section readiness, maximum admitted/emitting
width, and per-round completions. Five CPU fake-runtime tests pass, including
complete variable-card rosters, bounded admission, cancellation and preservation
of partial failure records.

`next_generated` includes prompt prefill, cache batching, token decode and
implicit waits. Its timer must not be called pure token-decode GPU time. The
candidate does not claim isolated GPU-kernel prefill/decode measurements.
Native token parity and latency remain to be checked by the authorized owner
after the pending approval is resolved and timing jobs are serialized.

## Concrete next options

| Change | Keeps text freshly generated per request? | What it can address | Important limit |
|---|---|---|---|
| Retain one pristine neutral common root in the resident engine | Yes | Avoid the repeatedly identical 328-token root prefill | The measured root costs about 0.53 s; this alone still leaves seconds of generation. Source, tokenizer, adapter and full-token identity must bind the cached state. |
| Profile and coalesce complete cache clone materialization | Yes | Reduce twenty sequential clone/sync boundaries | Materialize all bounded group clones together, preserve independent mutable caches, and verify native parity/memory. Savings are unmeasured. |
| Test width 20 with all twenty independent fields admitted initially | Yes | Avoid repeated small refill prefills and reduce possible sequential rounds | Wider matrix/cache work and memory may offset the gain. Same-width controls and real headroom inspection are required. No speedup has been measured. |
| Hold second-section admissions until the first section completes | Yes | Keep below-fold prompt prefill off the first-section critical path | It can delay full-six completion. Measure both first-three and full-six; never present first-three time as full completion. |
| Coalesce small refill groups or reuse a shared per-card fact prefix | Yes | Reduce small-prefill/batching overhead or repeated fact tokens | Scheduling changes need measured controls; changing field-message ordering changes the prompt seen by the fitted adapter and needs quality checks. |
| Smaller or genuinely non-autoregressive copy decoder | Yes | Change the multiple expensive 9B sequential token steps | This changes the model architecture/system tradeoff and needs training/quality work. A Python scheduling tweak does not provide it. |
| Explicitly precompute and cache native copy by public fact binding | **No: generated earlier** | Remove copy generation from the interactive request | Must expose cache provenance/generated time and call the metric cached-copy readiness, not fresh-generation latency. Ranking still takes its measured fresh-query time. |

Precomputation has a concrete small scope here because neutral copy prompts
contain no shopper query. Each card's title/badge/body depends only on its
selected name/category/price/color and fixed prompt/checkpoint. For all sixteen
products and three declared colors, that is 48 fact bindings × three fields,
plus two generic headings: **146 native field outputs**, with their raw
generation receipts and validation stored. The default-color catalog alone is
50 fields. This could provide a practical fast demo with fresh preference-based
ranking and transparently cached model-generated copy. It would change the
freshness contract and must be labelled explicitly; it is not silently proposed
as equivalent to newly sampled text for each interaction.

No one of these unmeasured optimizations supports promising a few milliseconds
for the current full fresh six-card request. The first focused experiment
should identify clone/suffix/refill/native-call costs, then test the largest
measured contribution while retaining genuine fresh text and honest card counts.
