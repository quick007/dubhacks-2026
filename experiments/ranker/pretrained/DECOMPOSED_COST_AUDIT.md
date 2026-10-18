# Decomposed generation cost audit

The saved independent-field path is slower chiefly inside native suffix prefill plus decoding. It sends substantially more row-specific prompt text through the model, and batching those short lanes does not amortize that work enough to beat the single manifest response. Measured Python cache-copy/materialization and strict assembly/validation costs are small. Output token counts alone do not explain the gap.

This is CPU-only analysis of preserved `PARALLEL_SEMANTIC_RESULTS.json`, `PARALLEL_SEMANTIC_SCALE_7G_RESULTS.json`, their matching case plans and the local implementations in `parallel_decode.py`, `fusion.py`, `full_page_reliability.py`, `heading_experiment.py` and installed mlx-lm `generate.py`. No inference, prompt variant, source edit or external quality-lane access occurred. Main JSON SHA-256 is `f3c4743f1d02524bdc726ed48854d73e3dcd02fe0826efdb24ceeed9679a12af`; separate scale JSON SHA-256 is `ce9d171ad3f973bd4c93891cd08f2e58d25604c5a4f201dd944cd4bc2a72fddf`.

## What the timers and token counts measure

`structuredMs` starts before ranking and includes public-input preparation, shared ranking-prefix tokenization/forward, trained decision heads, deterministic page grouping and plan validation. Every measured arm ranks again once. Decomposition does not rerun that prefix forward for every field: it forks the complete native recurrent/convolution/KV state.

`continuousDecodeWallMs` includes cache forks, suffix-boundary tokenization, native generator insertion/admission, native suffix prefill and generation, response bookkeeping, row detokenization and memory checks. `cacheForkMaterializeMs`, `suffixBoundaryPreparationMs` and `nativeSuffixPrefillAndDecodeMs` are nested inside it; do not add them again. The native timer surrounds `gen.next_generated()` and synchronization, combining suffix processing with generation. Saved data cannot retrospectively split that timer into exact native prefill and decode costs, or isolate native cache merging/padding/extraction.

Manifest `generationMs` measures its single native stream; prompt-boundary preparation is recorded separately. Manifest additionally records first-token/prefill and post-first-token times. Those finer metrics are absent for the independent-field batch decoder. The tables use a common additive accounting: structured region + generation region + recorded validation + residual = full completion. The manifest generation region excludes its approximately 1–4 ms prompt preparation; the decomposed generation region includes boundary preparation. That small definition difference is disclosed rather than hidden.

The residual is measured full completion minus these recorded regions. It includes row/prompt construction, generator construction/close, final checks and other untimed work; the existing data cannot attribute it exactly. It must not be labeled GPU prefill or cache-copy cost by assumption.

Token accounting needs care: installed `stream_generate` reports `generation_tokens=n+1` on the sampled EOS step. Thus saved manifest `generation.generatedTokens` **includes EOS**. Decomposed `actualOutputTokens` excludes EOS, while `actualRowArSteps` includes EOS and is the fair sampled-step comparison. For example, a stopped manifest count of 30 means 29 text tokens plus EOS. Earlier reports that called manifest `generatedTokens` emitted tokens excluding EOS used a different convention; the numbers below explicitly correct that interpretation without altering saved evidence. One `nativeDecodeRounds` iteration is a returned batch of generated responses and can include prompt work; it is not a count of every model forward/kernel invocation.

## Repeated prompt payload

| Cases / source | Cached ranking prefix tokens | Independent rows | Manifest suffix tokens | Sum of independent suffix tokens | Independent / manifest suffix ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| First accepted Grove, main | 1,009 | 3 | 275 | 668 | 2.43× |
| First accepted Cedar, main | 995 | 3 | 273 | 654 | 2.40× |
| Fixed one-card scale, main | 1,446 | 3 | 275 | 668 | 2.43× |
| Fixed three-card scale, separate seven-GiB | 1,446 | 7 | 405 | 1,734 | 4.28× |
| Fixed six-card scale, separate seven-GiB | 1,446 | 14 | 625 | 3,437 | 5.50× |

The one-card independent suffixes are heading 144 tokens, title 262 and badge 262 (Cedar title/badge are 255 each). Three-card suffixes are heading 158 plus title/badge pairs of 262/262, 267/267 and 259/259. Six cards add a second 157-token heading and pairs 253/253, 257/257 and 263/263. These are actual saved `suffixTokens`, identical across widths for each case.

Source inspection explains the repeated text: each title and badge independently carries selected evidence, approved titles, observed presentation preferences, an allowed-word list, locking instructions and output rules. They repeat most card-binding payload even though they produce different short values. The compact heading builder also calls the original row builder before replacing only heading messages. Ranking-prefix state is shared by exact forks, but none of these suffixes shares a separately cached common card prompt with its sibling lane.

The prefix lengths above are not repeated GPU prefill work. They are repeated logical histories/state carried by the independent lanes, with native batching also merging/padding/extracting complete caches. Multiplying prefix length by row count would overstate token encoding; ignoring the repeated state would understate memory/attention work. Existing receipts do not break that native state traffic out separately.

The independent runtime uses prefill step size 256; many card suffixes slightly exceed 256 (both one-card Grove rows, all six card rows in the three-card case, ten of twelve card rows in the six-card case). They therefore need more than one prompt chunk before the final-token generation transition. Manifest source still passes prefill step size 512: its 275/405-token suffixes fit below that step, while the 625-token suffix exceeds it. The benchmark records the 256 setting for the factorized path, but the two decoding APIs and chunk sizes differ. Attribution solely to independent-row factorization is therefore not a fully isolated causal result.

## Same two valid cases

These are the first Grove/Cedar cases, accepted in every compact width and in manifest. Values are means over the same two cases; with two observations these equal their medians. They are not twelve-case quality estimates.

| Arm | Structured / ranking region | Generation region | Strict validation / assembly | Residual | Full completion |
| --- | ---: | ---: | ---: | ---: | ---: |
| Manifest | 1,479.2 | 1,718.8 | 2.0 | 3.7 | 3,203.8 ms |
| Independent sequential | 1,362.1 | 2,193.6 | 0.6 | 53.0 | 3,609.3 ms |
| Width two | 1,365.5 | 1,964.2 | 0.9 | 31.2 | 3,361.9 ms |
| Width three | 1,409.8 | 1,854.9 | 1.5 | 56.4 | 3,322.6 ms |
| Width six | 1,409.3 | 1,887.6 | 1.9 | 50.5 | 3,349.3 ms |

Manifest ranking is actually slower in these saved pairs: its first Grove tokenization is 66.5 ms versus around 5–6 ms in later arms, and its average prefix-forward region is 1,440.7 ms versus approximately 1,356–1,404 ms for decomposed arms. This order/warmth variation favors the decomposed comparison. Even so, its generation region is 136.0 ms longer at width three and its total is 118.8 ms longer. Independent sequential is 405.5 ms slower overall, with 474.7 ms extra in generation partly offset by faster measured ranking.

Manifest samples 30/28 tokens including EOS. Independent sequential/width-three/width-six sample 27/27 total row steps; width two samples 27/28 because of the Cedar heading wording variation. Independent output has fewer sampled steps on average but a larger suffix burden and slower generation region. This directly rules out "more generated tokens" as the explanation for the valid-case slowdown.

Sequential native rounds are 27/27; widths three/six reduce them to 14/14 while admitting three simultaneous streams. Width six has no more than three available rows here. It does not double throughput versus width three, and its two-case mean is slightly slower. These single-attempt differences are not stable statistical throughput estimates.

## Fixed one-, three- and six-card cases

The one-card rows below come from the six-GiB main report. Three/six-card rows come from the separately declared seven-GiB supplement so every width has an actual completed attempt. This is a latency breakdown of named cases, **not merged denominators**. The main six-GiB three-card width-six failure and all six-card unattempted arms remain preserved.

| Fixed case / arm | Structured region | Generation region | Validation / assembly | Residual | Full completion |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 card manifest | 2,294.6 | 1,654.7 | 0.4 | 1.9 | 3,951.6 ms |
| 1 card sequential | 2,404.5 | 2,698.6 | 2.1 | 92.6 | 5,197.7 ms |
| 1 card width two | 2,347.9 | 2,341.2 | 2.4 | 69.5 | 4,761.0 ms |
| 1 card width three | 2,390.5 | 2,209.5 | 1.4 | 66.0 | 4,667.4 ms |
| 1 card width six | 2,239.2 | 2,161.4 | 1.3 | 40.9 | 4,442.8 ms |
| 3 cards manifest, seven GiB | 1,964.3 | 3,162.0 | 1.0 | 4.6 | 5,131.9 ms |
| 3 cards sequential, seven GiB | 1,883.7 | 5,535.5 | 0.8 | 44.8 | 7,464.8 ms |
| 3 cards width two, seven GiB | 1,889.6 | 4,912.7 | 0.9 | 45.2 | 6,848.3 ms |
| 3 cards width three, seven GiB | 1,900.6 | 4,453.4 | 2.2 | 33.1 | 6,389.3 ms |
| 3 cards width six, seven GiB | 1,912.2 | 3,946.8 | 1.6 | 33.6 | 5,894.2 ms |
| 6 cards manifest, seven GiB | 2,094.8 | 6,462.1 | 3.6 | 5.0 | 8,565.3 ms |
| 6 cards sequential, seven GiB | 1,946.2 | 11,640.8 | 2.6 | 75.1 | 13,664.7 ms |
| 6 cards width two, seven GiB | 2,054.8 | 10,296.0 | 3.3 | 73.0 | 12,427.0 ms |
| 6 cards width three, seven GiB | 2,045.4 | 9,825.6 | 3.9 | 29.6 | 11,904.4 ms |
| 6 cards width six, seven GiB | 2,085.7 | 8,900.2 | 2.5 | 46.1 | 11,034.6 ms |

Fixed-scale output remains rejected for every compact arm. Manifest accepts the one-/three-card pages and rejects the six-card page's factual section headings. These timings measure attempts, not new usable-page speedups.

At width six, decomposed generation exceeds manifest generation by 506.7 ms for one card, 784.8 ms for three cards and 2,438.2 ms for six cards. Ranking at width six is slightly faster than manifest for every named scale case. The generation region therefore accounts for essentially the full excess; ranking cannot explain the multi-second six-card gap.

For reference, manifest's saved native prompt times are 512.2/707.7/995.1 ms for fixed one/three/six cards. Its post-first-token time grows from 1,102.1 to 2,418.4 to 5,435.5 ms. The independent path has no equivalent native prefill-only receipt, so it is not defensible to assign its entire extra generation time to prefill. The payload growth and runtime structure are evidence of likely prefill/state overhead, with its exact split still unmeasured.

## Autoregressive work, batching and small overheads

| Fixed case | Manifest sampled steps including EOS | Independent sampled row steps, widths 1/2/3/6 | Independent returned native rounds, widths 1/2/3/6 |
| --- | ---: | --- | --- |
| One card | 30 | 33 / 33 / 33 / 33 | 33 / 20 / 15 / 15 |
| Three cards, seven GiB | 66 | 71 / 71 / 71 / 71 | 71 / 45 / 31 / 17 |
| Six cards, seven GiB | 140 | 143 / 142 / 142 / 142 | 143 / 83 / 60 / 36 |

The independent text-token counts excluding EOS are 30 for one card, 66 for three cards and 133/132/132/132 for six cards. The corresponding manifest text counts are 29/65/139. Hence the six-card decomposed output contains fewer text tokens than manifest and only two/three more sampled steps, yet takes substantially longer. Batched rounds fall by approximately three quarters at width six in the scale supplement, while the generation region falls about 29% for three cards and 24% for six cards from independent sequential. A returned round is more expensive when it carries multiple rows and may also perform queued prompt work; dividing the aggregate timer by rounds does not yield a clean per-token decode latency.

Across the named completed decomposed arms, cache fork/materialization costs approximately 0.4–4.3 ms, suffix-boundary preparation approximately 2.7–39.4 ms, and strict assembly/validation approximately 0.5–3.9 ms. At six-card width six, those are 3.9, 36.2 and 2.5 ms against 11,034.6 ms full completion. Eliminating the explicit Python cache-copy and validator work would not remove the 2,469.2 ms gap versus manifest. This conclusion does not declare native cache merging/state traffic free; such work lives inside the unseparated native timer.

Installed `BatchGenerator._next` advances an existing generation batch, admits/prepares queued prompts when capacity exists, merges complete states into prompt batches, and transitions prefetched rows into generation. Continuous refill removes a fixed-wave barrier but does not eliminate suffix forwards or their cache/state operations. `decode_rows` synchronizes after every `next_generated` call and performs a memory check before admitting replacements. These are real scheduling costs; their separate GPU/host shares were not saved.

## What the evidence supports next

The most useful local performance instrumentation would separate synchronized native prompt processing, native generation and cache merge/extract work, while preserving total latency, row tokens, state controls and strict quality. Existing `BatchGenerator` internal prompt/decode counters are potential instrumentation points, but their asynchronous timing semantics need checking before treating them as GPU measurements. This report does not run or implement that instrumentation.

The saved evidence already deprioritizes optimizing Python cache deep-copy or validation. Repeated row-suffix payload and native admission/prefill/state work deserve measurement before widening batch widths further. Any future shared suffix cache must preserve complete hybrid native state and matched-shape correctness; prefix trimming/splicing is not justified. Further prompt compression or card-field grouping would change the quality experiment and needs its own fixed paired comparison. The current local result remains that manifest is faster on shared valid cases and materially more reliable; actual independent six-stream execution is established, but a usable-page win is not.
