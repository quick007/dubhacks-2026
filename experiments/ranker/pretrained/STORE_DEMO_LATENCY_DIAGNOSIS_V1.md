# Integrated latency diagnosis

The integrated six-card path misses a few-millisecond end-to-end target. The
124–211ms earlier results were fresh query encoding, learned product scores
and guarded decisions. They produced no new titles, badges, section headings
or body text, did not load images, and did not render a page. They were never
full personalized text generation latency. This boundary must remain visible
in the pitch. A conventional LLM comparison will not change this shortfall.

## What actually runs

`store_demo_engine.decode_rows` uses one resident native MLX BatchGenerator
with completion/prefill width10. It starts ten independent fields for the first
three cards, then continuously refills freed slots with the other ten fields.
Each field has an exact private fork of the complete common Qwen recurrent/
convolution/KV state. Field decoding is genuinely batched; this is not20 serial
complete LLM calls. Each individual text remains autoregressive. On the five
final requests,173 native sampled tokens includingEOS is the median total,
with28–32 generator rounds. Width10 is capacity, not20 tokens emitted in one
forward or10 always-active rows. The longest field chains and refill/prefill
work remain on the critical path.

There are20 requested fields:2headings,6titles,6badges,6bodies. Native total
output cap416tokens; actual171–178tokens,621–639characters. Field suffixes
contain repeated bounded source evidence:905–911tokens in total. The common
328token neutral-copy root is freshly prefilled for EVERY request even though
its token hash is identical across these requests. This is avoidable repeated
work, not a shopper-dependent root. It was inherited from the exact fresh-
request benchmark contract. A serving cache can safely reuse it only with
adapter/tokenizer/prompt/source/position identity and native branch-isolation
checks; it is not implemented in the measured current service.

The resident model/checkpoint, immutable learned catalog features and tokenizer
are already reused. Warm catalog reads are verified each request. Generated
answers and per-product field caches are NOT reused. The ranking prefix is
not spliced into generation, which keeps neutral-copy provenance correct.

## Measured components

Five final actual-store requests, warm resident width10, same hardware:

| Component | Median | What is included |
| --- | ---: | --- |
| Fresh rank decision |224.6ms|HTTP admission/queue, query encoding, verified catalog, learned scores, guards/actions|
| Fresh common-copy root |529.5ms|328tokens, cache-only prefill in256token chunks|
| Combined generation loop |5074.8ms|cache forks/materialization, suffix construction, native suffix prefill, batched AR forwards, synchronization, memory checks and first3 callback|
| Whole copy stage |5637.6ms|root plus combined loop and remaining preparation|

These separate medians are not an additive trace. Raw per-request values are
in the JSON sidecar and exact request receipts. The current loop did NOT time
prefill, cache copies and AR GPU forwards separately, so attributing all5.07s
to token decoding would be false. No measured queue-only figure exists either.
The three final browser requests were sequential, so GPU contention was avoided.
Browser full-stage opportunity minus server readiness was39.2–55.6ms; browser
rank opportunity minus server rank was52.4–96.3ms. This combines dispatch,
network, React and frame scheduling and is not HTTP-only overhead.

First3 server readiness remains4.94–5.52s across the three instrumented cases;
all-six server completion5.69–6.10s. First3 waits for all ten fields, including
the long body chains. First3 acceptance8–9/10 and full6acceptance18/20 mean
these are attempted completion times, not complete all-fresh useful pages.
A title+body-per-card operational proxy is currently0/5 across final requests.
Metadata-backed cards are usable for cart actions but that does not fix fresh
copy quality or latency.

## Lowest-latency next options

1. Cache the identical common root per resident adapter/prompt revision. This
   can remove the measured~0.53s common-root work while retaining fresh sampled
   output, but does not plausibly turn a~6s path into milliseconds. Native
   source-cache hashes, exact fork/rebuild controls and output/quality receipts
   must accompany the first measurement.
2. Profile cache fork/materialization, suffix prefill, admission groups, native
   decode, synchronizations and callback costs before changing scheduling.
   Continuous single-slot refill may prefill tiny batches while bodies remain
   active; whether it dominates latency is currently unknown. Compare bounded
   width20 and group admissions only under measured memory headroom, with
   first3priority and all20fields retained. Neither option has a speed result.
3. Train genuinely shorter factual text and shorten repeated instructions/
   fact suffixes, then compare the same six-card workload and quality. Caps
   alone truncate output and are not a successful speed optimization.
4. Native model-generated catalog/variant copy can be precomputed or warmed
   ahead of a shopper. A cache hit can avoid request-time generation and preserve
   actual model-generated wording, but the text was generated earlier. Report
   cache-hit versus cache-miss time, age/source/adapter key, warming cost and
   hit rate; never label it fresh uncached request-time generation. The current
   neutral copy is shopper-independent, making this technically applicable.
   No precomputed copy is silently substituted in the live service.
5. A materially smaller/distilled generation model or non-autoregressive text
   decoder could lower uncached latency, but is a new model/quality experiment.
   The current9B autoregressive architecture has no measured route to a few
   milliseconds of fresh20-field generation.

Larger performance experiments are reserved for the later tuning window.
Usable integration and the shopper evaluation loop remain the immediate goal.
No new performance win is claimed from this source audit.
