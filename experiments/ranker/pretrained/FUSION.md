# Shared-backbone fusion proof

The same resident four-bit Qwen3.5-9B language model supplies final hidden states to genuinely trained product/presentation heads and retains its autoregressive vocabulary decoder for newly generated personalized titles or short copy. The decoder is frozen and untuned; this is not joint training or an architecture novelty claim. Image choice uses factual text descriptions, not image pixels.

A single complete observable conversation contains all candidate facts before eight candidate readouts. Heads require the separately trained `shared-prefix-v1` representation. The exact native cache boundary retains all DeltaNet recurrent and convolution state alongside attention KV. Generation appends verified token suffixes once, with no cache trimming, forking, speculative model, or second backbone. Shared-prefix retries are disabled.

```sh
pnpm pretrained:setup
pnpm pretrained:data
pnpm pretrained:download
pnpm pretrained:verify
pnpm pretrained:test
pnpm pretrained:test:mlx
pnpm pretrained:test:fusion
pnpm pretrained:smoke
pnpm pretrained:workflow
pnpm pretrained:fusion --mode staged --output data/ranker/pretrained/fusion/result.json
pnpm pretrained:fusion --mode full
```

The staged mode emits a structured decision first and a complete generated-copy result later. Its first event is not a guarantee of instant page latency. All timing must distinguish uncached prefill/head time, generation suffix preparation, first token, autoregressive decode, independent validation and fully completed response. Cold model loading and whole CLI process latency are separate measurements. No storefront integration is included.

Generation receives only the same whitelisted public context and selected product facts. A strict independent lexical guard rejects unsupported attributes, numerals, promotional/delivery claims, markup, truncated output and unapproved vocabulary. A title must differ from prewritten titles and, when supported explicit preferences exist, demonstrate one in its wording. Fields are checked separately: a valid novel title may be returned while an invalid optional sentence is removed and its rejection recorded. `generated-partial` does not claim that both fields passed. An approved-title fallback sets `generatedTitleFulfilled: false` and never counts as generation success. The guard is restrictive and is not a general semantic factuality proof.

All personas, outcomes and labels are synthetic. Sparse history does not establish hidden intent. Model quality, preference responsiveness, generation acceptance and latency must be reported with regressions. This module makes no human conversion, broad generalization or production page-load claim. Optional NONE/SELECT/GENERATE sub-label routing is a separate proposed experiment, not implemented by this two-head proof.
