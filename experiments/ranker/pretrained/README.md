# Pretrained language backbone with parallel decision heads

The primary proof uses **Qwen3.5-9B**, loaded through a pinned community four-bit MLX conversion, as a frozen language feature extractor. One shared conversation contains every candidate fact before eight query readouts. One backbone forward supplies those readouts to two shared linear heads for product suitability and presentation utility. The complete native recurrent/conv/KV cache is then reused by the retained autoregressive decoder. The structured path never calls `generate()` or asks an LLM to emit JSON. Ordinary code assembles the structured result.

This module is the structured portion of a shared-backbone fusion system. The retained MLX wrapper also contains its autoregressive decoder; `fusion.py` adds a generative path on the **same loaded backbone object** for new factual personalized title/short copy after the structured product/image decision. Approved title selection remains a baseline/fallback. The decoder may remain frozen and untuned; no joint-training or cross-path cache-reuse claim is made. Structured-only latency must be reported separately from the completed decision-plus-copy response.

Only the decision heads receive real local gradient updates. The language backbone remains frozen and quantized. Initial fitting and warm-start head adaptation use separate training families. This is training heads on pretrained features, not full-backbone fine-tuning, LoRA, or a replication of Jev/TypeSafe-AI. The earlier engineered-feature MLP in `../model/` remains a baseline. A larger backbone is a hypothesis to evaluate, not a guarantee of better ranking.

## Pinned model and provenance

- Official upstream model: [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), Apache-2.0. Its current source revision inspected during setup is `c202236235762e1c871ad0ccb60c8ee5ba337b9a`; the full-precision upstream weights are not used or downloaded for this proof.
- Actual artifact: [mlx-community/Qwen3.5-9B-4bit](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit), pinned revision **`8b2b98c00a6b4d291155e4890773ca8f769aee53`**, declared Apache-2.0. The card states conversion from Qwen/Qwen3.5-9B using mlx-vlm 0.3.12. The card metadata names a Base upstream while its prose says Qwen/Qwen3.5-9B; that source ambiguity remains unresolved. It is a community conversion, not an official Qwen quantized release. Its exact original upstream revision is unspecified; equivalence to the inspected upstream revision has not been established.
- The two expected weight files total **5,950,221,072 bytes**. Expected SHA-256 values from repository metadata are shown below; successful local download/hash verification is a separate step.

| Weight file                        |         Bytes | Expected SHA-256                                                   |
| ---------------------------------- | ------------: | ------------------------------------------------------------------ |
| `model-00001-of-00002.safetensors` | 5,349,771,222 | `a68b87558c6ef43f74c2bd63ce7e9092ceddc3101f3def0030774bae5f42aadd` |
| `model-00002-of-00002.safetensors` |   600,449,850 | `b0a770bf8469c7f3f18756a0e0283f1c1174344a83e059a4e483f6af4907352d` |

MLX-LM's [native Qwen3.5 implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py) exposes final normalized language hidden states through `model.model(...)` and skips vision weights in its text path. The proof uses image descriptions as text, never pixels. The LM vocabulary output head is never executed. Remote model code stays disabled; only native installed library code and data-only checkpoint files are used.

## Commands and evidence

Run from the repository root:

```sh
pnpm pretrained:setup
pnpm pretrained:data
pnpm pretrained:test
pnpm pretrained:test:mlx
pnpm pretrained:download
pnpm pretrained:verify
pnpm pretrained:smoke
pnpm pretrained:workflow
pnpm pretrained:infer
```

The wrapper pins the local snapshot path and revision. `pretrained:test:mlx` is an optional native Metal hardware check; it skips when no Metal device is available. The pure boundary/head tests do not need model weights. Direct CLI usage for custom local paths remains available:

```sh
.tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.cli smoke --model PATH --revision 8b2b98c00a6b4d291155e4890773ca8f769aee53
```

`smoke` performs one request's uncached eight-row pretrained forward, one real head update, checkpoint save and exact head reload. `workflow` extracts and caches training embeddings, fits initial heads for 80 fixed epochs, adapts for another 80 fixed epochs on other training families, saves both and then evaluates the same frozen checkpoints on held-out requests. Its fixed proof budget is ten minutes after loading, with no sweep. `infer` reloads adapted heads in a fresh process and performs an uncached forward. Optional flags are `--dataset`, `--run`, `--max-tokens`, `--memory-gib`, and `infer --request path.json`.

Defaults are `data/ranker/datasets/v2`, `data/ranker/pretrained/shared-prefix`, 1,536 selected shared-prefix tokens and a six-GiB process MLX allocator budget with a 128-MiB cache. MLX allocator limits are not an OS-level hard memory guarantee; active/peak allocation and available system memory are checked and reported. The loader refuses non-four-bit primary checkpoints and low available memory. Selected candidate facts are never silently truncated. There is no automatic model download, full-precision fallback, paid service or change to system GPU limits.

Evidence stays under the ignored run directory: smoke result, frozen manifest, training gradients/loss changes, initial/adapted head checkpoints and hashes, held-out predictions, quality/constraint metrics, uncached latency and memory. A native tiny randomly initialized Qwen3.5 hybrid test verifies final hidden-state access, causal right-padding equivalence and genuine CPU head training. That compatibility test is not a pretrained-model result. Actual larger pretrained training or performance must only be claimed after its completed evidence exists.

## Information and action boundaries

`boundary.py` whitelists task, shipping-inclusive budget, factual requirements, stated preferences, presentation surface/preferences and a compact summary of the latest cutoff-valid history. Selected product facts are title, category, price, shipping, stock and tags. Approved presentation title, facts, image-description text and style are included. Identifiers, positions, family/split metadata, labels, hidden truth and future actions never enter model text. Full historical descriptions and product descriptions are deliberately outside this compact representation.

Product suitability is averaged over its two variant rows. Presentation utility chooses an already-approved title/image-description variant after product selection. Known masks check stock, total price, exact requirement tags and presentation compatibility; no feasible product causes abstention. These masks are ordinary code, not learned safety. Richer private suitability and presentation rubrics are checked only by the independent offline grader.

The evaluation compares lexical heuristic, first-variant positional baseline, a freshly trained engineered-feature MLP on the same v2 train/test fixture, random pretrained-feature heads, initial heads and adapted heads. The engineered model uses its own frozen initial/adaptation schedule and a lexical presentation policy, never a claimed trained presentation head. Separate preference counterfactuals check whether changing observable preferences changes decisions; they are excluded from training and model selection. No IDs or product slots become model features. Unfavorable results, hidden-constraint failures and presentation errors must remain visible.

## Latency and scope

Training embeddings may be cached and that is disclosed. Reported uncached test timing includes text construction, tokenization, quantized backbone execution, heads and masks. It must not be confused with cached head-only timing or browser page rendering. There is no storefront integration in this module. A future integration should render a default immediately and apply only timely asynchronous enhancements. Do not claim this 9B local path meets a page-load latency target before measuring it.

All customer records, labels and outcomes are fictional. Tiny synthetic held-out improvements cannot establish human conversion, merchant revenue lift or broad generalization. Preference ranking and presentation-action policies are related but distinct outputs. New factual text generation belongs to the shared-backbone fusion path; image generation, real advertising accounts, browser shopping agents and real purchases remain outside this lane.

`requirements.txt` pins the exercised native Apple silicon environment. MLX/MLX-LM and PyTorch/library licenses apply to their respective dependencies; the checkpoint carries its declared Apache-2.0 license and this experiment's original code follows `../LICENSE`.

## Measured result

The fixed 9B workflow completed 42 forwards in 58.97 seconds. Only 8,194 decision-head parameters were trained. On 12 synthetic held-out requests, adapted NDCG@3 was 0.8743 versus heuristic 0.8397, but best-product accuracy fell from 75% to 66.7% and best-presentation accuracy from 50% to 41.7%. Only one of three preference-counterfactual pairs passed. This is a working training/fusion pipeline, not robust personalization.

Uncached structured decision p95 was 1.440 seconds over the held-out set. In eight complete fusion cases, resident full-response p50/p95 was 2.938/3.284 seconds; three fresh processes on one repeated request measured 6.073/6.293 seconds including startup. OS caches were uncontrolled. Five of eight cases returned validated new titles: one accepted both fields and four retained a valid title while rejecting the optional sentence. Three used explicitly labeled fallback. Full decoder timing includes rejected text, not only successful outputs. Fusion peak MLX allocation was 6.137 GB, below the configured 6 GiB budget. Process RSS and MLX allocation overlap and must not be added.

See [RESULTS.json](RESULTS.json) for exact hashes, all raw/guarded metrics and limitations, and [FUSION.md](FUSION.md) for the shared-cache/generation API. Reproduce complete-response measurements with `pnpm pretrained:benchmark:fusion`. Weights, datasets and full reports stay ignored under `data/` and `.cache/`.

The optional [NONE/SELECT/GENERATE label router](ROUTING.md) has also been trained and measured. It generalized poorly: zero generation attempts, seven of eight served NONE, and only 37.5% raw routing accuracy against its toy public policy. Its lower latency does not demonstrate useful selective generation.
