# Local ranking experiment boundary (v1)

All records, supervision, and evaluation are synthetic. No browser agent, UI, ad account, or real customer data is part of this milestone. Code lives in `experiments/ranker/`; generated datasets, base weights, adapters, predictions and reports live in ignored `data/ranker/` or `.cache/`.

## Dataset files

The standard-library generator writes `data/ranker/datasets/v1/` with:

- `catalog.json`: `{schemaVersion:1, synthetic:true, products:[Product]}`.
- `public/train.jsonl`, `public/validation.jsonl`, `public/test.jsonl`: merchant-observable Request records ONLY.
- `labels/train.jsonl`, `labels/validation.jsonl`: separate synthetic supervision `{recordId, relevance:{productId:0..3}, bestProductId:string|null}`. Never include label fields in request prompts.
- `private/truth.jsonl`: `{recordId,userId,personaFamilyId,split,hiddenPreferences,hiddenConstraints,relevance:{productId:0..3},bestProductId:string|null,...}` for independent grading; training code must not read this file.
- `manifest.json`: seed, generator version, family-level split assignment, public file hashes, counts, cutoff policy and synthetic provenance. Family/split IDs are metadata, never model features.

Product fields: `id`, `title`, `category`, `description`, `price`, `shippingCost`, `inStock`, `tags:string[]`, `presentationVariants:[{id,title,imageKey,facts:string[]}]`. Variants are factual preapproved fixture alternatives, not generated claims.

Request fields: `schemaVersion:1`, `synthetic:true`, `recordId`, `userId`, `asOf` (ISO8601), `context:{task,statedBudget:number|null,statedRequirements:string[],statedPreferences:string[]}`, `history:[{at,type,productId,dwellSeconds:number|null,source,confidence:number}]`, `candidateProductIds:string[]` (six), `missingFields:string[]`, `provenance:{source,generatorVersion,...}`. History timestamps must be <= asOf. Missing facts stay null/missing; do not fill from hidden truth. Inputs exclude sensitive/protected traits, family/split metadata, private truth and future outcomes.

Model prompt/feature builder uses ONLY context, cutoff-valid history, and candidate catalog facts. Never expose record/user identifiers as predictive features. Private preferences may influence synthetic labels but remain inaccessible as model inputs. No held-out family appears in training, and no test label is used for training/model selection. Include sparse/new/noisy/ambiguous records and cases where hidden information makes perfect inference impossible.

## Model/evaluation boundary

The initial implementation compares a transparent heuristic, a genuinely trained compact parallel decision network, and its warm-start adapted checkpoint on the SAME held-out requests/candidates. It uses an explicit observable-only feature encoder and one batched candidate-scoring pass, not autoregressive text/JSON generation. This is not Jev, a pretrained LLM, or a reproduction of a frozen-backbone fusion architecture. The fixed training schedule is chosen before reading final test metrics. Public labels are separate supervision, not input features. Report raw model ranking and constraint-guarded ranking separately; a deterministic constraint mask is not learned behavior.

A rank result is JSON `{schemaVersion:1,synthetic:true,recordId,modelId,rankedProducts:[{productId,score}],selectedProductId:string|null,presentationAction:{productId,variantId}|null,timing:{...},warnings:string[]}`. Scores need not be calibrated probabilities. A policy selecting a preapproved presentation variant is separate from preference ranking; this milestone may return `presentationAction:null`. Free-form request-time title/image generation is deferred. Abstain when known constraints leave no suitable item; never increase spend by violating suitability or hiding fees.

Use `data/ranker/runs/initial/` for the first bounded run, including adapter checkpoint, run manifest, evaluation and inference evidence. Root integrates commands/docs and commits a small result summary with checkpoint hashes; model weights and full generated datasets stay local and reproducible.

## First milestone resource/latency plan

Use one CPU/BLAS thread, a compact 32-hidden-unit candidate scorer (under 20,000 parameters), and two fixed training stages using disjoint training families. Initial fitting starts from random weights; adaptation loads that fitted checkpoint and performs real further updates on the second training partition. This demonstrates continued training mechanics, not foundation-model post-training. Freeze stages before test evaluation; do not sweep hyperparameters against held-out truth.

Training has a 60-second wall cap and a target below 256 MiB process RSS. Local inference preloads the checkpoint once. Use a configurable 5 ms decision deadline, immediate fallback on busy/unavailable/expired work, and no request-time text generation. Benchmark paired end-to-end loopback HTTP latency against the ready default path, including feature preparation, JSON and model execution, plus warm inference and startup/load separately. These are serving-interface measurements; the storefront is deliberately not edited or integrated in this milestone. Future page rendering must render the default immediately and request enhancement asynchronously with its own deadline.

Preference scores and a policy selecting preapproved presentation actions are separate decisions. This fixture has no presentation-action supervision; `presentationAction:null` must remain explicit until a policy is independently trained/evaluated. Optional free-form generation stays off the page-load path and is deferred.
