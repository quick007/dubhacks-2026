# Local ranking baseline

This is a small, genuinely trained feature-based baseline for testing the data, training, adaptation, evaluation and serving boundaries. It scores six candidates together using a 3,297-parameter tanh MLP with 32 hidden units and observable catalog/context/history features. Initial fitting starts from random weights; adaptation loads that checkpoint and trains further on different training families. This is not an LLM, a pretrained model or a Jev reproduction. The separate `pretrained/` experiment owns the frozen-pretrained-backbone proof; this baseline report supplies no evidence for that proof.

All records, labels and metrics are synthetic. They establish neither human preference accuracy nor conversion or revenue lift. Presentation selection has no supervision in this fixture and returns `presentationAction: null`. This experiment does not modify the storefront or its design.

## Run locally

From the repository root, use Node and an existing Python 3.12+ interpreter. The pinned NumPy 2.5.4 package requires Python 3.12+. Setup uses `uv` when available, otherwise Python's `venv` and the environment's `pip`. It installs only into `.tools/ranker-venv`; caches stay under `.cache/`. It does not install a system Python or modify system packages. Set `RANKER_PYTHON=/path/to/python3.12` before setup to choose a different interpreter for a new environment.

```sh
pnpm ranker:setup
pnpm ranker:data
pnpm ranker:test
pnpm ranker:train
pnpm ranker:evaluate
pnpm ranker:infer
pnpm ranker:benchmark
pnpm ranker:serve
```

Every command also works as `node scripts/ranker.mjs <command> [options]`. Generator options include `--output` and `--seed`; model commands accept `--dataset` and `--run`. Defaults are `data/ranker/datasets/v1`, seed 2026 and `data/ranker/runs/initial`. Inference accepts `--request path/to/request.json`; otherwise it uses the first held-out request. The test command runs 15 unit tests for generation, leakage boundaries, scoring, gradients, checkpoint updates/reload and deadline fallback. A first setup needs package download access. The dataset, trained weights and subsequent commands run locally without API keys.

`serve` binds only `127.0.0.1:4471`; `--port` changes that port. It loads the adapted checkpoint and warms the worker before listening. `GET /health` reports whether weights are loaded. `POST /rank` accepts `{"request": <public request>, "mode": "model"}`; mode `default` selects the deterministic heuristic. `GET /presentation-options` exposes factual approved fixture alternatives without claiming a trained presentation policy. If the checkpoint is absent, the service falls back to the heuristic. `--deadline-ms 5` is the default inference-worker wait budget. Busy, missing, failed or expired model work uses the immediate heuristic; the single busy worker is never allowed an unbounded queue.

The default output directories contain generated local artifacts and are ignored by Git. Regenerating data or training deliberately replaces files in the selected output/run directory. Use a different `--output` or `--run` to preserve a prior experiment. Keep generated weights, full datasets and predictions out of commits; the small [baseline-results.json](baseline-results.json) report records hashes and selected evidence.

## Boundaries and results

The v1 fixture has 360 requests from 60 families: 40 training, 10 validation and 10 held-out families. The two training stages use 20 disjoint training families each, with a fixed 160-epoch schedule per stage. Training reads only public training requests and separate training labels. Evaluation reads private truth solely for grading. Family IDs, request/user IDs, future outcomes and private preferences are excluded from model features. Known budget includes shipping. See [CONTRACT.md](CONTRACT.md) and the [dataset notes](dataset/README.md) for the exact wire and leakage boundaries.

The preserved initial run used 101 features, 3,297 parameters and about 52.8 MiB peak process RSS. Training recorded nonzero gradients, actual weight changes and lower training MSE at both stages. On the same 60 held-out requests, guarded NDCG@3 was 0.8311 for the heuristic, 0.8253 after initial fitting and 0.8399 after adaptation. Removing history from the adapted model reduced NDCG@3 to 0.7795. These are single fixed-run synthetic results, without uncertainty estimates or hyperparameter sweeps.

Adaptation did not improve every metric. Compared with the guarded heuristic, its mean relevance regret increased from 0.4833 to 0.5000 and its private-constraint violation rate increased from 5.0% to 8.3%. The deterministic known-constraint guard reduced known violations to zero for all models; that result is rule enforcement, not learned safety or knowledge of hidden constraints. Report raw and guarded rankings separately, and do not select a model based on NDCG alone.

The refreshed loopback benchmark warms the model worker before opening the socket, then makes one first HTTP model request followed by 60 paired default/model requests with alternating pair order. The report includes checkpoint loading, worker warm-up, first HTTP latency and steady paired latency separately. In the recorded run, model HTTP p95 was 0.312 ms and paired incremental p95 was 0.089 ms, with zero fallbacks. The first HTTP request still took 8.087 ms, exceeding the 5 ms worker budget in end-to-end wall time; that budget does not cap networking, JSON, startup or total page latency. These are measurements on one local machine under uncontrolled load, not storefront page-load measurements or a latency guarantee. See the report's timestamp, machine details and source hashes.

## Teammate ownership

- `dataset/`: reproducible fictional v1 catalog, observable records and separated supervision/private grading truth.
- `model/features.py`: observable-only feature construction; coordinate feature changes with checkpoint metadata.
- `model/network.py`: fixed tiny network, gradient fitting and checkpoint loading.
- `model/common.py`: transparent heuristic, constraint guard and independent metrics.
- `model/service.py`: preloaded single-worker deadline/fallback serving boundary.
- `model/cli.py`: bounded training, evaluation, inference and timing artifacts.
- `scripts/ranker.mjs`: isolated environment and stable developer commands.
- `pretrained/` and `dataset_v2/`: separately owned follow-up experiments; they are not part of the 15-test baseline or this result report.

Keep storefront changes and the pretrained proof separate from baseline evidence. No serving integration, browser agent, live customer data, ad account, deployment or purchase is part of this milestone.
