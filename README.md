# Personalization Labs

Please note that this repository is an experiment for a hackathon. Some of the info and content may not be exactly correct

Personalization Labs turns a shopper's request into a personalized product page.
It selects products and colors from a store's catalog, then generates titles,
badges and descriptions from the selected product facts.

Frozen Qwen3.5-9B features feed trained heads for product selection, color choice
and parallel copy generation. Built for DubHacks 2026, the project includes a
working storefront and AI shoppers that evaluate it from screenshots.

![Generic browsing and personalized product ordering in the demo storefront](docs/assets/shopper-sequence.png)

Figure 1. The demo storefront: (a) the initial collection, (b) a relevant blanket
below the first row, and (c) the blanket ranked first after personalization.

## Results

| Result | Measurement |
| --- | --- |
| **+3.78% value per session** | **$42.35 → $43.95** in simulated merchandise value across 60 paired shopping scenarios (120 sessions). |
| **86 ms median response** | Six products with fresh ranking and generated copy, using warm input encoding. |

These results come from separate synthetic shopper and serving benchmarks.
[Measurements and methodology](docs/RESULTS.md).

![Request latency: Our model at 86 ms, GPT-6.1 Sol at 6.2 s, and GPT-6 Luna at 4.3 s](docs/assets/latency-figure.png)

Figure 2. Our model serves six products with warm input encoding (backend time);
hosted models perform product ranking (client time).

## How it works

The selection heads choose products and colors. The text head predicts word slots
for titles, badges and descriptions in parallel, conditioned on the selected
catalog facts. Input features can be reused; every request computes fresh scores,
selections and text.

![Qwen3.5-9B feeds product selection, catalog facts and parallel text generation](docs/assets/architecture-figure.svg)

Figure 3. Product selection determines which catalog facts the text head receives.
Budgets and supported hard constraints restrict selection; generated descriptions
must match those facts.

The Fieldwork demo store has 35 products and 105 color variants, with galleries,
a cart and simulated checkout. AI shoppers browse both generic and personalized
versions using screenshots and visible controls. The evaluation records purchases,
abstentions, actions and merchandise value; the dashboard displays the results
with saved session evidence.

[Model and serving details](experiments/ranker/pretrained/STORE_EXPANDED_ENGINE_V1.md) ·
[Shopper evaluation](apps/runner/SHOPPER_COHORT.md)

## Run locally

Requires Node 24.11+ in the 24.x line or Node 26+, pnpm 12.5.1, Go 1.24+ with CGO,
and a C compiler.

```sh
pnpm install
pnpm seed
pnpm dev
```

Open the [storefront](http://shop.dubhacks.localhost:1356) or
[presentation](http://dubhacks.localhost:1356). The default shop runs without model
weights. Personalized serving needs the local checkpoints, and the results deck
needs a Tiger connection. [Setup instructions](docs/DEVELOPMENT.md#run-the-default-demo).

## Code

[Storefront](apps/storefront/) · [Model](experiments/ranker/pretrained/) ·
[Presentation](apps/web/src/CohortPresentation.tsx) · [Experiment API](apps/api/README.md)

## License

[MIT](LICENSE) © 2026 DubHacks project contributors. Third-party dependencies and
model artifacts retain their respective licenses and notices.
