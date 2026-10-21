# Personalization Labs: project vision

## One-sentence pitch

Make ecommerce sites better by serving each shopper a storefront built around their taste, using a model that picks the products, order, images and copy for that person from the store's real catalog.

Most storefronts show every shopper the same page. A shopper with specific taste scrolls past items they would never buy and may leave before finding the one that fits. Our model reads what the shopper has told the store and what they have done on the site, then returns a page plan for that shopper. The plan says which sections to show, which products go in each section and in what order, which product image to use, and what short titles, headings and badges to display.

Every claim on a personalized page comes from catalog facts. The model chooses among existing products and words existing facts. It never invents discounts, availability, reviews, free shipping or product capabilities.

The goal is that shoppers find a suitable item faster and merchants sell more items that fit. A larger basket counts only when the purchase is relevant, meets the shopper's requirements and stays within their budget.

Personalization Labs now includes the Fieldwork storefront, a local Qwen personalization service and a synthetic shopper evaluation dashboard. The [project overview](../../README.md) and [results guide](../RESULTS.md) describe the current implementation and measurements. The sections below retain the broader product direction and earlier architecture plans; human conversion and revenue remain future evaluation goals.

## Who it serves

The merchant is the customer. An ecommerce team wants more relevant purchases without hand-curating pages for every audience segment. They need to trust that personalized pages stay factual, respect shopper constraints and fall back to the normal page when the model is slow or unsure.

The shopper is the other stakeholder. A good page shows them things that match what they asked for and what they have shown interest in, explains cost and terms clearly, and lets them decide quickly. Sometimes the right page says the catalog has nothing that fits.

## The demo

The demo page exists to show how useful a taste-specific storefront is. It is a showcase for the model, not the product itself. The target demo:

1. Load one local fixture catalog and two or more shoppers with clearly different tastes, budgets and histories.
2. Show the default page and that shopper's personalized page side by side.
3. Let a viewer click any personalized element to see why it is there, meaning the catalog facts and observed history events that justify it.
4. Show what the model generated, what the validator rejected, where it fell back to approved catalog titles, and how long the response took.
5. Optionally, run simulated shoppers through both pages and compare how often they complete a suitable purchase. Label these results synthetic.

The fixture shop's checkout is fake. It takes no payment, address or real order.

## How it works

A request contains only what the store can observe. That means the current task, stated budget, requirements and preferences, history events up to a cutoff time, and the candidate products. Sensitive traits, hidden persona fields and future events are never model inputs.

The decision step uses [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) in a pinned 4-bit MLX conversion as a frozen backbone. One forward pass over the shared prompt feeds two small trained linear heads that score product suitability and presentation utility. Ordinary code turns those scores into product selection, order and image choice. The structured path does not ask the model to emit JSON.

The copy step reuses the same loaded backbone's decoder to write titles, section headings and optional badges. A validator checks each generated field against catalog facts and the shopper's stated constraints. Rejected text falls back to the approved catalog title or is left out.

The output follows the [page-plan v1 contract](../../experiments/ranker/pretrained/PAGE_PLAN.md). Each field records whether it was generated, rejected, left empty, filled from an approved fallback, or still pending.

When this reaches the storefront, the page renders the default immediately and requests the personalized plan asynchronously with its own deadline. If the plan is late, invalid or unavailable, the shopper keeps the default page.

## What exists today, and what comes next

| Area                  | Current repository behavior                                                                                                                                                            | Next step                                                                                   |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Personalization model | [`experiments/ranker/pretrained/`](../../experiments/ranker/pretrained/README.md) trains decision heads on a frozen Qwen3.5-9B backbone and generates validated page plans on synthetic data | Serve it behind the API with a latency budget and fallback                                  |
| Baseline ranker       | [`experiments/ranker/model/`](../../experiments/ranker/README.md) has a small engineered-feature network and a transparent heuristic with mixed synthetic results                            | Keep both as comparison baselines                                                           |
| Page-plan contract    | A v1 TypeScript interface, fixture and Python validator live under the model experiment                                                                                                | Move it into `packages/contracts/` once the storefront consumes it                          |
| Latency               | Earlier single-product responses took about three seconds, well short of near-normal page load. Batching and cache experiments are in progress                                         | Measure structured-only and full-page latency separately against a fixed target             |
| Storefront            | `apps/storefront/` is a fixture shop with config for headline, shipping visibility and recommendation order, plus a fake checkout                                                      | Render page plans per shopper, with the default page as fallback                            |
| Dashboard             | `apps/web/` shows runs from the earlier experiment loop, including persona splits and baseline/variant comparisons                                                                     | Become the demo view showing default and personalized pages with their evidence             |
| Shopper simulation    | `apps/api/` has a deterministic mock runner, a Playwright runner, an evaluator and an optimizer that suggests factual config changes                                                   | Compare default and personalized pages on held-out shoppers with independent outcome checks |

## Rules the model must follow

- Product facts, prices, fees and stock come from the catalog. Section headings may claim only attributes that every product in the section shares.
- Stated budgets and hard requirements are filters, not suggestions. When nothing fits, the page says so instead of showing unsuitable items.
- History reasons cite only observed, timestamp-valid actions such as a saved item. A badge may say the shopper saved something. It never claims to know their private intent.
- Personalization never hides fees, adds pressure tactics or pushes unsuitable items to raise spend.
- The default page is always available. A slow or failed model call must not break the store.

The current validator is lexical and deliberately strict. Passing it is not proof of semantic factuality, so we still review generated copy by hand.

## How we evaluate it

Compare arms on the same held-out shoppers, catalog and candidate products:

| Arm                         | What changes                                                |
| --------------------------- | ----------------------------------------------------------- |
| Default page                | No personalization                                          |
| Heuristic or baseline model | Transparent ranking from stated context and history         |
| Personalized, no history    | The model sees the current task and stated preferences only |
| Personalized, with history  | The same model also sees permitted observed history         |

Default versus no-history shows what the model adds from the request alone. No-history versus history shows what past behavior adds. Report context size and serving cost so a larger inference budget is not mistaken for better modeling.

Keep the data honest. Split training and test by shopper family, never train or select models on test labels, and freeze the configuration before reading final results. If we change the model after seeing held-out results, those results become development evidence and we need a fresh test set.

Report these together:

| Metric                 | What it tells us                                                                                        |
| ---------------------- | ------------------------------------------------------------------------------------------------------- |
| Ranking quality        | Whether suitable products rank above unsuitable ones against independently defined preferences          |
| Constraint violations  | Budget overruns, missing requirements, out-of-stock items and unsupported claims; any serious one fails |
| Copy outcomes          | How many generated fields passed, how many were rejected and how many fell back                         |
| Latency                | Structured decision time and full page-plan time, measured separately, plus warm and cold start         |
| Simulated task success | Suitable purchases among evaluable simulated runs, with technical failures and timeouts counted apart   |
| Basket value           | Fixture-priced total only for successful, constraint-satisfying purchases, shown with the success count |

Every displayed result names its source, whether deterministic mock, synthetic model run or later real-user evidence. Synthetic labels and simulated shoppers cannot establish human conversion or revenue lift. A neutral or negative result is valid and should be shown.

## Positioning and risks

[Shopify SimGym](https://help.shopify.com/en/manual/online-store/simgym) uses AI shoppers to test store changes. [Netflix's GenRec article](https://netflixtechblog.com/genrec-towards-llm-native-recommendation-at-netflix-f20be6f643e3) inspired how we represent history and weigh recommendation quality against serving cost. Our focus is generating a factual, per-shopper page from the store's own catalog. We do not claim to be the first to personalize storefronts or to reproduce Netflix's results.

The main risks:

- **Latency.** A personalized page that arrives seconds late is worse than the default page.
- **Factuality.** Generated copy can drift beyond what the catalog supports. The validator catches some of this, not all.
- **Synthetic data.** Our shoppers, preferences and labels are generated. A model can learn the generator's habits instead of real taste.
- **Simulator credibility.** Simulated shoppers may be unusually patient or literal, so their results only hint at human behavior.

Real-user research and a properly designed merchant A/B test are needed before any business claim.

## Scope

In scope for the demo are the local fixture catalog, the storefront, page-plan generation and validation, the side-by-side demo page and synthetic evaluation. Real customer data, real payments, catalog import, production integrations, deployment, authentication, arbitrary generated code and invented product claims are out of scope.

## Demo success criteria

- A teammate can start the local services and run the demo without secrets. Mock and synthetic outputs are labeled.
- Two shoppers with different tastes get visibly different personalized pages from the same catalog, and both differ from the default page.
- Every personalized product, heading and badge links back to the catalog facts or history events that support it.
- The demo shows rejected copy, fallbacks and latency instead of hiding them.
- Personalized pages never break a stated budget or requirement.
- The team can say what the results support, what is synthetic and what remains unvalidated.

## Decisions still open

1. Which merchant segment and catalog make the most convincing first demo?
2. What latency target counts as acceptable for a personalized page, and which serving setup meets it?
3. Which model provider or local backbone should the integrated version use?
4. How should the storefront and dashboard present the side-by-side comparison and its evidence?
5. Is there a permissioned or clearly synthetic history dataset realistic enough to test taste, not just generator patterns?
6. Does fine-tuning beyond the trained heads add value over better context and ranking, and what held-out test would show it?
7. What real-user study or A/B test would separate improvement for simulated shoppers from improvement for real ones?

Record these decisions with the experiment configuration as they are made.
