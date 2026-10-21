# Results and methodology

[Project overview](../README.md) · [Full development and experiment history](DEVELOPMENT.md)

## Synthetic merchandise value: +3.78%

The frozen-v5 cohort completed **60 paired scenarios / 120 assigned sessions**.
Valid simulated merchandise per assigned session rose from **$42.35 to $43.95**:
`(43.95 / 42.35 - 1) × 100 = 3.778040…%`, rounded to **3.78%**.
Totals were **$2,541 and $2,637**, a **$96** difference. One generic shopper
decision timeout on the $96 sneakers task accounts for that difference.

| Outcome | Generic | Personalized |
| --- | ---: | ---: |
| Intent-correct success | 58/60 | 57/60 |
| Suitable checkout on feasible assignments | 46/48 | 47/48 |
| Checkout per assigned session | 46/60 | 47/60 |
| Valid merchandise per assigned session | $42.35 | $43.95 |
| Timeouts, retained in assigned denominators | 1/60 | 1/60 |

These are synthetic shoppers and simulated purchases. The paired 95% interval for
the merchandise difference is **$0 to $4.80 per assigned session**. The result is
an observed benchmark uplift, with no demonstrated human conversion or causal lift.
The cohort used the earlier **16-product frozen-v5 service**; it has not been
rerun on the current 35-product demo. Eight separate controls are excluded.

Sources: [audited cohort report](../experiments/ranker/pretrained/store_shopper_cohort_v2/REPORT.md),
[machine-readable slide metrics](../experiments/ranker/pretrained/store_shopper_cohort_v2/slide-metrics.json),
[archive/readback verification](../apps/api/checkout-cohort.verification.json).

## Warm backend personalization: 86 ms

The source records **85.5174585012719 ms median**, rounded to **86 ms** on the
presentation slide. This is a median, in **milliseconds**, rather than an average
page-load time in seconds. Ten requests each returned six cards and 20 accepted
fresh fields with none withheld. All ten reused exact query features (zero native
query-encoding forwards); ranking, selection and copy were freshly computed.

The clock includes input checks, model selection, word predictions, factual gates
and serialized events. It excludes startup, network, browser and images. Two
separate fresh-query backend samples took **266.8 and 310.9 ms**.

Sources: [ten raw warm samples and source hashes](../apps/web/src/fieldwork/current-warm-speed.json),
[current live summary](../experiments/ranker/pretrained/expanded_store_results/v1/CURRENT_LIVE_SUMMARY.json).
The local original is `data/ranker/pretrained/expanded-suggestions-validation-v2/report.json`;
its SHA-256 is recorded in the checked-in warm-speed source.

## Visible products: 139 ms

Ten loaded-shop profile switches reached three fully visible cards with decoded
images in **138.75 ms median** (**138.8 ms** to one decimal, **139 ms** rounded).
The recorded p95 is **166.295 ms**. Profiles/query features were warm and browser
caches eligible; startup and initial navigation are separate. Full six-card image
completion was not established. This browser run is separate from the 85.5 ms
backend sample batch.

Source: [current live summary and raw browser samples](../experiments/ranker/pretrained/expanded_store_results/v1/CURRENT_LIVE_SUMMARY.json).

## Ranking and catalog coverage

The balanced known-catalog audit reproduced **29/32 correct first choices** and
**32/32 targets in six**, versus **14/32** and **29/32** for the prior ranker on
the same independently authored requests. This earlier audit is separate from
the expanded demo and shopper cohort.

Source: [balanced ranker audit](../experiments/ranker/pretrained/function_v5_audit_v1/INDEPENDENT_AUDIT.md).

The current catalog contains **35 products and 105 actual color variants**. Its
parallel word head fitted **1,575/1,575 supervised fields** after **400 CPU updates**.
That is known-catalog training fit; unseen-catalog quality remains a separate test.
The local training receipt is
`data/ranker/pretrained/expanded-parallel-copy-fit-v1-attempt-1/progress.json`
(final step 400: fields, accepted and exact all 1,575).

Sources: [expanded serving contract](../experiments/ranker/pretrained/STORE_EXPANDED_ENGINE_V1.md),
[preserved current-demo notes](DEVELOPMENT.md#current-personalized-demo).

## Speed graph

The README retains the actual exported presentation graph: **86 ms** for warm
local six-card backend personalization, alongside approximately **6.2 s** and
**4.3 s** medians for earlier hosted ranking tasks through the Codex client.
Each series is measured within its own request set and timing boundary. The chart
is descriptive; no matched speedup ratio is derived.

Sources: [chart series and boundaries](../apps/web/src/lib/latency.ts),
[hosted request measurements](../apps/web/src/fieldwork/llm-run.json),
[local warm measurements](../apps/web/src/fieldwork/current-warm-speed.json).

## The presentation's illustrative latency scenario

The deck retains **3.78% − 0.52% = 3.26 percentage points** as an illustrative
trade-off. The 0.52% term applies an assumed 0.6% revenue loss per 100 ms to an
86 ms input, rounding to two decimals before subtraction. This combines separate
experiments and a transferred delay assumption; it is not measured net revenue.
Warm backend time is not measured additional page-load delay.

Sources: [scenario definition and pinned evidence](../apps/web/src/fieldwork/illustrative-scenario.json),
[scenario calculation](../apps/web/src/lib/illustrative-scenario.ts).

## Visual provenance

The README uses compact figures with white backgrounds and captions outside the
image. The latency figure renders the existing `LatencyChart` and `LatencyRow`
components with the original raw samples, seeded bootstrap, linear axis and
markers. Only its export typography, colors and surrounding frame change.

| README figure | Source |
| --- | --- |
| [Storefront sequence](assets/shopper-sequence.png) | The existing public `browse.jpg`, `stuck.jpg` and `fit.jpg` captures, with panel letters |
| [Latency plot](assets/latency-figure.png) | Existing chart components and checked-in local/hosted measurements |
| [Architecture](assets/architecture-figure.svg) | The four stage labels from `ModelSlide.tsx`, redrawn as a compact vector diagram |

[Figure receipt](assets/figure-export.json) records source hashes, exact medians,
95% median intervals, sample counts and output hashes. Reproduce the exports with:

```sh
pnpm --filter @storefront/web build
TSX_TSCONFIG_PATH=apps/web/tsconfig.json node --import tsx scripts/export-readme-figures.mjs
```

Original full-slide exports remain at `personalization-labs.png`,
`speed-comparison.png` and `model-architecture.png` in this asset directory.
Their original source paths and hashes remain in [asset provenance](assets/provenance.json).

## Further evidence

All earlier README setup, model failures, audits, timing comparisons and shopper
results are preserved in the [development guide](DEVELOPMENT.md), with their
original source links. Historical experiment files and numerical evidence are
unchanged.
