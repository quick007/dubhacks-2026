# Storefront image library

The editorial hero and photographic category shortcuts come from the teammate storefront follow-ups. Search sits at the top right of the header; categories filter the visible catalog. The public persona selector retains its biography/history and current model integration. While a personalized request is pending, the hero and six product positions show placeholders; generic or previous-person results are not substituted. Current first-three cards can appear while the remaining selected cards finish.

Pages and filters live in the hash (`#collection?q=…&c=…`, `#cart`, `#confirmed`), so a link is shareable and the browser Back and Forward buttons step through the shop. Demo settings stay in the query string (`?showShipping=true&recommendationOrder=budget`); the Demo control keeps that query in step as settings change.

Every product has three color choices, defined in `src/variants.ts` with stable variant IDs. Selecting a color shows its corresponding photograph. Every color has the same gallery views as the original: studio angles, details, backgrounds and lifestyle scenes where available. Alternate-color contact sheets use explicit tile coordinates so main photos and thumbnails show the same square view. Alternate-color photos are generated using the original product as a reference. The lunch box varies its retaining band's color while keeping the steel box and bamboo lid unchanged.

The cart records product and variant IDs, groups repeated additions of the same variant and keeps different colors on separate lines. All colors of a product share its fixture price. Color selection survives category/search changes and visits to the cart within the current page session. No model selects colors automatically.

When shipping details are visible, the cart restates the same fixture terms as the hero — free over $50, $5 flat rate otherwise — and the demo total adds the flat rate. With shipping hidden, delivery terms stay out of the cart, which is the measurable difference the shipping-visible variant provides.

The demo keeps the same product, price and cart identity while changing the context in which it is shown. Search wording expresses an interest in a use case; it does not establish who a person is.

The UI catalog now has 35 fixture products: the original 16, 14 teammate additions, and five new rain/work-bag fixtures. The original 16 names, prices, variant IDs and galleries stay exact. `packages/contracts/data/expanded-store-catalog-v1.json` supplies the additional public facts and 3×3 image-sheet coordinates. Model and checkout adapters must use the matching catalog version.

| Product                | City commute       | Campus                  | Family days                 | Trails             | Other views                            |
| ---------------------- | ------------------ | ----------------------- | --------------------------- | ------------------ | -------------------------------------- |
| Everyday daypack       | City sidewalk      | Student carrying books  | Parent walking with a child | Forest hike        | Front, back, material detail           |
| Everywhere rain jacket | Rainy street       | Rainy college courtyard | Soccer sideline             | Misty forest trail | Front, back, material detail           |
| Trail mug              | Uses home fallback | Uses home fallback      | Uses campsite fallback      | Lakeside campsite  | Front, back, detail, breakfast at home |
| Pocket field notes     | Uses desk fallback | Uses desk fallback      | Uses outdoor fallback       | Lakeside writing   | Front, back, detail, home desk         |

## Try it

Start the shop and submit a phrase such as `rain jacket for campus`, `work bag`, `hiking gear` or `family park`, or pick one of the example chips below the search field. The headline, collection heading, ordering and lead image change together. Every photo remains available through the thumbnails. Clearing the search restores the baseline headline and order, with product-focused still-life photos leading each product gallery (textured/tabletop settings where available, otherwise the studio view). Studio, detail and alternate-setting views remain available through thumbnails.

The search policy is a local deterministic keyword demo, not a model or measured personalization result. A product-specific term outweighs a contextual ranking boost. Unknown queries preserve the baseline. Current search state stays in memory for this visit; there is no browsing-history collection or external inference request.

## Where the data lives

- `src/config.ts`: fixture product IDs, names, prices and variant parsing.
- `src/catalog-expansion.ts`: 12 additional products, descriptive search terms and image briefs.
- `src/images.ts`: image IDs, source paths, captions, alt text, context and descriptive tags. Studio triptychs use explicit crop positions; each lifestyle image is its own file.
- `src/personalization.ts`: replaceable query-to-context and ranking policy.
- `src/ProductGallery.tsx`: displays a chosen image and lets the shopper override it.
- `public/products/`: image assets and generation prompts. Images are AI-generated fixture media. Existing product reference images anchor identity across contexts.

Use `pnpm --filter @storefront/storefront test` to check intent ranking, baseline fallback, image compatibility and asset existence. `pnpm --filter @storefront/storefront build` checks types and bundles the page.

## Future model handoff

A future server adapter can select ordered product IDs and a known image ID per card, then return proposed text with provenance. Resolve IDs against the catalog; keep price and capabilities in the fixture facts. Generated wording should use approved product facts, while image captions describe what the image actually shows. Section ordering can be added independently of the image library.

The experimental `experiments/ranker/pretrained/page-plan.v1.ts` contract is not consumed here. Its synthetic catalog uses different IDs. Connect it through an explicit catalog/image mapping and validate membership before rendering; do not assume its fixture IDs match this shop. Keep baseline and candidate selections recorded and replayable when integrating with the experiment loop described in `VISION.md`.

The six new lifestyle assets intentionally vary setting, activity, clothing, adult age range and people represented. Those casting choices make a broader asset library; they are not rules for inferring a shopper's age, ethnicity, gender or family status. Selection currently uses only the submitted search words.

## Presentation is independent of use case

The best first image may be a product shot rather than a person using it. Each image therefore also carries `presentation` metadata: subject (`product` or `in-use`), background, tone, and composition. These attributes are separate from the context and descriptive tags.

The notebook has four additional product-only candidates: plain ivory, scratched concrete, warm oak, and dark charcoal. Its material, color and elastic closure stay consistent. The jacket has light and dark flat lays alongside its studio angles and four lifestyle scenes. These are candidate presentations, not proven preference matches.

All candidates are available through the thumbnails. There is no search-keyword rule choosing an aesthetic or declaring a winning lead image. The context demo remains a separate, replaceable policy. A model or controlled experiment should select individual image IDs based on evidence, with a neutral baseline for comparison.

Future experiments can vary an image ID independently of product ranking, text, section placement and other image IDs. Evaluate lead-image selection rather than assume a lifestyle scene is best. The current demo does not record or learn visual preferences and does not claim any candidate increases conversion.

## Expanded products and persona briefs

The additional catalog covers a steel bottle, canvas tote, woven throw, portable lamp, steel lunch box, over-ear headphones, canvas sneakers, cotton crewneck, weekender duffel, round sunglasses, organizer pouch and striped towel. Each has three gallery candidates: a clean studio view, an alternate still life and a photo in use. The three equal panels share one source board; `position` identifies the panel for display. Consumers need both the source and crop position when resolving image IDs.

`fixtures/shopper-personas.json` contains 11 synthetic asset briefs: campus student, focused commuter, family outing organizer, weekend wanderer, remote worker, active routine shopper, everyday walker, quiet homebody, light traveler, city explorer and creative hobbyist. Each records a shopping goal, compatible candidate product/image IDs and open visual questions. They are examples for exploring the catalog, not automatic recommendations, learned preference labels, or the development/held-out personas used by the experiment harness.

Lifestyle casting varies across the catalog, while product-only alternatives remain available for every item. No persona picker, new ranking model, model endpoint or automatic visual-preference policy is added. The existing local search demo simply recognizes the new product names and related terms. Prices are local fixture values, not real merchant offers.

## Automatic shopping telemetry

Configured shop deployments collect anonymous shopping events and masked session replay automatically, without a demo toggle or Start button. Ordinary local development/test builds stay off. Public build settings are in [`.env.example`](.env.example): `VITE_SHOP_TELEMETRY=production`, exact `VITE_SHOP_TELEMETRY_HOSTS`, and `VITE_SHOP_TELEMETRY_API_ORIGIN` require a production build, HTTPS and a non-local shop host. The backend must independently enable the matching environment and exact browser origin; no URL parameter can promote local/test traffic into real data. Never put credentials in `VITE_` settings.

Consent defaults to `required`: the Shopping privacy control offers Allow/Keep off before collection. A deployment approved for notice plus opt-out can set `VITE_SHOP_TELEMETRY_CONSENT=notice`; its visible “Anonymous recording on” control exposes the disclosure and Stop sharing. Saved opt-out, Global Privacy Control and Do Not Track prevent recorder loading/uploads. Opt-out discards queued data and stops future collection; it does not erase previously accepted data. The only persistent browser preference is granted/denied; session UUIDs stay in memory and do not identify a person across visits/sites.

[`shopping-events.ts`](src/shopping-events.ts) defines `fieldwork.shopping-batch.v1`. Each batch has UUID batch/session IDs, explicit environment/real-or-synthetic classification and epoch-ms start time; each event has UUID identity, sequence, epoch-ms timestamp, elapsed-ms, structured mode/order/shipping/model-version context and allowlisted action data. `catalog-view` records rendered product/color/position denominators; `product-view` requires at least half a card visible for 250 ms. Search records only submit/example/clear source, never wording. Selection, cart add/clear and **fake demo checkout** retain fixture IDs/counts. Model readiness reuses the existing PerformanceMeasure stages/durations and copy provenance; it means DOM commit plus two frame callbacks, not physical paint/image readiness. `VITE_SHOP_MODEL_VERSION` supplies a trusted artifact reference; unavailable local-model versions remain null.

All page text is masked; inputs, account/payment/checkout/cart/confirmation, iframes and `[data-trace-private]` regions are blocked. Query/hash, titles, arbitrary attributes, free text, console/network payloads/headers and credentials are excluded. Replay maps known responsive image derivatives back to their public original fixture. Official MIT [`@rrweb/record` / `@rrweb/replay`](https://github.com/rrweb-io/rrweb/blob/main/guide.md) 2.1.7 remain lazy loaded. Replay stops at 5 minutes, 2,000 events or 4 MiB; shopping collection is bounded to eight hours. New sensitive regions must carry `data-trace-private` before rendering content.

Uploads to the configured own backend use a volatile 1 MiB/1,000-item queue, batches up to 100 shopping/100 replay events, a two-second flush, eight-second request timeout and at most three attempts with the same immutable batch ID/body. Credentials/referrer are omitted, work does not block shopping, and queue/permanent failures stop a broken replay chain. Data can be lost on browser close, opt-out or exhausted bounds/retries. `202` confirms a durable local receipt; Tiger completion is asynchronous. Replay has separate 24-hour backend retention and never enters training features. [Backend configuration/export documentation](../api/SHOPPING_TELEMETRY.md) describes real-data selection, session-level splits and positive choice derivation. Ingestion/export does not train or deploy a model.

For isolated validation only, use `VITE_SHOP_TELEMETRY=synthetic`, an exact loopback host and `test`/`development` environment plus the matching explicitly synthetic backend policy. Synthetic observations are excluded from real training exports/builders by default. Build an isolated output with those settings and `pnpm --filter @storefront/storefront exec vp build --outDir ../../data/session-trace-auto-dist`, then run `node apps/storefront/scripts/verify-shopping-telemetry.mjs` during an agreed model-idle window. It serves loopback 4329, uses backend 4330, makes no model calls, verifies automatic collection/opt-out/DNT/canaries, and reconstructs the actual backend-sanitized replay. Evidence goes under ignored `data/session-trace-auto/`; the checked-in automatic verification summary records the tested IDs and results.

`pnpm --filter @storefront/storefront test` covers configuration/privacy gates, masks, catalog identity, bounds, stable retries, permanent errors and browser API binding. The earlier local-only debug mode remains available with `?trace=demo` for manual memory/export inspection; normal configured shoppers do not use it. Its original synthetic fixture/CPU proxy remain historical spike evidence, not automatic-ingestion performance claims.

## Teammate storefront follow-ups

Integrated fork `xclarityy/dubhacks-2026` branch `feat/storefront-follow-ups` through `e1134b2`: editorial hero with six available wide photo boards, category photography, header search, cart quantity/remove controls and keyboard gallery navigation. Canonical product IDs, names, prices, variants, responsive assets, model generation, telemetry and server-confirmed checkout are preserved. The normalized 35-product public packet is consumed through `src/expanded-catalog-adapter.ts`; activation requires all 35 entries. Fresh model quality and readiness are established separately by the serving owner’s bound validation receipts.

Hero assets and generation prompts are documented in [`public/products/HERO-PROMPTS.md`](public/products/HERO-PROMPTS.md). The hero keeps one factual editorial image selected from that person’s learned product order; it has no timer or slideshow. It hides during a new request and changes when the person changes. Galleries reveal additional views with a count button; arrow keys, Home and End navigate photographs. Cart mutations invalidate an earlier confirmation receipt. The loading-state rendering tests are in `src/shop-loading.test.ts`.

Suggested searches load only from `/search-suggestions.json` with schema `fieldwork.validated-search-suggestions.v1`. The serving owner publishes exact query strings after validating and preencoding them. Missing or malformed metadata yields no suggestions; the UI never invents browsing history. Arbitrary searches remain available.
