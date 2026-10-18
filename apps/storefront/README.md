# Storefront image library

The category row below search filters the visible catalog. Example-search chips offer the phrases the demo recognizes. The header's "Demo" control opens the structured settings the runner can also pass in the URL: store color, headline, shipping visibility and product order. The shopper's explicit color choice is saved locally under `fieldwork.store-color`; it is not inferred from their search or persona. These controls do not contact a model or alter image selection.

Pages and filters live in the hash (`#collection?q=…&c=…`, `#cart`, `#confirmed`), so a link is shareable and the browser Back and Forward buttons step through the shop. Demo settings stay in the query string (`?showShipping=true&recommendationOrder=budget`); the Demo control keeps that query in step as settings change.

Every product has three color choices, defined in `src/variants.ts` with stable variant IDs. Selecting a color shows its corresponding photograph. Every color has the same gallery views as the original: studio angles, details, backgrounds and lifestyle scenes where available. Alternate-color contact sheets use explicit tile coordinates so main photos and thumbnails show the same square view. Alternate-color photos are generated using the original product as a reference. The lunch box varies its retaining band's color while keeping the steel box and bamboo lid unchanged.

The cart records product and variant IDs, groups repeated additions of the same variant and keeps different colors on separate lines. All colors of a product share its fixture price. Color selection survives category/search changes and visits to the cart within the current page session. No model selects colors automatically.

When shipping details are visible, the cart restates the same fixture terms as the hero — free over $50, $5 flat rate otherwise — and the demo total adds the flat rate. With shipping hidden, delivery terms stay out of the cart, which is the measurable difference the shipping-visible variant provides.

The demo keeps the same product, price and cart identity while changing the context in which it is shown. Search wording expresses an interest in a use case; it does not establish who a person is.

The catalog now has 16 products. The daypack and rain jacket remain the detailed multi-context examples. Twelve additional versatile products broaden the range of shopping tasks, materials, prices and presentation candidates.

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
