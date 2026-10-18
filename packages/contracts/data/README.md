`expanded-store-catalog-v1.json` is the active public demo catalog. It preserves the original 16 product IDs, names, prices, color SKUs and gallery facts, adds the teammate's 14 fixtures, and adds five authored rainwear/work-bag fixtures. Fixture prices describe this local demo, not verified merchant offers.

`intentTags` mark eligible product types. A folding umbrella and cotton overshirt are not rain jackets; crossbody bags and laptop sleeves are not purpose-built work bags. `parentFamily` identifies the existing learned route used for family adaptation; it does not claim that the new individual products were included in the original 16-product training.

After changing the shared catalog, run `node apps/api/scripts/expanded-catalog.mjs`, then `node --test apps/api/scripts/expanded-catalog.test.mjs`. The helper generates the active checkout price/SKU catalog and telemetry SKU catalog. Original experiment snapshots and immutable saved checkout receipts remain separate and unchanged.
