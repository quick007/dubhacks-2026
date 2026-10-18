# Local demo checkout receipts

This endpoint confirms a fake store order against the frozen public catalog. It does not make a purchase, accept payment or customer details, change inventory, contact Tiger, or assign an experiment score. Both storefront arms use the same endpoint and prices.

From `apps/api`, run `node ../../scripts/go.mjs run ./cmd/demo-checkout-server`. It listens only on `127.0.0.1:4331` and archives receipts under `data/demo-checkout-cohort-v1` at the repository root. `PORT` and `DEMO_CHECKOUT_DIR` override these defaults. It needs no credentials or environment file. For the isolated storefront, set trusted Vite configuration `DEV_API_ORIGIN=http://127.0.0.1:4331`; the browser uses the same-origin `/api/demo-checkout` proxy. The normal Go application also registers these routes using its own data directory.

POST `/api/demo-checkout` with `Content-Type: application/json`:

```json
{"schema":"fieldwork.demo-checkout.v1","checkoutId":"00000000-0000-4000-8000-000000000001","items":[{"productId":"mug","variantId":"mug-cobalt","quantity":1}],"shippingVisible":true}
```

A `202` response contains `{receipt, duplicate}`. The receipt has schema `fieldwork.demo-checkout-receipt.v1`, the checkout UUID, deterministic `receiptId`, UTC `createdAt`, priced item lines, `subtotal`, `shipping: {status, cost}`, `total`, and `realPurchase: false`. Prices and shipping are calculated in integer cents, then serialized as dollars. For the example, subtotal is 24, shipping is 5, and total is 29. Visible shipping is free strictly above $50; hidden shipping has `cost: null` and total equals subtotal.

GET `/api/demo-checkout/{checkoutId}` returns `200 {receipt}` with the identical persisted receipt. Retrying an identical POST returns that same receipt with `duplicate: true`; reusing the checkout UUID with different items or shipping configuration returns `409`. Client price fields, unknown fields or products, mismatched variants, duplicate variant lines, and invalid quantities are rejected. A request can contain at most 48 lines and 1,000 total units in 16 KiB.

The server writes a private file, fsyncs it, publishes it exclusively under the checkout UUID, and fsyncs its directory before acknowledging. Receipt files survive restarts; the archive is capped at 10,000 entries. Storage failures return `503` so the UI must not claim server confirmation. No cross-origin CORS access is enabled. The local archive should stay on the same filesystem as its temporary files, with hard-link support.

`internal/democheckout/catalog.json` pins prices, variant IDs, and shipping constants from the four recorded storefront source hashes. Its source-parity test fails when those files change. Freeze the backend commit, catalog SHA-256, storefront source, and receipt directory for a cohort. Existing DOM-only runs remain DOM evidence; only a captured POST plus matching GET receipt is server-confirmed fake-order evidence. Local receipts are not hosted telemetry or real conversion.

Validation: `node ../../scripts/go.mjs test -race ./...` and `node ../../scripts/go.mjs vet ./...` from `apps/api`. Tests cover public-source parity, authoritative pricing, exact shipping threshold, immutable retries and conflicts, private durable files, restart/readback, and invalid inputs.
