# Documentation guide

The [root README](../README.md) is the current entry point for setup, capability
boundaries and measured status. Older design plans and model experiments use
catalogs and contracts that differ from the live 16-product store.

| Task | Reference |
|---|---|
| Store/query experiment evidence | [Results adapter](../apps/api/EXPERIMENT_RESULTS.md); local receipts and asynchronous hosted sync are distinct |
| Run the dashboard/API/default shop | [Root setup](../README.md), [Go API ownership](../apps/api/README.md) |
| Run the resident Qwen opt-in | [Local service guide](../experiments/ranker/pretrained/STORE_DEMO_SERVICE.md) |
| Inspect assets and generic search | [Storefront guide](../apps/storefront/README.md); its future-model notes predate the opt-in integration |
| Understand the scripted browser protocol | [Runner README](../apps/runner/README.md) |
| Prepare or aggregate independent shopper evidence | [Shopper CLI](../apps/runner/SHOPPER_EVAL.md), [session contract](../apps/runner/SHOPPER_CONTRACT_V1.md) |
| Inspect the fixed 10-pair scenarios | [Shopper packet](../experiments/ranker/pretrained/store_shopper_eval_v1/README.md) |
| Read the exploratory Chrome pair | [Smoke evidence](../experiments/ranker/pretrained/store_shopper_eval_smoke_v1/README.md) |
| Compare native conventional and hybrid execution | [Matched results](../experiments/ranker/pretrained/STORE_AR_COMPARISON_V1_RESULTS.md), [independent quality audit](../experiments/ranker/pretrained/STORE_AR_COMPARISON_V1_INDEPENDENT_AUDIT.md) |
| Check copy and budget disclosure coverage | [Transparency audit](../experiments/ranker/pretrained/STORE_TRANSPARENCY_AUDIT_V1.md) |
| Plan the later model performance window | [Overnight queue](../experiments/ranker/pretrained/STORE_OVERNIGHT_PERFORMANCE_QUEUE_V1.md); CPU preparation is not native execution |
| Follow the longer-term product direction | [Vision](../VISION.md) |
| Find earlier results, proposals or design plans | [Historical index](archive/README.md) |

## Evidence conventions

The dashboard mock, scripted checkout, native model requests and independent
shopper sessions have different success criteria. Keep their denominators and
timing scopes separate. A fast ranking result excludes fresh copy, and complete
copy serialization can still contain withheld fields. Browser measures cover
DOM commit and paint opportunities; image loading and physical paint remain
outside their scope.

Active weights, virtual environments and native serving checkpoints live under
ignored `.cache/`, `.tools/` and `data/`. Checked-in source, public fixtures and
selected receipts document how those local artifacts were produced. A CPU
preflight or test establishes its own checks, not a native run. Prepared shopper
assignments are not completed sessions.

## Keeping the docs useful

Update the current entry points when behavior changes. Keep experiment reports,
source-hashed Markdown and `AGENTS.md` at their existing paths so manifests and
evidence references continue to resolve. Link completed work from the historical
index instead of moving or rewriting it. The optional
[Tiger Cloud note](tiger-cloud.md) describes an earlier machine-specific setup;
the core Go application uses SQLite. Optional experiment evidence uses the
[current results adapter](../apps/api/EXPERIMENT_RESULTS.md) and `TIGER_DATABASE_URL`;
it does not replace storefront storage.
