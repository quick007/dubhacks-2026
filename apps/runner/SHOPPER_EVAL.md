# Independent shopper evidence

This separate CLI prepares the fixed ten-pair development packet and ingests
actual browser sidecars. It never launches Chrome, a model, GPU work or the
legacy scripted runner. Existing runner/API wire formats remain unchanged.

From the repository root, prepare a new assignment directory:

```sh
node --import tsx apps/runner/src/shopper-eval-cli.ts prepare \
  --packet experiments/ranker/pretrained/store_shopper_eval_v1 \
  --out /tmp/shopper-assignments-v1
```

The directory contains twenty private shopper assignments, twenty separate
public recommender contexts, and a manifest in the declared paired order.
Preparation reads only the schedule, public observations, private instructions
and manifest hashes. Evaluator gold and catalog bindings never enter an
assignment. The default origin is `http://127.0.0.1:4312/`; an allowed local
origin can be supplied with `--base-url`. Keep both arms on the same origin.

Dispatch assignments manually through authorized independent internal Chrome
CUA agents. Each shopper receives its private assignment, never evaluator
answers, expected product IDs, another arm's decisions or a prepared purchase
script. Only the public context may enter the recommender. The fixed budget is
24 actions and 120 seconds from navigation dispatch, including waits.

After execution, independently normalize observed actions and timings into the
`ShopperSession` contract. Save each sidecar at
`SESSION_ROOT/CASE_ID/ARM/session.json`, with referenced artifact files beside
it. Root's initial smoke directory can be used as `SESSION_ROOT`:
`experiments/ranker/pretrained/store_shopper_eval_smoke_v1`.
Raw `result.json` or Markdown logs are preserved but are not converted into
sessions by this CLI. A screenshot emitted in chat without a durable file is
missing evidence. CUA does not imply a Playwright trace; record missing traces
explicitly without inventing files, hashes, actions or timestamps.

```sh
node --import tsx apps/runner/src/shopper-eval-cli.ts aggregate \
  --packet experiments/ranker/pretrained/store_shopper_eval_v1 \
  --sessions experiments/ranker/pretrained/store_shopper_eval_smoke_v1 \
  --out /tmp/shopper-aggregate-v1.json
```

Aggregation reads separate frozen evaluator gold and catalog bindings. Optional
`--gold` and `--catalog` paths must contain the exact bytes declared by the
packet manifest. It checks current catalog source hashes, contract fields,
chosen product/color/price and independent need checks, then verifies saved
artifact byte lengths and SHA-256 hashes. Missing sessions remain unattempted;
invalid sidecars and technical failures remain visible. Each arm keeps its
fixed denominator of ten.

Set `protocol.passed` to false and list observed deviations for exploratory
runs, including a mismatched viewport or execution before the framework was
frozen. These remain recorded observations and may increase `reportedCompleted`;
they leave the formal slots unattempted and cannot increase formal completion
or verified outcome rates. Nullable provider metadata and unrecorded step
times stay null. An elapsed upper bound is labeled through `timingPrecision`.

Observed expected outcomes are reported separately from outcomes with every
screenshot/trace file verified. An honestly missing trace does not invalidate a
recorded session, but cannot count as complete verified evidence. Browser
readiness measures are after-paint opportunities; physical paint and image
transfer are outside their scope. Missing or failed fields remain visible.
Results are synthetic development evidence, not human conversion or unseen
generalization. Output files and assignment directories are never overwritten.

`src/shopper-runner.ts` adds a separate sequential execution path. It uses a new
headed installed Chrome process and isolated storage context for every arm,
captures viewport screenshots and a Playwright trace, and enforces the action
and wall budgets while a shopper is running. The schedule permutes the ten pairs
with a recorded seed while retaining their frozen, balanced arm orders:

```sh
node --import tsx apps/runner/src/shopper-runner.ts plan \
  --packet experiments/ranker/pretrained/store_shopper_eval_v1 --out /tmp/shopper-plan
```

Actual execution uses `run`. By default it discovers the official Codex CLI,
checks `model/list` for the requested `gpt-6.1-sol` model and image input, and
uses its existing CLI-managed ChatGPT login. `--codex-binary` selects an installed
CLI explicitly; `--model` requests a different available model without silently
substituting one. Each screenshot decision runs with `--ignore-user-config`,
`--ephemeral`, a read-only sandbox and an empty temporary working directory.
Shell, apps and delegation are disabled; evaluator data never enters the prompt.
No authentication token is read or copied by this harness.

With the storefront and local model service ready, run a fresh two-session smoke
or omit `--limit 2` for all twenty sessions. Use a new output directory each time:

```sh
node --import tsx apps/runner/src/shopper-runner.ts run \
  --packet experiments/ranker/pretrained/store_shopper_eval_v1 \
  --out /tmp/shopper-sol-new-smoke --seed 11 --limit 2
```

Alternatively, `--agent /absolute/installed/adapter` with optional
`--agent-args-json '["argument"]'` starts an external adapter process for each
session. It receives JSON lines with the private assignment, a viewport
screenshot, visible text/control refs, and remaining budgets. It returns one JSON
action line: `{"type":"click","ref":"control-1"}`, `{"type":"search","query":"assigned public query"}`,
`{"type":"scroll","direction":"down"}`, or `{"type":"abandon","reason":"observed reason"}`.
The adapter receives no evaluator catalog or gold data. Only the assigned public
query can enter search. Catalog identity and constraints are verified after the
shopper exits. Arbitrary scripts, selectors and navigation commands are rejected.

The installed app bundles the CLI at
`/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex` even when
`codex` is absent from PATH. If no CLI is installed or the requested model is
unavailable, execution stops before browser dispatch. For a missing login, use
the official `codex login` flow; do not copy authentication files. Existing CUA
shopper receipts remain exploratory evidence. Do not substitute the test fixture
adapter for a real shopper. The opt-in
`SHOPPER_CHROME_SMOKE=1` test checks Chrome isolation and artifact plumbing only;
it makes no shopper choices, model requests or checkout attempts. Coordinate
with the model owner before running any local-model arm.

After aggregation, an optional separate command posts an immutable snapshot to
the local results API. It runs outside shopper timing and never reads database
credentials or contacts the cloud database directly:

```sh
node --import tsx apps/runner/src/shopper-export.ts \
  --report /tmp/new-run/aggregate.json --report-id shopper-smoke-attempt-1
```

The endpoint is fixed to `http://127.0.0.1:4311/api/experiment-results`. A `202`
receipt confirms the local durable archive and asynchronous cloud handling; it
does not confirm a completed cloud write. Retry the same snapshot with its same
ID; use a new ID when evidence changes. Export requires the separate results API
to be running and is never invoked automatically by the shopper runner.

The real Sol smoke at `experiments/ranker/pretrained/store_shopper_sol_smoke_v2`
records two independently verified simulated checkouts: generic 12 steps/72.553 s,
local-model 8 steps/44.817 s. Both selected one qualifying $36 Graphite bottle.
The earlier repeated-add timeout pair remains in `store_shopper_sol_smoke_v1`.
The full development suite is saved in `store_shopper_sol_full_v1`: twenty actual
attempt receipts, including the two unchanged successful smoke receipts. The
remaining eighteen slots ran in the original seeded order with the same six
implementation hashes, viewport, model and budgets. Generic achieved eight of
ten verified expected outcomes (six simulated checkouts and two correct
abstentions); local-model achieved seven (five checkouts and two correct
abstentions). Both arms incorrectly abstained on the waterproof-jacket case.
Generic recorded one timeout. Local-model recorded one timeout and one browser
interruption after the Mac connection dropped. These attempts were preserved
without retries. Seventeen slots meet the full protocol; all twenty remain in
the fixed denominator. These are synthetic development observations and do not
establish human conversion lift.

`aggregate.json` is the unchanged raw aggregate. `aggregate-audited.json` and
`independent-audit.json` verify every receipt, available artifact hash, isolated
browser identity, frozen implementation/packet hash and independent outcome.
`summary.json` separates purchases from correct abstentions and summarizes
steps, preferences and measured runtime. The interrupted attempt's 349 ms
observation span cannot prove total runtime or budget compliance; the audit
marks its budget unverified and excludes it from total-runtime statistics.
All nineteen available trace archives passed ZIP integrity checks. Large trace
archives and duplicate observation images remain local; normalized observations,
final screenshots and artifact hashes are committed. One interrupted trace is
honestly missing. One local-model session recorded fallback use.

The result-only audit can be invoked once after a completed suite has saved all
twenty receipts. It never starts browsers or requests a model, and refuses to
overwrite its outputs:

```sh
node --import tsx experiments/ranker/pretrained/store_shopper_sol_full_v1/audit-final.mts
```

The full-suite snapshot has not been exported. The earlier smoke v1/v2 immutable
snapshots were separately approved and exported by the database owner. A new
full-suite export requires its own report ID and authorization for that payload.

CPU checks (fixture sidecars are not browser execution evidence):

```sh
node --import tsx --test apps/runner/src/shopper-eval-cli.test.ts
```
