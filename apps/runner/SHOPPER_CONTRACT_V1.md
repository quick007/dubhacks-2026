# Paired Chrome shopper evidence

`src/shopper-contract.ts` validates external case/session sidecars. It does not
change `RunSchema`, the CLI wire protocol, or the legacy scripted Playwright
adapter. It never launches a browser or model. The legacy first-button script is
not a substitute for these independent shopper sessions.

Prepare ten inspected synthetic cases before running either arm. Each case has
one private shopper goal, typed hard constraints/preferences, one separate public
recommender context, a seed, split, and declared arm order. The packet under
`experiments/ranker/pretrained/store_shopper_eval_v1` owns the cases and evaluator
gold; this module does not duplicate that dataset. Hand-authored inspected cases
remain development evidence, not unseen human conversion evidence.

Pass `buildShopperAssignment(case, arm, baseUrl?)` to each shopper agent. It
includes private shopper needs and the explicit public query in separate fields.
It excludes evaluator gold, target product/variant IDs, catalog answer keys and
the full `Run`. Manually audit prose for answer leakage too; schema validation
cannot detect a gold answer hidden in ordinary text. Agents may inspect and
choose independently but may only disclose the assigned observable query and
constraints to search/recommender inputs. Both arms use that same query. The
generic arm retains the storefront's existing deterministic search behavior.

Only headed Chrome actual navigation counts as a browser shopper attempt.
Allowed roots are the named local storefront and the declared localhost/127.0.0.1
port 4312 origins. The local arm adds `?model=local`; a generic arm has no model
parameter. Use a fresh tab per arm. Record `fresh-tab` with shared storage and no
claim of cleared cookies, or record a genuinely `fresh-storage-context` with
cleared, unshared storage. Verify the initial cart is empty. Record actual browser
version, tab/context identity and evidence. A provider may expose neither a Chrome
version nor a storage-context ID: keep `browser.version: null` and
`isolation.contextId: null`. A fresh tab still shares origin storage. A prelaunch technical failure instead
uses `not-created`, null identities/storage observations, no actions and no choice.

Session sidecars link to existing `Run` IDs/seeds/splits and nest existing
`StepEvent` and `Artifact` concepts. Every step includes an explicit actual action.
Observed elapsed times must be monotonic; missing action clocks and wall timestamps
remain `elapsedMs: null` and `event.at: null`. Required `timingPrecision` distinguishes
measured session time from a tool-wall upper bound, and measured, partial or
unrecorded per-action times. Record the actual chosen product, variant, original
name, unit catalog price, color, cart action and independent evidence. Version 1
evaluates one item per shopper. Confirm only the fake demo order; real purchases
and payment actions are forbidden.

Exploratory protocol deviations may declare `sessionElapsed: "estimated"` and
`startedAt: null` if no start clock was recorded. An estimate is never a proven
upper bound or a formal latency sample. With unrecorded per-action clocks, those
exploratory sessions may keep readiness `capturedAtElapsedMs: null` while retaining
the exact browser performance-entry start time/duration. Do not invent an ISO time
or a capture-relative clock to fill a schema field.

An independent verifier supplies an authenticated catalog snapshot with explicit
typed facts. `verifyShopperNeeds` compares the chosen item to those facts. Missing
stock or other facts stay unknown. `includes` means exact membership in a typed
array; it never extracts capabilities from substrings or negated prose. Do not
create absent facts from model claims. A completed outcome requires a real fake
confirmation action, an actual matching choice and every hard constraint met.
Preferences are reported separately and do not become mandatory silently.
An explicitly requested exact product name can be checked using canonical
`originalName`; a duplicated catalog fact must agree with that canonical name.

Capture screenshot and trace receipts with actual SHA-256 hashes and byte counts.
If CUA cannot export a trace, record a missing trace with a reason and null hash;
never synthesize a trace. References may point only to captured evidence.
`verifyShopperArtifactFiles` reads local files, checks exact bytes/hashes and
rejects path/symlink escapes. Missing or mismatched artifacts must stay visible in
the aggregate report even if the shopper completed the fake order.
If screenshots exist only in the CUA conversation, retain an explicit missing
screenshot receipt. A durable raw result JSON may support a choice labeled
`observationSource: "agent-reported"`; it does not become independent DOM evidence.

For model readiness, copy actual standard performance entries or the
`data-local-model-readiness` DOM attribute into an evidence artifact. Preserve the
request ID, measure stage/name, duration, captured session elapsed time and fresh/
withheld field counts. These measure React commit plus two animation frames: an
after-paint opportunity, excluding physical paint and image transfer/decode.
Ranking-only timing cannot be called complete generated-copy readiness. Record
model failures, missing observations and any catalog/last-safe fallback explicitly.

`validatePairedShopperSuite` requires ten cases and twenty receipts, preserves
declared pair order and failed attempts, and uses ten as each arm's denominator.
The preparation/aggregation CLI may retain missing sessions as unattempted rows;
it must not fabricate successful receipts to satisfy this full-suite validator.
Every session declares `protocol: { passed, deviations }`. An exploratory smoke
with a viewport mismatch or other protocol deviation keeps `passed: false` and
cannot contribute to formal completion counts. `reportedCompleted` retains all
reported outcomes; `verifiedDomCompleted` separately counts formal sessions whose
choice has independent DOM verification. This distinction must survive reporting.
Do not tune model checkpoints against these shopper outcomes.

CPU validation commands, from `apps/runner`:

```sh
../../node_modules/.bin/tsc --noEmit
node --import tsx --test src/shopper-contract.test.ts
```

The existing package test command intentionally remains unchanged.
