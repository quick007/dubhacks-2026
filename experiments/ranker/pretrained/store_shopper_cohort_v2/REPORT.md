# Synthetic shopper cohort v2

The frozen 60-pair cohort did not demonstrate an improvement in primary intent
success. Generic scored 58/60 and local-model scored 57/60 after independent
negative-outcome review. Local-model recorded one additional checkout; that
single discordant purchase came from a generic shopper decision that timed out.

All 120 assigned sessions ran. No attempt was replaced or retried. The raw
outcomes are 93 completed checkouts, 25 abstentions and two timeouts. There were
118 protocol-passing receipts; both timeout receipts remain in every assigned
denominator. All results describe synthetic agents and simulated purchases.

## Outcomes

Differences are local-model minus generic. Intervals are the preregistered
20,000-replicate paired-case percentile bootstrap, seed 20261018.

| Measure | Generic | Local-model | Paired difference, 95% interval |
| --- | ---: | ---: | ---: |
| Intent-correct success | 58/60 (96.7%) | 57/60 (95.0%) | −1.67 pp [−8.33, +3.33] |
| Prespecified feasible checkout | 46/48 (95.8%) | 47/48 (97.9%) | +2.08 pp [0, +6.25] |
| Checkout per assigned session | 46/60 (76.7%) | 47/60 (78.3%) | +1.67 pp [0, +5.00] |
| Valid simulated merchandise per assigned session | $42.35 | $43.95 | +$1.60 [$0, +$4.80] |
| Correct abstention on negative assignments | 12/12 | 10/12 | Included in primary intent success |
| Preference match among verified purchasers | 46/46 | 47/47 | Synthetic preference proxy |
| Timeout | 1/60 | 1/60 | Both retained |
| Median observed steps, including navigation | 7 | 9 | Descriptive |
| Median measured total elapsed | 32.07 s | 40.02 s | Descriptive |

Primary paired counts are one local-model win, two generic wins and 57 ties.
The checkout and merchandise differences each have one local-model win and
59 ties across all assigned pairs. The $96 sneakers task accounts for the
entire $96 merchandise-total difference: generic timed out during an unfinished
shopper decision, while local-model completed checkout. This observation does
not establish that personalization caused the extra purchase.

All accepted server orders also passed the independent purchase checks, so
valid-purchase and all-server merchandise totals coincide: $2,541 generic and
$2,637 local-model. Quantity is one. Delivery was hidden in all 93 orders;
delivery charges remain unknown. These amounts are simulated merchandise
subtotals, with no real payments or revenue.

Private preference checks were met for all 93 purchasers. Across the 48
preference-bearing assignments per arm, generic had 46 met and two unknown;
local-model had 47 met and one unknown. Assignment-wide purchased preference
credit is 46/60 versus 47/60. This tracks purchase completion as well as fit;
it is not observed customer satisfaction.

The elapsed clock starts before fresh Chrome setup and entry navigation and
includes agent decisions, browser actions and waits through the observed
outcome. It excludes cleanup and scoring. All 120 total durations are measured
monotonically, including the bounded failures. These are synthetic task times.

## Fixed design and execution

The six manifest-bound packets contain 48 prespecified purchase-feasible and
12 negative pairs: 60 scenarios, 120 arm assignments. Each of the 16 known
products appears with three private preferred colors in the positive panel.
Public queries and private shopper needs are separate from evaluator gold.
Only the limited assigned public query can enter storefront search. Shoppers
received screenshots and visible controls, with no correct product IDs or
purchase scripts.

Both arms used the installed Codex CLI with gpt-6.1-sol, low reasoning effort,
24 actions, 120 seconds, quantity one and concurrency one. Each session used
a fresh headed Chrome process and storage context at 1280×960. Pair order was
randomized with a frozen seed, with balanced within-pair treatment order.
Catalog, prices, agent settings, model weights and storefront sources were
held fixed. The 670 source pins, three runtime identities and actual model
health were checked around execution. The study used the post-checkout v11 UI.

The main run started at 08:42:31 UTC and the last session finished at
09:57:53 UTC on 2026-10-18. The original plan was committed before outcomes.
The complete raw cohort was committed as e4559a7 before audit adjustments.
There was no outcome-based retuning, replacement, resizing or early stopping.

The first five randomized pairs remained included. Each arm had 5/5 intent
success and four checkouts in that pilot. In the remaining 55 pairs, intent
success was 53/55 generic and 52/55 local-model; checkout among the remaining
44 prespecified feasible assignments was 42/44 versus 43/44. The full result
and both sensitivity panels are saved in [slide-metrics.json](slide-metrics.json).

## Independent verification and failures

All 480 captured artifact byte lengths and SHA-256 hashes verified. All 120
trace archives passed ZIP CRC checks, all 120 final screenshot signatures
passed, and all 120 recorder-consistency checks passed. All contexts and tabs
were distinct, shared storage was false, and initial empty carts were verified.
Large traces and intermediate screenshots remain local; their hashes and the
normalized observations are preserved.

The shopper completed checkout only through the real storefront control.
The local receiver validated catalog identities and prices and persisted an
immutable receipt. Saved POST, saved GET, visible receipt identity and fresh
independent GET all matched for 93 orders. The server index retains all 120
assigned rows, with zero readback errors. Post-run verification issued GETs
only and created no orders.

A fresh reviewer received all 24 negative assignments with arm labels, run
identities and URLs removed. Generated wording could still reveal the
experience; this was label blinding. The fixed rubric accepted 22 justified
abstentions and rejected two outcomes: the local lunch-price timeout and the
local bottle refusal in holdout-v4-09, whose saved visible evidence did not
establish the claimed price impossibility. The separate audit lowered exactly
one previous expected-outcome credit. Raw reports, sessions, formal eligibility
and checkout receipts were unchanged.

The full raw-failure census found four original failures:

- Generic holdout-v2-05: a second shopper decision started after the sneakers
  were visible and did not finish before timeout. The underlying stall cause
  is unknown.
- Local-model holdout-v2-09: repeated alternating scrolling ended in timeout
  without an abstention. Assigned-query readiness had already completed.
- Both holdout-v3-07 arms: the private hard goal required a visibly identified
  Canvas travel bag. Frozen gold/catalog included that category, but the
  frozen product card did not render it. Both shoppers saw Weekender duffel,
  $84 and Sand and explicitly cited the missing canvas fact. This is a
  task-visibility/evaluator mismatch. The frozen feasibility label, denominator
  and failure scores remain unchanged; it limits absolute-rate interpretation.

See [failure-review.json](failure-review.json) and
[negative-review-v1/review.json](negative-review-v1/review.json). Future UI or
packet repairs are separate revisions and cannot repair this cohort's scores.

## Readiness and controls

Local-model completion readiness has 61 verified request observations across
58 sessions: median 322.1 ms, p95 383.0 ms. All 61 require positive generated
fields; none is a zero-field completion. There were 1,199 accepted and two
withheld fields out of 1,201 required fields. These are browser after-paint
opportunities, not physical paint, image transfer, native GPU timing or shopper
task time. Requests are clustered within sessions.

Fifty-eight local sessions recorded complete status and two recorded no
assigned-query readiness observation; none recorded model failed status.
Thirty-one sessions recorded an ever-observed last-safe-result flag. All 31
later recorded complete status. The frozen UI displays that status while a new
rank is pending and the recorder retains the observation; the flags do not
establish 31 service outages. The full counts, clocks and field scopes are in
[readiness-report.json](readiness-report.json).

After the main cohort, four generic A/A sessions and four reversed-order
repeat sessions ran in independently isolated browsers. All eight completed
server-confirmed, visible checkouts. A/A labels each had 2/2 purchases and
equal merchandise totals; both repeated treatment arms had 2/2 purchases and
equal totals. These are small descriptive checks over two original scenario
clusters per group. They are not pooled with the 120 main assignments. See
[controls-summary.json](../store_shopper_controls_v2/controls-summary.json).

All shopper browsers were confirmed closed at 10:04:43 UTC; model health was
ready with zero queued requests and all 670 pins still matched. The model
owner then gracefully retired that worker with a separate 927-tensor integrity
receipt. No later browser or native-model request was made by this task.

## Scope and artifacts

These are new scenario wordings/private preferences for a catalog already
known to training. Products repeat across cases. Bootstrap intervals are
conditional on this hand-authored panel and do not make the cases independent
customers or products. A lower bootstrap bound of zero or an all-tie subset
does not establish a positive effect or equivalence. The treatment combines
ranking, copy, layout and timing; mechanisms are not isolated. Human conversion
lift, retention, satisfaction and real revenue were not measured.

The original [20-attempt development suite](../store_shopper_sol_full_v1)
remains unchanged and separate. Its earlier no-uplift result is preserved;
the two cohorts are not a controlled before/after comparison. Evaluation
personas, gold and traces remain held out of training.

- [Full metrics and paired intervals](shopper-metrics.json)
- [Server receipt index](server-outcomes.json)
- [Artifact integrity audit](artifact-audit.json)
- [Frozen plan](cohort-plan.json) and [browser idle receipt](browser-idle-receipt.json)
- [Final file hashes and export scope](final-artifact-index.json)
- [Focused harness setup and commands](../../../../apps/runner/SHOPPER_COHORT.md)

Independent metric verification passed 1,879 assertions, including the actual
API exporter's unchanged CPU reanalysis reproducing the saved metrics exactly.
See [metrics-verification.json](metrics-verification.json).

The hosted archive is verified under immutable report ID
`store-shopper-checkout-cohort-v2-final`, experiment ID
`actual-store-checkout-cohort-v2`, digest
`31afcba97aa08096b805960f6b6d5f8871fe5fde258467e9ff6ac451d9394172`.
Fresh strict `source=tiger` reads matched the prepared report and the actual
frontend parser accepted it. A separate readback at 10:11:58 UTC confirmed the
metrics and slide projection still equal the local final files. Controls and
raw browser logs were excluded from the hosted payload. See
[archive-verification.json](archive-verification.json) and the preserved
[presentation response](presentation-readback.json).
