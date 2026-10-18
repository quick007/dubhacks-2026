# First independent Chrome shopper pair

Two independent agents, each with no parent history, navigated separate actual
Chrome tabs for shopper-pair-02. Both received the same shopper need and public
query, no correct product ID or evaluator gold, and no other shopper outcome.
The generic arm ran first; the local-model arm followed, matching the declared
AB order and seed2026101802. No source/catalog files were available to either
shopper. Both observed an empty initial cart and completed only fake demo
checkout with the $36 Graphite Everyday steel bottle, quantity1.

Generic used10 input actions and confirmed within a38.456s measured upper
bound. Local used9 input actions and estimated~35s; its first-browser-action
clock was not recorded, so this is not an exact paired elapsed-time estimate.
Do not infer speed, conversion lift or statistical advantage from this one
exploratory pair. The generic storefront already uses deterministic search
heuristics; it is not an empty/random baseline.

The local agent observed rank before choosing, first3copy before inspection,
and fullcopy by Add to cart. Its actual submitted query was
`a bottle for drinking water under $40`. Requesta516a44d-1cd6-4dae-b962-6f537fcac17b
reached browser rank opportunity421.9ms, first3opportunity5539.8ms with10/10
accepted native fields, and full6opportunity6515.4ms with19/20 accepted fields.
These are React DOMcommit+two-frame opportunities, not images/physicalpaint.
The later fullcopywithholding and this newquery are preserved separately
from the original frozenfive-request baseline-preparation readout; neither
changes that denominator retroactively.

## Evidence and protocol deviations

Each arm has original resultJSON and action notes. The local raw readiness
DOMattribute is saved. Native CUA screenshots exist in tool transcripts, but
no durable screenshot files or Playwright traces were provided by that API.
Do not create hashes for nonexistent screenshot/trace artifacts. Choice facts
can be checked independently against the16-product catalog, but durable
choice observations remain shopper-agent reports, not independentDOMcaptures.

These runs began while the compatible formal contract was being completed.
Actual provider-default viewport was used rather than the later packet's
1280x960 setting; exact browser version, full storage-context isolation and
per-action clocks were unavailable. Each tab had fresh React/cart state but
shared origin storage, which was never cleared. These are exploratory actual
browser smoke results; they do not satisfy a fully controlled20-run evaluation.
The fixedten-pair suite must preserve these unknowns/deviations and retain
unattempted runs. Re-run under the finalized protocol for comparative claims.
