# Store demo v1: qualitative receipt audit

Reviewed all **136 non-null native fields** supplied for display in the nine
completed request receipts: 15 headings, 43 titles, 43 badges and 35 bodies.
Eight other bodies were withheld out of 144 attempted fields. First-three
payload text was unchanged in the corresponding complete events, so repeated
stream snapshots are counted once. This is an exhaustive offline LLM review of
the supplied payloads, not independent human review or browser-paint inspection.
The companion JSON preserves all accepted quotes, bound facts, receipt hashes,
source pointers, withheld native text and per-query judgments.

## Ranking and hard-policy results

| Public request | Observations | Result |
|---|---:|---|
| `a bottle for drinking water` | 2 | Bottle first, appropriate basic function. |
| `a light for my bedside table` | 2 | Portable table light first, appropriate basic function; no special bedside safety claim inferred. |
| `a notebook for quick notes` | 2 | **Miss: organizer first**, followed by lamp and lunchbox. Preserve this failure. |
| `Browse the store` | 1 | Open browse; no unique correct first choice assigned. |
| `a waterproof rain jacket` | 1 | Jacket only, consistent with declared `Waterproof hooded shell`; final event reports one eligible product and five shortfall. This is a deterministic hard-policy result. |
| `Trail mug under $20` | 1 | Correct empty result: the exact named mug is $24 and cannot satisfy the ceiling. Final event reports zero eligible and six shortfall. |

Basic semantic query matches are 4/6 request occurrences, or 2/3 unique queries.
These repeats are not independent unseen test cases. The waterproof and exact
identity/budget cases are reported separately from learned semantic matching.
Final counts above come from complete events, not the earlier base-head ranking
stage that precedes extra hard/exact policy.

## Accepted text and practical-screen gap

**129/136 accepted fields are intelligible and source-compatible.** No directly
wrong product name, selected color or item price was found; no unsupported
performance, stock or shipping claim, incomplete value or redundant `selected`
construction was displayed among the accepted fields.

The remaining **seven accepted bodies** all repeat:

> This collection of Portable table light in Ivory has an item price of $64.00.

Portable table light, Ivory and $64 agree with the bound item. However,
`collection of` introduces an unsupported scope and could suggest multiple
items or a bundle, despite the later `item price` wording. Flag this for a
prospective practical-screen change; it is an ambiguous scope claim, not a
proven wrong dollar amount. It appears in all seven six-card payloads: the two
bottle requests, two light requests, two notebook requests and open browse.
For example, receipt `002a042d-5344-4946-9664-6642285e84e0`, complete event
`/events/2/sections/0/cards/1/bodyCopy`, contains this exact accepted text.

All 15 headings are `Your selected collection`. Titles/badges and most other
bodies accurately repeat selected public name/color/price facts. Their modest
editorial variety is apparent; they do not demonstrate query-specific generated
wording. The receipt correctly discloses that the shopper query influences
ranking, while generation receives a neutral browsing request and selected
public facts. Hidden personas do not enter this copy conditioning.

The practical screen withheld eight caption-style bodies such as
`Steel lunch box with Olive band selected`, `Market canvas tote shown in Olive`
and `Everyday round sunglasses with Tortoise selected`, recording
`unfinished-body`. They remained null in display payloads; no replacement or
template was inserted. Those phrases are readable as captions, but are separate
from the accepted-body denominator and are not silently promoted by this audit.

## Scope and preservation

Sources are the fixed nine `data/ranker/pretrained/store-demo-v1/requests/*/receipt.json`
files listed with SHA-256 in `STORE_DEMO_V1_QUALITATIVE_AUDIT.json`. No native
model, GPU, tokenizer, frontend, Git operation, old strict gate or v1 receipt
was changed. The seven ambiguous bodies and notebook ranking misses remain
preserved. Images/colors are deterministic metadata bindings; browser readiness,
visual rendering, rich profile reasoning and conversion are outside this audit.
