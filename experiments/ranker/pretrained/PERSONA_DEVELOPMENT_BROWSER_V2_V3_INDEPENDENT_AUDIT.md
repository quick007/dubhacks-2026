# Public persona demo: independent semantic audit

The ten loaded public profile cases have usable, factual short copy and source-matched images. Personalization remains uneven: four profiles receive an already-owned product, and notebook, organizer and lunchbox appear in every six-card selection. This is a known-development demo, not a model quality promotion.

I reviewed V2 attempts 2–11 against the serialized public requests, public ten-person source and 16-product catalog, then verified the same ten selected payloads and the functional outcomes in the separate V3 run. All ten screenshots were inspected. The [JSON evidence](PERSONA_DEVELOPMENT_BROWSER_V2_V3_INDEPENDENT_AUDIT.json) contains request IDs, all 60 card texts and source facts, variant/image bindings and individual receipt hashes. No future evaluation labels or outputs were read for this audit.

| Check | Fixed denominator and result |
| --- | --- |
| Public requests and selected order | 10/10 exact; rank/first-three/complete order agrees |
| Card identity, selected color, price and stated budget | 60/60 source-supported |
| Short title, badge and body | 180/180 factual and coherent; no unsupported claim or unfinished text found |
| Generated section headings | 20/20 safe neutral wording |
| Images | 60/60 source-bound and decoded; 30 first-three images visually reviewed |
| Loaded view | First three whole cards visible in all 10 screenshots; second row partly below fold |
| V2 versus V3 profile payload | 10/10 same six-card semantic payload |

The service accepted 200 generated fields. `ShopView` displays the first section heading and all 18 card fields per profile: 190 text strings in total. Ten second-section headings were generated but unused. Six decoded cards do not mean six cards were simultaneously wholly visible; additional DOM catalog cards use canonical metadata. These are loaded, pre-scrolled views, with no cold page-top, physical-paint or independently recomputed latency claim.

| Public profile | First selection and material semantic note |
| --- | --- |
| Bob | Blanket: coherent sofa complement; no owned mug repeat. |
| Maya | Blanket; owned notebook repeats at rank 5. |
| Priya | Notebook then organizer: writing/loose-pen needs are plausible; owned pack avoided. |
| Lena | Jacket; viewed daypack absent from six and notebook rank 2 weakly motivated. First three Navy variants miss the soft green/natural palette despite offered Forest/Olive/Sage alternatives. |
| June | Headphones; owned notebook repeats at rank 2. |
| Grace | Bottle; meal-prep lunchbox is only rank 4, while sunglasses rank 3 lack a stated sunny-use need. |
| Sam | Duffel and pouch fit travel; owned bottle avoided. |
| Ana | Notebook; owned mug repeats at rank 3. |
| Rosa | Bottle; owned organizer repeats at rank 3, towel absent from six despite swim context, sweatshirt lacks a stated cold-use need. |
| Felix | Sneakers fit walking; notebook and lamp at ranks 2–3 have weak public support. |

Owned repeats affect 4/10 profiles and 4/60 cards, including 3/30 first-three cards. No replacement request supports these repeats. The three universal products occupy 30/60 slots, with seven distinct top choices. These are ranking/usefulness misses; the Navy selection is a soft-preference miss, not a hard-constraint violation or a false product fact.

Copy is clear but repetitive: titles name the item and color, badges say “Color selected,” and bodies say “Shown in Color at $N.00.” It supplies no personal functional explanation. Receipts identify fresh learned word-slot prediction from cached public fact encodings, one feedforward pass and zero autoregressive steps; the copy head explicitly trained all 48 catalog bindings. Repeated wording does not establish runtime templates, and this evidence does not establish unseen-catalog generation. Large checkpoint hashes were not independently recomputed by this auditor.

The separate V3 functional receipts pass rapid Bob→Maya→Bob stale-response checks and per-person cart/completed-checkout receipt invalidation. The middle request abort is observed in the browser; immediate GPU interruption is not established. The local demo checkout had already settled before switching (`checkoutRaceExercised: false`), so an in-flight checkout race was not exercised. It is simulated checkout, not payment or conversion evidence. V2’s original aborted-body probe failure remains preserved; V3 is a new run, not a retrospective regrade.

The current model remains `qualityGatePassed: false`, `modelPromotion: false`: known-development primary quality is 17/32 versus frozen base 20/32. Ten factual demo views do not overturn that result. Raw files, UI, models and data were unchanged; this auditor used no browser, GPU or Git.

Evidence: V2 report SHA-256 `6487f4c9c9ba792cd291b795225d7a05557e26f19c30dc0cd5989b240ef4a749`; V3 `5f2135156560c83d0c82d077af39f8c78b3c924787e6809c1bc5cc256e46520f`; accompanying JSON `84b756ad75e1f240d38088529d32a628148559567247d0b0c8e4dae052777cd6`.
