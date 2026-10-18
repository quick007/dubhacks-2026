# Known DEV error analysis: learned adapter versus frozen backbone

This is post-adjudication analysis of the same 32 requests, four replacements and ten known history pairs. Labels, thresholds, product order and exclusions are unchanged. Counts below describe saved choices; they do not establish why training changed them or implicate model capacity.

| Canonical comparison, 32 cases | Raw | Fused |
|---|---:|---:|
| Base correct → learned wrong | 5 | 4 |
| Base wrong → learned correct | 2 | 2 |
| Both correct | 15 | 15 |
| Both wrong | 10 | 11 |
| Same top choice | 23 | 25 |
| Different top choice | 9 | 7 |

The two gains are shared: sneakers after the duffel purchase, and towel for the immediate swimming need. Four losses are also shared: lamp twice and organizer twice become mug. Raw additionally changes a correct pack to notebook. Two raw choices change while both remain wrong; one such change remains in the fused pipeline. All complete per-case records and fixed counts are saved in `data/ranker/pretrained/persona-choice-dev-error-analysis-v1-attempt-1/ANALYSIS.json`.

## Priority and completed-purchase behavior

Raw base primary accuracy is 9/16 before and 11/16 after the controlled purchase; learned is 8/16 and 9/16. Both sides are correct in four of sixteen pairs for either model. The learned model introduces three pairs where both sides are wrong: pack/lamp, blanket/organizer and lamp/bottle. Fused base is 9/16 before and 10/16 after; learned remains 8/16 and 9/16.

Six raw learned errors choose the explicitly less urgent secondary job before the first gap is filled; the base has the same six. Three learned errors repeat the newly purchased tote, lunchbox or headphones. The base additionally repeats the purchased lamp. Changing that owned lamp to mug reduces the owned-repeat count from four to three, but still misses the requested bottle; it does not demonstrate useful complementarity in that case.

Across the sixteen current matched pairs, both models demote the acquired product in eight pairs. On the separate, previously inspected ten-pair history diagnostic, raw base has eight demotions and five novel relevant choices afterwards; learned has six and seven. Fused base has eight/five and learned seven/seven. The behavior is mixed: these results do not support a blanket claim that either model learned or ignored purchase history.

All four explicit replacements are unchanged between learned and base, raw and fused: cup, headphones and shoes are correct; the filled notebook request incorrectly selects towel. The public request explicitly says every notebook page is full and fresh writing pages take priority over drying off after the pool. The error cannot be credited as a correct replacement without regrading.

## Concentrated transfer toward mug

Raw mug choices increase from two to eight of 32; fused mug choices increase from two to seven. Mug's frozen TRAIN target prior is only 0.03795, while organizer is the largest at 0.125. The observed mug drift therefore is not simply repetition of the highest global TRAIN prior. It remains an association, not an identified training cause.

The following raw margins are the annotated primary score minus the best *eligible* competitor. They use the same canonical product set and item guards, not the maximum over ineligible products. Fused margins use a different score scale and must not be compared numerically with these logits.

| Lost primary | Public situation | Base margin | Learned margin |
|---|---|---:|---:|
| Lamp | “The corner where I read after dark needs a tabletop light before I spend money on the rest of my list.” | 8.375 | −2.875 |
| Organizer | Blanket already purchased; “Loose charging leads … a little zipped case to keep them together … afterwards.” | 7.4375 | −0.250 |
| Lamp | Pack already purchased; “A light on the small table where I read at night … after I have dealt with carrying those things.” | 5.750 | −0.375 |
| Organizer | “The cables and little adapters … need a zipped pouch first”; a cup is explicitly secondary. | 3.375 | −2.500 |
| Pack, raw only | Towel already purchased; hands-free bag for daytime books remains the next job. | 0.3125 | −1.8125 |

These losses are not uniformly devoid of plausible utility. The pack/lamp case has a mug cart event, graded 1; the organizer/mug case explicitly gives the cup secondary utility, graded 2. Those facts explain partial relevance without changing the stated priority target. In the other shared lamp/organizer losses, mug has no positive annotation or directly stated missing function. Generic home/desk associations might be plausible, but do not override the explicit need order under the frozen judgments. No cases were excluded or regraded.

## Constraints, facts and the router

All annotated primary products remain eligible under the budgets and typed constraints. Both models have zero product-budget or typed-eligibility violations. Each is primary-correct on two of the four hard-Navy cases. This rules out an ineligible target or different guard result as the explanation for the counted primary losses; it says nothing about selected variant, image or generated-copy quality, which this offline output does not contain.

For these authored generic-query contexts, `pSpecific` ranges from 0.0183 to 0.2611, median 0.09365. These are descriptive continuous probabilities, not a newly selected routing threshold. The four common losses exist in the raw comparison before query fusion, so a router change alone cannot explain the adapter's negative transfer. Fusion modestly changes relevance/coverage but retains primary accuracy 17/32.

The next diagnosis should reconcile these public need/priority and purchase failures with the separately prepared TRAIN objective/convergence and label audit. This report adds no experiment, capacity conclusion, serving exception or promotion recommendation.
