The first genuine Qwen public-context candidate fails the frozen offline gate. The byte-frozen adjudicator exited 2 as expected: 4/32 primary first choices, 7/32 novel relevant first choices and 15/32 strong-positive first-three results, versus required 20/24/28. Source budget, typed constraints, selected variants/colors and canonical image bindings had zero violations. The known development history diagnostic had 5/10 raw-rank demotions and 10/10 novel relevant results after purchase, failing the demotion minimum of eight. All four strict baseline comparisons pass, but those small improvements do not satisfy the quality minima. Serving acceptance remains pending and this candidate should not become the default.

The adjudicator exactly replayed 208 learned score rows and 64 simple baseline rows. It verified the checkpoint/reload, all eight changed tensors against seeded initialization, native 432-row roster, 864 serialized feature hashes, source/annotation/catalog pins, public-only demo cache and native 927-integrity/shutdown receipts. It did not recreate native frozen-tensor hashes. Original checkpoint SHA-256 is `ad641a62da8845e5a904d9a326fc67eaf32d739792f09f73436761d834b9163a`.

Independent saved-checkpoint TRAIN replay found substantial remaining underfit:

| TRAIN subset | Grade-3 first choice | Novel grade ≥2 first choice | Strong positive in first three | Owned first choice |
|---|---:|---:|---:|---:|
| Original 160 | 45/160 | 151/160 | 159/160 | 0/160 |
| Contrast 160 | 145/160 | 145/160 | 160/160 | 8/160 |
| All 320 | 190/320 | 296/320 | 319/320 | 8/320 |

All 320 target/profile IDs, public ownership facts, original public profile contents and consumer product/variant targets aligned exactly. There are 80 TRAIN groups of four sessions. Sixteen contrast labels have two legitimate grade-3 alternatives; all other rows have one. Grade-3 metrics accept any such alternative. Partial relevance makes the aggregate results look stronger than the original requests' primary choices.

The final saved checkpoint has product cross-entropy 1.26259 and variant cross-entropy 1.30781, weighted by 0.5, giving joint loss 1.91650. The corresponding joint target-entropy lower bound is 0.96254, leaving excess 0.95396. The recorded loss was still falling from 2.62241 at step 226 to 1.99561 before update 250. This supports further optimization tests before a capacity claim; the lower bound need not be attainable with shared parameters and constraints.

Raw vectors have a strong common component. The TRAIN context mean has norm 0.98821; average off-diagonal cosine is 0.97648. Pairs sharing a grade-3 label average 0.97810 versus 0.97633 for other pairs; original/contrast pairs average 0.99456. Catalog mean norm is 0.99747 and pair cosine 0.99461. Centering exposes some class separation: context shared-label cosine 0.06708 versus −0.00706. These are descriptive geometry measurements, not proof that centering improves quality. The learned context and catalog centered effective ranks are 3.63 and 4.45. TRAIN original/contrast raw-rank demotions occur in 102/160 pairs.

Evidence is saved under `data/ranker/pretrained/persona-qwen-independent-audit-v1-attempt-1` and `persona-qwen-train-independent-audit-v1-attempt-2`. The companion JSON inventories their hashes and the new diagnostic source. Three fabricated CPU tests passed in 0.429 seconds, including multi-positive target handling, label-order invariance, bad joins/ownership, and shared-positive geometry. The initial diagnostic attempt is preserved separately with its source and failure receipt: the reviewer initially assumed every TRAIN row had a unique grade-3 target, which the actual annotations correctly rejected.

No optimizer, native model, browser or Git action ran in this review. Original model/data/adjudicator files were preserved. The former fresh 32 cases are now inspected development replay; no further unseen-evaluation claim is valid for them.
