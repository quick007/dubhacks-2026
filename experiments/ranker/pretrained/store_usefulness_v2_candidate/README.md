# Equivalent-dollar usefulness candidate

This is a **CPU-only offline score**, version `store-usefulness-offline-candidate.v2`, of the six archived native comparison arms. It was designed after reading the v1 failures. It is ready for review before any future native run; it is not a preregistered benchmark, serving change or observed readiness result. Original output, gates and timestamps remain unchanged.

The [scorer](../store_usefulness_v2.py) reuses the exact hash-pinned v1 practical screen and native-receipt checks through AST extraction of CPU functions/constants. It imports no model runtime. For body fields only, a private scoring string can replace a numerically correct `36 dollars` or `36.00 dollars` span with the equivalent dollar-sign representation. `Decimal` compares the amount to the selected public item's price first. The generated wording and original token/EOS receipt remain intact; scorer output contains the original text and normalization span evidence, not replacement storefront copy.

Existing `$36` and `$36.00` already pass. Wrong amounts, duplicate price mentions, extra numbers, foreign currencies, malformed/signed/grouped amounts, unsupported claims, identity/color errors, unfinished text and ambiguous collection pricing still reject. `USD` and spelled-out numbers are deliberately unsupported. Original protocol/length, EOS/cap and repetition failures cannot disappear during normalization. No hidden persona or private shopper truth is used.

| Archived system | Offline title/body core first3 | Offline title/body core full6 | Accepted fields, including optional headings/badges |
| --- | --- | --- | --- |
| A: frozen ordinary AR | 3/3 | 3/3 | 48/60 |
| B: adapted hybrid | 0/3 | 0/3 | 54/60 |

The only newly accepted fields are A's **18 numerically correct body prices**. B is unchanged: three ambiguous collection/item-price bodies and three unfinished caption bodies remain rejected. All six payloads still fail an all-20-fields requirement. Optional headings/badges are reported separately from the directional title/body core; permissive abstention is not counted as fresh generated copy. Missing or failed arms retain their fixed denominator.

The v1 frozen first3/full6 usable timestamps remain null for all six arms. This report has **no v2 readiness measurements**, and supplies no useful-latency or speedup ratio. The original numeric-dollar formatting issue cannot be used as evidence that A produced false prices. Conversely, this lexical score does not repair A's misleading section headings or B's notebook selection miss. The [independent qualitative audit](../STORE_AR_COMPARISON_V1_INDEPENDENT_AUDIT.md) remains necessary: it finds public-query top1 relevance A3/3 versus B2/3 and distinguishes plausible wording from unverified subjective claims. All three queries are development examples; none tests budgets, stock or sizes.

The [report](report.json) preserves all 120 raw fields, rejection reasons, source hashes and fixed six-arm denominator. Run and test with ordinary CPU Python:

```sh
python3 -B -m unittest experiments.ranker.pretrained.test_store_usefulness_v2 -v
python3 -B -m experiments.ranker.pretrained.store_usefulness_v2 \
  --output /tmp/store-usefulness-v2-new-candidate.json
```

Thirteen tests passed. They cover equivalent and wrong prices, decimal source swaps, extra numbers, foreign/malformed currencies, retained identity/color/claim/unfinished/collection checks, native-receipt failure, unchanged nonprice decisions, missing-arm denominator behavior and archive byte immutability. The scorer refuses to overwrite an existing report. No live engine, source manifest, README or historical result was edited.
