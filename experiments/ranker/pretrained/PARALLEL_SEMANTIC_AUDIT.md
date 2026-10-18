# Independent semantic audit of measured parallel outputs

This CPU-only review reads the actual saved benchmark fields, token arrays, strict-validation results and matching case plans. It performs no native inference and changes no validator or generated wording. Accepted fields remain those reported by the unchanged validator; descriptive reviews of globally rejected fields do not rescue pages. These are previously inspected synthetic cases, not human conversion evidence or an untouched generalization evaluation.

## Six-GiB main comparison

Source: `PARALLEL_SEMANTIC_RESULTS.json`, SHA-256 `f3c4743f1d02524bdc726ed48854d73e3dcd02fe0826efdb24ceeed9679a12af`, backed by ignored `parallel-semantic-256-compact-v1/benchmark.json` and its fixed twelve-case plan. This is the compact-heading prompt at native prefill size 256. There are 55 executed attempts: 54 completed and one memory-guard failure. The six-card case remains unattempted in every arm. Every arm keeps its twelve-case denominator.

| Arm | Completed / failed / unattempted | Valid pages / fixed cases | Accepted titles / headings / generated badges | Full-attempt median | Valid-page median |
| --- | --- | --- | --- | ---: | ---: |
| Manifest | 11 / 0 / 1 | 7/12 | 10 / 9 / 10 | 3,354.0 ms | 3,294.7 ms, n=7 |
| Independent sequential | 11 / 0 / 1 | 2/12 | 5 / 2 / 0 | 4,230.2 ms | 3,609.3 ms, n=2 |
| Width two | 11 / 0 / 1 | 2/12 | 5 / 2 / 0 | 3,868.8 ms | 3,361.9 ms, n=2 |
| Width three | 11 / 0 / 1 | 2/12 | 5 / 2 / 0 | 3,736.1 ms | 3,322.6 ms, n=2 |
| Width six | 10 / 1 / 1 | 2/12 | 5 / 2 / 0 | 3,587.3 ms | 3,349.3 ms, n=2 |

Accepted titles/headings can occur on rejected pages, so field totals are not page counts. `NONE` is a permitted badge omission and does not count as a generated badge. The width-six full-attempt median retains the failed attempt's 5,476.0 ms latency; it does not assign that failure a valid page.

The only two accepted compact cases are Grove and Cedar, already reviewed in `HEADING_EXPERIMENT.md`. Their output remains coherent, preserves selected product identity, and uses supported tags. Width two's Cedar heading changes from `Compact and durable pack` to `Compact and Durable Choice`; both are coherent and factual. The seven accepted manifest pages were inspected for bound identity, supported attributes, coherence and raw protocol. No accepted lexical false positive was found in this main result. This does not eliminate the validator gaps demonstrated by the fabricated CPU-only probe in `FACTORIZED_OUTPUT_AUDIT.md`.

On the two shared accepted cases, manifest's median is 3,203.8 ms; every compact arm is slower (sequential 3,609.3; widths two/three/six 3,361.9/3,322.6/3,349.3 ms). Native width three improves latency versus independent sequential on those two cases, but there is no measured quality/latency win over manifest. Each has only three independent fields, so these valid-page results do not demonstrate six-stream or multi-card usable-page scaling.

## Remaining mechanisms in the main comparison

Across each sequential/width-two/width-three arm there are thirteen title rows, thirteen badge rows and eleven heading rows. All thirteen titles stop on EOS. Ten headings stop and one exhausts its sixteen-token cap. Seven badge rows stop and six exhaust their twelve/eight-token caps. Width six has ten completed one-card attempts: ten stopped titles, nine stopped headings plus one truncated heading, six stopped badges plus four truncated badges; the failed three-card attempt contributes no completed row evidence.

Representative actual invalid outputs:

```text
S1: Water-resistant organizer daypack
T1: Meadow 17L Daypack — water-resistant
B1: B1: Meadow 17L Day
```

The heading is factual and stopped. The title copies an approved title and the badge repeats its label and truncates. Widths two/three/six use `backpack` instead of `daypack` in this heading; that change is factual and coherent. It is not itself evidence of a cache defect.

```text
S1: Roomy and water-resistant Summit
T1: Summit 23L Daypack — roomy
B1: B1: Summit 23L Day
```

The heading includes the partial product name Summit despite the intervention's name prohibition. Similar violations include `Water-resistant roomy Field Daypack`, `Field 20L Daypack — water-resistant`, `Cedar 26L Daypack with laptop sleeve and roomy`, `Compact and Durable Cedar`, and `Compact and water-resistant Grove 23L Daypack`. These are six distinct one-card cases. Full-name-only `catalogNamesFound` misses partial names Summit/Field/Cedar; its empty list must not be treated as proof of name-free headings. The long Cedar heading also truncates.

Some such names and attributes are factual for the selected card. Their heading rejection partly reflects the existing section-proxy lexicon and the explicit name/number-free heading protocol, rather than proof of wrong selected-item facts. That does not justify weakening the validator. The explicit name anchors in the prompt improved the first three inspected cases but also encouraged name copying elsewhere.

The no-history/no-preferences control emits a correctly stopped `NONE` in every compact arm. Its page is rejected because the heading says `Compact and Durable Cedar`, not because badge omission is wrong. The Pine case has a coherent stopped heading, `Compact and durable daily pack`, but its badge copies the identity and truncates. Global non-stop rejection masks otherwise factual stopped titles/headings in these cases; report row mechanics separately from official accepted-field counts.

The completed three-card arms produce:

```text
S1: Durable Compact Travel Packs
T1: Grove 23L Daypack — compact
B1: NONE
T2: Cedar 26L Daypack — durable compact
B2: Cedar 26L
T3: River 26L Daypack — durable everyday
B3: River 26L
```

This heading is a substantive factual failure. Selected tags are Grove: water-resistant/compact; Cedar: durable/compact; River: durable/organizer. The shared intersection is empty: Grove lacks durable and River lacks compact. Asserting both at section level is invalid. The two identity-only badges also truncate at eight tokens and show no supported attribute. Strict rejection is necessary. Prompting with the shared-tag intersection did not reliably prevent importing non-universal attributes from member names or the common prefix.

Exact sequential token/finish/prompt parity is 8/11 width-two completed attempts, 10/11 width-three, and 9/10 width-six. The report retains the mismatches rather than replacing strings. Other observed variation, such as a width-two Summit title ending `roomy detail-view`, is shopper-copy/protocol drift requiring review; it appears on an already rejected page and does not create an accepted false positive.

## Cost and memory of the main comparison

| Arm | Saved emitted tokens, excluding EOS | Sum of full-attempt wall times |
| --- | ---: | ---: |
| Manifest | 380 | 38,778.8 ms |
| Sequential | 372 | 50,737.8 ms |
| Width two | 374 | 46,810.0 ms |
| Width three | 371 | 45,123.9 ms |
| Width six | 305 from completed attempts; failed rows unavailable | 42,705.0 ms including failure |

The main run's total wall time is 231,489.8 ms, with 2,582.0 ms resident model load recorded separately. These costs retain rejected attempts. Faster rejected output cannot count as a usable-page speedup.

The three-card width-six attempt failed the adjustable repository process guard: peak MLX allocation 6,495,112,752 bytes versus the declared six-GiB limit 6,442,450,944 bytes. The main run stopped there, preserving the failed attempt and all six-card unattempted arms. This is not reported as an operating-system OOM or a user/environment quota. A separately declared seven-GiB scale supplement cannot erase that failure, backfill the main report, or change any twelve-case denominator.

## Separate seven-GiB scale supplement

The parent subsequently completed the separately declared two-case supplement for the fixed three-card and six-card cases. Source: `PARALLEL_SEMANTIC_SCALE_7G_RESULTS.json`, SHA-256 `ce9d171ad3f973bd4c93891cd08f2e58d25604c5a4f201dd944cd4bc2a72fddf`, backed by ignored `parallel-semantic-256-compact-scale-7g-v1/benchmark.json` and its own case plan. All ten attempts completed without abort. Its denominator is **two**, its process guard is **seven GiB**, and its results are not merged into the main twelve-case report.

| Arm | Valid pages / separate fixed cases | Three-card / six-card completion | Emitted tokens, three / six cards |
| --- | --- | --- | --- |
| Manifest | 1/2 | 5,131.9 / 8,565.3 ms | 66 / 140 |
| Independent sequential | 0/2 | 7,464.8 / 13,664.7 ms | 66 / 133 |
| Width two | 0/2 | 6,848.3 / 12,427.0 ms | 66 / 132 |
| Width three | 0/2 | 6,389.3 / 11,904.4 ms | 66 / 132 |
| Width six | 0/2 | 5,894.2 / 11,034.6 ms | 66 / 132 |

Manifest has nine accepted titles, one accepted heading and seven generated badges across the two attempts; all compact arms have zero accepted titles/headings/generated badges because their unfinished badges trigger global rejection. The manifest's three-card accepted page has a generic, coherent heading and supported product-bound titles/badges. Its six-card page correctly rejects both headings: they assert compact/water-resistant or durable/compact despite each bound section having no universal tag. The six-card manifest titles and badges themselves preserve selected identities and supported facts. There is no accepted lexical false positive in this supplement.

The compact three-card output is the same invalid output quoted above in every width. The six-card sequential raw output is:

```text
S1: Durable Compact Travel Packs
S2: Three Everyday Travel Bags
T1: Grove 23L Daypack — compact
B1: NONE
T2: Cedar 26L Daypack — durable compact
B2: Cedar 26L
T3: River 26L Daypack — durable everyday
B3: River 26L
T4: Field 14L Daypack — everyday travel companion
B4: Field 14L
T5: Ridge 23L Daypack — durable lightweight
B5: NONE
T6: River 14L Daypack — durable compact
B6: River 14L
```

Section one binds Grove water-resistant/compact, Cedar durable/compact and River durable/organizer; section two binds Field lightweight/roomy, Ridge lightweight/roomy and River durable/compact. Both intersections are empty. The first heading falsely generalizes attributes in every width. Sequential/width-two second heading is coherent and factually counts three members, but the spelled word `Three` is outside the restrictive framing lexicon; that is a lexical-policy rejection rather than evidence of an incorrect count. Widths three/six change the second heading to `Three Compact Travel Options`, introducing a substantive false universal compact claim as well.

There is also a card-level factual failure: Ridge 23L has only lightweight/roomy tags. Its title asserts unsupported durable in all widths: `durable lightweight` in sequential/two/six and `durable everyday` in width three. This cannot be repaired merely by expanding the badge budget or shortening headings. Titles preserve product names, but identity preservation alone does not ensure their attributed facts belong to the selected item.

Each three-card compact arm has one stopped heading, three stopped titles, one stopped badge and two length-capped badges. Each six-card compact arm has two stopped headings, six stopped titles, two stopped badges and four length-capped badges. `B4: Field 14L` violates the no-support `NONE` requirement and truncates, whereas `B5: NONE` is individually correct; the global `correctNone=false` covers both and must not be interpreted as showing the fifth raw badge is wrong. These per-row distinctions preserve strict page rejection.

The native runtime records maximum simultaneous emitting rows of one/two/three/six at the respective widths, in both scale cases. This establishes actual six-stream execution in the supplement; it does not establish six-card usable-page acceleration. The three-card strings reproduce exactly across widths. Six-card token/finish equality differs from sequential at every batched width, with all mismatch evidence retained. The observed lexical/semantic differences above are reviewed directly rather than treated as equivalent solely because page acceptance is unchanged.

Peak allocation is 6,940,200,480 bytes against the separate 7,516,192,768-byte seven-GiB process guard. Overall wall time is 92,851.5 ms, with 2,727.1 ms resident load recorded separately. More memory allows the scale attempts to finish, but every compact scale page remains rejected. The main six-GiB failure and its unattempted six-card entries remain unchanged.

## Implication for the next experiment

Keep manifest as the measured quality baseline. The compact prompt is useful evidence that one source of truncation can be removed without changing the card rows, but its broader quality is inadequate for promotion. A next bounded quality experiment should target the independent badge prompt/protocol on the same fixed cases, explicitly prohibiting product names and generated labels inside the badge value, keeping caps and strict validation fixed. Preserve genuine generated wording and raw failures; do not turn a badge repair into a claim that the false universal headings or Ridge's unsupported durable title are solved. Those substantive fact-binding failures remain separate release blockers for multi-card factorization.
